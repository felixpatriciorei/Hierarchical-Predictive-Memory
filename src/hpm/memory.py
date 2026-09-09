from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .differentiable_topk import differentiable_topk_mask, sparsemax

MEMORY_CONTROLS = {"normal", "shuffle_values", "shuffled_values", "random_keys", "corrupt_values", "no_retrieval"}
READ_MODES = {"hard_topk", "ste_topk", "sparsemax"}


def gather_token_positions(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather ``x[b, positions[b, ...]]`` for batched positions."""

    batch = torch.arange(x.size(0), device=x.device)
    flat_positions = positions.reshape(positions.size(0), -1)
    gathered = x[batch[:, None], flat_positions]
    return gathered.reshape(*positions.shape, x.size(-1))


def retrieve_topk(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_values: torch.Tensor,
    memory_mask: torch.Tensor,
    top_k: int,
    use_null_slot: bool = False,
    null_score: Optional[torch.Tensor] = None,
    select_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Brute-force top-k memory retrieval.

    When ``use_null_slot`` is enabled, a zero-valued null memory is appended
    to the retrieved pool before the softmax. This prevents top-k retrieval
    from being forced to assign probability mass to bad matches when no real
    slot is useful.

    ``select_bias``, if given, is an additive (..., m) log-bias applied on
    top of the hard ``memory_mask`` -- e.g. the differentiable writer-selection
    bias from ``differentiable_topk_bias``. It lets a loss computed on
    ``retrieved``/``scores`` (downstream of this softmax) route gradient back
    into whatever produced ``select_bias``, which the boolean ``memory_mask``
    alone can never do. ``memory_mask`` still hard-excludes structurally
    invalid slots (e.g. padding) regardless of ``select_bias``.
    """

    scores = torch.einsum("bd,bmd->bm", query, memory_keys) / math.sqrt(query.size(-1))
    if select_bias is not None:
        scores = scores + select_bias
    scores = scores.masked_fill(~memory_mask, -1.0e9)
    k = min(top_k, memory_keys.size(1))
    top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
    top_values = memory_values.gather(
        1,
        top_indices.unsqueeze(-1).expand(-1, -1, memory_values.size(-1)),
    )

    null_weight = query.new_zeros(query.size(0))
    if use_null_slot:
        if null_score is None:
            null_score = query.new_zeros(())
        null_scores = null_score.to(device=query.device, dtype=query.dtype).expand(query.size(0), 1)
        null_values = query.new_zeros(query.size(0), 1, memory_values.size(-1))
        top_scores = torch.cat([top_scores, null_scores], dim=1)
        top_values = torch.cat([top_values, null_values], dim=1)

    weights = torch.softmax(top_scores, dim=-1)
    retrieved = torch.sum(weights.unsqueeze(-1) * top_values, dim=1)
    if use_null_slot:
        null_weight = weights[:, -1]
    return retrieved, scores, top_indices, weights, null_weight


def _soft_topk_membership_on_valid_slots(
    scores: torch.Tensor,
    memory_mask: torch.Tensor,
    k: int,
    *,
    eps: float,
    n_iters: int,
) -> torch.Tensor:
    """Return an OT-relaxed top-``k`` membership over only valid slots.

    ``differentiable_topk_mask`` requires ``k < n``. Episodic batches can
    contain padding and can legitimately have exactly ``k`` valid slots in a
    row, so applying it directly to the ``-1e9``-padded score matrix is both
    mathematically wrong (padding would be part of the transport problem) and
    numerically fragile.

    The normal episodic benchmark has a fixed number of valid slots per
    batch row.  In that case this helper gathers the valid domain for every
    row and runs one batched Sinkhorn solve.  That is mathematically identical
    to independent per-row solves but avoids launching ``batch_size`` tiny GPU
    kernels per Sinkhorn iteration.  A conservative row-wise fallback remains
    for genuinely ragged batches, where there is no one rectangular transport
    domain to solve over.
    """

    if scores.ndim != 2 or memory_mask.shape != scores.shape:
        raise ValueError("scores and memory_mask must both have shape (batch, slots)")

    valid_counts = memory_mask.sum(dim=-1)
    if bool((valid_counts < k).any().item()):
        min_valid = int(valid_counts.min().item())
        raise ValueError(
            "ste_topk requires at least top_k valid episodic slots in every batch row; "
            f"got {min_valid} valid slots for top_k={k}"
        )

    # Fast path: a rectangular active candidate domain.  This includes both
    # fully populated exact-slot memory and batches whose padding locations
    # differ by row but have the same active count.
    if bool(valid_counts.eq(valid_counts[0]).all().item()):
        valid_count = int(valid_counts[0].item())
        # Valid positions sort before padding. The order among valid slots is
        # irrelevant to OT top-k and gets scattered back below.
        positions = torch.argsort((~memory_mask).to(torch.int64), dim=-1)
        valid_positions = positions[:, :valid_count]
        active_scores = scores.gather(1, valid_positions)
        if valid_count == k:
            active_membership = torch.ones_like(active_scores)
        else:
            active_membership = differentiable_topk_mask(
                active_scores,
                k=k,
                eps=eps,
                n_iters=n_iters,
                hard=False,
            )
        return torch.zeros_like(scores).scatter(1, valid_positions, active_membership)

    memberships = []
    for row_scores, row_mask in zip(scores.unbind(0), memory_mask.unbind(0)):
        valid_indices = row_mask.nonzero(as_tuple=False).squeeze(-1)
        valid_count = valid_indices.numel()
        if valid_count == k:
            # There is no selection boundary to relax: every valid slot is
            # already in the read set.  Keeping this branch independent of
            # scores gives the correct zero selection-gradient in that case.
            valid_membership = torch.ones_like(row_scores[valid_indices])
        else:
            valid_membership = differentiable_topk_mask(
                row_scores[valid_indices],
                k=k,
                eps=eps,
                n_iters=n_iters,
                hard=False,
            )
        memberships.append(torch.zeros_like(row_scores).scatter(0, valid_indices, valid_membership))

    return torch.stack(memberships, dim=0)


def retrieve_ste_topk(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_values: torch.Tensor,
    memory_mask: torch.Tensor,
    top_k: int,
    use_null_slot: bool = False,
    null_score: Optional[torch.Tensor] = None,
    select_bias: Optional[torch.Tensor] = None,
    ste_eps: float = 0.3,
    ste_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact-forward hard top-k retrieval with a relaxed selection gradient.

    Forward semantics, returned ``scores``, ``top_indices``, ``weights``, and
    null-slot behavior are exactly those of :func:`retrieve_topk`.  Therefore
    this mode can be evaluated and deployed as ordinary finite-slot episodic
    retrieval.  Its difference is only the derivative of ``retrieved``:

    ``hard_read + (soft_OT_read - stopgrad(soft_OT_read))``.

    ``soft_OT_read`` is the average of the entropic-OT relaxed top-k
    membership over valid slots.  This supplies a gradient to a near-miss
    slot across the hard cutoff, while the ordinary hard branch retains the
    correct within-selected-set softmax gradient.  It is a straight-through
    estimator, hence biased; it is deliberately opt-in and not a claim of an
    unbiased gradient for discrete retrieval.

    With a null slot, its normal hard-path gradient remains intact.  The
    relaxed selection term is scaled by the detached hard non-null mass so a
    read that forwards almost entirely to null does not receive a large
    artificial pressure to select a memory slot in the backward surrogate.

    ``select_bias`` is intentionally rejected.  The learned full-candidate
    writer produces that bias with its own straight-through selection
    estimator; composing the two estimators is a different, untested method.
    R1 therefore has a fixed candidate set by construction.
    """

    if ste_eps <= 0.0:
        raise ValueError(f"ste_eps must be positive, got {ste_eps}")
    if ste_iters <= 0:
        raise ValueError(f"ste_iters must be positive, got {ste_iters}")
    if select_bias is not None:
        raise ValueError(
            "ste_topk is currently validated only for a fixed episodic candidate set and "
            "cannot be combined with select_bias from the learned full-candidate writer. "
            "Use hard_topk for that path until the composed two-estimator objective has its "
            "own controlled experiment."
        )

    scores = torch.einsum("bd,bmd->bm", query, memory_keys) / math.sqrt(query.size(-1))
    scores = scores.masked_fill(~memory_mask, -1.0e9)
    k = min(top_k, memory_keys.size(1))
    top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
    top_values = memory_values.gather(
        1,
        top_indices.unsqueeze(-1).expand(-1, -1, memory_values.size(-1)),
    )

    null_weight = query.new_zeros(query.size(0))
    if use_null_slot:
        if null_score is None:
            null_score = query.new_zeros(())
        null_scores = null_score.to(device=query.device, dtype=query.dtype).expand(query.size(0), 1)
        null_values = query.new_zeros(query.size(0), 1, memory_values.size(-1))
        top_scores = torch.cat([top_scores, null_scores], dim=1)
        top_values = torch.cat([top_values, null_values], dim=1)

    weights = torch.softmax(top_scores, dim=-1)
    hard_retrieved = torch.sum(weights.unsqueeze(-1) * top_values, dim=1)
    if use_null_slot:
        null_weight = weights[:, -1]

    # The relaxation only defines a training-time surrogate derivative.  In
    # torch.no_grad() evaluation/inference, its forward contribution cancels
    # identically, so skip the O(batch * Sinkhorn iterations) work and return
    # the already-computed exact hard read directly.
    if not torch.is_grad_enabled():
        return hard_retrieved, scores, top_indices, weights, null_weight

    soft_membership = _soft_topk_membership_on_valid_slots(
        scores,
        memory_mask,
        k,
        eps=ste_eps,
        n_iters=ste_iters,
    )
    soft_retrieved = torch.sum(
        (soft_membership / float(k)).unsqueeze(-1) * memory_values,
        dim=1,
    )
    if use_null_slot:
        soft_retrieved = soft_retrieved * (1.0 - null_weight.detach()).unsqueeze(-1)

    # Straight-through identity: exact hard read in forward, OT membership
    # gradient in backward.  The subtraction cancels bit-for-bit because its
    # right operand is a detached view of the same floating-point values.
    retrieved = hard_retrieved + (soft_retrieved - soft_retrieved.detach())
    return retrieved, scores, top_indices, weights, null_weight


def retrieve_sparsemax(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_values: torch.Tensor,
    memory_mask: torch.Tensor,
    top_k: int,
    use_null_slot: bool = False,
    null_score: Optional[torch.Tensor] = None,
    select_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous, exactly-sparse alternative to :func:`retrieve_topk`.

    Not a drop-in in the strictest sense -- ``weights`` has a different
    shape, see the NOT a fully drop-in replacement note below -- but same
    function signature and never calls ``torch.topk`` to decide which slots
    participate. Instead:
      1. scores over ALL valid slots go through ``sparsemax`` (see
         ``differentiable_topk.py``), whose support (which slots get
         nonzero weight) is not pinned to a fixed ``top_k`` the way
         ``retrieve_topk``'s gather is. A near-tied runner-up that a fixed
         ``top_k`` would drop from the computation graph entirely (and so
         get exactly zero gradient, structurally) instead ends up with real
         weight and a real, closed-form gradient here. Slots sparsemax
         itself excludes (weight exactly 0) still get exactly zero local
         gradient -- correctly so, not a shortcoming: that is the true
         derivative of a threshold an infinitesimal score change hasn't
         crossed yet, and is inherent to any exactly-sparse operator,
         ``retrieve_topk`` included.
      2. ``retrieved`` is the sparsemax-weighted sum over ALL slots directly
         (no gather-then-softmax two-step), so the memory-value contraction
         and the slot-selection decision share one differentiable operator.

    ``top_indices`` is still returned (for logging/diagnostics parity with
    ``retrieve_topk`` -- e.g. ``writer_missed_fact_rate``-style checks), taken
    as the ``top_k`` highest-weight slots, but it is NOT used to gather
    values: it plays no role in computing ``retrieved`` or ``weights``.

    NOT a fully drop-in replacement: ``weights`` here has shape
    ``(batch, n_slots [+1 if use_null_slot])`` -- one weight per valid slot,
    since sparsemax sparsity is emergent rather than a fixed count -- whereas
    ``retrieve_topk`` returns one weight per gathered slot. Callers must not
    assume sparsemax weights align elementwise with ``top_indices``.
    """

    scores = torch.einsum("bd,bmd->bm", query, memory_keys) / math.sqrt(query.size(-1))
    if select_bias is not None:
        scores = scores + select_bias
    scores = scores.masked_fill(~memory_mask, -1.0e9)

    null_weight = query.new_zeros(query.size(0))
    if use_null_slot:
        if null_score is None:
            null_score = query.new_zeros(())
        null_scores = null_score.to(device=query.device, dtype=query.dtype).expand(query.size(0), 1)
        null_values = query.new_zeros(query.size(0), 1, memory_values.size(-1))
        all_scores = torch.cat([scores, null_scores], dim=1)
        all_values = torch.cat([memory_values, null_values], dim=1)
    else:
        all_scores = scores
        all_values = memory_values

    weights = sparsemax(all_scores)
    retrieved = torch.sum(weights.unsqueeze(-1) * all_values, dim=1)
    if use_null_slot:
        null_weight = weights[:, -1]

    k = min(top_k, scores.size(-1))
    with torch.no_grad():
        top_indices = torch.topk(scores, k=k, dim=-1).indices

    return retrieved, scores, top_indices, weights, null_weight


def apply_memory_control(
    memory_keys: torch.Tensor,
    memory_values: torch.Tensor,
    memory_control: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if memory_control not in MEMORY_CONTROLS:
        raise ValueError(f"unknown memory_control: {memory_control}")
    if memory_control in {"shuffle_values", "shuffled_values"}:
        return memory_keys, torch.roll(memory_values, shifts=1, dims=1)
    if memory_control == "corrupt_values":
        rolled = torch.roll(memory_values, shifts=1, dims=1)
        corrupt = torch.arange(memory_values.size(1), device=memory_values.device) % 2 == 1
        mixed = torch.where(corrupt[None, :, None], rolled, memory_values)
        return memory_keys, mixed
    if memory_control == "random_keys":
        random_keys = torch.randn_like(memory_keys)
        return random_keys, memory_values
    return memory_keys, memory_values


class EpisodicMemory(nn.Module):
    """Tiny in-sequence episodic memory with brute-force retrieval.

    This v0 deliberately uses token-level fact representations: the memory key
    is projected from the declared key token representation and the memory
    value is projected from the declared value token representation. That keeps
    the experiment focused on whether an explicit memory path helps long-gap
    exact recall, not on learning a write detector.
    """

    def __init__(
        self,
        d_model: int,
        use_null_slot: bool = False,
        null_score_init: float = 0.0,
        read_mode: str = "hard_topk",
        read_ste_eps: float = 0.3,
        read_ste_iters: int = 50,
    ):
        super().__init__()
        if read_mode not in READ_MODES:
            raise ValueError(f"unknown read_mode: {read_mode} (expected one of {sorted(READ_MODES)})")
        if read_ste_eps <= 0.0:
            raise ValueError(f"read_ste_eps must be positive, got {read_ste_eps}")
        if read_ste_iters <= 0:
            raise ValueError(f"read_ste_iters must be positive, got {read_ste_iters}")
        self.use_null_slot = use_null_slot
        self.read_mode = read_mode
        self.read_ste_eps = float(read_ste_eps)
        self.read_ste_iters = int(read_ste_iters)
        self.null_score = nn.Parameter(torch.tensor(float(null_score_init)))
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.hop_query_proj = nn.Linear(d_model, d_model, bias=False)
        self._init_identity()

    def _init_identity(self) -> None:
        for layer in (self.query_proj, self.key_proj, self.value_proj, self.hop_query_proj):
            nn.init.eye_(layer.weight)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        memory_token_positions: torch.Tensor,
        memory_mask: torch.Tensor,
        query_key_positions: torch.Tensor,
        top_k: int = 1,
        num_hops: int = 1,
        hop_positive_indices: Optional[torch.Tensor] = None,
        memory_control: str = "normal",
        select_bias: Optional[torch.Tensor] = None,
        read_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        active_read_mode = read_mode if read_mode is not None else self.read_mode
        if active_read_mode not in READ_MODES:
            raise ValueError(f"unknown read_mode: {active_read_mode} (expected one of {sorted(READ_MODES)})")
        if active_read_mode == "hard_topk":
            retrieve_fn = retrieve_topk
        elif active_read_mode == "ste_topk":
            retrieve_fn = retrieve_ste_topk
        else:
            retrieve_fn = retrieve_sparsemax
        memory_repr = gather_token_positions(token_embeddings, memory_token_positions)
        memory_key_source = memory_repr[:, :, 0, :]
        memory_value_source = memory_repr[:, :, 1, :]
        memory_keys = self.key_proj(memory_key_source)
        memory_values = self.value_proj(memory_value_source)
        query_source = gather_token_positions(token_embeddings, query_key_positions)

        if memory_control == "no_retrieval":
            return torch.zeros_like(query_source), {"retrieval_loss": token_embeddings.new_zeros(())}
        memory_keys, memory_values = apply_memory_control(memory_keys, memory_values, memory_control)

        hop_infos: List[Dict[str, torch.Tensor]] = []
        retrieval_losses = []

        retrieved = torch.zeros_like(query_source)
        for hop in range(num_hops):
            query = self.query_proj(query_source)
            retrieve_kwargs = {
                "query": query,
                "memory_keys": memory_keys,
                "memory_values": memory_values,
                "memory_mask": memory_mask,
                "top_k": top_k,
                "use_null_slot": self.use_null_slot,
                "null_score": self.null_score,
                "select_bias": select_bias,
            }
            if active_read_mode == "ste_topk":
                retrieve_kwargs["ste_eps"] = self.read_ste_eps
                retrieve_kwargs["ste_iters"] = self.read_ste_iters
            retrieved, scores, top_indices, weights, null_weight = retrieve_fn(**retrieve_kwargs)
            hop_infos.append({
                "scores": scores,
                "top_indices": top_indices,
                "weights": weights,
                "null_weight": null_weight,
            })

            if hop_positive_indices is not None and hop < hop_positive_indices.size(1):
                positive = hop_positive_indices[:, hop]
                valid = positive >= 0
                if valid.any():
                    # retrieval_loss must not use `scores` (from retrieve_topk)
                    # directly: that tensor already has select_bias added, and
                    # select_bias subtracts 1e4 from every valid-but-unselected
                    # candidate -- including the oracle's true-fact position
                    # whenever the writer hasn't selected it yet (common early
                    # in training; see writer_missed_fact_rate in the raw sweep
                    # logs). Cross-entropy against a target whose own logit has
                    # been artificially suppressed by an unrelated discrete gate
                    # is not a proper scoring rule there -- it produced ~1e4
                    # magnitude loss spikes the instant
                    # learned_writer_teacher_forcing turned off, identically for
                    # jepa_bias_on/off (JEPA was never involved).
                    #
                    # But plain raw (select_bias-free) scores are also wrong:
                    # they cut the ONLY gradient path from a real downstream
                    # loss into the writer at top_k=1, where softmax-of-one
                    # makes answer_loss alone exactly zero-gradient into
                    # select_bias (see test_answer_loss_alone_still_zero_at_top_k_1).
                    # That gradient path was the entire point of building the
                    # differentiable full-candidate path in the first place --
                    # losing it is a regression, not a fix (caught by
                    # test_writer_gradient_from_real_training_loss).
                    #
                    # Fix: a gradient-only bias. `select_bias - select_bias.detach()`
                    # is exactly 0 in the forward pass (identical values, so the
                    # subtraction cancels exactly -- no floating-point residue,
                    # no discrete jump, target's logit is never suppressed) but
                    # its gradient w.r.t. select_bias's inputs (and therefore
                    # w.r.t. the writer's scorer) is untouched, since only the
                    # second copy is detached. This gets both properties at
                    # once: retrieval_loss's VALUE measures pure content
                    # matching (sound cross-entropy, no spike), while its
                    # GRADIENT still reaches the writer exactly as the
                    # differentiable path intended.
                    raw_scores = torch.einsum("bd,bmd->bm", query, memory_keys) / math.sqrt(query.size(-1))
                    raw_scores = raw_scores.masked_fill(~memory_mask, -1.0e9)
                    if select_bias is not None:
                        grad_only_bias = select_bias - select_bias.detach()
                        raw_scores = raw_scores + grad_only_bias
                    retrieval_losses.append(F.cross_entropy(raw_scores[valid], positive[valid]))
            query_source = self.hop_query_proj(retrieved)

        if retrieval_losses:
            retrieval_loss = torch.stack(retrieval_losses).mean()
        else:
            retrieval_loss = token_embeddings.new_zeros(())

        final_info = hop_infos[-1]
        info = {
            "scores": final_info["scores"],
            "top_indices": final_info["top_indices"],
            "weights": final_info["weights"],
            "null_weight": final_info.get("null_weight", token_embeddings.new_zeros(token_embeddings.size(0))),
            "retrieval_loss": retrieval_loss,
        }
        return retrieved, info


class HebbianMemory(nn.Module):
    """Dense associative memory baseline.

    Memory is built in declaration order as:
        M = decay * M + eta * k * v^T
    Retrieval uses:
        r = M^T q
    """

    def __init__(self, d_model: int, decay: float = 0.9, eta: float = 1.0):
        super().__init__()
        self.decay = decay
        self.eta = eta
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.hop_query_proj = nn.Linear(d_model, d_model, bias=False)
        self._init_identity()

    def _init_identity(self) -> None:
        for layer in (self.query_proj, self.key_proj, self.value_proj, self.hop_query_proj):
            nn.init.eye_(layer.weight)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        memory_token_positions: torch.Tensor,
        memory_mask: torch.Tensor,
        query_key_positions: torch.Tensor,
        top_k: int = 1,
        num_hops: int = 1,
        hop_positive_indices: Optional[torch.Tensor] = None,
        memory_control: str = "normal",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory_repr = gather_token_positions(token_embeddings, memory_token_positions)
        memory_keys = self.key_proj(memory_repr[:, :, 0, :])
        memory_values = self.value_proj(memory_repr[:, :, 1, :])
        query_source = gather_token_positions(token_embeddings, query_key_positions)

        if memory_control == "no_retrieval":
            return torch.zeros_like(query_source), {"retrieval_loss": token_embeddings.new_zeros(())}
        memory_keys, memory_values = apply_memory_control(memory_keys, memory_values, memory_control)

        bsz, slots, d_model = memory_keys.shape
        matrix = token_embeddings.new_zeros(bsz, d_model, d_model)
        valid = memory_mask.float()
        for slot in range(slots):
            outer = torch.einsum("bd,be->bde", memory_keys[:, slot], memory_values[:, slot])
            matrix = self.decay * matrix + self.eta * outer * valid[:, slot, None, None]

        hop_infos: List[Dict[str, torch.Tensor]] = []
        retrieval_losses = []
        retrieved = torch.zeros_like(query_source)
        for hop in range(num_hops):
            query = self.query_proj(query_source)
            retrieved = torch.einsum("bde,bd->be", matrix, query) / math.sqrt(d_model)
            scores = torch.einsum("bd,bmd->bm", query, memory_keys) / math.sqrt(d_model)
            scores = scores.masked_fill(~memory_mask, -1.0e9)
            k = min(top_k, slots)
            top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
            weights = torch.softmax(top_scores, dim=-1)
            hop_infos.append({"scores": scores, "top_indices": top_indices, "weights": weights})

            if hop_positive_indices is not None and hop < hop_positive_indices.size(1):
                positive = hop_positive_indices[:, hop]
                valid_positive = positive >= 0
                if valid_positive.any():
                    retrieval_losses.append(F.cross_entropy(scores[valid_positive], positive[valid_positive]))
            query_source = self.hop_query_proj(retrieved)

        if retrieval_losses:
            retrieval_loss = torch.stack(retrieval_losses).mean()
        else:
            retrieval_loss = token_embeddings.new_zeros(())

        final_info = hop_infos[-1]
        return retrieved, {
            "scores": final_info["scores"],
            "top_indices": final_info["top_indices"],
            "weights": final_info["weights"],
            "retrieval_loss": retrieval_loss,
        }
