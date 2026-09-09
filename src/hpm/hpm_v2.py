from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

import math


@dataclass
class BlockMemoryConfig:
    """Configuration for HPM v2 memory modules.

    These modules are intentionally small and kernel-free so they run on a single
    RTX 4060 / Kaggle T4 before any custom CUDA/Mamba kernels exist.
    """

    d_model: int = 128
    block_size: int = 128
    dropout: float = 0.0
    fast_decay_init: float = 0.95


class BlockwiseSelectiveRecurrentState(nn.Module):
    """Mamba-inspired selective state-space recurrence at block granularity.

    This is a compact research implementation of selective state updates, not
    the official Mamba/Mamba-2 kernel or an implementation-equivalence claim.

    h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
    y_t = C_t . h_t + D * x_t

    A is a learned per-channel, per-state-dim decay rate, stored in log-space
    and negated on use (A = -exp(A_log)) so the system is stable by
    construction. delta, B, and C are all computed fresh from the input at
    every block.

    delta_proj is initialized the way the real Mamba implementation does it:
    bias set so delta starts in a small, controlled range (dt_min..dt_max),
    weight zeroed so delta starts input-INdependent (a gentle, near-fixed
    decay) and only becomes selective as training shapes the weight away
    from zero. Skipping this (using PyTorch's default init instead) makes
    delta an uncontrolled ~O(1) value at init, which combined with A ranging
    to -state_dim collapses exp(delta*A) to ~0 in a single block -- the state
    gets wiped every time, not selectively updated. That was the bug in the
    first version of this class.
    """

    def __init__(self, d_model: int, block_size: int = 128, dropout: float = 0.0,
                 state_dim: int = 16, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.d_model = int(d_model)
        self.block_size = int(block_size)
        self.state_dim = int(state_dim)

        self.ln = nn.LayerNorm(d_model)

        a_init = torch.arange(1, self.state_dim + 1, dtype=torch.float32).log()
        self.A_log = nn.Parameter(a_init.unsqueeze(0).repeat(self.d_model, 1))

        self.delta_proj = nn.Linear(d_model, d_model)
        # Mamba-style dt init: small controlled range, weight zeroed so delta
        # starts near-constant (stable) and only becomes input-dependent
        # once training moves the weight away from zero.
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.delta_proj.bias.copy_(inv_softplus_dt)
            self.delta_proj.weight.zero_()

        self.B_proj = nn.Linear(d_model, self.state_dim)
        self.C_proj = nn.Linear(d_model, self.state_dim)
        self.D = nn.Parameter(torch.ones(d_model))

        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, seq_len, d_model]")
        bsz, seq_len, d_model = hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        hidden = self.ln(hidden)
        state = hidden.new_zeros(bsz, d_model, self.state_dim)
        outputs = hidden.new_zeros(bsz, seq_len, d_model)

        A = -torch.exp(self.A_log)

        for start in range(0, seq_len, self.block_size):
            end = min(seq_len, start + self.block_size)
            summary = hidden[:, start:end, :].mean(dim=1)

            C_t = self.C_proj(summary)
            readout = (state * C_t.unsqueeze(1)).sum(dim=-1) + self.D * summary
            outputs[:, start:end, :] = readout[:, None, :]

            delta = F.softplus(self.delta_proj(summary))
            B_t = self.B_proj(summary)
            A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = delta.unsqueeze(-1) * B_t.unsqueeze(1)
            state = A_bar * state + B_bar * summary.unsqueeze(-1)

        return self.out(outputs)


class FastWeightBlockMemory(nn.Module):
    """Small differentiable fast-weight memory path.

    The update combines a delta-rule correction with input-dependent
    channel-wise decay, inspired by the modern DeltaNet / Gated DeltaNet / KDA
    family. It is intentionally simpler than current Gated DeltaNet-2: erase
    and write strength remain coupled through one scalar ``beta``. This lets
    the module forget more precisely
    (each memory channel can decay at its own rate) and overwrite stale
    associations at a given key instead of only ever accumulating on top of them:

        M <- (I - beta * k k^T) @ Diag(alpha) @ M + beta * outer(k, v)
        read_t = q_t @ M

    Diag(alpha) is a learned per-channel forget gate (one rate per memory
    dimension, not one rate for the whole matrix). The (I - beta k k^T) term is
    the delta-rule correction: before writing the new (k, v) association it
    first removes whatever the memory currently outputs for that same key, so a
    repeated/similar key overwrites cleanly instead of just piling noise on top
    of the old value. beta plays the role of both the write strength and the
    correction strength. That coupling is a known expressivity limitation,
    not a claim of exact equivalence to KDA or Gated DeltaNet-2.

    The update happens once per completed block. Current block tokens read the
    memory state produced by earlier blocks only.
    """

    def __init__(self, d_model: int, block_size: int = 128, decay_init: float = 0.95):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.d_model = int(d_model)
        self.block_size = int(block_size)
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.eta = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        # Input-dependent per-channel decay gate (this is the actual
        # "selective"/KDA part -- matching the real paper's g, which is a
        # function of the input at every step, not a fixed constant). Weight
        # is zero-initialized and bias set to logit(decay_init), so training
        # starts out identical to the old fixed-decay behavior and only
        # learns to deviate from it if doing so actually helps -- this keeps
        # early training stable instead of starting from noisy random gating.
        decay_init = min(max(float(decay_init), 1e-4), 0.9999)
        init_bias = torch.logit(torch.full((d_model,), decay_init))
        self.decay_proj = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.decay_proj.weight.zero_()
            self.decay_proj.bias.copy_(init_bias)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, seq_len, d_model]")
        bsz, seq_len, d_model = hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        h = self.ln(hidden)
        memory = h.new_zeros(bsz, d_model, d_model)
        reads = h.new_zeros(bsz, seq_len, d_model)

        for start in range(0, seq_len, self.block_size):
            end = min(seq_len, start + self.block_size)
            q = F.normalize(self.q_proj(h[:, start:end, :]), dim=-1)
            reads[:, start:end, :] = torch.bmm(q, memory)

            summary = h[:, start:end, :].mean(dim=1)
            key = F.normalize(self.k_proj(summary), dim=-1)
            value = self.v_proj(summary)
            beta = self.eta(summary).view(bsz, 1, 1)
            # decay is now a function of THIS block's content, not a fixed
            # per-channel constant -- (bsz, d_model, 1) so it broadcasts over
            # the memory's row (key) dimension per-example.
            decay = torch.sigmoid(self.decay_proj(summary)).unsqueeze(-1)

            # Diag(alpha) @ M : per-channel, input-dependent decay.
            decayed = decay * memory
            # (I - beta * k k^T) @ decayed : delta-rule correction. First
            # compute what the decayed memory currently associates with this
            # key (k^T @ decayed), then subtract beta times that projected
            # back along k, which removes the stale component before writing.
            existing_for_key = torch.bmm(key.unsqueeze(1), decayed)  # (bsz, 1, d_model)
            correction = beta * torch.bmm(key.unsqueeze(2), existing_for_key)  # (bsz, d_model, d_model)
            write = beta * torch.bmm(key.unsqueeze(2), value.unsqueeze(1))  # (bsz, d_model, d_model)

            memory = decayed - correction + write

        return self.out(reads)


class HpmV2PathRouter(nn.Module):
    """Route among local, selective recurrent, fast-weight, and episodic paths."""

    def __init__(self, d_model: int, num_paths: int = 4, logit_clamp: Optional[float] = None):
        super().__init__()
        self.num_paths = int(num_paths)
        self.proj = nn.Linear(num_paths * d_model, num_paths)
        if logit_clamp is not None and (not math.isfinite(float(logit_clamp)) or float(logit_clamp) <= 0.0):
            raise ValueError("logit_clamp must be a finite positive value or None")
        self.logit_clamp = logit_clamp
        """If set, forward() smoothly bounds |logit| <= this value via
        C * tanh(raw / C) before softmax.

        This is an architectural forward bound, not a training incentive and
        not a guarantee of optimizer recoverability.

        tanh bounds the EFFECTIVE pre-softmax logits unconditionally: no
        matter how large the raw projection grows,

            |effective| = |C * tanh(raw/C)| < C.

        This prevents exact probability collapse at the softmax output. It
        does *not* by itself guarantee optimization recoverability. The
        derivative from effective logit back to raw projection is

            d effective / d raw = sech(raw / C)^2,

        which tends to zero as |raw| grows. A router may therefore remain
        strictly inside the probability simplex while its raw projection is
        effectively gradient-locked behind the tanh clamp. The router-health
        diagnostics expose both raw/effective logits and this local clamp
        Jacobian so those failure modes cannot be conflated.

        With num_paths=4 and C=3, the *forward* worst-case one-path dominance
        is bounded at ~99.3%; that statement is intentionally narrower than
        a claim that a non-negligible parameter gradient always remains.

        Default None = off, exactly reproducing prior (unclamped) behavior.
        """

    def _route(
        self,
        *paths: torch.Tensor,
        fixed_weights: Optional[torch.Tensor] = None,
        contribution_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Internal routing implementation with non-invasive diagnostics.

        The public :meth:`forward` method deliberately keeps the historical
        three-tensor return contract. ``route_with_diagnostics`` exposes the
        fourth diagnostics object for model-level telemetry without breaking
        direct callers/tests that already unpack ``(mixed, weights, logits)``.
        """
        if len(paths) != self.num_paths:
            raise ValueError(f"expected {self.num_paths} paths, got {len(paths)}")

        raw_logits = self.proj(torch.cat(list(paths), dim=-1))
        if self.logit_clamp is not None:
            scaled = raw_logits / self.logit_clamp
            tanh_scaled = torch.tanh(scaled)
            logits = self.logit_clamp * tanh_scaled
            # Exact elementwise derivative d(C*tanh(raw/C))/d(raw).
            clamp_jacobian = 1.0 - tanh_scaled.square()
        else:
            logits = raw_logits
            clamp_jacobian = torch.ones_like(raw_logits)

        learned_weights = torch.softmax(logits, dim=-1)
        if fixed_weights is None:
            weights = learned_weights
        else:
            if fixed_weights.shape != learned_weights.shape:
                raise ValueError(
                    "fixed_weights must have shape [batch, seq_len, num_paths] matching router logits; "
                    f"got {tuple(fixed_weights.shape)}, expected {tuple(learned_weights.shape)}"
                )
            if not torch.isfinite(fixed_weights).all() or (fixed_weights < 0.0).any():
                raise ValueError("fixed_weights must be finite and non-negative")
            if not torch.allclose(
                fixed_weights.sum(dim=-1),
                torch.ones_like(fixed_weights[..., 0]),
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                raise ValueError("fixed_weights must sum to one across paths")
            weights = fixed_weights.to(dtype=learned_weights.dtype)

        if contribution_mask is None:
            mask = learned_weights.new_ones(self.num_paths)
        else:
            mask = contribution_mask.to(device=learned_weights.device, dtype=learned_weights.dtype)
            try:
                mask = torch.broadcast_to(mask, learned_weights.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "contribution_mask must broadcast to [batch, seq_len, num_paths]; "
                    f"got {tuple(contribution_mask.shape)}, expected compatible with {tuple(learned_weights.shape)}"
                ) from exc
            if not torch.isfinite(mask).all() or (mask < 0.0).any() or (mask > 1.0).any():
                raise ValueError("contribution_mask must be finite and lie in [0, 1]")

        mixed = sum(weights[..., i : i + 1] * mask[..., i : i + 1] * path for i, path in enumerate(paths))
        diagnostics = {
            "raw_logits": raw_logits,
            "effective_logits": logits,
            "clamp_jacobian": clamp_jacobian,
        }
        return mixed, weights, logits, diagnostics

    def forward(
        self,
        *paths: torch.Tensor,
        fixed_weights: Optional[torch.Tensor] = None,
        contribution_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix path states, optionally under a frozen-gate contribution intervention.

        ``fixed_weights`` and ``contribution_mask`` exist for evaluation
        only.  They make it possible to ask whether a path's contribution was
        load-bearing *under the gate the unablated model actually selected*:

        ``sum_i fixed_weights_i * contribution_mask_i * path_i``.

        The mask is deliberately applied after softmax and never
        renormalized.  Renormalizing (or computing the gate from already
        ablated path states) would let the router compensate for the missing
        path and would turn a necessity test into a different architecture.
        Both arguments default to ``None``, preserving established training
        and inference behavior exactly.
        """
        mixed, weights, logits, _ = self._route(
            *paths,
            fixed_weights=fixed_weights,
            contribution_mask=contribution_mask,
        )
        # Also return the pre-softmax logits (not just the normalized weights).
        # Raw/effective logits are returned for router-health telemetry. Sharp
        # routing can represent useful specialization, so concentration alone
        # is not classified as failure; the diagnostics distinguish probability
        # concentration, optimization lock, and functional path dependence.
        return mixed, weights, logits

    def route_with_diagnostics(
        self,
        *paths: torch.Tensor,
        fixed_weights: Optional[torch.Tensor] = None,
        contribution_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Route exactly as :meth:`forward`, plus raw-router diagnostics.

        Diagnostics are tensors on the same graph as the ordinary forward:

        ``raw_logits``
            Linear projection output before any clamp.
        ``effective_logits``
            Actual pre-softmax logits (identical to the third ordinary return).
        ``clamp_jacobian``
            Elementwise local derivative from raw to effective logits. It is
            exactly one with no clamp and ``sech(raw/C)^2`` with the tanh clamp.

        This method intentionally changes no routing, losses, or intervention
        semantics; it only makes previously hidden optimization state visible.
        """
        return self._route(
            *paths,
            fixed_weights=fixed_weights,
            contribution_mask=contribution_mask,
        )


class JepaLiteAuxiliary(nn.Module):
    """JEPA-lite auxiliary objective for chunk representations.

    This is deliberately auxiliary. It should never replace the exact episodic
    memory path for key-value facts. It predicts the latent representation of a
    future block from a context block using stop-gradient targets.
    """

    def __init__(self, d_model: int, latent_dim: Optional[int] = None):
        super().__init__()
        latent_dim = int(latent_dim or d_model)
        self.context = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim), nn.GELU())
        self.predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.target = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))
        # Token-level counterparts of the three modules above. Separate
        # weights from the block-level path on purpose: block summaries are
        # mean-pooled and much lower-variance than raw per-token hidden
        # states, so sharing one set of projections would make one of the
        # two objectives fight the other's scale during training.
        self.token_context = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim), nn.GELU())
        self.token_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.token_target = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))

    def forward(self, block_summaries: torch.Tensor) -> Dict[str, torch.Tensor]:
        if block_summaries.ndim != 3:
            raise ValueError("block_summaries must have shape [batch, blocks, d_model]")
        if block_summaries.size(1) < 2:
            zero = block_summaries.new_zeros(())
            return {"jepa_loss": zero, "jepa_cosine": zero, "jepa_target_std": zero}

        ctx = block_summaries[:, :-1, :]
        tgt = block_summaries[:, 1:, :]
        pred = self.predictor(self.context(ctx))
        target = self.target(tgt).detach()
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()
        cosine = (pred * target).sum(dim=-1).mean()
        target_std = target.std(dim=(0, 1)).mean()
        return {"jepa_loss": loss, "jepa_cosine": cosine, "jepa_target_std": target_std}

    def token_surprise(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-token predictive-surprise path, for biasing writer selection.

        Same JEPA-lite recipe as ``forward`` above (predict a stop-gradient
        target from context, score with cosine similarity), but at token
        granularity instead of block granularity: position ``t`` predicts
        position ``t + 1``'s own (projected) hidden state. A position the
        model could *not* predict well from its own token-level context is
        "surprising" -- exactly the kind of position a memory writer should
        be biased toward keeping, since a well-predicted position is by
        definition redundant with what local context already carries.

        Returns ``token_jepa_loss`` (scalar, for training the token-level
        predictor/target heads) and ``token_surprise`` (``[batch, seq_len]``,
        non-negative, higher = more surprising, 0 at the last position since
        there is no future token to compare against there).
        """
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, seq_len, d_model]")
        bsz, seq_len, _ = hidden.shape
        if seq_len < 2:
            zero = hidden.new_zeros(())
            return {"token_jepa_loss": zero, "token_surprise": hidden.new_zeros(bsz, seq_len)}

        ctx = hidden[:, :-1, :]
        tgt = hidden[:, 1:, :]
        pred = self.token_predictor(self.token_context(ctx))
        target = self.token_target(tgt).detach()
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        cosine = (pred * target).sum(dim=-1)  # [B, T-1]
        token_loss = (1.0 - cosine).mean()
        surprise = (1.0 - cosine).clamp_min(0.0)
        # Position t's surprise is "how well was t+1 predicted from t" --
        # attach it to position t (the candidate write-start position), and
        # pad the last position (no t+1 to score) with zero surprise.
        surprise = F.pad(surprise, (0, 1), value=0.0)
        return {"token_jepa_loss": token_loss, "token_surprise": surprise}


def block_summaries(hidden: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool hidden states into block summaries [B, num_blocks, D]."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    chunks = []
    for start in range(0, hidden.size(1), block_size):
        end = min(hidden.size(1), start + block_size)
        chunks.append(hidden[:, start:end, :].mean(dim=1))
    return torch.stack(chunks, dim=1)
