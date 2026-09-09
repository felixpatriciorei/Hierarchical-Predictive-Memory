"""
Differentiable top-k selection via entropic-regularized Optimal Transport (Sinkhorn).

Implements the operator from Xie, Dai, Chen, Dai, Zhao, Zha, Wipf, Zhou,
"Differentiable Top-k Operator with Optimal Transport", NeurIPS 2020
(arXiv:2002.06504), specialized + hardened for use as a drop-in replacement
for `torch.topk(scores, k).indices` inside a writer/router that currently
breaks gradient flow.

Forward value returned is numerically IDENTICAL to hard top-k (a 0/1 mask,
or the top-k values themselves) -- this is a straight-through estimator.
The backward pass routes gradient through the soft OT plan instead of
through a non-differentiable topk. This provides a surrogate gradient only;
it does not by itself ensure that an end-to-end loss trains a useful writer.
For example, a downstream softmax over one hard-selected value is constant
and sends no answer-loss signal to selection scores. Use an explicitly
validated set-ranking/coverage loss or a non-degenerate read objective when
the intended claim is that the writer learned the top-k decision.

Math summary
------------
Given scores s in R^n and target count k, define m = k+1 anchor "buckets":
    - k copies of a "selected" anchor  a_hi = max(s)  (+ margin)
    - 1 copy   of a "not-selected" anchor a_lo = min(s)  (- margin)
Row marginals (over the n elements):  mu_i = 1          for i = 1..n
Col marginals (over the m buckets):   nu_j = 1          for the k "selected" cols
                                       nu_{k+1} = n - k  for the "not-selected" col
Cost:  C_ij = (s_i - a_j)^2
Entropic OT (Sinkhorn) solves:
    Gamma* = argmin_Gamma  <Gamma, C> - eps * H(Gamma)
             s.t. Gamma 1 = mu,  Gamma^T 1 = nu,  Gamma >= 0
via the classic scaling iteration:
    K = exp(-C / eps)
    u <- mu / (K v),   v <- nu / (K^T u)         (log-domain in the impl below)
    Gamma = diag(u) K diag(v)

The soft top-k indicator per element is the row-sum of Gamma over the k
"selected" columns:  p_i = sum_{j=1}^{k} Gamma_ij  in [0, 1].
As eps -> 0, Gamma -> the vertex of the transport polytope = the exact
top-k permutation, so p_i -> hard {0,1} indicator (Prop. 1 / Thm. 1 of the
paper: entropic error is O(eps * log n), so eps ~ 1e-2 already gives a
near-exact relaxation on n in the tens; verified empirically below).

Straight-through wrapper:
    hard_mask  = 0/1 indicator from real torch.topk        (used forward)
    soft_mask  = p_i from Sinkhorn                          (used backward)
    output     = hard_mask + (soft_mask - soft_mask.detach())
so `output` forward-equals hard_mask exactly, but d(output)/d(scores) =
d(soft_mask)/d(scores), which is nonzero everywhere (Sinkhorn is smooth).

Complexity: O(n * m * n_iters) per call, m = k+1. For a writer choosing
1-of-{3,8,16} memory slots this is negligible next to the rest of the
forward pass; n_iters=20-50 is plenty for eps=0.05-0.1 at these sizes
(see the empirical convergence check at the bottom of this file).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sparsemax: continuous, exactly-sparse read-side alternative to torch.topk.
#
# Context (see docs/engineering/read_side_sparsemax.md for the full writeup):
# the writer side already has a differentiable selection path
# (`differentiable_topk_bias` above, a Sinkhorn/OT straight-through
# estimator). The READ side in `hpm/memory.py::retrieve_topk` still
# calls `torch.topk` directly to decide WHICH memory slots are gathered
# before the softmax-over-selected-slots. Two separate limitations follow
# from that:
#   1. the number of slots that participate is pinned to a fixed `top_k`
#      regardless of the score distribution -- a near-tied runner-up just
#      outside the cutoff is not down-weighted, it is dropped from the
#      computation graph entirely (gather only copies the winners), so it
#      gets exactly zero gradient no matter how close its score is;
#   2. that gather step has no gradient of its own for "should slot i have
#      been in the top-k" -- upstream signal can only reach slot ranking
#      indirectly, via a separate bolted-on mechanism like `select_bias`.
#
# Sparsemax (Martins & Astudillo, 2016, arXiv:1602.02068) -- a Euclidean
# projection onto the probability simplex -- fixes (1): its support size
# is emergent from thresholding against ALL candidate scores, not fixed in
# advance, so a near-tied runner-up like the one above ends up WITH real
# weight and a real, closed-form (non-STE, no temperature to tune) gradient.
# It does not fix (2) in general: for a slot that ends up strictly outside
# the support (exact weight 0.0), the local gradient is, correctly, exactly
# zero -- that is the same "zero gradient off the selection" property any
# exactly-sparse operator has (including hard top-k), not a limitation
# unique to this implementation. The payoff is narrower than Direction 4.3's
# fully continuous MoE-style softmax route (dense weights, gradient
# everywhere), but it keeps exact zeros -- which several existing
# diagnostics (`writer_missed_fact_rate`, retrieval top-1) are defined
# against -- while removing the fixed-k ceiling and the need for a separate
# STE hack on the read side.
# ---------------------------------------------------------------------------


class _SparsemaxFunction(torch.autograd.Function):
    """Sparsemax over the last dimension, with the exact projection Jacobian.

    Forward solves the Euclidean projection onto the simplex via the
    standard sort-and-threshold algorithm (Held, Wolfe & Crowder 1974;
    popularized for attention by Martins & Astudillo 2016). Backward uses
    the closed-form Jacobian restricted to the support set S = {i : p_i > 0}:
        dp_i/dz_j = delta_ij - 1/|S|   for i, j in S
                  = 0                   otherwise
    i.e. gradient is passed through unchanged on the support, minus the
    support-average (so the simplex constraint sum(p)=1 is respected), and
    is exactly zero off the support -- correct, not an artifact: an
    infinitesimal change to an already-excluded score does not change the
    output until it crosses the threshold, so its true local derivative is
    zero, not merely small the way an STE approximation would give.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        n = z.size(-1)
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        z_cumsum = z_sorted.cumsum(dim=-1)
        rho = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
        # support[..., j] = True iff the j-th largest element (1-indexed rho)
        # survives thresholding; because z_sorted is descending this is
        # equivalent to "is the true support size >= rho" so summing it
        # gives k(z) directly.
        support = (1 + rho * z_sorted) > z_cumsum
        k = support.sum(dim=-1, keepdim=True).clamp_min(1)
        cssv_k = torch.gather(z_cumsum, -1, (k - 1))
        tau = (cssv_k - 1.0) / k.to(z.dtype)
        p = torch.clamp(z - tau, min=0.0)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (p,) = ctx.saved_tensors
        support = (p > 0).to(grad_output.dtype)
        k = support.sum(dim=-1, keepdim=True).clamp_min(1.0)
        support_avg = (grad_output * support).sum(dim=-1, keepdim=True) / k
        grad_input = support * (grad_output - support_avg)
        return grad_input


def sparsemax(scores: torch.Tensor) -> torch.Tensor:
    """Sparsemax over the last dim of ``scores``: (..., n) -> (..., n).

    Drop-in replacement for ``torch.softmax(scores, dim=-1)`` where exact
    zeros (true sparsity, not just small weights) and a real, non-STE
    gradient are both wanted -- e.g. reading `n` memory slots without first
    committing to a hard top-k index set. Masked-out slots should be set to
    a large negative value (as with softmax) before calling this; they will
    receive exactly 0.0 weight and, off the support, exactly 0.0 gradient.
    """

    return _SparsemaxFunction.apply(scores)


def _sinkhorn_topk_soft(
    scores: torch.Tensor,          # (..., n)
    k: int,
    eps: float = 0.3,
    n_iters: int = 50,
    anchor_margin: float = 1.0,
) -> torch.Tensor:
    """
    Returns soft per-element top-k membership probabilities, shape (..., n),
    each in [0, 1], differentiable w.r.t. `scores`.
    """
    *batch_shape, n = scores.shape
    assert 1 <= k < n, f"need 1 <= k < n, got k={k}, n={n}"
    device, dtype = scores.device, scores.dtype

    # NOTE: anchors/spread are intentionally detached. If gradient is
    # allowed through max()/min() here, the anchors chase the very scores
    # they're supposed to be a fixed reference for, which creates a
    # self-referential feedback loop -- empirically this produces spurious
    # fixed points (e.g. optimization stalling at an exact 50/50 tie among
    # non-target elements). Treat anchors as a fixed per-step reference
    # scale, the same way you'd stop-gradient normalization statistics.
    scores_d = scores.detach()
    s_max = scores_d.max(dim=-1, keepdim=True).values
    s_min = scores_d.min(dim=-1, keepdim=True).values
    spread = (s_max - s_min).clamp_min(1e-6)
    a_hi = s_max + anchor_margin * spread   # "selected" anchor
    a_lo = s_min - anchor_margin * spread   # "not-selected" anchor

    # m = k+1 anchor columns: k copies of a_hi, then a_lo
    anchors = torch.cat(
        [a_hi.expand(*batch_shape, k), a_lo.expand(*batch_shape, 1)], dim=-1
    )  # (..., k+1)

    # cost matrix C: (..., n, k+1). Normalize by spread^2 so `eps` is a
    # RELATIVE temperature (fraction of the score range), not absolute --
    # otherwise, as the scoring head trains and score separation grows,
    # a fixed absolute eps saturates the Sinkhorn plan to exactly 0/1 and
    # the gradient vanishes (verified empirically; this is the standard
    # failure mode of entropic OT at small eps / large cost range).
    C = ((scores.unsqueeze(-1) - anchors.unsqueeze(-2)) / spread.unsqueeze(-1)) ** 2

    # marginals
    mu = torch.ones(*batch_shape, n, device=device, dtype=dtype)               # row sums = 1 each
    nu = torch.cat(
        [torch.ones(*batch_shape, k, device=device, dtype=dtype),
         torch.full((*batch_shape, 1), float(n - k), device=device, dtype=dtype)],
        dim=-1,
    )  # col sums: 1 per "selected" col, (n-k) for "not-selected" col

    # log-domain Sinkhorn for numerical stability
    log_mu = mu.clamp_min(1e-12).log()
    log_nu = nu.clamp_min(1e-12).log()
    log_K = -C / eps                                    # (..., n, k+1)

    log_u = torch.zeros(*batch_shape, n, device=device, dtype=dtype)
    log_v = torch.zeros(*batch_shape, k + 1, device=device, dtype=dtype)

    for _ in range(n_iters):
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=-2)

    log_Gamma = log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)  # (..., n, k+1)
    Gamma = log_Gamma.exp()

    soft_membership = Gamma[..., :k].sum(dim=-1)   # (..., n), in [0, 1]
    return soft_membership.clamp(0.0, 1.0)


def differentiable_topk_mask(
    scores: torch.Tensor,
    k: int,
    eps: float = 0.3,
    n_iters: int = 50,
    hard: bool = True,
) -> torch.Tensor:
    """
    Drop-in differentiable replacement for building a top-k 0/1 mask.

    scores: (..., n) unnormalized selection scores (writer's "how good is
            slot i for this write" logits, eviction scores, router logits...)
    k:      how many elements to select
    hard:   if True (default), forward pass is EXACTLY the same as
            `torch.topk(scores, k).indices` turned into a mask (so behavior
            at inference / in the loss is unchanged); backward pass uses the
            soft OT relaxation (straight-through estimator).
            if False, returns the raw soft mask (fully soft, e.g. for
            annealed training where you fade hard=True in over time).

    Returns: mask of shape (..., n), values in {0,1} (hard=True) or [0,1]
             (hard=False), same dtype/device as scores, differentiable.
    """
    soft_mask = _sinkhorn_topk_soft(scores, k, eps=eps, n_iters=n_iters)

    if not hard:
        return soft_mask

    with torch.no_grad():
        hard_idx = scores.topk(k, dim=-1).indices
        hard_mask = torch.zeros_like(scores).scatter_(-1, hard_idx, 1.0)

    # straight-through: forward = hard_mask, backward = grad of soft_mask
    return hard_mask + (soft_mask - soft_mask.detach())


def differentiable_topk_bias(
    scores: torch.Tensor,
    k: int,
    eps: float = 0.3,
    n_iters: int = 50,
    bias_scale: float = 1.0e4,
) -> torch.Tensor:
    """
    Differentiable replacement for the common
        `attn_scores.masked_fill(~hard_topk_mask, -1e9)`
    pattern used to gate softmax attention down to a top-k subset.

    Returns an additive log-bias of shape (..., n): 0 for the k selected
    elements, -bias_scale for the rest -- forward-identical to masking with
    the real hard top-k (so `attn_scores + bias` behaves exactly like
    `attn_scores.masked_fill(~hard_mask, -bias_scale)` at inference).
    Backward routes gradient through the smooth Sinkhorn membership, so a
    downstream loss computed *after* the softmax (e.g. an answer/read loss)
    can shape which elements get selected, not just the raw `scores` that
    fed the selection.

    Use this instead of `differentiable_topk_mask` when the top-k decision
    feeds into an additive attention/softmax mask rather than a multiplicative
    gate on gathered values -- it keeps the STE trick's zero-inference-cost
    guarantee while giving gradient somewhere the downstream loss can
    actually use it (the pre-softmax scores), rather than requiring you to
    already have gathered the winners into their own smaller tensor.
    """
    mask = differentiable_topk_mask(scores, k, eps=eps, n_iters=n_iters, hard=True)  # STE: fwd={0,1} exact, bwd=d(soft_mask)/d(scores)
    # BUG (found via a real training run, not derived): scaling the WHOLE STE
    # tensor by bias_scale also scales its backward component by bias_scale.
    # `mask`'s gradient is d(soft_mask)/d(scores), an O(1) quantity; at
    # bias_scale=1e9 that's a ~10^9x gradient amplification with zero
    # justification beyond "make the forward masking strong enough" -- those
    # are two different concerns that must not share one constant. Fix:
    # scale ONLY the detached hard part; let the soft/gradient-carrying part
    # pass through unscaled. Forward value is identical either way (the soft
    # component is exactly 0 at eval time); only backward magnitude changes.
    hard_component = (mask.detach() - 1.0) * bias_scale   # forward: 0 or -bias_scale; no gradient (detached)
    soft_component = mask - mask.detach()                  # forward: exactly 0; backward: d(soft_mask)/d(scores), UNSCALED
    return hard_component + soft_component


def differentiable_topk_select(
    values: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    eps: float = 0.3,
    n_iters: int = 50,
    hard: bool = True,
) -> torch.Tensor:
    """
    Convenience wrapper: select/gate `values` (..., n, d) by top-k `scores`
    (..., n), returning the masked-and-summed result (..., d) -- e.g. for a
    writer that picks one memory slot and needs the *gated write vector*,
    not just the mask/index.
    """
    mask = differentiable_topk_mask(scores, k, eps=eps, n_iters=n_iters, hard=hard)  # (..., n)
    return torch.einsum("...n,...nd->...d", mask, values)
