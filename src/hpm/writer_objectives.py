"""Objectives for a finite-capacity supervised episodic writer.

These objectives are deliberately separate from ``SupervisedMemoryWriter``.
The model can therefore retain its historical BCE-only behavior unless a
training run explicitly opts into an additional objective and its capacity
contract has been checked.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class _CappedSigmoidTopK(torch.autograd.Function):
    """Smooth Top-C membership with an exact forward mass constraint.

    For scores ``z`` and target count ``k``, solve a scalar threshold ``tau``
    such that ``sum(sigmoid((z_i - tau) / temperature)) == k``.  The backward
    is the implicit derivative of that constraint, so score perturbations
    redistribute a fixed selection budget rather than independently raising
    every membership.  This is a training-only relaxation used by the set
    objective; it does not change the writer's deployed hard Top-C operation.
    """

    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        k: int,
        temperature: float,
        n_iters: int,
    ) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError("_CappedSigmoidTopK expects scores with shape [batch, candidates]")
        valid_mask = valid_mask.bool()
        # Treat scale and centering as fixed per optimization step.  This makes
        # temperature relative to the scorer's current range, preventing an
        # otherwise harmless growth in raw logits from saturating the loss.
        detached = scores.detach()
        valid_count = valid_mask.sum(dim=-1, keepdim=True).to(scores.dtype)
        row_max = detached.masked_fill(~valid_mask, -torch.inf).max(dim=-1, keepdim=True).values
        row_min = detached.masked_fill(~valid_mask, torch.inf).min(dim=-1, keepdim=True).values
        scale = (row_max - row_min).clamp_min(1.0e-6)
        mean = (detached * valid_mask).sum(dim=-1, keepdim=True) / valid_count
        centered = (scores - mean) / scale
        # sigmoid((x-tau)/T) is already within exp-safe range beyond +/-40.
        lower = centered.detach().masked_fill(~valid_mask, torch.inf).min(dim=-1, keepdim=True).values - 40.0 * temperature
        upper = centered.detach().masked_fill(~valid_mask, -torch.inf).max(dim=-1, keepdim=True).values + 40.0 * temperature
        for _ in range(n_iters):
            threshold = (lower + upper) / 2.0
            membership = torch.sigmoid((centered.detach() - threshold) / temperature) * valid_mask
            mass = membership.sum(dim=-1, keepdim=True)
            # Mass decreases monotonically as threshold rises.
            lower = torch.where(mass > k, threshold, lower)
            upper = torch.where(mass > k, upper, threshold)
        threshold = (lower + upper) / 2.0
        membership = torch.sigmoid((centered.detach() - threshold) / temperature) * valid_mask
        ctx.save_for_backward(membership, scale, valid_mask)
        ctx.temperature = temperature
        return membership

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        membership, scale, valid_mask = ctx.saved_tensors
        slope = membership * (1.0 - membership)
        slope_sum = slope.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        # Implicitly differentiate sum(membership) = k:
        # d tau / d z_j = slope_j / sum(slope).
        correction = (grad_output * slope).sum(dim=-1, keepdim=True) / slope_sum
        grad_scores = (slope / ctx.temperature) * (grad_output - correction) / scale
        return grad_scores * valid_mask, None, None, None, None


def capped_sigmoid_topk_membership(
    scores: torch.Tensor,
    k: int,
    *,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 0.3,
    n_iters: int = 48,
) -> torch.Tensor:
    """Return smooth fixed-budget memberships for one or more candidate rows.

    ``scores`` is ``[candidates]`` or ``[batch, candidates]``.  ``valid_mask``
    lets padded candidates remain exactly zero.  The bisection is tensorized
    over batch rows: it contains no host-device synchronization in its
    iteration loop, which matters when the optional writer loss is run on a
    GPU.
    """
    was_vector = scores.ndim == 1
    if was_vector:
        scores = scores.unsqueeze(0)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [candidates] or [batch, candidates], got {tuple(scores.shape)}")
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    elif was_vector:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must have the same shape as scores")
    valid_mask = valid_mask.bool()
    active_counts = valid_mask.sum(dim=-1)
    if torch.any(active_counts <= k) or k < 1:
        raise ValueError(
            "need 1 <= k < valid candidates in every row, got "
            f"k={k}, active_counts={active_counts.detach().cpu().tolist()}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if n_iters < 1:
        raise ValueError(f"n_iters must be positive, got {n_iters}")
    membership = _CappedSigmoidTopK.apply(scores, valid_mask, float(k), temperature, n_iters)
    return membership.squeeze(0) if was_vector else membership


def topc_required_margin_diagnostics(
    selection_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_mask: torch.Tensor,
) -> Dict[str, float]:
    """Measure the exact hard Top-C inclusion margin for required records.

    In a row with ``p`` required candidates and a hard selection budget ``C``,
    every required candidate is retained iff the least-scored required logit
    exceeds the ``(C-p+1)``-th highest distractor logit.  Equivalently, with
    zero-indexed descending distractor scores, the relevant cutoff is index
    ``C-p``.  The returned margin is this required-minus-cutoff value.  Its
    sign is therefore a continuous, causally linked progress measure for the
    discrete all-required-in-Top-C event; it is not a proxy based on router
    weights, output confidence, or the final answer.

    ``selected_mask`` is the deployed hard writer's slot mask, so the
    diagnostic supports variable per-row capacities without duplicating an
    architecture-level capacity setting.  Rows without an actual selection
    boundary (no positives, all candidates selected, or an invalid capacity)
    are omitted rather than assigned an arbitrary margin.
    """
    if selection_logits.ndim != 2:
        raise ValueError("selection_logits must have shape [batch, candidates]")
    if labels.shape != selection_logits.shape or valid_mask.shape != selection_logits.shape:
        raise ValueError("labels and valid_mask must have the same shape as selection_logits")
    if selected_mask.ndim != 2 or selected_mask.size(0) != selection_logits.size(0):
        raise ValueError("selected_mask must have shape [batch, selected_slots]")
    if not torch.isfinite(selection_logits).all():
        raise ValueError("Top-C margin received non-finite selection logits")

    valid_mask = valid_mask.bool()
    if not torch.all((labels[valid_mask] == 0) | (labels[valid_mask] == 1)):
        raise ValueError("Top-C margin requires binary writer labels")
    labels = labels.bool()
    positive = labels & valid_mask
    negative = valid_mask & ~positive
    required_counts = positive.sum(dim=-1)
    distractor_counts = negative.sum(dim=-1)
    active_counts = valid_mask.sum(dim=-1)
    capacities = selected_mask.bool().sum(dim=-1)
    eligible = (
        (required_counts > 0)
        & (capacities >= required_counts)
        & (capacities < active_counts)
        & (distractor_counts > capacities - required_counts)
    )
    if not eligible.any():
        return {
            "samples": 0.0,
            "mean_required_topc_margin": 0.0,
            "all_required_topc_rate": 0.0,
        }

    min_required = selection_logits.masked_fill(~positive, torch.inf).min(dim=-1).values
    sorted_distractors = selection_logits.masked_fill(~negative, -torch.inf).sort(dim=-1, descending=True).values
    cutoff_index = (capacities - required_counts).clamp_min(0)
    cutoff_distractor = sorted_distractors.gather(1, cutoff_index.unsqueeze(1)).squeeze(1)
    margins = min_required - cutoff_distractor
    eligible_margins = margins[eligible]
    return {
        "samples": float(eligible.sum().item()),
        "mean_required_topc_margin": float(eligible_margins.mean().item()),
        "all_required_topc_rate": float((eligible_margins > 0.0).float().mean().item()),
    }


def topc_set_coverage_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    capacity: int,
    *,
    eps: float = 0.3,
    n_iters: int = 48,
    mass_tolerance: float = 1.0e-3,
    require_strict_selection: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Penalize each required candidate that falls outside a finite Top-C set.

    For each batch row, let ``A`` be the valid candidates, ``P`` the positive
    labels in ``A``, and ``k = min(capacity, |A|)``.  A smooth capped-sigmoid
    membership vector ``m = SoftTopK(logits[A], k)`` gives the objective

        ``-mean_{i in P} log(max(m_i, 1e-12))``.

    This is a set objective: it requires every positive candidate to receive
    a place in the Top-C set, but does not impose an order among positives.
    It is valid only when ``1 <= |P| <= k``.  If ``require_strict_selection``
    is true, it also requires ``k < |A|``: otherwise every candidate fits,
    the loss is identically zero, and a supposedly selective experiment would
    silently be testing only proposal/read plumbing.

    The scalar threshold is solved by bisection and the numerical mass check
    fails closed if its fixed-budget invariant is not met.  This objective
    intentionally does not use the project's Sinkhorn straight-through
    operator: that operator remains the deployed differentiable-read
    surrogate, but its 2-D transport solve was needlessly expensive and had a
    measurable finite-iteration mass error for this 1-D ranking objective.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, candidates], got {tuple(logits.shape)}")
    if labels.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("labels and valid_mask must have the same shape as logits")
    if capacity < 1:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if n_iters < 1:
        raise ValueError(f"n_iters must be positive, got {n_iters}")
    if mass_tolerance < 0.0:
        raise ValueError(f"mass_tolerance must be non-negative, got {mass_tolerance}")
    if not torch.isfinite(logits).all():
        raise ValueError("Top-C set coverage received non-finite writer logits")

    valid_mask = valid_mask.bool()
    labels = labels.to(dtype=logits.dtype)
    active_counts = valid_mask.sum(dim=-1)
    positive_mask = labels.bool() & valid_mask
    required_counts = positive_mask.sum(dim=-1)
    if torch.any(active_counts == 0):
        row = int((active_counts == 0).nonzero(as_tuple=False)[0].item())
        raise ValueError(f"batch row {row} has no valid writer candidates")
    if not torch.all((labels[valid_mask] == 0.0) | (labels[valid_mask] == 1.0)):
        raise ValueError("Top-C set coverage requires binary writer labels")
    if torch.any(required_counts == 0):
        row = int((required_counts == 0).nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"batch row {row} has no required candidates; "
            "the coverage objective is undefined for this task"
        )
    if torch.any(required_counts > capacity):
        row = int((required_counts > capacity).nonzero(as_tuple=False)[0].item())
        required = int(required_counts[row].item())
        raise ValueError(
            f"batch row {row} requires {required} candidates but "
            f"Top-C capacity is only {capacity}; repair the causal-capacity contract"
        )
    if require_strict_selection and torch.any(active_counts <= capacity):
        row = int((active_counts <= capacity).nonzero(as_tuple=False)[0].item())
        active = int(active_counts[row].item())
        raise ValueError(
            f"batch row {row} has {active} valid candidates and "
            f"capacity {capacity}; every candidate fits, so Top-C coverage would be zero"
        )
    if not require_strict_selection:
        # No current caller needs a mixed full-fit/finite-capacity batch.  Do
        # not silently produce a near-one approximation in those rows: a
        # nonselective control should use no coverage loss at all.
        raise ValueError("Top-C set coverage currently requires strict capacity < valid candidates")

    membership = capped_sigmoid_topk_membership(
        logits,
        k=capacity,
        valid_mask=valid_mask,
        temperature=eps,
        n_iters=n_iters,
    )
    mass_error = (membership.sum(dim=-1).detach() - capacity).abs()
    max_mass_error = float(mass_error.max().item())
    if max_mass_error > mass_tolerance:
        raise RuntimeError(
            "Top-C membership did not satisfy its mass contract: "
            f"max_error={max_mass_error:.6g}, tolerance={mass_tolerance:.6g}. "
            "Increase threshold-solver iterations."
        )
    per_row_loss = -(membership.clamp_min(1.0e-12).log() * positive_mask).sum(dim=-1) / required_counts
    loss = per_row_loss.mean()
    diagnostics = {
        "samples": float(logits.size(0)),
        "mean_active_candidates": float(active_counts.float().mean().item()),
        "mean_required_candidates": float(required_counts.float().mean().item()),
        "mean_soft_positive_coverage": float(
            (membership.detach() * positive_mask).sum(dim=-1).div(required_counts).mean().item()
        ),
        "max_membership_mass_error": max_mass_error,
    }
    return loss, diagnostics
