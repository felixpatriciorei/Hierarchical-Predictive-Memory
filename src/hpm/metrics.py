from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def answer_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    losses = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none")
    mask = loss_mask.reshape(-1)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def answer_exact_accuracy(
    logits: torch.Tensor,
    answer_positions: torch.Tensor,
    answer_tokens: torch.Tensor,
) -> torch.Tensor:
    batch = torch.arange(logits.size(0), device=logits.device)
    predictions = logits[batch, answer_positions].argmax(dim=-1)
    return (predictions == answer_tokens).float().mean()


def answer_span_correct_mask(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample exact match over all masked answer target positions."""

    predictions = logits.argmax(dim=-1)
    answer_mask = loss_mask > 0
    token_correct = (predictions == target_ids) | ~answer_mask
    has_answer = answer_mask.any(dim=1)
    return token_correct.all(dim=1) & has_answer


def answer_span_exact_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    return answer_span_correct_mask(logits, target_ids, loss_mask).float().mean()


def answer_predictions(logits: torch.Tensor, answer_positions: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(logits.size(0), device=logits.device)
    return logits[batch, answer_positions].argmax(dim=-1)


def retrieval_correct_mask(
    retrieval: Dict[str, torch.Tensor],
    positive_indices: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not retrieval or "top_indices" not in retrieval:
        return None
    top_indices = retrieval["top_indices"]
    if positive_mask is not None:
        valid = positive_mask.any(dim=1)
        selected = torch.zeros_like(positive_mask, dtype=torch.bool)
        selected.scatter_(1, top_indices.clamp_min(0), True)
        correct = valid & ((selected & positive_mask).sum(dim=1) == positive_mask.sum(dim=1))
        return correct
    if positive_indices is None:
        return None
    valid = positive_indices >= 0
    correct = torch.zeros_like(valid, dtype=torch.bool)
    correct[valid] = (top_indices[valid] == positive_indices[valid, None]).any(dim=1)
    return correct


def retrieval_metrics(
    retrieval: Dict[str, torch.Tensor],
    positive_indices: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    if not retrieval or "top_indices" not in retrieval:
        return {}
    top_indices = retrieval["top_indices"]
    if positive_mask is not None:
        valid = positive_mask.any(dim=1)
        if not valid.any():
            return {}
        selected = torch.zeros_like(positive_mask, dtype=torch.bool)
        selected.scatter_(1, top_indices.clamp_min(0), True)
        top1 = positive_mask.gather(1, top_indices[:, :1]).squeeze(1)
        contains_any = (selected & positive_mask).any(dim=1)
        contains_all = (selected & positive_mask).sum(dim=1) == positive_mask.sum(dim=1)
        out = {
            "retrieval_top1": top1[valid].float().mean().item(),
            "retrieval_topk": contains_all[valid].float().mean().item(),
            "retrieval_topk_any": contains_any[valid].float().mean().item(),
        }
        if "scores" in retrieval:
            scores = retrieval["scores"][valid]
            pos_mask = positive_mask[valid]
            positive_scores = scores.masked_fill(~pos_mask, 1.0e9).min(dim=-1).values
            negative_scores = scores.masked_fill(pos_mask, -1.0e9).max(dim=-1).values
            out["retrieval_margin"] = (positive_scores - negative_scores).mean().item()
        return out

    if positive_indices is None:
        return {}
    valid = positive_indices >= 0
    if not valid.any():
        return {}

    top1 = (top_indices[valid, 0] == positive_indices[valid]).float().mean().item()
    contains = (top_indices[valid] == positive_indices[valid, None]).any(dim=1).float().mean().item()
    out = {
        "retrieval_top1": top1,
        "retrieval_topk": contains,
    }
    if "scores" in retrieval:
        scores = retrieval["scores"][valid]
        pos = positive_indices[valid]
        row = torch.arange(scores.size(0), device=scores.device)
        positive_scores = scores[row, pos]
        negative_scores = scores.clone()
        negative_scores[row, pos] = -1.0e9
        margin = positive_scores - negative_scores.max(dim=-1).values
        out["retrieval_margin"] = margin.mean().item()
    return out


def router_entropy_loss(router_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-token entropy of the path router's softmax distribution.

    router_weights: [batch, seq_len, num_paths], already softmax-normalized
    (see HpmV2PathRouter.forward).

    This is deliberately *not* a load-balancing loss (Switch-Transformer style
    losses push usage toward *uniform* across paths, which is the wrong target
    here -- episodic memory is already established as genuinely load-bearing,
    so forcing balance would fight that). Minimizing this instead pushes each
    token's distribution toward one-hot, i.e. it removes the router's ability
    to hedge by spreading a little weight onto paths that don't earn it. A
    path that's truly useless should see its weight driven toward zero as
    this sharpens; a path that's genuinely useful for a subset of tokens can
    still win those tokens outright. Aggregate usage (already logged as
    eval_router_weight_*) becomes a much more honest signal once this is
    applied, since dense softmax mixing can no longer disguise "unused" as
    "diffusely used a little everywhere."
    """
    if router_weights.dim() != 3:
        raise ValueError(f"expected router_weights shaped [batch, seq_len, num_paths], got {tuple(router_weights.shape)}")
    entropy = -(router_weights.clamp_min(eps) * router_weights.clamp_min(eps).log()).sum(dim=-1)
    return entropy.mean()


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE-style router z-loss: penalizes logsumexp(logits)**2, i.e. the
    logits' own scale, independent of which path wins.

    router_logits: [batch, seq_len, num_paths], the PRE-softmax output of
    HpmV2PathRouter.forward (its third return value).

    This is not a substitute for router_entropy_loss above and does the
    opposite job. router_entropy_loss operates on the post-softmax weights
    and governs confidence/entropy (sharp vs. hedged) -- it has no way to
    tell "confident because the logits are well-separated at a modest scale"
    apart from "confident because logit magnitude blew up and saturated
    softmax regardless of separation," since both land at the same
    post-softmax entropy. z-loss operates on the pre-softmax logits
    directly and only constrains their magnitude, so it can't fight
    legitimate sharp/confident routing the way a naive entropy penalty in
    the wrong direction would -- two logits at (+3, -3) and two logits at
    (+30, -30) both saturate softmax to the same one-hot output and the
    same entropy, but z-loss penalizes only the latter.

    Added 2026-08-02 after a real sweep showed router_logit_abs_mean sitting
    flat at ~1.2-1.7 for steps 50-200, then jumping 3-7x in a single 50-step
    window exactly at step 250 -- the first eval after
    learned_writer_teacher_forcing_steps=200 expired and the differentiable
    full-candidate writer path engaged for the first time -- coincident with
    a loss spike (0.001 -> 5-24) and router_weights snapping to an exact
    one-hot (1,0,0,0) that never recovered through step 600. That flat-then-
    shock shape (not a gradual rise) is the signature of a destabilizing
    gradient event driving logits into a self-reinforcing corner (softmax's
    Jacobian vanishes near the simplex corners, so once saturated there's
    no gradient left to walk it back), not the router earning a sharp
    decision through ordinary training.
    """
    if router_logits.dim() != 3:
        raise ValueError(f"expected router_logits shaped [batch, seq_len, num_paths], got {tuple(router_logits.shape)}")
    log_z = torch.logsumexp(router_logits, dim=-1)
    return log_z.square().mean()


def router_health_metrics(
    router_weights: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    router_raw_logits: torch.Tensor | None = None,
    router_clamp_jacobian: torch.Tensor | None = None,
    eps: float = 1.0e-12,
) -> Dict[str, torch.Tensor]:
    """Non-loss telemetry for distinguishing three router failure modes.

    The HPM router can look "collapsed" for materially different reasons:

    1. *Probability concentration*: softmax weights are near a simplex vertex.
    2. *Raw projection growth*: the linear router logits themselves become huge.
    3. *Clamp lock*: with ``C*tanh(raw/C)``, effective logits stay bounded but
       the local derivative back to raw logits, ``sech(raw/C)^2``, vanishes.

    Existing ``router_weight_*`` and ``router_logit_abs_mean`` telemetry only
    observes (1) and, when a clamp is enabled, the *effective* part of (2).
    This helper makes the missing state explicit without changing any loss or
    routing decision.

    Of particular interest is ``router_raw_to_weight_jacobian_fro_mean``. For
    each token, if ``p = softmax(z)`` and ``d`` is the elementwise derivative
    from raw logits ``a`` to effective logits ``z``, then

        d p / d a = (diag(p) - p p^T) diag(d).

    Its Frobenius norm is an exact local sensitivity of the router weights to
    the raw projection. It becomes small either because softmax saturates or
    because the tanh clamp's derivative saturates, so it is a useful single
    "can the gate still move locally?" diagnostic while the decomposed terms
    below tell us *why* it became small.

    Returned values are scalar tensors so callers may log ``.item()`` under
    ``torch.no_grad()`` without forcing this diagnostic into a training loss.
    """
    if router_weights.dim() != 3:
        raise ValueError(
            f"expected router_weights shaped [batch, seq_len, num_paths], got {tuple(router_weights.shape)}"
        )
    if router_logits.shape != router_weights.shape:
        raise ValueError(
            "router_logits must match router_weights shape; "
            f"got {tuple(router_logits.shape)} vs {tuple(router_weights.shape)}"
        )

    raw_logits = router_logits if router_raw_logits is None else router_raw_logits
    if raw_logits.shape != router_weights.shape:
        raise ValueError(
            "router_raw_logits must match router_weights shape; "
            f"got {tuple(raw_logits.shape)} vs {tuple(router_weights.shape)}"
        )

    clamp_jacobian = (
        torch.ones_like(router_weights)
        if router_clamp_jacobian is None
        else router_clamp_jacobian
    )
    if clamp_jacobian.shape != router_weights.shape:
        raise ValueError(
            "router_clamp_jacobian must match router_weights shape; "
            f"got {tuple(clamp_jacobian.shape)} vs {tuple(router_weights.shape)}"
        )
    if not torch.isfinite(router_weights).all() or not torch.isfinite(router_logits).all():
        raise ValueError("router health inputs must be finite")
    if not torch.isfinite(raw_logits).all() or not torch.isfinite(clamp_jacobian).all():
        raise ValueError("router raw-logit/clamp-jacobian diagnostics must be finite")

    p = router_weights
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)
    max_weight = p.max(dim=-1).values

    # Softmax Jacobian J_z = diag(p) - p p^T.
    softmax_jacobian = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    softmax_jacobian_fro = softmax_jacobian.square().sum(dim=(-2, -1)).sqrt()

    # Chain rule through the optional elementwise clamp:
    # J_raw = J_z @ diag(dz/da). Right-multiplication by a diagonal matrix
    # scales each Jacobian column by the corresponding clamp derivative.
    raw_to_weight_jacobian = softmax_jacobian * clamp_jacobian.unsqueeze(-2)
    raw_to_weight_jacobian_fro = raw_to_weight_jacobian.square().sum(dim=(-2, -1)).sqrt()
    clamp_attenuation_ratio = raw_to_weight_jacobian_fro / softmax_jacobian_fro.clamp_min(eps)

    return {
        "router_weight_max_mean": max_weight.mean(),
        "router_entropy_mean": entropy.mean(),
        # exp(H) is the entropy-equivalent number of active paths: 1 at a
        # one-hot vertex, K for a uniform K-path router.
        "router_effective_paths_mean": entropy.exp().mean(),
        "router_logit_abs_mean": router_logits.abs().mean(),
        "router_raw_logit_abs_mean": raw_logits.abs().mean(),
        "router_logit_span_mean": (router_logits.max(dim=-1).values - router_logits.min(dim=-1).values).mean(),
        "router_raw_logit_span_mean": (raw_logits.max(dim=-1).values - raw_logits.min(dim=-1).values).mean(),
        "router_clamp_jacobian_mean": clamp_jacobian.mean(),
        "router_clamp_jacobian_min": clamp_jacobian.min(),
        "router_clamp_jacobian_lt_0p01_frac": (clamp_jacobian < 1.0e-2).float().mean(),
        "router_clamp_jacobian_lt_0p001_frac": (clamp_jacobian < 1.0e-3).float().mean(),
        "router_softmax_jacobian_fro_mean": softmax_jacobian_fro.mean(),
        "router_softmax_jacobian_fro_min": softmax_jacobian_fro.min(),
        "router_raw_to_weight_jacobian_fro_mean": raw_to_weight_jacobian_fro.mean(),
        "router_raw_to_weight_jacobian_fro_min": raw_to_weight_jacobian_fro.min(),
        "router_clamp_attenuation_ratio_mean": clamp_attenuation_ratio.mean(),
        "router_clamp_attenuation_ratio_min": clamp_attenuation_ratio.min(),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
