"""Writer-transition diagnostics: instrumentation to find the true origin of
the step-250-style collapse, upstream of the router.

Context: router_logit_clamp (hpm_v2.HpmV2PathRouter) showed that the observed
teacher-forcing-cutover failure survives even when the EFFECTIVE pre-softmax
logits are unconditionally bounded. That rules out unbounded effective-logit
scale as a sufficient explanation, but it does not prove the router is fully
recoverable: the tanh clamp can itself suppress gradients to an extreme raw
router projection. The newer router-health telemetry therefore records raw
and effective logits plus the clamp/raw-to-weight Jacobian separately. This
module remains focused on the two upstream subsystems that changed sharply at
the cutover and were already implicated by the experiment:
the learned writer's candidate scorer and the episodic memory read, both of
which change behavior sharply at the exact same step -- teacher forcing
stops gating memory_token_positions/memory_mask onto the oracle-supervised
`writer_memory_token_positions`/`writer_memory_mask`, and the differentiable
full-candidate Sinkhorn read (`writer_select_bias`) engages for the first
time.

Two complementary measurements, both centered on the cutover step:

1. Per-subsystem gradient norms (local_blocks, selective_recurrent,
   fast_memory, router, writer, episodic_memory, jepa, tied
   embedding/answer_head) on the REAL training batch each step, captured
   right after loss.backward() and before grad clipping -- clipping rescales
   every parameter's grad by one global factor, which would erase the
   relative-magnitude signal this is meant to expose. Whichever subsystem's
   norm jumps first, in the step immediately before the loss spike rather
   than the same step or after, is upstream of it.

2. Writer candidate-distribution shift, measured by re-running the writer
   (in eval mode, no_grad, non-teacher-forced so the differentiable
   full-candidate path is exercised even for steps still inside the teacher
   forcing window) against a SINGLE FIXED probe batch sampled once before
   training starts. Using the same input every diagnosed step is what makes
   entropy/KL/Jaccard comparable step-to-step -- comparing distributions
   over freshly-sampled training batches would conflate "the input changed"
   with "the model's candidate scoring changed", which is exactly the
   ambiguity this instrumentation exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# Named submodule groups probed for gradient norms. Every hpm_lite_v2 model
# has local_blocks, episodic_memory, token_embedding, position_embedding,
# final_ln; the rest are optional (None under episodic_only=True or
# use_learned_writer=False / use_jepa_aux=False) and skipped when absent.
# answer_head.weight is tied to token_embedding.weight (see
# HpmLiteV2Model.__init__), so it is intentionally folded into "embed_head"
# rather than double-counted as a separate group.
_MODULE_GROUPS = (
    "local_blocks",
    "selective_recurrent",
    "fast_memory",
    "router",
    "writer",
    "episodic_memory",
    "delta_writer",
    "jepa",
)


def module_grad_norm(module: Optional[torch.nn.Module]) -> Optional[float]:
    """L2 norm of the current .grad across all of a module's parameters.

    Returns None (not 0.0) if the module is absent or has no grad yet, so
    callers/CSV rows can distinguish "this subsystem doesn't exist in this
    config" from "this subsystem's grad is genuinely zero".
    """
    if module is None:
        return None
    total = 0.0
    saw_any = False
    for p in module.parameters():
        if p.grad is not None:
            saw_any = True
            total += p.grad.detach().float().pow(2).sum().item()
    return total**0.5 if saw_any else None


def collect_module_grad_norms(model: torch.nn.Module) -> Dict[str, Optional[float]]:
    """Per-subsystem grad norms, plus 'embed_head' (tied token embedding +
    position embedding + final layernorm) and 'total' (all parameters,
    matching what clip_grad_norm_ sees) for context."""
    norms: Dict[str, Optional[float]] = {}
    for name in _MODULE_GROUPS:
        norms[f"grad_norm_{name}"] = module_grad_norm(getattr(model, name, None))

    embed_total = 0.0
    embed_saw_any = False
    for module in (model.token_embedding, model.position_embedding, model.final_ln):
        for p in module.parameters():
            if p.grad is not None:
                embed_saw_any = True
                embed_total += p.grad.detach().float().pow(2).sum().item()
    norms["grad_norm_embed_head"] = embed_total**0.5 if embed_saw_any else None

    total = 0.0
    saw_any = False
    for p in model.parameters():
        if p.grad is not None:
            saw_any = True
            total += p.grad.detach().float().pow(2).sum().item()
    norms["grad_norm_total_preclip"] = total**0.5 if saw_any else None
    return norms


def masked_candidate_distribution(selection_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the writer's candidate scores, masked to valid positions
    per example (invalid positions get exactly zero probability mass, not
    just a very negative logit, so entropy/KL below aren't skewed by however
    many invalid positions happen to exist)."""
    masked = selection_logits.masked_fill(~valid_mask, torch.finfo(selection_logits.dtype).min)
    probs = torch.softmax(masked, dim=-1)
    return probs * valid_mask.float()


def candidate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Per-example entropy (nats) of an already-masked candidate distribution."""
    safe = probs.clamp_min(1e-12)
    return -(probs * safe.log()).sum(dim=-1)


def candidate_kl(probs_t: torch.Tensor, probs_prev: torch.Tensor) -> torch.Tensor:
    """Per-example KL(probs_t || probs_prev). Both distributions must come
    from the SAME fixed probe batch (see module docstring) so position i
    means the same candidate token in both -- KL between distributions over
    different inputs would just measure input difference, not model drift."""
    p = probs_t.clamp_min(1e-12)
    q = probs_prev.clamp_min(1e-12)
    return (probs_t * (p.log() - q.log())).sum(dim=-1)


def topk_jaccard(starts_t: torch.Tensor, mask_t: torch.Tensor, starts_prev: torch.Tensor, mask_prev: torch.Tensor) -> torch.Tensor:
    """Per-example Jaccard overlap of selected candidate start positions
    between two steps' top-k writer selections on the same fixed probe
    batch -- how abruptly the actual selected memory slots change, as
    opposed to the full-distribution entropy/KL above."""
    bsz = starts_t.size(0)
    out = torch.zeros(bsz, dtype=torch.float32)
    for b in range(bsz):
        set_t = set(starts_t[b][mask_t[b]].tolist())
        set_prev = set(starts_prev[b][mask_prev[b]].tolist())
        union = set_t | set_prev
        if not union:
            out[b] = 1.0
            continue
        out[b] = len(set_t & set_prev) / len(union)
    return out


@dataclass
class ProbeState:
    """Carries the previous diagnosed step's probe-batch candidate
    distribution and selection, so each new step can be compared against
    the immediately preceding one rather than only against the fixed batch
    in isolation."""

    probs: Optional[torch.Tensor] = None
    starts: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


DIAGNOSTIC_CSV_COLUMNS: List[str] = [
    "step",
    "teacher_forcing",
    "teacher_forcing_prob",
    "loss",
    "answer_loss",
    "writer_loss",
    "sinkhorn_warmup_loss",
    "sinkhorn_warmup_weight_effective",
    "primary_loss_weight_effective",
    "grad_norm_total_preclip",
    "grad_norm_local_blocks",
    "grad_norm_selective_recurrent",
    "grad_norm_fast_memory",
    "grad_norm_router",
    "grad_norm_writer",
    "grad_norm_episodic_memory",
    "grad_norm_delta_writer",
    "grad_norm_jepa",
    "grad_norm_embed_head",
    "probe_candidate_entropy_mean",
    "probe_candidate_kl_vs_prev_mean",
    "probe_topk_jaccard_vs_prev_mean",
    "probe_router_logit_abs_mean",
    "probe_router_raw_logit_abs_mean",
    "probe_router_clamp_jacobian_mean",
    "probe_router_raw_to_weight_jacobian_fro_mean",
]
