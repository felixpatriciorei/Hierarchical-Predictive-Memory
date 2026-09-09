import torch
from torch import nn

from hpm.diagnostics import (
    ProbeState,
    candidate_entropy,
    candidate_kl,
    collect_module_grad_norms,
    masked_candidate_distribution,
    module_grad_norm,
    topk_jaccard,
)


def test_module_grad_norm_none_module_and_none_grad():
    """None module (e.g. episodic_only=True drops selective_recurrent/
    fast_memory/router entirely) and a module with no grad yet must both
    return None, not 0.0 -- these mean different things (subsystem absent
    vs subsystem present but genuinely untouched by this step's backward)."""
    assert module_grad_norm(None) is None

    linear = nn.Linear(4, 4)
    assert module_grad_norm(linear) is None  # no backward() run yet, .grad is None


def test_module_grad_norm_matches_manual_l2_norm():
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    linear(x).sum().backward()
    expected = torch.cat([p.grad.flatten() for p in linear.parameters()]).norm().item()
    assert abs(module_grad_norm(linear) - expected) < 1e-5


def test_collect_module_grad_norms_skips_absent_submodules():
    """Pins the episodic_only=True case: selective_recurrent/fast_memory/
    router are None on the real model, and collect_module_grad_norms must
    not crash or fabricate a 0.0 for them -- getattr(model, name, None)
    combined with module_grad_norm(None) -> None handles this."""

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.local_blocks = nn.Linear(2, 2)
            self.selective_recurrent = None
            self.fast_memory = None
            self.router = None
            self.writer = nn.Linear(2, 1)
            self.episodic_memory = nn.Linear(2, 2)
            self.jepa = None
            self.token_embedding = nn.Embedding(5, 2)
            self.position_embedding = nn.Embedding(5, 2)
            self.final_ln = nn.LayerNorm(2)

    model = FakeModel()
    x = torch.randn(1, 2)
    loss = model.local_blocks(x).sum() + model.writer(x).sum() + model.episodic_memory(x).sum()
    loss.backward()

    norms = collect_module_grad_norms(model)
    assert norms["grad_norm_local_blocks"] is not None and norms["grad_norm_local_blocks"] > 0
    assert norms["grad_norm_writer"] is not None and norms["grad_norm_writer"] > 0
    assert norms["grad_norm_episodic_memory"] is not None and norms["grad_norm_episodic_memory"] > 0
    # Absent submodules: None, not 0.0.
    assert norms["grad_norm_selective_recurrent"] is None
    assert norms["grad_norm_fast_memory"] is None
    assert norms["grad_norm_router"] is None
    assert norms["grad_norm_jepa"] is None
    # Untouched-by-this-backward submodules (token/position embedding, final_ln
    # never entered the loss above): also None, not 0.0.
    assert norms["grad_norm_embed_head"] is None
    # grad_norm_total_preclip must reflect only the three modules that were
    # actually part of the backward pass.
    assert norms["grad_norm_total_preclip"] is not None and norms["grad_norm_total_preclip"] > 0


def test_masked_candidate_distribution_zeros_invalid_and_sums_to_one():
    logits = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    valid = torch.tensor([[True, True, True, False]])
    probs = masked_candidate_distribution(logits, valid)
    assert probs[0, 3].item() == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_candidate_entropy_uniform_is_higher_than_peaked():
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    assert candidate_entropy(uniform).item() > candidate_entropy(peaked).item()


def test_candidate_kl_zero_for_identical_distributions():
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    assert candidate_kl(probs, probs).item() < 1e-6


def test_candidate_kl_nonzero_and_asymmetric_for_shifted_distributions():
    """Pins that this is genuinely KL (order matters), not a symmetric
    distance -- callers comparing 'step t vs step t-1' need direction."""
    p = torch.tensor([[0.9, 0.05, 0.03, 0.02]])
    q = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    kl_pq = candidate_kl(p, q).item()
    kl_qp = candidate_kl(q, p).item()
    assert kl_pq > 0.0
    assert kl_qp > 0.0
    assert abs(kl_pq - kl_qp) > 1e-4


def test_topk_jaccard_identical_selection_is_one():
    starts = torch.tensor([[2, 5, 9]])
    mask = torch.tensor([[True, True, True]])
    assert topk_jaccard(starts, mask, starts, mask).item() == 1.0


def test_topk_jaccard_disjoint_selection_is_zero():
    starts_t = torch.tensor([[2, 5, 9]])
    starts_prev = torch.tensor([[1, 3, 7]])
    mask = torch.tensor([[True, True, True]])
    assert topk_jaccard(starts_t, mask, starts_prev, mask).item() == 0.0


def test_topk_jaccard_partial_overlap():
    starts_t = torch.tensor([[2, 5, 9]])
    starts_prev = torch.tensor([[2, 5, 7]])
    mask = torch.tensor([[True, True, True]])
    # intersection={2,5}=2, union={2,5,9,7}=4 -> 0.5
    assert abs(topk_jaccard(starts_t, mask, starts_prev, mask).item() - 0.5) < 1e-6


def test_probe_state_starts_empty():
    state = ProbeState()
    assert state.probs is None
    assert state.starts is None
    assert state.mask is None
