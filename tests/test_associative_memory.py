"""Tests for DeltaRuleEpisodicMemory (Direction 4.1: replace the writer's
hard torch.topk candidate-position selection with a continuous delta-rule
write). See hpm/associative_memory.py's module docstring for why this
targets the WRITER rather than the read side that
tests/test_sparsemax_read.py covers.

The core property under test: every candidate must receive a real, nonzero
gradient through its write_gate, not just whichever candidates a hard top-k
would have kept. That gradient path is exactly what
`SupervisedMemoryWriter`'s `torch.topk(masked_logits, k=slots)` structurally
cannot give -- excluded candidates are never gathered, so they are outside
the computation graph for `retrieved` entirely, no matter how close their
score was to the cutoff.
"""
import torch

from hpm.associative_memory import DeltaRuleEpisodicMemory, budgeted_sigmoid_write_gate


def _toy_batch(bsz=1, seq_len=None, d_model=4, num_candidates=6, seed=0):
    torch.manual_seed(seed)
    # Non-overlapping [start, start+1] pairs per candidate -- overlapping
    # candidates would make the "does candidate i's content affect the
    # output" isolation tests below ambiguous (perturbing one candidate's
    # token positions would also perturb its neighbor's).
    if seq_len is None:
        seq_len = 2 * num_candidates + 2
    token_embeddings = torch.randn(bsz, seq_len, d_model)
    starts = torch.arange(num_candidates) * 2
    candidate_positions = torch.stack([starts, starts + 1], dim=-1)[None].expand(
        bsz, num_candidates, 2
    ).contiguous()
    candidate_valid_mask = torch.ones(bsz, num_candidates, dtype=torch.bool)
    query_key_positions = torch.full((bsz,), seq_len - 1, dtype=torch.long)
    return token_embeddings, candidate_positions, candidate_valid_mask, query_key_positions


def test_forward_shape():
    mem = DeltaRuleEpisodicMemory(d_model=4)
    token_embeddings, positions, valid, query_pos = _toy_batch()
    write_gate = torch.full((1, positions.size(1)), 0.3)
    retrieved, info = mem(token_embeddings, positions, valid, write_gate, query_pos)
    assert retrieved.shape == (1, 4)
    assert "write_gate_mean" in info and "memory_frobenius_norm" in info


def test_no_torch_topk_used():
    import inspect

    src = inspect.getsource(DeltaRuleEpisodicMemory)
    assert "topk" not in src


def test_gradient_reaches_every_candidates_write_gate_not_just_a_hard_subset():
    torch.manual_seed(0)
    mem = DeltaRuleEpisodicMemory(d_model=4)
    token_embeddings, positions, valid, query_pos = _toy_batch(num_candidates=8)
    write_gate = torch.rand(1, positions.size(1), requires_grad=True) * 0.8 + 0.05  # in (0.05, 0.85)

    retrieved, _ = mem(token_embeddings, positions, valid, write_gate, query_pos)
    retrieved.sum().backward()

    assert write_gate.grad is not None
    # Every single candidate -- not just a top-k subset -- must have a
    # nonzero gradient; this is the whole point of the delta-rule rewrite.
    assert torch.all(write_gate.grad.abs() > 0), write_gate.grad


def test_zero_gate_candidate_content_does_not_affect_output():
    # A candidate with write_gate exactly 0 is fully excluded from the
    # memory write -- changing its key/value content should not change the
    # retrieved output at all (same "hard exclude" semantics EpisodicMemory's
    # memory_mask has, just expressed continuously here).
    torch.manual_seed(1)
    mem = DeltaRuleEpisodicMemory(d_model=4)
    token_embeddings, positions, valid, query_pos = _toy_batch(num_candidates=5)
    write_gate = torch.tensor([[0.4, 0.0, 0.6, 0.2, 0.1]])

    retrieved_a, _ = mem(token_embeddings, positions, valid, write_gate, query_pos)

    perturbed = token_embeddings.clone()
    # scramble the token embeddings feeding candidate index 1's key/value
    # positions -- candidate 1 has write_gate == 0.
    key_pos, value_pos = positions[0, 1].tolist()
    perturbed[0, key_pos] = torch.randn(4)
    perturbed[0, value_pos] = torch.randn(4)

    retrieved_b, _ = mem(perturbed, positions, valid, write_gate, query_pos)
    assert torch.allclose(retrieved_a, retrieved_b, atol=1e-5)


def test_invalid_candidate_is_hard_excluded_regardless_of_write_gate():
    # candidate_valid_mask forces the gate to 0 even if the caller passed a
    # nonzero write_gate for a structurally invalid position (mirrors
    # EpisodicMemory's memory_mask hard-exclude semantics).
    torch.manual_seed(2)
    mem = DeltaRuleEpisodicMemory(d_model=4)
    token_embeddings, positions, valid, query_pos = _toy_batch(num_candidates=5)
    valid = valid.clone()
    valid[0, 1] = False
    write_gate = torch.tensor([[0.4, 0.9, 0.6, 0.2, 0.1]])  # nonzero for candidate 1 anyway

    retrieved_a, _ = mem(token_embeddings, positions, valid, write_gate, query_pos)

    perturbed = token_embeddings.clone()
    key_pos, value_pos = positions[0, 1].tolist()
    perturbed[0, key_pos] = torch.randn(4)
    perturbed[0, value_pos] = torch.randn(4)
    retrieved_b, _ = mem(perturbed, positions, valid, write_gate, query_pos)

    assert torch.allclose(retrieved_a, retrieved_b, atol=1e-5)


def test_budgeted_sigmoid_gate_caps_total_mass_and_keeps_all_candidates_differentiable():
    logits = torch.zeros(1, 6, requires_grad=True)
    valid = torch.tensor([[True, True, False, True, True, True]])
    gate = budgeted_sigmoid_write_gate(logits, valid, write_budget=2)

    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)
    assert gate[0, 2].item() == 0.0
    assert torch.allclose(gate.sum(dim=-1), torch.tensor([2.0]))

    # A non-constant downstream signal receives gradient from every valid
    # candidate. This is a continuous capacity constraint, not hard top-k.
    (gate * torch.arange(1.0, 7.0)).sum().backward()
    assert torch.all(logits.grad[0, valid[0]].abs() > 0)
    assert logits.grad[0, 2].item() == 0.0
