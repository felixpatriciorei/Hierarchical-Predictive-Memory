"""Tests for the read-side sparsemax path (Direction 4.2/4.3 in the research
brief: replace the hard `torch.topk` gather in `retrieve_topk` with a
continuous, exactly-sparse alternative -- see differentiable_topk.sparsemax
and hpm.memory.retrieve_sparsemax).

These tests target the specific gap `retrieve_topk` has and `retrieve_sparsemax`
is meant to close: a slot outside the current hard top-k has EXACTLY zero
gradient with respect to the scores that produced it, no matter how close it
is to the cutoff. Sparsemax should give that slot a real (possibly still
zero-valued, but not zero-*gradient*) path.
"""
import torch

from hpm.differentiable_topk import sparsemax
from hpm.memory import EpisodicMemory, retrieve_sparsemax, retrieve_topk


def test_sparsemax_rows_sum_to_one_and_are_nonnegative():
    torch.manual_seed(0)
    scores = torch.randn(4, 6)
    p = sparsemax(scores)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_sparsemax_gives_exact_zeros_not_just_small_weights():
    # A few very negative (masked-out) slots should get *exactly* 0.0, not
    # merely small values the way a plain softmax would.
    scores = torch.tensor([[5.0, 4.9, -1.0e9, -1.0e9, -1.0e9]])
    p = sparsemax(scores)
    assert torch.equal(p[0, 2:], torch.zeros(3))
    assert p[0, 0] > 0 and p[0, 1] > 0


def test_sparsemax_matches_argmax_when_one_score_dominates():
    scores = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    p = sparsemax(scores)
    assert torch.allclose(p, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6)


def test_hard_topk_gives_zero_gradient_to_a_near_miss_slot():
    # This is the exact limitation the research brief calls the "STE bias
    # floor" / hard-selection gradient gap: a slot just below the top-k
    # cutoff gets no gradient from retrieve_topk no matter how close its
    # score is to making the cut.
    query = torch.tensor([[1.0, 0.0]])
    memory_keys = torch.tensor([[[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]], requires_grad=True)
    memory_values = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    retrieved, *_ = retrieve_topk(query, memory_keys, memory_values, mask, top_k=1)
    retrieved.sum().backward()
    # slot 1 (the near-miss, second-highest score) must have received
    # exactly zero gradient -- it was never inside the top-1 selection.
    assert torch.equal(memory_keys.grad[0, 1], torch.zeros(2))


def test_sparsemax_read_includes_and_grads_a_slot_hard_top1_would_fully_discard():
    # With top_k=1, retrieve_topk keeps ONLY the single highest-scoring slot
    # -- a near-tied runner-up is not merely down-weighted, it is dropped
    # from the computation graph entirely (see the previous test: exactly
    # zero gradient, not just a small one). Sparsemax's support size is not
    # pinned to top_k; when two scores are close, both can end up in the
    # support and both receive real, nonzero gradient. (A slot that sparsemax
    # itself excludes -- weight exactly 0 -- still gets exactly zero
    # gradient, same as hard top-k: sparsemax removes the *fixed-k*
    # limitation, not the "zero grad off the support" property, which is
    # inherent to any exactly-sparse operator.)
    query = torch.tensor([[1.0, 0.0]])
    memory_keys = torch.tensor([[[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]], requires_grad=True)
    # values chosen so slot 0 and slot 1 contribute differently to the loss
    # (a degenerate equal-contribution case would correctly get zero
    # gradient even inside the support, since sparsemax's gradient only
    # reflects how weight would shift *among* support members).
    memory_values = torch.tensor([[[1.0, 0.5], [0.0, 1.0], [0.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    retrieved, _, _, weights, _ = retrieve_sparsemax(query, memory_keys, memory_values, mask, top_k=1)
    assert weights[0, 1] > 0.0  # slot 1 is inside the support despite top_k=1
    retrieved.sum().backward()
    assert memory_keys.grad[0, 1].abs().sum().item() != 0.0


def test_retrieve_sparsemax_return_shapes():
    # NOTE: `weights` is intentionally NOT the same shape as retrieve_topk's
    # -- retrieve_topk's weights live over the k gathered slots (bsz, k),
    # while retrieve_sparsemax's weights live over ALL valid slots
    # (bsz, n_slots), since sparsity is emergent rather than a fixed k.
    # Everything else (retrieved, scores, top_indices, null_weight) matches.
    # Callers that pair `weights` elementwise with a (bsz, k) `top_scores`
    # tensor are not
    # yet compatible with read_mode="sparsemax" -- see
    # docs/engineering/read_side_sparsemax.md.
    torch.manual_seed(0)
    bsz, slots, d = 2, 5, 4
    query = torch.randn(bsz, d)
    memory_keys = torch.randn(bsz, slots, d)
    memory_values = torch.randn(bsz, slots, d)
    mask = torch.ones(bsz, slots, dtype=torch.bool)

    retrieved, scores, top_indices, weights, null_weight = retrieve_sparsemax(
        query, memory_keys, memory_values, mask, top_k=2
    )
    assert retrieved.shape == (bsz, d)
    assert scores.shape == (bsz, slots)
    assert top_indices.shape == (bsz, 2)
    assert weights.shape == (bsz, slots)
    assert null_weight.shape == (bsz,)


def test_episodic_memory_forward_runs_with_sparsemax_read_mode():
    torch.manual_seed(0)
    memory = EpisodicMemory(d_model=4, read_mode="sparsemax")
    token_embeddings = torch.randn(1, 6, 4)
    positions = torch.tensor([[[1, 2], [3, 4]]])
    mask = torch.tensor([[True, True]])
    query_key_positions = torch.tensor([3])

    retrieved, info = memory(token_embeddings, positions, mask, query_key_positions, top_k=1)
    assert retrieved.shape == (1, 4)
    assert "weights" in info


def test_episodic_memory_forward_accepts_per_call_read_mode_override():
    # NOTE: this fixture is deliberately crafted, not random. With only 2
    # memory slots and top_k=1, retrieve_topk's softmax-of-one always
    # returns exactly the single gathered slot's value, whatever its score
    # -- and with generic random inputs, retrieve_sparsemax can *coincidentally*
    # collapse onto that same one-hot slot too (this happened with
    # torch.manual_seed(0) on a randn fixture), making the two read modes
    # agree by chance and producing a flaky test. Here the two candidate
    # slots are built with deliberately near-tied keys (EpisodicMemory's
    # projections are identity-initialized, see _init_identity) so
    # retrieve_sparsemax's support provably spans both slots (real,
    # non-degenerate weight on each) while retrieve_topk still only ever
    # sees the single top-scoring one -- guaranteeing the two modes diverge.
    memory = EpisodicMemory(d_model=4)  # default read_mode="hard_topk"
    token_embeddings = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0],   # 0: unused
            [1.0, 0.0, 0.0, 0.0],   # 1: slot 0 key source
            [1.0, 0.0, 0.0, 0.0],   # 2: slot 0 value source
            [0.999, 0.001, 0.0, 0.0],  # 3: slot 1 key source (near-tied with slot 0)
            [0.0, 1.0, 0.0, 0.0],   # 4: slot 1 value source (differs from slot 0's value)
            [1.0, 0.0, 0.0, 0.0],   # 5: query source (matches slot 0 closely, but not exactly)
        ]]
    )
    positions = torch.tensor([[[1, 2], [3, 4]]])
    mask = torch.tensor([[True, True]])
    query_key_positions = torch.tensor([5])

    hard_out, _ = memory(token_embeddings, positions, mask, query_key_positions, top_k=1)
    soft_out, _ = memory(
        token_embeddings, positions, mask, query_key_positions, top_k=1, read_mode="sparsemax"
    )
    # Different read operators over the same inputs generally disagree.
    assert not torch.allclose(hard_out, soft_out)


def test_unknown_read_mode_raises():
    try:
        EpisodicMemory(d_model=4, read_mode="not_a_real_mode")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown read_mode")
