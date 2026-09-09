"""Contract tests for the opt-in exact-forward STE episodic read.

The point of ``ste_topk`` is deliberately narrow: retain the finite-slot
hard-top-k read that B1 writer experiments use, while restoring a training
gradient to a candidate immediately below the read cutoff.  These tests make
both parts of that contract executable.
"""

import pytest
import torch

from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model
from hpm.memory import EpisodicMemory, retrieve_ste_topk, retrieve_topk


@pytest.mark.parametrize("use_null_slot", [False, True])
def test_ste_topk_is_bitwise_hard_topk_in_forward(use_null_slot: bool):
    """The estimator must not change deployed/read-time behavior."""

    torch.manual_seed(7)
    query = torch.randn(2, 4)
    memory_keys = torch.randn(2, 5, 4)
    memory_values = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    null_score = torch.tensor(0.4)

    hard = retrieve_topk(
        query,
        memory_keys,
        memory_values,
        mask,
        top_k=2,
        use_null_slot=use_null_slot,
        null_score=null_score,
    )
    ste = retrieve_ste_topk(
        query,
        memory_keys,
        memory_values,
        mask,
        top_k=2,
        use_null_slot=use_null_slot,
        null_score=null_score,
    )

    for hard_tensor, ste_tensor in zip(hard, ste):
        assert torch.equal(hard_tensor, ste_tensor)


def test_ste_topk_gives_a_near_miss_slot_a_real_selection_gradient():
    """Hard top-1 disconnects slot 1; STE top-1 must not."""

    query = torch.tensor([[1.0, 0.0]])
    memory_keys = torch.tensor(
        [[[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]], requires_grad=True
    )
    # The first two values have different loss contributions; otherwise a
    # reallocation of membership would correctly have no effect on the loss.
    memory_values = torch.tensor([[[1.0, 0.5], [0.0, 1.0], [0.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    hard, *_ = retrieve_topk(query, memory_keys, memory_values, mask, top_k=1)
    hard.sum().backward()
    hard_near_miss_grad = memory_keys.grad[0, 1].clone()

    memory_keys.grad = None
    ste, *_ = retrieve_ste_topk(query, memory_keys, memory_values, mask, top_k=1)
    ste.sum().backward()
    ste_near_miss_grad = memory_keys.grad[0, 1]

    assert torch.equal(hard_near_miss_grad, torch.zeros(2))
    assert ste_near_miss_grad.abs().sum().item() > 1.0e-8


def test_ste_topk_no_grad_fast_path_is_exactly_hard_topk():
    torch.manual_seed(19)
    query = torch.randn(2, 3)
    keys = torch.randn(2, 4, 3)
    values = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    with torch.no_grad():
        hard = retrieve_topk(query, keys, values, mask, top_k=2)
        ste = retrieve_ste_topk(query, keys, values, mask, top_k=2)

    for hard_tensor, ste_tensor in zip(hard, ste):
        assert torch.equal(hard_tensor, ste_tensor)


def test_ste_topk_never_selects_or_backpropagates_through_padding():
    query = torch.tensor([[1.0, 0.0]])
    memory_keys = torch.tensor(
        [[[1.0, 0.0], [0.999, 0.001], [100.0, 0.0], [99.0, 0.0]]], requires_grad=True
    )
    memory_values = torch.tensor([[[1.0, 0.5], [0.0, 1.0], [8.0, 8.0], [9.0, 9.0]]])
    mask = torch.tensor([[True, True, False, False]])

    retrieved, _, top_indices, _, _ = retrieve_ste_topk(
        query, memory_keys, memory_values, mask, top_k=1
    )
    retrieved.sum().backward()

    assert torch.equal(top_indices, torch.tensor([[0]]))
    assert torch.equal(memory_keys.grad[0, 2:], torch.zeros(2, 2))


def test_ste_topk_batched_variable_padding_matches_independent_rows_and_gradients():
    """The vectorized equal-count path must preserve the row-wise estimator."""

    torch.manual_seed(29)
    query = torch.randn(2, 3)
    keys = torch.randn(2, 5, 3, requires_grad=True)
    values = torch.randn(2, 5, 3)
    # Three active slots in each row, but at different padded positions.
    mask = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])

    batched, _, batched_indices, _, _ = retrieve_ste_topk(query, keys, values, mask, top_k=1)
    batched.sum().backward()
    batched_grad = keys.grad.detach().clone()

    row_outputs = []
    row_grads = []
    for row in range(2):
        row_keys = keys.detach()[row : row + 1].clone().requires_grad_(True)
        row_output, _, row_indices, _, _ = retrieve_ste_topk(
            query[row : row + 1], row_keys, values[row : row + 1], mask[row : row + 1], top_k=1
        )
        row_output.sum().backward()
        row_outputs.append(row_output.detach())
        row_grads.append(row_keys.grad.detach())
        assert torch.equal(batched_indices[row : row + 1], row_indices)

    assert torch.allclose(batched.detach(), torch.cat(row_outputs, dim=0), atol=1.0e-6, rtol=0.0)
    assert torch.allclose(batched_grad, torch.cat(row_grads, dim=0), atol=1.0e-6, rtol=0.0)


def test_ste_topk_rejects_an_underspecified_padded_read():
    query = torch.randn(1, 3)
    keys = torch.randn(1, 3, 3)
    values = torch.randn(1, 3, 3)
    mask = torch.tensor([[True, False, False]])

    with pytest.raises(ValueError, match="at least top_k valid episodic slots"):
        retrieve_ste_topk(query, keys, values, mask, top_k=2)


def test_ste_topk_rejects_unvalidated_learned_writer_select_bias_composition():
    query = torch.randn(1, 3)
    keys = torch.randn(1, 3, 3)
    values = torch.randn(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="fixed episodic candidate set"):
        retrieve_ste_topk(
            query,
            keys,
            values,
            mask,
            top_k=1,
            select_bias=torch.zeros(1, 3),
        )


def test_episodic_memory_ste_override_preserves_multi_hop_forward_result():
    torch.manual_seed(11)
    memory = EpisodicMemory(d_model=4, read_mode="ste_topk")
    token_embeddings = torch.randn(1, 9, 4)
    positions = torch.tensor([[[0, 1], [2, 3], [4, 5]]])
    mask = torch.tensor([[True, True, True]])
    query_key_positions = torch.tensor([6])

    hard_out, hard_info = memory(
        token_embeddings,
        positions,
        mask,
        query_key_positions,
        top_k=2,
        num_hops=2,
        read_mode="hard_topk",
    )
    ste_out, ste_info = memory(
        token_embeddings,
        positions,
        mask,
        query_key_positions,
        top_k=2,
        num_hops=2,
    )

    assert torch.equal(hard_out, ste_out)
    for field in ("scores", "top_indices", "weights", "null_weight", "retrieval_loss"):
        assert torch.equal(hard_info[field], ste_info[field])


def test_hpm_v2_config_wires_opt_in_ste_read_into_episodic_memory():
    config = HpmLiteV2Config(
        d_model=16,
        layers=1,
        heads=2,
        window=8,
        episodic_read_mode="ste_topk",
        episodic_read_ste_eps=0.2,
        episodic_read_ste_iters=31,
    )
    model = HpmLiteV2Model(config)

    assert model.episodic_memory.read_mode == "ste_topk"
    assert model.episodic_memory.read_ste_eps == 0.2
    assert model.episodic_memory.read_ste_iters == 31


def test_hpm_v2_rejects_unvalidated_ste_read_plus_learned_writer_composition():
    with pytest.raises(ValueError, match="fixed, oracle-written candidate set"):
        HpmLiteV2Model(
            HpmLiteV2Config(
                d_model=16,
                layers=1,
                heads=2,
                window=8,
                episodic_read_mode="ste_topk",
                use_learned_writer=True,
            )
        )
