import torch

from hpm.data import FactRecallConfig, FactRecallDataset
from hpm.memory import EpisodicMemory, HebbianMemory, retrieve_topk


def test_retrieve_topk_returns_expected_slot():
    query = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    memory_keys = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    memory_values = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True]])

    retrieved, scores, top_indices, weights, null_weight = retrieve_topk(query, memory_keys, memory_values, mask, top_k=1)
    assert top_indices.tolist() == [[1]]
    assert torch.allclose(retrieved, memory_values[:, 1, :])
    assert scores[0, 1] > scores[0, 0]
    assert torch.allclose(weights, torch.ones_like(weights))
    assert torch.allclose(null_weight, torch.zeros_like(null_weight))


def test_retrieve_topk_null_slot_can_absorb_bad_matches():
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    memory_keys = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    memory_values = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0]]])
    mask = torch.tensor([[False, False]])

    retrieved, _, _, weights, null_weight = retrieve_topk(
        query,
        memory_keys,
        memory_values,
        mask,
        top_k=1,
        use_null_slot=True,
        null_score=torch.tensor(0.0),
    )
    assert weights.shape == (1, 2)
    assert null_weight.item() > 0.999
    assert torch.allclose(retrieved, torch.zeros_like(retrieved), atol=1e-6)


def test_memory_module_topk_contains_expected_slot():
    memory = EpisodicMemory(d_model=4)
    token_embeddings = torch.zeros(1, 6, 4)
    token_embeddings[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    token_embeddings[0, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    token_embeddings[0, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    token_embeddings[0, 4] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    positions = torch.tensor([[[1, 2], [3, 4]]])
    mask = torch.tensor([[True, True]])
    query_key_positions = torch.tensor([3])

    _, info = memory(token_embeddings, positions, mask, query_key_positions, top_k=2)
    assert info["top_indices"][0, 0].item() == 1
    assert 1 in info["top_indices"][0].tolist()


def test_memory_controls_break_slot_alignment():
    memory = EpisodicMemory(d_model=4)
    token_embeddings = torch.zeros(1, 6, 4)
    token_embeddings[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    token_embeddings[0, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    token_embeddings[0, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    token_embeddings[0, 4] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    positions = torch.tensor([[[1, 2], [3, 4]]])
    mask = torch.tensor([[True, True]])
    query_key_positions = torch.tensor([3])

    normal, _ = memory(token_embeddings, positions, mask, query_key_positions, top_k=1)
    shuffled, _ = memory(token_embeddings, positions, mask, query_key_positions, top_k=1, memory_control="shuffle_values")
    corrupt, _ = memory(token_embeddings, positions, mask, query_key_positions, top_k=1, memory_control="corrupt_values")
    no_retrieval, info = memory(token_embeddings, positions, mask, query_key_positions, top_k=1, memory_control="no_retrieval")
    assert not torch.allclose(normal, shuffled)
    assert not torch.allclose(normal, corrupt)
    assert torch.allclose(no_retrieval, torch.zeros_like(no_retrieval))
    assert "top_indices" not in info


def test_hebbian_memory_returns_expected_direction():
    memory = HebbianMemory(d_model=4, decay=0.0, eta=1.0)
    token_embeddings = torch.zeros(1, 4, 4)
    token_embeddings[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    token_embeddings[0, 2] = torch.tensor([0.0, 0.0, 0.0, 2.0])
    positions = torch.tensor([[[1, 2]]])
    mask = torch.tensor([[True]])
    query_key_positions = torch.tensor([1])

    retrieved, info = memory(token_embeddings, positions, mask, query_key_positions, top_k=1)
    assert info["top_indices"].tolist() == [[0]]
    assert retrieved[0, 3] > 0.0


def test_no_future_answer_span_in_memory_writes():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=192, window=64, seed=4, oracle_memory=False))
    batch = dataset.sample_batch(8)
    assert torch.all(batch["memory_spans"][:, :, 1] < batch["answer_positions"][:, None])
    assert torch.all(batch["memory_token_positions"] < batch["answer_positions"][:, None, None])
