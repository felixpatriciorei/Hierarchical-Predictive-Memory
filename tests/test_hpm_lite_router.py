import torch

from hpm.data import FactRecallConfig, FactRecallDataset, VOCAB_SIZE
from hpm.metrics import answer_cross_entropy
from hpm.model import HpmLiteConfig, HpmLiteModel, MemoryPathRouter


def test_router_weights_sum_to_one():
    router = MemoryPathRouter(d_model=16)
    local = torch.randn(2, 5, 16)
    recurrent = torch.randn(2, 5, 16)
    episodic = torch.randn(2, 5, 16)
    mixed, weights = router(local, recurrent, episodic)
    assert mixed.shape == local.shape
    assert weights.shape == (2, 5, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_hpm_lite_uses_gru_memory_and_router():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=128, window=32, seed=12, num_facts=3))
    batch = dataset.sample_batch(2)
    model = HpmLiteModel(
        HpmLiteConfig(
            model_type="hpm_lite",
            vocab_size=VOCAB_SIZE,
            d_model=32,
            layers=1,
            heads=4,
            window=32,
            max_seq_len=256,
        )
    )
    output = model(
        batch["input_ids"],
        memory_token_positions=batch["memory_token_positions"],
        memory_mask=batch["memory_mask"],
        answer_positions=batch["answer_positions"],
        query_key_positions=batch["query_key_positions"],
        task="kv",
        hop_positive_memory_indices=batch["hop_positive_memory_indices"],
    )
    logits = output["logits"]
    router_weights = output["retrieval"]["router_weights"]
    assert logits.shape == (2, 128, VOCAB_SIZE)
    assert router_weights.shape == (2, 128, 3)
    assert torch.allclose(router_weights.sum(dim=-1), torch.ones(2, 128), atol=1e-6)
    loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
    loss = loss + 0.1 * output["retrieval"]["retrieval_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.hpm_gru.parameters())
    assert any(parameter.grad is not None for parameter in model.router.parameters())
