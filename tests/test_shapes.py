import torch

from hpm.data import FactRecallConfig, FactRecallDataset, VOCAB_SIZE
from hpm.metrics import answer_cross_entropy
from hpm.model import HpmLiteConfig, HpmLiteModel, make_local_causal_mask


def test_local_mask_blocks_future_and_outside_window():
    mask = make_local_causal_mask(seq_len=6, window=2)
    assert mask[4, 4]
    assert mask[4, 3]
    assert mask[4, 2]
    assert not mask[4, 1]
    assert not mask[2, 3]


def test_models_forward_and_backward():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=128, window=32, seed=5, num_facts=3))
    batch = dataset.sample_batch(2)
    for model_type in ["local", "epmem", "hpm_lite", "hebbian"]:
        model = HpmLiteModel(
            HpmLiteConfig(
                model_type=model_type,
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
        assert logits.shape == (2, 128, VOCAB_SIZE)
        loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        if output["retrieval"]:
            loss = loss + 0.1 * output["retrieval"]["retrieval_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        finite_grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        assert finite_grads and all(finite_grads)
