import torch

from hpm.data import FactRecallConfig, FactRecallDataset
from hpm.model import HpmLiteConfig, HpmLiteModel
from hpm.write_modes import batch_from_memory_selection, writer_metrics


def test_learned_writer_forward_shapes_and_loss():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=128, window=32, num_facts=4, seed=7))
    batch = dataset.sample_batch(3)
    model = HpmLiteModel(
        HpmLiteConfig(
            model_type="hpm_lite",
            d_model=32,
            layers=1,
            heads=4,
            window=32,
            max_seq_len=129,
            use_learned_writer=True,
            use_null_slot=True,
        )
    )
    output = model(
        batch["input_ids"],
        memory_token_positions=batch["memory_token_positions"],
        memory_mask=batch["memory_mask"],
        answer_positions=batch["answer_positions"],
        query_key_positions=batch["query_key_positions"],
        hop_positive_memory_indices=batch["hop_positive_memory_indices"],
        use_learned_writer=True,
        learned_writer_teacher_forcing=False,
    )
    retrieval = output["retrieval"]
    assert torch.isfinite(retrieval["writer_loss"])
    assert retrieval["writer_memory_token_positions"].shape == batch["memory_token_positions"].shape
    assert retrieval["writer_memory_mask"].shape == batch["memory_mask"].shape

    written = batch_from_memory_selection(
        batch,
        retrieval["writer_memory_token_positions"],
        retrieval["writer_memory_mask"],
    )
    stats = writer_metrics(batch, written)
    assert set(stats) == {"avg_written_slots", "true_fact_written_rate", "false_write_rate", "missed_fact_rate"}


def test_learned_writer_teacher_forcing_keeps_retrieval_target_valid():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=128, window=32, num_facts=4, seed=8))
    batch = dataset.sample_batch(2)
    model = HpmLiteModel(
        HpmLiteConfig(
            model_type="hpm_lite",
            d_model=32,
            layers=1,
            heads=4,
            window=32,
            max_seq_len=129,
            use_learned_writer=True,
        )
    )
    output = model(
        batch["input_ids"],
        memory_token_positions=batch["memory_token_positions"],
        memory_mask=batch["memory_mask"],
        answer_positions=batch["answer_positions"],
        query_key_positions=batch["query_key_positions"],
        hop_positive_memory_indices=batch["hop_positive_memory_indices"],
        use_learned_writer=True,
        learned_writer_teacher_forcing=True,
    )
    retrieval = output["retrieval"]
    assert torch.isfinite(retrieval["writer_loss"])
    assert torch.isfinite(retrieval["retrieval_loss"])
