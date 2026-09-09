import torch

from hpm.data import ANSWER, QUERY, FactRecallConfig, FactRecallDataset
from hpm.write_modes import apply_write_mode, batch_from_memory_selection, first_positions


def test_fact_token_writer_matches_clean_oracle_without_metadata_writes():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=192, window=64, seed=21, task="kv"))
    batch = dataset.sample_batch(4)
    written, stats = apply_write_mode(batch, "fact_token")

    assert torch.equal(written["memory_token_positions"], batch["memory_token_positions"])
    assert torch.equal(written["memory_mask"], batch["memory_mask"])
    assert stats["true_fact_written_rate"] == 1.0
    assert stats["false_write_rate"] == 0.0
    assert stats["missed_fact_rate"] == 0.0


def test_fact_token_writer_preserves_multi_positive_mask():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=224, window=64, seed=23, task="coexisting"))
    batch = dataset.sample_batch(4)
    written, stats = apply_write_mode(batch, "fact_token")

    assert torch.equal(written["positive_memory_mask"], batch["positive_memory_mask"])
    assert torch.all(written["positive_memory_mask"].sum(dim=1) == 2)
    assert stats["true_fact_written_rate"] == 1.0


def test_random_write_never_uses_query_answer_or_post_query_tokens():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=192, window=64, seed=22, task="kv"))
    batch = dataset.sample_batch(8)
    written, _ = apply_write_mode(batch, "random_write")
    query_positions = first_positions(written["input_ids"], QUERY)

    for b in range(written["input_ids"].size(0)):
        valid = written["memory_mask"][b]
        positions = written["memory_token_positions"][b, valid]
        assert torch.all(positions < query_positions[b])
        stored_tokens = written["input_ids"][b, positions.reshape(-1)]
        assert QUERY not in stored_tokens.tolist()
        assert ANSWER not in stored_tokens.tolist()


def test_selected_memory_uses_selected_slot_space_for_positive_labels():
    """Regression: N oracle candidates and C writer slots must not share masks."""
    batch = FactRecallDataset(
        FactRecallConfig(
            seq_len=192,
            window=64,
            seed=29,
            task="causal_salience_kv",
            num_facts=4,
            writer_required_facts=2,
        )
    ).sample_batch(2)

    # Select an original positive candidate that is deliberately beyond slot C
    # and place it in rewritten slot zero.
    selected = torch.zeros(2, 2, 2, dtype=batch["memory_token_positions"].dtype)
    selected_mask = torch.zeros(2, 2, dtype=torch.bool)
    for row in range(2):
        original = int(batch["positive_memory_indices"][row].item())
        selected[row, 0] = batch["memory_token_positions"][row, original]
        selected_mask[row, 0] = True

    rewritten = batch_from_memory_selection(batch, selected, selected_mask)
    assert rewritten["positive_memory_mask"].shape == (2, 2)
    assert torch.equal(rewritten["positive_memory_indices"], torch.zeros(2, dtype=torch.long))
    assert torch.equal(rewritten["positive_memory_mask"], torch.tensor([[True, False], [True, False]]))
