import pytest
import torch

from hpm.data import (
    ALIAS_RANGE,
    ALIASED_KEY_RANGE,
    ANSWER,
    FACT,
    IRRELEVANT,
    NOISE_WORD,
    NO_VALUE,
    REMEMBER,
    FactRecallConfig,
    FactRecallDataset,
)


def test_batch_shapes_and_answer_mapping():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=192, window=64, seed=1, num_facts=4))
    batch = dataset.sample_batch(8)

    assert batch["input_ids"].shape == (8, 192)
    assert batch["target_ids"].shape == (8, 192)
    assert batch["loss_mask"].shape == (8, 192)
    assert batch["memory_token_positions"].shape == (8, 4, 2)
    assert batch["memory_spans"].shape == (8, 4, 2)
    assert batch["answer_positions"].shape == (8,)

    for b in range(8):
        answer_pos = batch["answer_positions"][b].item()
        assert batch["input_ids"][b, answer_pos].item() == ANSWER
        assert batch["target_ids"][b, answer_pos].item() == batch["answer_tokens"][b].item()

        query_key = batch["query_key_tokens"][b].item()
        found_value = None
        for slot in range(batch["memory_token_positions"].size(1)):
            key_pos, value_pos = batch["memory_token_positions"][b, slot].tolist()
            if batch["input_ids"][b, key_pos].item() == query_key:
                found_value = batch["input_ids"][b, value_pos].item()
        assert found_value == batch["answer_tokens"][b].item()


def test_fact_is_outside_local_window():
    window = 64
    dataset = FactRecallDataset(FactRecallConfig(seq_len=192, window=window, seed=2, num_facts=4))
    batch = dataset.sample_batch(16)
    fact_ends = batch["memory_spans"][:, :, 1].max(dim=1).values
    assert torch.all(fact_ends < batch["answer_positions"] - window)


def test_aliased_kv_removes_literal_query_key_match_but_preserves_one_correct_slot():
    batch = FactRecallDataset(
        FactRecallConfig(seq_len=192, window=64, task="aliased_kv", num_facts=8, seed=97)
    ).sample_batch(24)

    for row in range(batch["input_ids"].size(0)):
        query_alias = int(batch["query_key_tokens"][row].item())
        query_position = int(batch["query_key_positions"][row].item())
        positive_slot = int(batch["positive_memory_indices"][row].item())
        key_positions = batch["memory_token_positions"][row, :, 0]
        memory_keys = batch["input_ids"][row, key_positions]
        positive_key = int(memory_keys[positive_slot].item())
        value_position = int(batch["memory_token_positions"][row, positive_slot, 1].item())

        assert ALIAS_RANGE[0] <= query_alias < ALIAS_RANGE[1]
        assert torch.all((memory_keys >= ALIASED_KEY_RANGE[0]) & (memory_keys < ALIASED_KEY_RANGE[1]))
        assert query_alias not in memory_keys.tolist()
        assert batch["input_ids"][row, query_position].item() == query_alias
        assert query_alias == ALIAS_RANGE[0] + (positive_key - ALIASED_KEY_RANGE[0])
        assert batch["input_ids"][row, value_position].item() == batch["answer_tokens"][row].item()


def test_twohop_data_maps_to_final_value():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=224, window=64, seed=3, task="twohop", num_facts=4))
    batch = dataset.sample_batch(4)
    assert batch["hop_positive_memory_indices"].shape == (4, 2)
    assert torch.all(batch["positive_memory_indices"] == batch["hop_positive_memory_indices"][:, 1])
    for b in range(4):
        final_slot = batch["positive_memory_indices"][b].item()
        final_value_pos = batch["memory_token_positions"][b, final_slot, 1].item()
        assert batch["input_ids"][b, final_value_pos].item() == batch["answer_tokens"][b].item()


def test_memfail_diagnostic_tasks_have_expected_answer_structure():
    coexisting = FactRecallDataset(
        FactRecallConfig(seq_len=224, window=64, seed=31, task="coexisting", num_facts=4)
    ).sample_batch(4)
    assert coexisting["answer_target_positions"].shape == (4, 2)
    assert torch.all(coexisting["positive_memory_mask"].sum(dim=1) == 2)
    for b in range(4):
        positions = coexisting["answer_target_positions"][b]
        assert torch.equal(coexisting["target_ids"][b, positions], coexisting["answer_token_spans"][b])

    conditional = FactRecallDataset(
        FactRecallConfig(seq_len=224, window=64, seed=32, task="conditional", num_facts=4)
    ).sample_batch(8)
    assert conditional["answer_target_positions"].shape == (8, 1)
    for b in range(8):
        answer = conditional["answer_tokens"][b].item()
        positive_slots = conditional["positive_memory_mask"][b].nonzero(as_tuple=False).reshape(-1)
        if answer == NO_VALUE:
            assert positive_slots.numel() == 0
        else:
            assert positive_slots.numel() == 1
            slot = positive_slots[0].item()
            value_pos = conditional["memory_token_positions"][b, slot, 1].item()
            assert conditional["input_ids"][b, value_pos].item() == answer

    longhop = FactRecallDataset(
        FactRecallConfig(seq_len=224, window=64, seed=33, task="longhop", num_facts=4)
    ).sample_batch(4)
    assert longhop["hop_positive_memory_indices"].shape == (4, 2)
    assert torch.all(longhop["positive_memory_indices"] == longhop["hop_positive_memory_indices"][:, 1])


def test_repaired_conditional_variants_remove_or_balance_no_value_shortcut():
    balanced = FactRecallDataset(
        FactRecallConfig(seq_len=224, window=64, seed=41, task="conditional_balanced", num_facts=4)
    ).sample_batch(8)
    assert torch.sum(balanced["answer_tokens"] == NO_VALUE).item() == 4
    assert torch.sum(balanced["answer_tokens"] != NO_VALUE).item() == 4

    positive_only = FactRecallDataset(
        FactRecallConfig(seq_len=224, window=64, seed=42, task="conditional_positive_only", num_facts=4)
    ).sample_batch(8)
    assert not torch.any(positive_only["answer_tokens"] == NO_VALUE)
    assert torch.all(positive_only["positive_memory_mask"].sum(dim=1) == 1)

    contrastive = FactRecallDataset(
        FactRecallConfig(seq_len=224, window=64, seed=43, task="conditional_contrastive", num_facts=4)
    ).sample_batch(8)
    assert not torch.any(contrastive["answer_tokens"] == NO_VALUE)
    for b in range(8):
        query_key = contrastive["query_key_tokens"][b].item()
        query_condition = contrastive["input_ids"][b, contrastive["query_key_positions"][b].item() + 1].item()
        matching_key_slots = []
        for slot in range(contrastive["memory_token_positions"].size(1)):
            key_pos, value_pos = contrastive["memory_token_positions"][b, slot].tolist()
            if contrastive["input_ids"][b, key_pos].item() == query_key:
                matching_key_slots.append(slot)
                if bool(contrastive["positive_memory_mask"][b, slot].item()):
                    assert contrastive["input_ids"][b, value_pos + 2].item() == query_condition
                    assert contrastive["input_ids"][b, value_pos].item() == contrastive["answer_tokens"][b].item()
        assert len(matching_key_slots) >= 2


def test_hard_twohop_has_no_answer_leak_and_query_key_noise_control():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=512,
            window=64,
            seed=11,
            task="twohop",
            num_facts=16,
            repeated_keys=True,
            similar_values=True,
            distractor_fact_spans=4,
            fact_order="query_last",
        )
    )
    batch = dataset.sample_batch(4)
    query_positions = batch["answer_positions"] - 2
    answer_value_positions = batch["answer_positions"] + 1
    assert torch.all(batch["memory_spans"][:, :, 1] < query_positions[:, None])
    assert not torch.any(batch["memory_token_positions"] == answer_value_positions[:, None, None])

    no_match = FactRecallDataset(
        FactRecallConfig(
            seq_len=512,
            window=64,
            seed=12,
            task="twohop",
            num_facts=16,
            repeated_keys=True,
            query_key_noise_only=True,
        )
    ).sample_batch(4)
    for b in range(4):
        query_key = no_match["query_key_tokens"][b].item()
        fact_key_positions = no_match["memory_token_positions"][b, :, 0]
        fact_keys = no_match["input_ids"][b, fact_key_positions]
        query_pos = no_match["answer_positions"][b].item() - 2
        assert query_key not in fact_keys.tolist()
        assert query_key in no_match["input_ids"][b, :query_pos].tolist()


def test_conditional_stress_slots_are_pre_query_and_typed():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=71,
            task="conditional_contrastive_stress",
            num_facts=16,
            num_positive=1,
            num_hard_negatives=8,
            similarity_mode="mixed",
            slot_order="random",
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(4)

    assert batch["stress_slot_types"].shape == (4, 16)
    assert torch.all(batch["positive_memory_mask"].sum(dim=1) == 1)
    assert torch.all(batch["stress_slot_types"].eq(1).sum(dim=1) == 1)

    query_positions = batch["answer_positions"] - 3
    assert torch.all(batch["memory_spans"][:, :, 1] < query_positions[:, None])
    condition_positions = batch["memory_token_positions"][:, :, 1] + 2
    assert torch.all(condition_positions[batch["memory_mask"]] < query_positions[:, None].expand_as(condition_positions)[batch["memory_mask"]])
    assert not torch.any(batch["memory_token_positions"] == (batch["answer_positions"] + 1)[:, None, None])


def test_coexisting_stress_answer_set_metadata():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=72,
            task="coexisting_stress",
            num_facts=12,
            num_positive=3,
            num_hard_negatives=4,
            similarity_mode="adjacent",
            slot_order="random",
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(4)

    assert batch["answer_target_positions"].shape == (4, 3)
    assert torch.all(batch["positive_memory_mask"].sum(dim=1) == 3)
    assert torch.all(batch["stress_slot_types"].eq(1).sum(dim=1) == 3)
    for b in range(4):
        positions = batch["answer_target_positions"][b]
        assert torch.equal(batch["target_ids"][b, positions], batch["answer_token_spans"][b])


def test_causal_salience_kv_exposes_keep_drop_role_before_write_and_queries_only_keep():
    batch = FactRecallDataset(
        FactRecallConfig(
            seq_len=192,
            window=64,
            seed=83,
            task="causal_salience_kv",
            num_facts=6,
            writer_required_facts=2,
            fact_order="random",
        )
    ).sample_batch(24)

    assert torch.all(batch["memory_mask"].sum(dim=1) == 2)
    assert torch.all(batch["positive_memory_mask"].sum(dim=1) == 1)
    assert torch.all(batch["memory_mask"] & batch["positive_memory_mask"] == batch["positive_memory_mask"])

    required_slots = []
    for row in range(batch["input_ids"].size(0)):
        required_slots.append(tuple(batch["memory_mask"][row].nonzero(as_tuple=False).flatten().tolist()))
        positive_slot = int(batch["positive_memory_indices"][row].item())
        assert bool(batch["memory_mask"][row, positive_slot].item())
        key_pos, value_pos = batch["memory_token_positions"][row, positive_slot].tolist()
        assert batch["input_ids"][row, key_pos].item() == batch["query_key_tokens"][row].item()
        assert batch["input_ids"][row, value_pos].item() == batch["answer_tokens"][row].item()
        for slot in range(batch["memory_token_positions"].size(1)):
            key_pos, _ = batch["memory_token_positions"][row, slot].tolist()
            assert batch["input_ids"][row, key_pos - 1].item() == FACT
            expected_role = REMEMBER if batch["memory_mask"][row, slot] else IRRELEVANT
            assert batch["input_ids"][row, key_pos - 2].item() == expected_role

    # Fixed ordinal position is not a viable shortcut: randomization changes
    # which fact-record slots carry the visible REMEMBER role.
    assert len(set(required_slots)) > 1


def test_causal_salience_kv_rejects_hidden_or_ordinal_role_controls():
    with pytest.raises(ValueError, match="fact_order='random'"):
        FactRecallDataset(
            FactRecallConfig(
                task="causal_salience_kv",
                num_facts=4,
                writer_required_facts=2,
                fact_order="query_last",
            )
        )
    with pytest.raises(ValueError, match="oracle_memory=True"):
        FactRecallDataset(
            FactRecallConfig(
                task="causal_salience_kv",
                num_facts=4,
                writer_required_facts=2,
                oracle_memory=False,
            )
        )


def test_causal_salience_masked_role_negative_control_preserves_records_but_hides_utility():
    visible = FactRecallDataset(
        FactRecallConfig(
            seq_len=192,
            window=64,
            seed=84,
            task="causal_salience_kv",
            num_facts=4,
            writer_required_facts=2,
            writer_role_mode="visible",
        )
    ).sample_batch(4)
    masked = FactRecallDataset(
        FactRecallConfig(
            seq_len=192,
            window=64,
            seed=84,
            task="causal_salience_kv",
            num_facts=4,
            writer_required_facts=2,
            writer_role_mode="masked",
        )
    ).sample_batch(4)

    assert torch.equal(visible["memory_token_positions"], masked["memory_token_positions"])
    assert torch.equal(visible["memory_mask"], masked["memory_mask"])
    assert torch.equal(visible["query_key_tokens"], masked["query_key_tokens"])
    assert torch.equal(visible["answer_tokens"], masked["answer_tokens"])
    for row in range(masked["input_ids"].size(0)):
        key_positions = masked["memory_token_positions"][row, :, 0]
        assert torch.all(masked["input_ids"][row, key_positions - 2] == NOISE_WORD)
