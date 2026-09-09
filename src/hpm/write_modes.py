from __future__ import annotations

from typing import Dict, Tuple

import torch

from .data import ANSWER, FACT, IF, QUERY

WRITE_MODES = {"oracle", "fact_token", "random_write", "learned"}


def parse_write_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [mode for mode in modes if mode not in WRITE_MODES]
    if unknown:
        raise ValueError(f"unknown write mode(s): {unknown}")
    return modes


def apply_write_mode(batch: Dict[str, torch.Tensor], write_mode: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    if write_mode not in WRITE_MODES:
        raise ValueError(f"unknown write_mode: {write_mode}")
    if write_mode == "oracle":
        return clone_batch(batch), writer_metrics(batch, batch)
    if write_mode == "learned":
        # Learned writes are selected inside the model from learned scores.
        # Return the oracle batch here because the writer still needs support
        # labels for supervised warm-start training.
        return clone_batch(batch), writer_metrics(batch, batch)

    if write_mode == "fact_token":
        rewritten = build_fact_token_writes(batch)
    else:
        fact_token = build_fact_token_writes(batch)
        rewritten = build_random_writes(batch, fact_token["memory_mask"].sum(dim=1))
    return rewritten, writer_metrics(batch, rewritten)


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def first_positions(input_ids: torch.Tensor, token: int) -> torch.Tensor:
    positions = []
    for row in input_ids:
        found = (row == token).nonzero(as_tuple=False)
        if found.numel() == 0:
            raise ValueError(f"token {token} not found in input_ids")
        positions.append(found[0, 0])
    return torch.stack(positions)


def build_fact_token_writes(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    rewritten = clone_batch(batch)
    input_ids = batch["input_ids"]
    bsz, max_slots, _ = batch["memory_token_positions"].shape
    device = input_ids.device
    query_positions = first_positions(input_ids, QUERY)

    memory_positions = torch.zeros_like(batch["memory_token_positions"])
    memory_spans = torch.zeros_like(batch["memory_spans"])
    memory_condition_positions = torch.full_like(batch.get("memory_condition_positions", batch["memory_token_positions"][:, :, 0]), -1)
    memory_mask = torch.zeros_like(batch["memory_mask"])

    for b in range(bsz):
        slot = 0
        query_pos = int(query_positions[b].item())
        for pos in range(query_pos):
            if int(input_ids[b, pos].item()) != FACT:
                continue
            if pos + 2 >= query_pos:
                continue
            if slot >= max_slots:
                break
            span_end = pos + 4 if pos + 4 < query_pos and int(input_ids[b, pos + 3].item()) == IF else pos + 2
            memory_positions[b, slot] = torch.tensor([pos + 1, pos + 2], device=device)
            memory_spans[b, slot] = torch.tensor([pos, span_end], device=device)
            if pos + 4 < query_pos and int(input_ids[b, pos + 3].item()) == IF:
                memory_condition_positions[b, slot] = pos + 4
            memory_mask[b, slot] = True
            slot += 1

    rewritten["memory_token_positions"] = memory_positions
    rewritten["memory_condition_positions"] = memory_condition_positions
    rewritten["memory_spans"] = memory_spans
    rewritten["memory_mask"] = memory_mask
    update_positive_indices(rewritten, batch)
    assert_pre_query_writes(rewritten)
    return rewritten


def build_random_writes(batch: Dict[str, torch.Tensor], target_counts: torch.Tensor) -> Dict[str, torch.Tensor]:
    rewritten = clone_batch(batch)
    input_ids = batch["input_ids"]
    bsz, max_slots, _ = batch["memory_token_positions"].shape
    device = input_ids.device
    query_positions = first_positions(input_ids, QUERY)

    memory_positions = torch.zeros_like(batch["memory_token_positions"])
    memory_spans = torch.zeros_like(batch["memory_spans"])
    memory_condition_positions = torch.full_like(batch.get("memory_condition_positions", batch["memory_token_positions"][:, :, 0]), -1)
    memory_mask = torch.zeros_like(batch["memory_mask"])

    forbidden = torch.tensor([QUERY, ANSWER], device=device)
    for b in range(bsz):
        query_pos = int(query_positions[b].item())
        candidates = []
        for start in range(max(0, query_pos - 1)):
            pair = input_ids[b, start : start + 2]
            if torch.isin(pair, forbidden).any():
                continue
            candidates.append(start)
        if not candidates:
            continue
        count = min(int(target_counts[b].item()), max_slots, len(candidates))
        order = torch.randperm(len(candidates), device=device)[:count]
        for slot, candidate_index in enumerate(order.tolist()):
            start = candidates[candidate_index]
            memory_positions[b, slot] = torch.tensor([start, start + 1], device=device)
            memory_spans[b, slot] = torch.tensor([start, start + 1], device=device)
            memory_mask[b, slot] = True

    rewritten["memory_token_positions"] = memory_positions
    rewritten["memory_condition_positions"] = memory_condition_positions
    rewritten["memory_spans"] = memory_spans
    rewritten["memory_mask"] = memory_mask
    update_positive_indices(rewritten, batch)
    assert_pre_query_writes(rewritten)
    return rewritten



def batch_from_memory_selection(
    oracle: Dict[str, torch.Tensor],
    memory_token_positions: torch.Tensor,
    memory_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build a batch-style memory view from learned writer selections."""

    rewritten = clone_batch(oracle)
    rewritten["memory_token_positions"] = memory_token_positions.detach().clone()
    rewritten["memory_mask"] = memory_mask.detach().clone()
    rewritten["memory_spans"] = memory_token_positions.detach().clone()
    # memory_condition_positions must be shaped [bsz, writer_slots] to match
    # memory_token_positions/memory_mask above -- NOT [bsz, num_facts] like the
    # oracle batch's own tensor. When --memory-slots < --num-facts these two
    # slot counts legitimately differ, so the template has to come from the
    # writer's own selection, not from the oracle.
    condition_dtype = oracle.get(
        "memory_condition_positions", oracle["memory_token_positions"][:, :, 0]
    ).dtype
    rewritten["memory_condition_positions"] = torch.full(
        memory_token_positions.shape[:2],
        -1,
        dtype=condition_dtype,
        device=memory_token_positions.device,
    )
    update_positive_indices(rewritten, oracle)
    assert_pre_query_writes(rewritten)
    return rewritten


def update_positive_indices(rewritten: Dict[str, torch.Tensor], oracle: Dict[str, torch.Tensor]) -> None:
    bsz, slots, _ = rewritten["memory_token_positions"].shape
    positive = torch.full_like(oracle["positive_memory_indices"], -1)
    hop_positive = torch.full_like(oracle["hop_positive_memory_indices"], -100)
    # The rewritten writer may have C slots while the oracle candidate list
    # has N slots.  Labels returned here must live in rewritten slot space;
    # copying the oracle shape silently misaligns retrieval diagnostics when
    # C != N.  This is especially important for the finite-capacity causal
    # writer control, where a selected positive can originate at any one of N
    # randomly ordered candidate records.
    positive_mask = torch.zeros(
        (bsz, slots),
        dtype=torch.bool,
        device=rewritten["memory_mask"].device,
    )

    for b in range(bsz):
        for hop in range(oracle["hop_positive_memory_indices"].size(1)):
            oracle_slot = int(oracle["hop_positive_memory_indices"][b, hop].item())
            if oracle_slot < 0:
                continue
            oracle_pair = oracle["memory_token_positions"][b, oracle_slot]
            matched = find_matching_slot(rewritten, b, oracle_pair)
            if matched >= 0:
                hop_positive[b, hop] = matched
        oracle_positive = int(oracle["positive_memory_indices"][b].item())
        if oracle_positive >= 0:
            oracle_pair = oracle["memory_token_positions"][b, oracle_positive]
            matched = find_matching_slot(rewritten, b, oracle_pair)
            if matched >= 0:
                positive[b] = matched
        for oracle_slot in range(oracle["memory_token_positions"].size(1)):
            if not bool(oracle["positive_memory_mask"][b, oracle_slot].item()):
                continue
            oracle_pair = oracle["memory_token_positions"][b, oracle_slot]
            matched = find_matching_slot(rewritten, b, oracle_pair)
            if matched >= 0:
                positive_mask[b, matched] = True

    rewritten["positive_memory_indices"] = positive
    rewritten["positive_memory_mask"] = positive_mask
    rewritten["hop_positive_memory_indices"] = hop_positive


def find_matching_slot(batch: Dict[str, torch.Tensor], sample_index: int, pair: torch.Tensor) -> int:
    for slot in range(batch["memory_token_positions"].size(1)):
        if not bool(batch["memory_mask"][sample_index, slot].item()):
            continue
        if torch.equal(batch["memory_token_positions"][sample_index, slot], pair):
            return slot
    return -1


def writer_metrics(oracle: Dict[str, torch.Tensor], written: Dict[str, torch.Tensor]) -> Dict[str, float]:
    true_total = int(oracle["memory_mask"].sum().item())
    written_total = int(written["memory_mask"].sum().item())
    true_written = 0
    false_written = 0

    for b in range(written["memory_token_positions"].size(0)):
        true_pairs = {
            slot_signature(oracle, b, slot)
            for slot in range(oracle["memory_token_positions"].size(1))
            if bool(oracle["memory_mask"][b, slot].item())
        }
        for slot in range(written["memory_token_positions"].size(1)):
            if not bool(written["memory_mask"][b, slot].item()):
                continue
            pair = slot_signature(written, b, slot)
            if pair in true_pairs:
                true_written += 1
            else:
                false_written += 1

    missed = max(true_total - true_written, 0)
    batch_size = max(int(written["memory_token_positions"].size(0)), 1)
    return {
        "avg_written_slots": written_total / batch_size,
        "true_fact_written_rate": true_written / max(true_total, 1),
        "false_write_rate": false_written / max(written_total, 1),
        "missed_fact_rate": missed / max(true_total, 1),
    }


def slot_signature(batch: Dict[str, torch.Tensor], sample_index: int, slot: int) -> tuple[int, int, int]:
    key_pos = int(batch["memory_token_positions"][sample_index, slot, 0].item())
    value_pos = int(batch["memory_token_positions"][sample_index, slot, 1].item())
    cond_pos = -1
    if "memory_condition_positions" in batch:
        cond_pos = int(batch["memory_condition_positions"][sample_index, slot].item())
    return key_pos, value_pos, cond_pos


def assert_pre_query_writes(batch: Dict[str, torch.Tensor]) -> None:
    query_positions = first_positions(batch["input_ids"], QUERY)
    valid_positions = batch["memory_token_positions"][batch["memory_mask"]]
    if valid_positions.numel() == 0:
        return
    for b in range(batch["memory_token_positions"].size(0)):
        valid = batch["memory_mask"][b]
        if not valid.any():
            continue
        positions = batch["memory_token_positions"][b, valid]
        if torch.any(positions >= query_positions[b]):
            raise AssertionError("write mode stored QUERY/ANSWER or post-query token")
        if "memory_condition_positions" in batch:
            conditions = batch["memory_condition_positions"][b, valid]
            valid_conditions = conditions >= 0
            if torch.any(conditions[valid_conditions] >= query_positions[b]):
                raise AssertionError("write mode stored post-query condition token")
