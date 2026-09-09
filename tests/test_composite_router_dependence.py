"""Integration test for evaluate_fixed_router_dependencies on CPU.

This is the structural contract of router_dependency.json -- the artifact
the cross-seed analyzer and the ts-gate certification both consume. It runs
the REAL evaluator end to end (real CompositeSkillDataset batches, real
hpm_lite_v2 forward passes, all 16 frozen-router conditions) at a tiny size
and asserts the identities every downstream consumer silently assumes:

- all 2^4 = 16 removal conditions are present and keyed via removal_key;
- singleton_exact_drop is exactly full-minus-singleton-drop per subtask;
- Shapley attribution sums to A(full) - A(all removed) per subtask
  (efficiency identity, the thing that makes the attributions interpretable);
- answer-position router weights are a convex combination per subtask;
- accuracies live in [0, 1] and the payload carries its provenance kind.

First written 2026-08-26 -- before this date the diagnostic had never been
executed anywhere (the 8 CSVs under runs/composite_four_path came from the
pre-diagnostic harness).
"""
import pytest
import torch

from hpm.composite_harness import (
    build_model,
    evaluate_fixed_router_dependencies,
)
from hpm.composite_skill_data import (
    SUBTASKS,
    CompositeSkillConfig,
    CompositeSkillDataset,
)
from hpm.router_diagnostics import (
    ROUTER_PATH_NAMES,
    all_removed_path_sets,
    removal_key,
)


def _tiny_setup():
    config = CompositeSkillConfig(
        seq_len=96,
        window=16,
        seed=1234,
        copy_span=8,
        mood_block_len=16,
        num_facts=2,
    )
    dataset = CompositeSkillDataset(config)
    model = build_model("hpm_lite_v2", d_model=16, layers=1, heads=2, window=16, seq_len=96, device=torch.device("cpu"))
    return model, dataset, config


def test_evaluate_fixed_router_dependencies_structural_contract():
    model, dataset, config = _tiny_setup()
    result = evaluate_fixed_router_dependencies(
        model=model,
        dataset=dataset,
        config=config,
        model_type="hpm_lite_v2",
        top_k=1,
        batch_size=2,
        batches=2,
        device=torch.device("cpu"),
    )

    assert result["diagnostic_kind"] == "fixed_router_post_gate_conditional_coablation"
    assert tuple(result["path_names"]) == ROUTER_PATH_NAMES

    expected_keys = {removal_key(removed) for removed in all_removed_path_sets(ROUTER_PATH_NAMES)}
    assert set(result["condition_exact"]) == expected_keys
    assert "full" in result["condition_exact"]

    for condition, per_subtask in result["condition_exact"].items():
        assert set(per_subtask) == set(SUBTASKS)
        for value in per_subtask.values():
            assert 0.0 <= value <= 1.0

    # singleton drop identity: drop_s[p] == A(full) - A(drop p)
    for subtask in SUBTASKS:
        full = result["condition_exact"]["full"][subtask]
        for path in ROUTER_PATH_NAMES:
            dropped = result["condition_exact"][removal_key((path,))][subtask]
            assert result["singleton_exact_drop"][subtask][path] == pytest.approx(full - dropped, abs=1e-9)

    # efficiency identity: attributions sum to A(full) - A(all removed)
    all_removed_key = removal_key(ROUTER_PATH_NAMES)
    for subtask in SUBTASKS:
        total = sum(result["shapley_exact_attribution"][subtask].values())
        gap = (
            result["condition_exact"]["full"][subtask]
            - result["condition_exact"][all_removed_key][subtask]
        )
        assert total == pytest.approx(gap, abs=1e-9)

    # answer-position gate weights remain a convex combination per subtask
    for subtask in SUBTASKS:
        weights = result["answer_position_router_weights"][subtask]
        assert set(weights) == set(ROUTER_PATH_NAMES)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
        for value in weights.values():
            assert -1e-6 <= value <= 1.0 + 1e-6


def test_evaluate_fixed_router_dependencies_requires_batches():
    model, dataset, config = _tiny_setup()
    try:
        evaluate_fixed_router_dependencies(
            model=model,
            dataset=dataset,
            config=config,
            model_type="hpm_lite_v2",
            top_k=1,
            batch_size=2,
            batches=0,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "at least one batch" in str(exc)
    else:
        raise AssertionError("batches=0 must be rejected")
