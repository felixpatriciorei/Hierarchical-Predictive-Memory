"""Tests for the aggregate R1 episodic-read candidate-count audit."""

import importlib.util
from pathlib import Path

import pytest


def _analysis_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_episodic_read_scale.py"
    spec = importlib.util.spec_from_file_location("episodic_read_scale_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paired_row(num_facts: int, seed: int, exact_delta: float, top1_delta: float):
    hard_exact, hard_top1, hard_ce, hard_margin, hard_wall = 0.10, 0.12, 5.0, -2.0, 10.0
    return {
        "num_facts": str(num_facts),
        "seed": str(seed),
        "hard_eval_answer_exact": str(hard_exact),
        "ste_eval_answer_exact": str(hard_exact + exact_delta),
        "delta_ste_minus_hard_eval_answer_exact": str(exact_delta),
        "hard_eval_answer_ce": str(hard_ce),
        "ste_eval_answer_ce": "2.0",
        "delta_ste_minus_hard_eval_answer_ce": "-3.0",
        "hard_eval_retrieval_top1": str(hard_top1),
        "ste_eval_retrieval_top1": str(hard_top1 + top1_delta),
        "delta_ste_minus_hard_eval_retrieval_top1": str(top1_delta),
        "hard_eval_retrieval_margin": str(hard_margin),
        "ste_eval_retrieval_margin": "1.0",
        "delta_ste_minus_hard_eval_retrieval_margin": "3.0",
        "hard_train_wall_time_sec": str(hard_wall),
        "ste_train_wall_time_sec": "15.0",
        "delta_ste_minus_hard_train_wall_time_sec": "5.0",
    }


def _pretrain_row(num_facts: int, seed: int):
    return {
        "num_facts": str(num_facts),
        "seed": str(seed),
        "task": "aliased_kv",
        "pretrain_eval_retrieval_top1": "0.18",
        "max_allowed_retrieval_top1": "0.50",
    }


def _summary_row(num_facts: int, exact: float, top1: float):
    return {
        "num_facts": str(num_facts),
        "paired_seed_count": "1",
        "mean_delta_ste_minus_hard_eval_answer_exact": str(exact),
        "mean_delta_ste_minus_hard_eval_retrieval_top1": str(top1),
        "mean_ste_over_hard_wall_time_ratio": "1.5",
    }


def test_scale_summary_validation_accepts_matching_per_m_paired_audits():
    analysis = _analysis_module()
    reports = {
        4: analysis.paired_effect_summary([_paired_row(4, 1, 0.5, 0.4)], [_pretrain_row(4, 1)]),
        8: analysis.paired_effect_summary([_paired_row(8, 1, 0.6, 0.3)], [_pretrain_row(8, 1)]),
    }

    analysis.validate_scale_summary(
        [_summary_row(4, 0.5, 0.4), _summary_row(8, 0.6, 0.3)], reports
    )


def test_scale_summary_validation_rejects_mismatched_root_mean():
    analysis = _analysis_module()
    reports = {
        4: analysis.paired_effect_summary([_paired_row(4, 1, 0.5, 0.4)], [_pretrain_row(4, 1)]),
    }

    with pytest.raises(ValueError, match="scale_summary mismatch"):
        analysis.validate_scale_summary([_summary_row(4, 0.4, 0.4)], reports)
