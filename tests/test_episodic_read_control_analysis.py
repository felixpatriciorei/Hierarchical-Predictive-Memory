"""Tests for the paired episodic-read result audit."""

import importlib.util
from pathlib import Path

import pytest


def _analysis_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_episodic_read_control.py"
    spec = importlib.util.spec_from_file_location("episodic_read_control_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paired_row(seed: int, exact_delta: float, top1_delta: float):
    hard_exact, hard_top1, hard_ce, hard_margin, hard_wall = 0.10, 0.12, 5.0, -2.0, 10.0
    ste_exact = hard_exact + exact_delta
    ste_top1 = hard_top1 + top1_delta
    ste_ce, ste_margin, ste_wall = 2.0, 1.0, 90.0
    return {
        "seed": str(seed),
        "hard_eval_answer_exact": str(hard_exact),
        "ste_eval_answer_exact": str(ste_exact),
        "delta_ste_minus_hard_eval_answer_exact": str(exact_delta),
        "hard_eval_answer_ce": str(hard_ce),
        "ste_eval_answer_ce": str(ste_ce),
        "delta_ste_minus_hard_eval_answer_ce": str(ste_ce - hard_ce),
        "hard_eval_retrieval_top1": str(hard_top1),
        "ste_eval_retrieval_top1": str(ste_top1),
        "delta_ste_minus_hard_eval_retrieval_top1": str(top1_delta),
        "hard_eval_retrieval_margin": str(hard_margin),
        "ste_eval_retrieval_margin": str(ste_margin),
        "delta_ste_minus_hard_eval_retrieval_margin": str(ste_margin - hard_margin),
        "hard_train_wall_time_sec": str(hard_wall),
        "ste_train_wall_time_sec": str(ste_wall),
        "delta_ste_minus_hard_train_wall_time_sec": str(ste_wall - hard_wall),
    }


def _pretrain_row(seed: int, top1: float = 0.18, ceiling: float = 0.50):
    return {
        "seed": str(seed),
        "task": "aliased_kv",
        "pretrain_eval_retrieval_top1": str(top1),
        "max_allowed_retrieval_top1": str(ceiling),
    }


def test_audit_reports_consistent_positive_paired_gain_and_cost():
    analysis = _analysis_module()
    report = analysis.paired_effect_summary(
        [_paired_row(1, 0.50, 0.55), _paired_row(2, 0.40, 0.45)],
        [_pretrain_row(1), _pretrain_row(2)],
    )

    assert report["audit_status"] == "pass"
    assert report["interpretation"] == "proof_of_mechanism_only"
    assert report["delta_ste_minus_hard"]["eval_answer_exact"]["mean"] == pytest.approx(0.45)
    assert report["delta_ste_minus_hard"]["eval_retrieval_top1"]["positive_seed_count"] == 2
    assert report["ste_over_hard_wall_time_ratio"]["mean"] == pytest.approx(9.0)


def test_audit_refuses_saturated_or_unpaired_input():
    analysis = _analysis_module()
    with pytest.raises(ValueError, match="anti-saturation"):
        analysis.paired_effect_summary([_paired_row(1, 0.2, 0.2)], [_pretrain_row(1, top1=0.8)])
    with pytest.raises(ValueError, match="seed sets differ"):
        analysis.paired_effect_summary([_paired_row(1, 0.2, 0.2)], [_pretrain_row(2)])


def test_audit_marks_nonuniform_paired_improvement_as_failure_without_hiding_it():
    analysis = _analysis_module()
    report = analysis.paired_effect_summary(
        [_paired_row(1, 0.2, 0.1), _paired_row(2, -0.1, 0.1)],
        [_pretrain_row(1), _pretrain_row(2)],
    )

    assert report["audit_status"] == "fail"
    assert report["interpretation"] == "no_consistent_pilot_gain"
    assert report["delta_ste_minus_hard"]["eval_answer_exact"]["positive_seed_count"] == 1
