import importlib.util
from pathlib import Path

import pytest


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_causal_writer_control.py"
    spec = importlib.util.spec_from_file_location("causal_writer_control_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scale_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_causal_writer_scale.py"
    spec = importlib.util.spec_from_file_location("causal_writer_scale_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_runner_computes_seed_matched_deltas():
    runner = _runner_module()
    baseline = {metric: 1.0 for metric in runner.METRICS}
    coverage = {metric: 1.25 for metric in runner.METRICS}
    rows = runner.paired_summary_rows(
        [
            {"seed": 8, "condition": "bce_plus_coverage", **coverage},
            {"seed": 8, "condition": "bce_only", **baseline},
        ]
    )

    assert rows[0]["seed"] == 8
    assert rows[0]["delta_eval_true_fact_written_rate"] == 0.25
    assert rows[0]["delta_train_writer_set_coverage_mass_error"] == 0.25


def test_paired_runner_refuses_missing_counterfactual():
    runner = _runner_module()
    record = {metric: 0.0 for metric in runner.METRICS}
    with pytest.raises(ValueError, match="missing one paired condition"):
        runner.paired_summary_rows([{"seed": 9, "condition": "bce_only", **record}])


def test_runner_parses_an_unpaired_masked_control_without_relaxing_pair_validation():
    runner = _runner_module()
    assert runner.parse_conditions("bce_only") == ["bce_only"]
    assert runner.parse_conditions("bce_only,bce_plus_coverage") == [
        "bce_only",
        "bce_plus_coverage",
    ]
    with pytest.raises(Exception, match="unknown condition"):
        runner.parse_conditions("not_a_condition")

    record = {metric: 0.25 for metric in runner.METRICS}
    rows = runner.condition_rows([{"seed": 4, "condition": "bce_only", **record}])
    assert rows == [{"seed": 4, "condition": "bce_only", **record}]


def test_visible_role_analysis_requires_a_gain_over_its_masked_counterfactual():
    runner = _runner_module()
    visible = {
        "causal_writer_capacity_contract": "causal_salience_aligned",
        "eval_true_fact_written_rate": 0.83,
        "eval_answer_exact": 0.81,
    }
    masked = {
        "causal_writer_capacity_contract": "masked_role_negative_control",
        "eval_true_fact_written_rate": 0.52,
        "eval_answer_exact": 0.53,
    }
    summary = runner.assert_visible_role_beats_masked_control(visible, masked)
    assert summary["coverage_gap"] == pytest.approx(0.31)

    no_gain_visible = {**visible, "eval_true_fact_written_rate": 0.52}
    with pytest.raises(AssertionError, match="did not exceed"):
        runner.assert_visible_role_beats_masked_control(no_gain_visible, masked)


def test_scale_runner_persists_configuration_specific_records_and_pairwise_deltas():
    scale = _scale_module()
    assert scale.parse_positive_ints("4,8,16") == [4, 8, 16]
    assert scale.parse_scale_conditions("visible_bce,masked_bce") == [
        "visible_bce",
        "masked_bce",
    ]
    with pytest.raises(Exception, match="unknown condition"):
        scale.parse_scale_conditions("unknown")

    control = scale.build_arg_parser().parse_args([])
    train_args, config = scale.condition_configuration(
        control, num_facts=8, seed=5, condition="masked_bce"
    )
    assert train_args.num_facts == 8
    assert train_args.writer_role_mode == "masked"
    assert train_args.lambda_writer_set_coverage == 0.0
    assert config["num_facts"] == 8

    base = {
        "num_facts": 8,
        "seed": 5,
        "memory_slots": 2,
        "writer_required_facts": 2,
        "eval_answer_exact": 0.5,
        "eval_retrieval_top1": 0.5,
        "eval_true_fact_written_rate": 0.5,
        "eval_false_write_rate": 0.5,
        "eval_missed_fact_rate": 0.5,
    }
    visible = {**base, "condition": "visible_bce", "eval_true_fact_written_rate": 0.7}
    coverage = {**base, "condition": "visible_coverage", "eval_true_fact_written_rate": 0.8, "eval_answer_exact": 0.6}
    masked = {**base, "condition": "masked_bce", "eval_true_fact_written_rate": 0.25}
    pairwise = scale.pairwise_rows([visible, coverage, masked])
    assert pairwise == [
        {
            "num_facts": 8,
            "seed": 5,
            "delta_coverage_true_fact": pytest.approx(0.1),
            "delta_coverage_answer_exact": pytest.approx(0.1),
            "visible_minus_masked_true_fact": pytest.approx(0.45),
        }
    ]
    summary = scale.summary_rows([masked])
    assert summary[0]["masked_analytic_ceiling"] == pytest.approx(0.25)
