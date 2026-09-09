"""Unit tests for the paired exact-memory read control runner."""

import importlib.util
from pathlib import Path

import pytest


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_episodic_read_control.py"
    spec = importlib.util.spec_from_file_location("episodic_read_control_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_control_constructs_an_aliased_reader_only_top1_comparison():
    runner = _runner_module()
    control = runner.build_arg_parser().parse_args([])
    args = runner.make_train_args(control, seed=17, condition="ste_topk")

    assert args.model == "hpm_lite_v2"
    assert args.task == "aliased_kv"
    assert args.write_mode == "oracle"
    assert args.episodic_only is True
    assert args.top_k == 1
    assert args.episodic_read_mode == "ste_topk"
    assert args.lambda_ret == 0.0  # answer-loss-only selection-gradient control
    assert args.lambda_writer == 0.0
    assert args.lambda_jepa == 0.0


def test_read_control_pretraining_gate_scales_with_candidate_count_and_allows_override():
    runner = _runner_module()
    control = runner.build_arg_parser().parse_args([])
    control.num_facts = 8
    assert runner.pretrain_retrieval_limit(control) == pytest.approx(0.275)

    control.num_facts = 4
    assert runner.pretrain_retrieval_limit(control) == pytest.approx(0.40)

    control.max_pretrain_retrieval_top1 = 0.31
    assert runner.pretrain_retrieval_limit(control) == pytest.approx(0.31)


def test_read_control_exposes_a_non_training_preflight_mode():
    runner = _runner_module()
    control = runner.build_arg_parser().parse_args(["--preflight-only"])
    assert control.preflight_only is True


def test_read_control_computes_paired_ste_minus_hard_deltas():
    runner = _runner_module()
    hard = {metric: 1.0 for metric in runner.METRICS}
    ste = {metric: 1.25 for metric in runner.METRICS}

    rows = runner.paired_summary_rows(
        [
            {"seed": 4, "condition": "ste_topk", **ste},
            {"seed": 4, "condition": "hard_topk", **hard},
        ]
    )

    assert rows[0]["seed"] == 4
    assert rows[0]["delta_ste_minus_hard_eval_answer_exact"] == pytest.approx(0.25)
    assert rows[0]["delta_ste_minus_hard_train_loss"] == pytest.approx(0.25)


def test_read_control_refuses_an_unpaired_summary_or_unknown_condition():
    runner = _runner_module()
    record = {metric: 0.0 for metric in runner.METRICS}
    with pytest.raises(ValueError, match="missing one paired condition"):
        runner.paired_summary_rows([{"seed": 3, "condition": "hard_topk", **record}])
    with pytest.raises(Exception, match="unknown condition"):
        runner.parse_conditions("hard_topk,not_a_mode")
