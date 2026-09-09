"""Unit tests for the resumable R1 episodic-read candidate-count sweep."""

import importlib.util
from pathlib import Path

import pytest


def _scale_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_episodic_read_scale.py"
    spec = importlib.util.spec_from_file_location("episodic_read_scale_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(scale, *, num_facts: int, seed: int, condition: str, value: float):
    return {
        "num_facts": num_facts,
        "seed": seed,
        "condition": condition,
        **{metric: value for metric in scale.METRICS},
    }


def test_scale_runner_locks_each_arm_to_the_aliased_reader_only_control(tmp_path):
    scale = _scale_module()
    control = scale.build_arg_parser().parse_args(
        ["--out-dir", str(tmp_path), "--num-facts-values", "4,8"]
    )
    train_args, config = scale.run_configuration(
        control, num_facts=4, seed=17, condition="ste_topk"
    )

    assert train_args.task == "aliased_kv"
    assert train_args.episodic_only is True
    assert train_args.write_mode == "oracle"
    assert train_args.top_k == 1
    assert train_args.episodic_read_mode == "ste_topk"
    assert train_args.lambda_ret == 0.0
    assert train_args.lambda_writer == 0.0
    assert train_args.lambda_jepa == 0.0
    assert Path(train_args.out_dir) == tmp_path / "M004"
    assert config["max_pretrain_retrieval_top1"] == pytest.approx(0.40)


def test_scale_runner_requires_complete_within_seed_pairs_and_summarizes_them():
    scale = _scale_module()
    records = [
        _record(scale, num_facts=8, seed=3, condition="hard_topk", value=0.10),
        _record(scale, num_facts=8, seed=3, condition="ste_topk", value=0.75),
        _record(scale, num_facts=8, seed=4, condition="hard_topk", value=0.20),
    ]

    pairwise = scale.scale_pairwise_rows(records)
    summary = scale.scale_summary_rows(pairwise)

    assert len(pairwise) == 1
    assert pairwise[0]["seed"] == 3
    assert pairwise[0]["delta_ste_minus_hard_eval_answer_exact"] == pytest.approx(0.65)
    assert summary == [
        {
            "num_facts": 8,
            "chance_retrieval_top1": 0.125,
            "paired_seed_count": 1,
            "mean_delta_ste_minus_hard_eval_answer_exact": pytest.approx(0.65),
            "mean_delta_ste_minus_hard_eval_retrieval_top1": pytest.approx(0.65),
            "mean_delta_ste_minus_hard_train_wall_time_sec": pytest.approx(0.65),
            "mean_ste_over_hard_wall_time_ratio": pytest.approx(7.5),
        }
    ]


def test_scale_runner_rejects_unsupported_candidate_counts():
    scale = _scale_module()
    with pytest.raises(Exception, match="4 <= M <= 50"):
        scale.parse_candidate_counts("3,8")
