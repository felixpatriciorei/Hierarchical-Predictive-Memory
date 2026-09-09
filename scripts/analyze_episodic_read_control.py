"""Audit a paired ``run_episodic_read_control.py`` experiment directory.

The read-side experiment is valid only when three conditions hold together:

1. every paired seed passed the pre-training anti-saturation ceiling;
2. every reported hard/STE result can be matched to exactly one completed
   ``metrics.json`` artifact; and
3. the paired STE-minus-hard deltas improve both exact answer accuracy and
   top-1 slot retrieval on every seed.

This is intentionally an audit, not a p-value generator.  A three-seed pilot
can establish a proof-of-mechanism but not a robust scaling claim.  The report
also records the speed cost of the current row-wise OT implementation, which
must not be hidden by reporting accuracy alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PAIRED_METRICS = (
    "eval_answer_exact",
    "eval_answer_ce",
    "eval_retrieval_top1",
    "eval_retrieval_margin",
    "train_wall_time_sec",
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row is missing finite numeric field {key!r}: {dict(row)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"field {key!r} is non-finite: {value!r}")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty series")
    return float(statistics.fmean(values))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty series")
    return float(statistics.median(values))


def paired_effect_summary(
    paired_rows: Iterable[Mapping[str, str]],
    pretraining_rows: Iterable[Mapping[str, str]],
) -> Dict[str, Any]:
    """Validate matching seeds and calculate paired effect/cost summaries."""

    paired_by_seed: Dict[int, Mapping[str, str]] = {}
    for row in paired_rows:
        seed = int(row["seed"])
        if seed in paired_by_seed:
            raise ValueError(f"duplicate paired-summary row for seed={seed}")
        paired_by_seed[seed] = row
    pretrain_by_seed: Dict[int, Mapping[str, str]] = {}
    for row in pretraining_rows:
        seed = int(row["seed"])
        if seed in pretrain_by_seed:
            raise ValueError(f"duplicate pretraining row for seed={seed}")
        pretrain_by_seed[seed] = row

    if not paired_by_seed:
        raise ValueError("paired_summary.csv has no rows")
    if set(paired_by_seed) != set(pretrain_by_seed):
        raise ValueError(
            "paired and pretraining seed sets differ: "
            f"paired={sorted(paired_by_seed)}, pretraining={sorted(pretrain_by_seed)}"
        )

    per_seed = []
    for seed in sorted(paired_by_seed):
        paired = paired_by_seed[seed]
        pretrain = pretrain_by_seed[seed]
        task = pretrain.get("task")
        if task != "aliased_kv":
            raise ValueError(f"seed={seed} pretraining task is {task!r}, expected 'aliased_kv'")
        baseline_top1 = _float(pretrain, "pretrain_eval_retrieval_top1")
        ceiling = _float(pretrain, "max_allowed_retrieval_top1")
        if baseline_top1 > ceiling:
            raise ValueError(
                f"seed={seed} fails the pre-training anti-saturation gate: "
                f"{baseline_top1:.6f} > {ceiling:.6f}"
            )

        values = {
            "seed": seed,
            "pretrain_retrieval_top1": baseline_top1,
            "pretrain_retrieval_ceiling": ceiling,
        }
        for metric in PAIRED_METRICS:
            hard = _float(paired, f"hard_{metric}")
            ste = _float(paired, f"ste_{metric}")
            delta = _float(paired, f"delta_ste_minus_hard_{metric}")
            if not math.isclose(delta, ste - hard, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"seed={seed} has an inconsistent delta for {metric}: "
                    f"reported={delta}, expected={ste - hard}"
                )
            values[f"hard_{metric}"] = hard
            values[f"ste_{metric}"] = ste
            values[f"delta_{metric}"] = delta
        if values["hard_train_wall_time_sec"] <= 0.0:
            raise ValueError(f"seed={seed} has non-positive hard training duration")
        values["ste_over_hard_wall_time_ratio"] = (
            values["ste_train_wall_time_sec"] / values["hard_train_wall_time_sec"]
        )
        per_seed.append(values)

    def summarize_delta(metric: str) -> Dict[str, float | int]:
        values = [float(row[f"delta_{metric}"]) for row in per_seed]
        return {
            "mean": _mean(values),
            "median": _median(values),
            "minimum": min(values),
            "maximum": max(values),
            "positive_seed_count": sum(value > 0.0 for value in values),
            "seed_count": len(values),
        }

    exact = summarize_delta("eval_answer_exact")
    top1 = summarize_delta("eval_retrieval_top1")
    ce = summarize_delta("eval_answer_ce")
    latency_ratios = [float(row["ste_over_hard_wall_time_ratio"]) for row in per_seed]
    all_exact_positive = exact["positive_seed_count"] == exact["seed_count"]
    all_top1_positive = top1["positive_seed_count"] == top1["seed_count"]

    return {
        "audit_status": "pass" if all_exact_positive and all_top1_positive else "fail",
        "interpretation": (
            "proof_of_mechanism_only" if all_exact_positive and all_top1_positive else "no_consistent_pilot_gain"
        ),
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "pretraining_retrieval_top1": {
            "mean": _mean([float(row["pretrain_retrieval_top1"]) for row in per_seed]),
            "maximum": max(float(row["pretrain_retrieval_top1"]) for row in per_seed),
        },
        "delta_ste_minus_hard": {
            "eval_answer_exact": exact,
            "eval_retrieval_top1": top1,
            "eval_answer_ce": ce,
        },
        "ste_over_hard_wall_time_ratio": {
            "mean": _mean(latency_ratios),
            "median": _median(latency_ratios),
            "minimum": min(latency_ratios),
            "maximum": max(latency_ratios),
        },
    }


def _matches_expected(metric: Mapping[str, Any], expected: Mapping[str, str], mode: str) -> bool:
    if metric.get("episodic_read_mode") != f"{mode}_topk":
        return False
    if int(metric.get("seed", -1)) != int(expected["seed"]):
        return False
    for field in ("eval_answer_exact", "eval_retrieval_top1", "train_wall_time_sec"):
        metric_value = float(metric.get(field, float("nan")))
        summary_value = _float(expected, f"{mode}_{field}")
        if not math.isclose(metric_value, summary_value, rel_tol=0.0, abs_tol=1.0e-6):
            return False
    return True


def audit_completed_artifacts(run_dir: Path, paired_rows: Iterable[Mapping[str, str]]) -> Dict[str, Any]:
    """Match each CSV result to its exact completed artifact.

    Reusing an output directory leaves older completed runs beside the new
    ones.  The paired summary is authoritative, but this audit confirms its
    selected metrics match one and only one completed run, then reports older
    artifacts as orphans instead of silently treating them as extra seeds.
    """

    metric_artifacts = []
    for path in run_dir.rglob("metrics.json"):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metric_artifacts.append({"path": str(path), "metrics": metrics})

    selected_paths = []
    for row in paired_rows:
        for mode in ("hard", "ste"):
            matches = [
                artifact
                for artifact in metric_artifacts
                if _matches_expected(artifact["metrics"], row, mode)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"seed={row['seed']} mode={mode} expected exactly one completed metrics.json, "
                    f"found {len(matches)}"
                )
            selected_paths.append(matches[0]["path"])

    selected = set(selected_paths)
    orphan_paths = sorted(artifact["path"] for artifact in metric_artifacts if artifact["path"] not in selected)
    return {
        "selected_metrics_paths": sorted(selected_paths),
        "orphan_metrics_paths": orphan_paths,
        "completed_metrics_count": len(metric_artifacts),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    deltas = report["delta_ste_minus_hard"]
    exact = deltas["eval_answer_exact"]
    top1 = deltas["eval_retrieval_top1"]
    ce = deltas["eval_answer_ce"]
    speed = report["ste_over_hard_wall_time_ratio"]
    artifact = report["artifact_audit"]
    lines = [
        "# Episodic Read Control Audit",
        "",
        f"Status: **{report['audit_status']}** ({report['interpretation']}).",
        "",
        f"Paired seeds: {report['seed_count']}. All pre-training baselines passed the configured ceiling.",
        "",
        "| Metric | Mean STE − hard | Min | Positive seeds |",
        "| --- | ---: | ---: | ---: |",
        f"| Answer exact | {exact['mean']:+.4f} | {exact['minimum']:+.4f} | {exact['positive_seed_count']}/{exact['seed_count']} |",
        f"| Retrieval top-1 | {top1['mean']:+.4f} | {top1['minimum']:+.4f} | {top1['positive_seed_count']}/{top1['seed_count']} |",
        f"| Answer CE | {ce['mean']:+.4f} | {ce['minimum']:+.4f} | n/a |",
        "",
        f"Observed STE wall-time cost: {speed['mean']:.2f}× hard top-k on average ",
        f"(range {speed['minimum']:.2f}×–{speed['maximum']:.2f}×).",
        "",
        f"Matched {len(artifact['selected_metrics_paths'])} selected completed artifacts. "
        f"Ignored {len(artifact['orphan_metrics_paths'])} older/orphan artifact(s) in this reused output directory.",
        "",
        "Interpretation: this establishes a controlled read-side proof of mechanism only. "
        "It does not establish learned-writer behavior, router non-collapse, JEPA benefit, or scaling.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    paired_rows = read_csv(run_dir / "paired_summary.csv")
    pretraining_rows = read_csv(run_dir / "pretraining_check.csv")
    report = paired_effect_summary(paired_rows, pretraining_rows)
    report["run_dir"] = str(run_dir)
    report["artifact_audit"] = audit_completed_artifacts(run_dir, paired_rows)

    json_path = run_dir / "read_control_audit.json"
    markdown_path = run_dir / "read_control_audit.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(
        "status={status} mean_delta_exact={exact:+.4f} mean_delta_top1={top1:+.4f} speed_ratio={speed:.2f}x".format(
            status=report["audit_status"],
            exact=report["delta_ste_minus_hard"]["eval_answer_exact"]["mean"],
            top1=report["delta_ste_minus_hard"]["eval_retrieval_top1"]["mean"],
            speed=report["ste_over_hard_wall_time_ratio"]["mean"],
        )
    )
    return report


if __name__ == "__main__":
    main()
