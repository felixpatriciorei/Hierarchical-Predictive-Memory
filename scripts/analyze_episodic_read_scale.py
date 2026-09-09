"""Audit all candidate-count arms of an R1 episodic-read scale sweep.

This composes the fixed-M paired audit over the ``M###`` directories written
by :mod:`run_episodic_read_scale`.  It checks that every candidate count has
matching hard/STE artifacts, passed its own anti-saturation gate, and retained
positive exact-answer and retrieval-top-1 deltas for every matched seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_episodic_read_control import (
    audit_completed_artifacts,
    paired_effect_summary,
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


def _group_by_num_facts(rows: Iterable[Mapping[str, str]], artifact_name: str) -> Dict[int, List[Mapping[str, str]]]:
    grouped: Dict[int, List[Mapping[str, str]]] = {}
    for row in rows:
        try:
            num_facts = int(row["num_facts"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{artifact_name} has an invalid num_facts row: {dict(row)!r}") from exc
        grouped.setdefault(num_facts, []).append(row)
    if not grouped:
        raise ValueError(f"{artifact_name} has no rows")
    return grouped


def validate_scale_summary(
    summary_rows: Iterable[Mapping[str, str]],
    reports_by_m: Mapping[int, Mapping[str, Any]],
) -> None:
    """Verify root means against the canonical seed-paired audit calculations."""

    summary_by_m = _group_by_num_facts(summary_rows, "scale_summary.csv")
    if set(summary_by_m) != set(reports_by_m):
        raise ValueError(
            "scale_summary candidate counts differ from paired records: "
            f"summary={sorted(summary_by_m)}, paired={sorted(reports_by_m)}"
        )
    for num_facts, report in reports_by_m.items():
        rows = summary_by_m[num_facts]
        if len(rows) != 1:
            raise ValueError(f"scale_summary.csv has {len(rows)} rows for M={num_facts}, expected one")
        row = rows[0]
        expected = {
            "paired_seed_count": float(report["seed_count"]),
            "mean_delta_ste_minus_hard_eval_answer_exact": float(
                report["delta_ste_minus_hard"]["eval_answer_exact"]["mean"]
            ),
            "mean_delta_ste_minus_hard_eval_retrieval_top1": float(
                report["delta_ste_minus_hard"]["eval_retrieval_top1"]["mean"]
            ),
            "mean_ste_over_hard_wall_time_ratio": float(
                report["ste_over_hard_wall_time_ratio"]["mean"]
            ),
        }
        for field, expected_value in expected.items():
            actual = _float(row, field)
            if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"M={num_facts} scale_summary mismatch for {field}: "
                    f"reported={actual}, expected={expected_value}"
                )


def scale_effect_summary(
    run_dir: Path,
    paired_rows: Sequence[Mapping[str, str]],
    pretraining_rows: Sequence[Mapping[str, str]],
    summary_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    """Build and validate the candidate-count-scale audit report."""

    paired_by_m = _group_by_num_facts(paired_rows, "scale_pairwise.csv")
    pretraining_by_m = _group_by_num_facts(pretraining_rows, "pretraining_check.csv")
    if set(paired_by_m) != set(pretraining_by_m):
        raise ValueError(
            "paired/pretraining candidate-count sets differ: "
            f"paired={sorted(paired_by_m)}, pretraining={sorted(pretraining_by_m)}"
        )

    reports_by_m: Dict[int, Dict[str, Any]] = {}
    for num_facts in sorted(paired_by_m):
        report = paired_effect_summary(paired_by_m[num_facts], pretraining_by_m[num_facts])
        report["artifact_audit"] = audit_completed_artifacts(
            run_dir / f"M{num_facts:03d}", paired_by_m[num_facts]
        )
        reports_by_m[num_facts] = report

    validate_scale_summary(summary_rows, reports_by_m)
    statuses = {num_facts: report["audit_status"] for num_facts, report in reports_by_m.items()}
    return {
        "audit_status": "pass" if all(status == "pass" for status in statuses.values()) else "fail",
        "interpretation": (
            "candidate_set_scale_proof_of_mechanism_only"
            if all(status == "pass" for status in statuses.values())
            else "no_consistent_candidate_set_scale_gain"
        ),
        "candidate_counts": sorted(reports_by_m),
        "by_num_facts": reports_by_m,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Episodic Read R1 Candidate-Set Scale Audit",
        "",
        f"Status: **{report['audit_status']}** ({report['interpretation']}).",
        "",
        "| Candidate slots M | Paired seeds | Max pretrain top-1 | Mean exact Δ | Min exact Δ | Mean top-1 Δ | Min top-1 Δ | STE / hard wall time |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for num_facts in report["candidate_counts"]:
        current = report["by_num_facts"][num_facts]
        exact = current["delta_ste_minus_hard"]["eval_answer_exact"]
        top1 = current["delta_ste_minus_hard"]["eval_retrieval_top1"]
        pretrain = current["pretraining_retrieval_top1"]
        speed = current["ste_over_hard_wall_time_ratio"]
        lines.append(
            f"| {num_facts} | {current['seed_count']} | {pretrain['maximum']:.4f} | "
            f"{exact['mean']:+.4f} | {exact['minimum']:+.4f} | "
            f"{top1['mean']:+.4f} | {top1['minimum']:+.4f} | {speed['mean']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "Every row passed its own hard-read anti-saturation gate and matched each selected "
            "hard/STE CSV result to exactly one completed `metrics.json` artifact.",
            "",
            "Interpretation: this establishes only a fixed-candidate, oracle-written read-gradient "
            "mechanism across the tested candidate counts. It does not establish learned-writer "
            "behavior, router non-collapse, JEPA benefit, sequence-length scaling, or HPM-wide scaling.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    report = scale_effect_summary(
        run_dir,
        read_csv(run_dir / "scale_pairwise.csv"),
        read_csv(run_dir / "pretraining_check.csv"),
        read_csv(run_dir / "scale_summary.csv"),
    )
    report["run_dir"] = str(run_dir)

    json_path = run_dir / "read_scale_audit.json"
    markdown_path = run_dir / "read_scale_audit.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(
        "status={status} candidate_counts={counts}".format(
            status=report["audit_status"], counts=",".join(map(str, report["candidate_counts"]))
        )
    )
    return report


if __name__ == "__main__":
    main()
