"""Aggregate ``router_dependency.json`` artifacts from the composite four-path
frozen-router dependence diagnostic (``composite_harness.py
--router-dependence-eval``, one file per seed).

Each artifact records exact accuracy under every fixed-router post-gate
removal condition (all 16 subsets of local/recurrent/fast-weight/episodic),
per-subtask singleton drops, exact Shapley attributions, and the gate weights
at each subtask's answer position -- all measured with the unablated model's
own per-token gate held fixed, so no ablation effect can come from the router
re-routing around the intervention.

This script is an aggregator/auditor, not a statistical gate:

1. validates every artifact (provenance kind, complete 16-condition coverage,
   consistent path names, unique seeds);
2. reports per-subtask load-bearing evidence across seeds (mean singleton
   drop / Shapley attribution, and on how many of n seeds each path's removal
   hurt that subtask at all);
3. optionally emits a paired CSV shaped exactly like the paired summaries the
   TorchScope assay bridge consumes, so the headline claim (e.g. full vs
   drop-episodic on fact exact) can be certified::

       py scripts/analyze_router_dependency.py \\
           --inputs runs/composite_four_path \\
           --emit-paired-csv runs/composite_four_path/paired_summary.csv
       py scripts/assay_submit.py runs/composite_four_path/paired_summary.csv \\
           --baseline full --candidate drop_episodic --metric fact_exact

Certification itself is deliberately out of scope here -- run assay_submit
against real sweep data only (never smoke/synthetic artifacts).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from hpm.evidence_provenance import evidence_set_sha256, sha256_file

EXPECTED_DIAGNOSTIC_KIND = "fixed_router_post_gate_conditional_coablation"
DEFAULT_INPUT_DIR = Path("runs") / "composite_four_path"


def load_artifacts(inputs: Sequence[Path]) -> List[Mapping[str, Any]]:
    """Resolve files/directories into validated, seed-sorted artifacts."""

    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            # Two on-disk layouts exist: the Modal entrypoint writes flat
            # seed<N>_router_dependency.json files, while a local
            # composite_harness --out-dir nests one router_dependency.json
            # per <timestamp>_composite_<model>_seed<N> run directory.
            # Accept either; provenance/diagnostic-kind validation below
            # rejects anything unrelated.
            found = sorted(path.glob("seed*_router_dependency.json"))
            if not found:
                found = sorted(p for p in path.rglob("router_dependency.json"))
            if not found:
                raise FileNotFoundError(
                    f"no router_dependency.json artifacts under {path} "
                    "(looked for seed*_router_dependency.json and */router_dependency.json)"
                )
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"--inputs entry does not exist: {path}")

    artifacts: List[Mapping[str, Any]] = []
    seen_seeds: Dict[int, Path] = {}
    reference_path_names = None
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        kind = artifact.get("diagnostic_kind")
        if kind != EXPECTED_DIAGNOSTIC_KIND:
            raise ValueError(
                f"{path}: diagnostic_kind={kind!r}, expected {EXPECTED_DIAGNOSTIC_KIND!r}"
            )
        path_names = tuple(artifact.get("path_names", ()))
        if reference_path_names is None:
            reference_path_names = path_names
        elif path_names != reference_path_names:
            raise ValueError(
                f"{path}: path_names {path_names!r} disagrees with earlier artifact {reference_path_names!r}"
            )
        conditions = artifact.get("condition_exact")
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError(f"{path}: missing/empty condition_exact")
        for condition, per_subtask in conditions.items():
            for subtask, value in per_subtask.items():
                value = float(value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{path}: condition {condition!r} subtask {subtask!r} accuracy {value!r} outside [0, 1]"
                    )
        seed = int(artifact["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed={seed}: {seen_seeds[seed]} and {path}")
        seen_seeds[seed] = path
        artifact = dict(artifact)
        artifact["_source_path"] = str(path)
        artifact["_source_sha256"] = sha256_file(path)
        artifacts.append(artifact)

    return sorted(artifacts, key=lambda a: int(a["seed"]))


def aggregate(artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-subtask, per-path across-seed summary of removal effects."""

    subtasks = sorted(next(iter(artifacts[0]["condition_exact"].values())))
    path_names = tuple(artifacts[0]["path_names"])
    seeds = [int(a["seed"]) for a in artifacts]

    def series(pick) -> List[float]:
        return [float(pick(a)) for a in artifacts]

    singleton: Dict[str, Dict[str, Dict[str, float]]] = {}
    shapley: Dict[str, Dict[str, Dict[str, float]]] = {}
    for subtask in subtasks:
        singleton[subtask] = {}
        shapley[subtask] = {}
        for path in path_names:
            drops = series(lambda a, p=path: a["singleton_exact_drop"][subtask][p])
            attrs = series(lambda a, p=path: a["shapley_exact_attribution"][subtask][p])
            singleton[subtask][path] = {
                "mean": statistics.fmean(drops),
                "minimum": min(drops),
                "maximum": max(drops),
                # strictly positive drop == removing the path cost accuracy
                # on that seed; 0/n means the model never needed it there.
                "hurt_seed_count": sum(drop > 0.0 for drop in drops),
            }
            shapley[subtask][path] = {
                "mean": statistics.fmean(attrs),
                "minimum": min(attrs),
                "maximum": max(attrs),
                "positive_seed_count": sum(attr > 0.0 for attr in attrs),
            }

    return {
        "seed_count": len(artifacts),
        "seeds": seeds,
        "path_names": list(path_names),
        "singleton_exact_drop": singleton,
        "shapley_exact_attribution": shapley,
    }


def emit_paired_csv(
    artifacts: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    baseline_condition: str,
    candidate_condition: str,
    subtasks: Sequence[str],
) -> None:
    """Write an assay-bridge-compatible paired summary.

    Columns follow the explicit-form convention assay_submit.py understands:
    ``{baseline}_{subtask}_exact``, ``{candidate}_{subtask}_exact``, and
    ``delta_{candidate}_minus_{baseline}_{subtask}_exact`` -- submit with
    ``--baseline <baseline_condition> --candidate <candidate_condition>
    --metric <subtask>_exact``.
    """

    first_conditions = set(artifacts[0]["condition_exact"])
    for condition in (baseline_condition, candidate_condition):
        if condition not in first_conditions:
            raise ValueError(
                f"condition {condition!r} not present in artifacts; "
                f"available: {sorted(first_conditions)}"
            )
    for artifact in artifacts:
        if set(artifact["condition_exact"]) != first_conditions:
            raise ValueError(f"seed={artifact['seed']}: condition set differs between artifacts")

    metric_suffix = "_exact"
    header = ["seed"]
    for subtask in subtasks:
        header.append(f"{baseline_condition}_{subtask}{metric_suffix}")
    for subtask in subtasks:
        header.append(f"{candidate_condition}_{subtask}{metric_suffix}")
    for subtask in subtasks:
        header.append(
            f"delta_{candidate_condition}_minus_{baseline_condition}_{subtask}{metric_suffix}"
        )

    rows = []
    for artifact in artifacts:
        row: Dict[str, Any] = {"seed": int(artifact["seed"])}
        for subtask in subtasks:
            base = float(artifact["condition_exact"][baseline_condition][subtask])
            cand = float(artifact["condition_exact"][candidate_condition][subtask])
            row[f"{baseline_condition}_{subtask}{metric_suffix}"] = base
            row[f"{candidate_condition}_{subtask}{metric_suffix}"] = cand
            row[
                f"delta_{candidate_condition}_minus_{baseline_condition}_{subtask}{metric_suffix}"
            ] = cand - base
        rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    seed_hashes = {int(a["seed"]): a["_source_sha256"] for a in artifacts}
    provenance = {
        "schema": "hpm_paired_router_evidence_v1",
        "paired_csv": str(destination),
        "baseline_condition": baseline_condition,
        "candidate_condition": candidate_condition,
        "subtasks": list(subtasks),
        "seeds": sorted(seed_hashes),
        "source_sha256_by_seed": {str(k): v for k, v in sorted(seed_hashes.items())},
        "evidence_set_sha256": evidence_set_sha256(seed_hashes),
    }
    sidecar = destination.with_suffix(destination.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    agg = report["aggregate"]
    lines = [
        "# Router Dependence Audit (frozen-router conditional co-ablation)",
        "",
        f"Seeds: {agg['seed_count']} {agg['seeds']}. Provenance kind verified: "
        f"`{report['diagnostic_kind']}`.",
        f"Evidence set SHA-256: `{report.get('evidence_set_sha256', 'unavailable')}`.",
        "",
        "All numbers are exact-match accuracy changes under the unablated model's own "
        "frozen per-token gate; no renormalization, no retraining.",
        "",
    ]
    for title, block in (
        ("Singleton removal drop", agg["singleton_exact_drop"]),
        ("Shapley attribution", agg["shapley_exact_attribution"]),
    ):
        lines.append(f"## {title} (mean across seeds)")
        lines.append("")
        lines.append("| Subtask | " + " | ".join(agg["path_names"]) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * len(agg["path_names"])) + " |")
        for subtask, per_path in block.items():
            cells = [
                f"{per_path[path]['mean']:+.4f} ({per_path[path].get('hurt_seed_count', per_path[path].get('positive_seed_count', 0))}/{agg['seed_count']} seeds)"
                for path in agg["path_names"]
            ]
            lines.append(f"| {subtask} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "Interpretation: `(k/n seeds)` counts seeds where the effect has the "
        "load-bearing sign (drop > 0 for removals, attribution > 0 for Shapley).",
        "",
        "This audit does NOT certify anything. Submit the paired CSV through the ",
        "assay bridge for a certified verdict.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=[DEFAULT_INPUT_DIR],
        help=f"router_dependency.json files or directories of them (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--emit-paired-csv",
        type=Path,
        default=None,
        help="write a paired summary CSV ready for scripts/assay_submit.py",
    )
    parser.add_argument("--baseline-condition", type=str, default="full")
    parser.add_argument("--candidate-condition", type=str, default="drop_episodic")
    parser.add_argument(
        "--subtasks",
        type=str,
        default="fact",
        help="comma-separated subtasks for the paired CSV (default: fact)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the audit report (default: first input directory, else cwd)",
    )
    parser.add_argument(
        "--emit-all-paired-dir",
        type=Path,
        default=None,
        help=(
            "optional directory for one full-vs-drop_<path> paired CSV per "
            "subtask/path combination, using only the already-produced frozen-router artifacts"
        ),
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    artifacts = load_artifacts(args.inputs)
    aggregate_block = aggregate(artifacts)
    seed_hashes = {int(a["seed"]): a["_source_sha256"] for a in artifacts}
    report = {
        "diagnostic_kind": EXPECTED_DIAGNOSTIC_KIND,
        "sources": [a["_source_path"] for a in artifacts],
        "source_sha256_by_seed": {str(k): v for k, v in sorted(seed_hashes.items())},
        "evidence_set_sha256": evidence_set_sha256(seed_hashes),
        "aggregate": aggregate_block,
    }

    # Optional no-training expansion: materialize every single-path necessity
    # comparison into an assay-ready paired CSV.  This does not certify the
    # comparisons; it only prevents repeated hand-written analyzer commands and
    # keeps all comparisons on the exact same frozen-router artifacts.
    all_pair_commands: List[str] = []
    if args.emit_all_paired_dir is not None:
        args.emit_all_paired_dir.mkdir(parents=True, exist_ok=True)
        available_conditions = set(artifacts[0]["condition_exact"])
        for path in aggregate_block["path_names"]:
            drop_condition = f"drop_{path}"
            if drop_condition not in available_conditions:
                continue
            for subtask in sorted(aggregate_block["singleton_exact_drop"]):
                dest = args.emit_all_paired_dir / f"paired_{subtask}_{path}.csv"
                emit_paired_csv(
                    artifacts,
                    dest,
                    baseline_condition=drop_condition,
                    candidate_condition="full",
                    subtasks=[subtask],
                )
                cmd = (
                    f"py scripts/assay_submit.py {dest} --baseline {drop_condition} "
                    f"--candidate full --metric {subtask}_exact --direction improved"
                )
                all_pair_commands.append(cmd)
        report["all_paired_assay_commands"] = all_pair_commands

    output_dir = args.output_dir
    if output_dir is None:
        first = args.inputs[0]
        output_dir = first if first.is_dir() else first.parent
    json_path = output_dir / "router_dependence_audit.json"
    markdown_path = output_dir / "router_dependence_audit.md"

    csv_note = ""
    if args.emit_paired_csv is not None:
        emit_paired_csv(
            artifacts,
            args.emit_paired_csv,
            baseline_condition=args.baseline_condition,
            candidate_condition=args.candidate_condition,
            subtasks=[s.strip() for s in args.subtasks.split(",") if s.strip()],
        )
        csv_note = (
            f"paired_csv={args.emit_paired_csv} "
            f"(submit with: assay_submit.py <csv> --baseline {args.baseline_condition} "
            f"--candidate {args.candidate_condition} --metric <subtask>_exact)"
        )
        print(f"wrote {args.emit_paired_csv}")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    fact_drop = aggregate_block["singleton_exact_drop"].get("fact", {}).get("episodic", {})
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(
        f"status=aggregated seeds={aggregate_block['seed_count']} "
        f"episodic_singleton_drop_on_fact_mean={fact_drop.get('mean', float('nan')):+.4f} "
        f"hurt={fact_drop.get('hurt_seed_count', 0)}/{aggregate_block['seed_count']} seeds"
    )
    if csv_note:
        print(csv_note)
    if all_pair_commands:
        print(f"wrote {len(all_pair_commands)} full-vs-single-drop paired CSVs under {args.emit_all_paired_dir}")
        print("assay commands were also recorded in router_dependence_audit.json")
    return report


if __name__ == "__main__":
    main()
