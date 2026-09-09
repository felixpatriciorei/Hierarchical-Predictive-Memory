#!/usr/bin/env python3
"""Submit an HPM paired-seed result to TorchScope's assay-gate for a certified verdict.

Reads a ``paired_summary.csv`` written by a paired-seed control script
(e.g. ``run_causal_writer_control.py``), extracts the per-seed
baseline/candidate pair for one metric, and pipes a claim document to the
Rust ``assay-gate`` binary (TorchScope ``ts-gate`` crate). The gate runs the
fixed-sample ``metric_delta`` checker through the ts-assay kernel and
persists the minted cert; this helper prints a human-readable one-line
verdict and exits with the gate's code so it drops straight into a
PowerShell pass/fail check::

    python scripts/assay_submit.py runs/tui_test2/paired_summary.csv
    if ($LASTEXITCODE -eq 0) { ... }

Exit codes mirror assay-gate: 0 = Accept, 1 = Reject or Inconclusive
(do-not-advance), 2 = error (bad input, missing columns, gate not found).

Nothing here trains or modifies model code — this is read-only plumbing over
an already-produced paired-run artifact.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

# Search order after --gate and $ASSAY_GATE: the sibling TorchScope checkout,
# release profile first (a real gate would ship optimized).
GATE_SEARCH_PATHS = [
    REPO_ROOT.parent
    / "Torchscope_Project"
    / "torchscope"
    / "host"
    / "target"
    / "release"
    / "assay-gate.exe",
    REPO_ROOT.parent
    / "Torchscope_Project"
    / "torchscope"
    / "host"
    / "target"
    / "debug"
    / "assay-gate.exe",
]

DEFAULT_STORE = REPO_ROOT / "runs" / "assay-store" / "certs.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paired_csv", type=Path, help="paired_summary.csv from a paired-seed control run")
    parser.add_argument("--baseline", default="bce", help="condition prefix of the baseline columns (default: bce)")
    parser.add_argument("--candidate", default="coverage", help="condition prefix of the candidate columns (default: coverage)")
    parser.add_argument("--metric", default="eval_answer_exact", help="metric suffix shared by both condition columns")
    parser.add_argument("--direction", choices=("improved", "not_worse"), default="improved")
    parser.add_argument("--alpha", type=float, default=0.05, help="per-decision false-accept level (default: 0.05)")
    parser.add_argument("--range", type=float, default=1.0, dest="value_range", help="declared bound on |per-seed delta| (default: 1.0)")
    parser.add_argument("--gate", type=Path, default=None, help="path to the assay-gate executable (default: search sibling TorchScope checkout)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help=f"durable cert store jsonl (default: {DEFAULT_STORE})")
    return parser


def find_gate(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"--gate {explicit} does not exist")
    env_gate = os.environ.get("ASSAY_GATE")
    if env_gate:
        p = Path(env_gate)
        if p.exists():
            return p
        raise FileNotFoundError(f"$ASSAY_GATE={env_gate} does not exist")
    for candidate in GATE_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in GATE_SEARCH_PATHS)
    raise FileNotFoundError(
        "assay-gate executable not found; build it with "
        "`cargo build -p ts-gate` in the TorchScope host workspace, or pass --gate "
        f"(searched: {searched})"
    )


def extract_trials(
    rows: List[Dict[str, str]], baseline: str, candidate: str, metric: str, value_range: float
) -> List[Dict[str, Any]]:
    base_col = f"{baseline}_{metric}"
    cand_col = f"{candidate}_{metric}"
    # The paired writers use either shorthand (`delta_<metric>`, e.g. the
    # causal-writer control) or the explicit form
    # (`delta_<candidate>_minus_<baseline>_<metric>`, e.g. the episodic-read
    # control/scale runners). Check whichever is present.
    delta_cols = (
        f"delta_{candidate}_minus_{baseline}_{metric}",
        f"delta_{metric}",
    )
    required = {"seed", base_col, cand_col}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            f"{base_col}/{cand_col} columns not all present (missing: {missing}); "
            "check --baseline/--candidate/--metric against the CSV header"
        )

    trials: List[Dict[str, Any]] = []
    for row in rows:
        seed_raw = row["seed"].strip()
        if not seed_raw:
            continue
        try:
            baseline_value = float(row[base_col])
            candidate_value = float(row[cand_col])
        except ValueError as exc:
            raise ValueError(f"non-numeric value for seed={seed_raw}: {exc}") from exc
        trial: Dict[str, Any] = {
            "seed": int(seed_raw),
            "baseline": baseline_value,
            "candidate": candidate_value,
        }
        recorded_delta = None
        delta_source = None
        for col in delta_cols:
            if col in row and row[col].strip():
                recorded_delta = float(row[col])
                delta_source = col
                break
        if recorded_delta is not None:
            recomputed = candidate_value - baseline_value
            if abs(recorded_delta - recomputed) > 1e-9:
                raise ValueError(
                    f"seed={seed_raw}: recorded {delta_source}={recorded_delta} disagrees with "
                    f"{cand_col}-{base_col}={recomputed}"
                )
        trials.append(trial)

    if not trials:
        raise ValueError("no data rows found in the paired summary")

    worst = max(abs(t["candidate"] - t["baseline"]) for t in trials)
    if worst > value_range:
        raise ValueError(
            f"largest |delta| ({worst:.6g}) exceeds the declared --range {value_range}; "
            f"re-run with --range >= {math.ceil(worst * 1e6) / 1e6} "
            "(the checker refuses evidence outside its declared range)"
        )
    return trials


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        gate = find_gate(args.gate)
    except FileNotFoundError as exc:
        print(f"assay_submit: error: {exc}", file=sys.stderr)
        return 2

    if not args.paired_csv.exists():
        print(f"assay_submit: error: {args.paired_csv} does not exist", file=sys.stderr)
        return 2
    with args.paired_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    try:
        trials = extract_trials(rows, args.baseline, args.candidate, args.metric, args.value_range)
    except ValueError as exc:
        print(f"assay_submit: error: {exc}", file=sys.stderr)
        return 2

    claim = {
        "metric": args.metric,
        "baseline_state": args.baseline,
        "candidate_state": args.candidate,
        "trials": trials,
        "direction": args.direction,
    }

    result = subprocess.run(
        [
            str(gate),
            "--store",
            str(args.store),
            "--range",
            repr(args.value_range),
            "--alpha",
            repr(args.alpha),
        ],
        input=json.dumps(claim),
        capture_output=True,
        text=True,
    )
    if result.returncode == 2:
        print("assay_submit: gate error:", file=sys.stderr)
        sys.stderr.write(result.stderr)
        return 2
    if result.returncode not in (0, 1):
        print(
            f"assay_submit: error: unexpected gate exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.stderr.write(result.stderr)
        return 2

    try:
        verdict = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"assay_submit: error: unparseable gate output ({exc})", file=sys.stderr)
        return 2

    parts = [
        f"ASSAY {verdict['verdict'].upper()}",
        f"metric={verdict['metric']}",
        f"n={verdict['n']}",
        f"mean={verdict['mean']:+.6f}",
        f"baseline={claim['baseline_state']}",
        f"candidate={claim['candidate_state']}",
        f"cert={verdict['cert_id'][:12]}",
    ]
    if "lower_bound" in verdict.get("evidence", {}):
        ev = verdict["evidence"]
        parts.append(f"hoeffding=[{ev['lower_bound']:+.6f}, {ev['upper_bound']:+.6f}]@{ev['alpha']}")
    if verdict.get("reason"):
        parts.append(f'reason="{verdict["reason"]}"')
    parts.append(f"store={verdict['store']}")
    print(" ".join(parts))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
