#!/usr/bin/env python3
"""Measure the finite-horizon router starvation guard from existing logs.

No training is performed.  The calibration uses the same whole-sequence mean
per-path routing mass already stored in the composite step logs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

PATHS = ("local", "recurrent", "fast_weight", "episodic")
NECESSARY = {
    "copy": "recurrent",
    "mood": "fast_weight",
    "fact": "episodic",
}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty log: {path}")
    return sorted(rows, key=lambda r: int(float(r["step"])))


def path_probs(row):
    return {p: float(row[f"router_weight_{p}"]) for p in PATHS}


def adverse_gap_from_mean_probs(row, path):
    probs = path_probs(row)
    target = probs[path]
    competitor = max(v for k, v in probs.items() if k != path)
    if target <= 0.0 or competitor <= 0.0:
        raise ValueError("router weights must be strictly positive")
    return math.log(competitor / target)


def task_probs(row, task):
    return {p: float(row[f"router_weight_{task}_{p}"]) for p in PATHS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        type=Path,
        default=Path("runs/composite_four_path"),
        help="Directory containing seed*_composite_step_log.csv",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("runs/composite_four_path/router_starvation_guard_calibration.json"),
    )
    args = ap.parse_args()

    files = sorted(args.inputs.glob("seed*_composite_step_log.csv"))
    if not files:
        raise FileNotFoundError(f"no seed composite logs under {args.inputs}")
    rows_by_seed = {}
    hashes = {}
    for path in files:
        rows = read_rows(path)
        seed = int(float(rows[0]["seed"]))
        rows_by_seed[seed] = rows
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    common_steps = sorted(set.intersection(*[{int(float(r["step"])) for r in rows} for rows in rows_by_seed.values()]))
    if not common_steps:
        raise ValueError("no common logged steps across seeds")

    majority = len(rows_by_seed) // 2 + 1
    discovery_step = None
    discovery_votes = None
    for step in common_steps:
        votes = {}
        all_majority = True
        for task, required_path in NECESSARY.items():
            count = 0
            for rows in rows_by_seed.values():
                row = next(r for r in rows if int(float(r["step"])) == step)
                probs = task_probs(row, task)
                if max(probs, key=probs.get) == required_path:
                    count += 1
            votes[f"{task}:{required_path}"] = count
            if count < majority:
                all_majority = False
        if all_majority:
            discovery_step = step
            discovery_votes = votes
            break
    if discovery_step is None:
        raise RuntimeError("no empirical majority discovery checkpoint found")

    first_step = common_steps[0]
    initial_gaps = []
    p_values = []
    drift_rows = []
    for seed, rows in rows_by_seed.items():
        selected = [r for r in rows if int(float(r["step"])) <= discovery_step]
        first = selected[0]
        for path in PATHS:
            gap = adverse_gap_from_mean_probs(first, path)
            initial_gaps.append({"seed": seed, "path": path, "gap": gap})
        for row in selected:
            step = int(float(row["step"]))
            for path, probability in path_probs(row).items():
                p_values.append({"seed": seed, "step": step, "path": path, "probability": probability})
        for before, after in zip(selected, selected[1:]):
            s0 = int(float(before["step"]))
            s1 = int(float(after["step"]))
            dt = s1 - s0
            for path in PATHS:
                g0 = adverse_gap_from_mean_probs(before, path)
                g1 = adverse_gap_from_mean_probs(after, path)
                drift_rows.append(
                    {
                        "seed": seed,
                        "path": path,
                        "from_step": s0,
                        "to_step": s1,
                        "drift_per_step": (g1 - g0) / dt,
                    }
                )

    max_initial = max(initial_gaps, key=lambda x: x["gap"])
    max_drift = max(drift_rows, key=lambda x: x["drift_per_step"])
    min_prob = min(p_values, key=lambda x: x["probability"])

    report = {
        "schema_version": "hpm.router_starvation_guard.empirical.v1",
        "source": {
            "directory": str(args.inputs),
            "seed_count": len(rows_by_seed),
            "seeds": sorted(rows_by_seed),
            "sha256_by_file": hashes,
        },
        "measurement_statistic": (
            "whole-sequence mean routing probability per path; adverse gap is the "
            "log ratio max_competitor_mean_probability / target_mean_probability"
        ),
        "definitions": {
            "T_discover": (
                "earliest common logged checkpoint where each frozen R2 necessary task/path "
                "pair is top-routed in a strict majority of seeds"
            ),
            "A0": "maximum adverse aggregate log-probability gap at the first logged checkpoint",
            "d_max": "maximum observed increase in adverse aggregate gap per optimizer step before T_discover",
            "p_min": "minimum observed whole-sequence mean path probability through T_discover",
        },
        "measured": {
            "A0": max_initial["gap"],
            "A0_witness": max_initial,
            "d_max": max_drift["drift_per_step"],
            "d_max_witness": max_drift,
            "T_discover": discovery_step,
            "T_discover_votes": discovery_votes,
            "majority_required": majority,
            "p_min": min_prob["probability"],
            "p_min_witness": min_prob,
            "first_logged_step": first_step,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["measured"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
