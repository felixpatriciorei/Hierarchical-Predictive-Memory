#!/usr/bin/env python3
"""Enumerate faithfulness/completeness/minimality of HPM path supports.

Consumes frozen-router 16-condition `seed*_router_dependency.json` artifacts.
No model execution or training is performed.
"""
from __future__ import annotations

import argparse, itertools, json, math
from pathlib import Path
from statistics import mean

PATHS = ("local", "recurrent", "fast_weight", "episodic")
TASKS = ("copy", "mood", "fact", "xref")


def condition_for_drops(drops):
    ds = [p for p in PATHS if p in set(drops)]
    return "full" if not ds else "drop_" + "__".join(ds)


def load_artifacts(root: Path):
    files = sorted(root.glob("seed*_router_dependency.json"))
    if not files:
        raise SystemExit(f"no seed*_router_dependency.json under {root}")
    out = []
    seen = set()
    for p in files:
        obj = json.loads(p.read_text())
        seed = int(obj["seed"])
        if seed in seen:
            raise SystemExit(f"duplicate seed {seed}")
        seen.add(seed)
        out.append(obj)
    return out


def stats(xs):
    return {"mean": mean(xs), "min": min(xs), "max": max(xs), "positive_seed_count": sum(x > 1e-12 for x in xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--completeness-tol", type=float, default=0.05)
    ap.add_argument("--faithfulness-min", type=float, default=0.05)
    ap.add_argument("--minimality-min", type=float, default=0.02)
    args = ap.parse_args()
    arts = load_artifacts(args.inputs)
    result = {"seed_count": len(arts), "thresholds": {
        "completeness_tol": args.completeness_tol,
        "faithfulness_min": args.faithfulness_min,
        "minimality_min": args.minimality_min,
    }, "tasks": {}}

    all_paths = set(PATHS)
    for task in TASKS:
        entries = []
        for r in range(1, len(PATHS)+1):
            for support_tuple in itertools.combinations(PATHS, r):
                support = set(support_tuple)
                complement = all_paths - support
                faith = []
                complete_loss = []
                minimal = {p: [] for p in support_tuple}
                for a in arts:
                    c = a["condition_exact"]
                    full = float(c["full"][task])
                    drop_support = float(c[condition_for_drops(support)][task])
                    keep_only = float(c[condition_for_drops(complement)][task])
                    faith.append(full - drop_support)
                    complete_loss.append(full - keep_only)
                    for p in support_tuple:
                        keep_without_p = float(c[condition_for_drops(complement | {p})][task])
                        minimal[p].append(keep_only - keep_without_p)
                fstat, cstat = stats(faith), stats(complete_loss)
                mstat = {p: stats(xs) for p, xs in minimal.items()}
                passed = (
                    fstat["mean"] >= args.faithfulness_min
                    and cstat["max"] <= args.completeness_tol
                    and all(s["mean"] >= args.minimality_min for s in mstat.values())
                )
                entries.append({
                    "support": list(support_tuple),
                    "faithfulness_full_minus_drop_support": fstat,
                    "completeness_full_minus_keep_only": cstat,
                    "minimality_keep_only_minus_without_member": mstat,
                    "mature_under_thresholds": passed,
                })
        # Prefer smallest mature support, then best completeness, then faithfulness.
        entries.sort(key=lambda e: (not e["mature_under_thresholds"], len(e["support"]), e["completeness_full_minus_keep_only"]["mean"], -e["faithfulness_full_minus_drop_support"]["mean"]))
        result["tasks"][task] = entries

    out = args.output or args.inputs / "path_support_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for task in TASKS:
        best = next((e for e in result["tasks"][task] if e["mature_under_thresholds"]), None)
        if best:
            print(f"{task}: mature support={'+'.join(best['support'])}")
        else:
            print(f"{task}: no support clears configured thresholds")

if __name__ == "__main__":
    main()
