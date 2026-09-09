#!/usr/bin/env python3
"""Describe pairwise path interactions from HPM frozen-router co-ablation JSONs.

This script uses the already-produced 16-condition diagnostic cube.  It does
not train, alter the router, or mint a statistical certificate.

For accuracy v, define single-removal losses

    d_a  = v(full) - v(drop_a)
    d_b  = v(full) - v(drop_b)

and joint-removal loss

    d_ab = v(full) - v(drop_a_b).

The reported interaction is

    I_loss = d_ab - d_a - d_b
           = v(drop_a) + v(drop_b) - v(full) - v(drop_a_b).

I_loss > 0 means joint removal hurts *more* than the sum of the two singleton
losses (complementarity / super-additive necessity on this metric).
I_loss < 0 means the singleton losses overlap (redundancy / substitutability).
Exact-match accuracy is nonlinear, so treat this as descriptive mechanism
analysis, not an algebraic decomposition of latent computation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from hpm.evidence_provenance import evidence_set_sha256, sha256_file

ZERO_TOL = 1.0e-12

EXPECTED_KIND = "fixed_router_post_gate_conditional_coablation"


def discover(inputs: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            out.append(p)
            continue
        if not p.is_dir():
            raise FileNotFoundError(p)
        found = sorted(p.glob("seed*_router_dependency.json"))
        if not found:
            found = sorted(p.rglob("router_dependency.json"))
        if not found:
            raise FileNotFoundError(f"no router dependency artifacts under {p}")
        out.extend(found)
    return out


def load(paths: List[Path]) -> List[Mapping[str, Any]]:
    artifacts: List[Mapping[str, Any]] = []
    seen = set()
    for p in paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("diagnostic_kind") != EXPECTED_KIND:
            raise ValueError(f"{p}: wrong diagnostic kind")
        seed = int(obj["seed"])
        if seed in seen:
            raise ValueError(f"duplicate seed {seed}")
        seen.add(seed)
        obj["_source_path"] = str(p)
        obj["_source_sha256"] = sha256_file(p)
        artifacts.append(obj)
    return sorted(artifacts, key=lambda x: int(x["seed"]))


def pair_condition(a: str, b: str, available: set[str]) -> str:
    c1 = f"drop_{a}__{b}"
    c2 = f"drop_{b}__{a}"
    if c1 in available:
        return c1
    if c2 in available:
        return c2
    raise KeyError(f"joint removal condition missing for {a}, {b}")


def analyze(artifacts: List[Mapping[str, Any]]) -> Dict[str, Any]:
    first = artifacts[0]
    paths = list(first["path_names"])
    conditions = set(first["condition_exact"])
    subtasks = sorted(first["condition_exact"]["full"])
    seed_hashes = {int(a["seed"]): a["_source_sha256"] for a in artifacts}
    result: Dict[str, Any] = {
        "diagnostic_kind": EXPECTED_KIND,
        "definition": "joint_loss_minus_sum_single_losses",
        "zero_tolerance": ZERO_TOL,
        "seed_count": len(artifacts),
        "seeds": [int(a["seed"]) for a in artifacts],
        "source_sha256_by_seed": {str(k): v for k, v in sorted(seed_hashes.items())},
        "evidence_set_sha256": evidence_set_sha256(seed_hashes),
        "subtasks": {},
    }
    for task in subtasks:
        task_rows: Dict[str, Any] = {}
        for a, b in combinations(paths, 2):
            joint = pair_condition(a, b, conditions)
            vals = []
            for art in artifacts:
                c = art["condition_exact"]
                full = float(c["full"][task])
                va = float(c[f"drop_{a}"][task])
                vb = float(c[f"drop_{b}"][task])
                vab = float(c[joint][task])
                d_a = full - va
                d_b = full - vb
                d_ab = full - vab
                interaction = d_ab - d_a - d_b
                if abs(interaction) <= ZERO_TOL:
                    interaction = 0.0
                vals.append(interaction)
            task_rows[f"{a}+{b}"] = {
                "mean": statistics.fmean(vals),
                "min": min(vals),
                "max": max(vals),
                "positive_seed_count": sum(v > ZERO_TOL for v in vals),
                "negative_seed_count": sum(v < -ZERO_TOL for v in vals),
                "zero_seed_count": sum(abs(v) <= ZERO_TOL for v in vals),
                "per_seed": [
                    {"seed": int(artifacts[i]["seed"]), "interaction": vals[i]}
                    for i in range(len(vals))
                ],
            }
        result["subtasks"][task] = task_rows
    return result


def render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Router Pairwise Interaction Audit",
        "",
        "Descriptive only; **not** a TorchScope certificate.",
        "",
        "Interaction is `joint removal loss - sum(single removal losses)`. ",
        "Positive = complementarity/super-additive necessity; negative = overlapping/redundant singleton effects.",
        "",
        f"Seeds: {report['seed_count']} {report['seeds']}",
        f"Evidence set SHA-256: `{report.get('evidence_set_sha256', 'unavailable')}`",
        f"Numerical zero tolerance: `{report.get('zero_tolerance', ZERO_TOL):.1e}`",
        "",
    ]
    for task, rows in report["subtasks"].items():
        lines += [
            f"## {task.upper()}",
            "",
            "| Pair | Mean interaction | Min | Max | + / - / 0 seeds |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for pair, x in rows.items():
            lines.append(
                f"| {pair} | {x['mean']:+.6f} | {x['min']:+.6f} | {x['max']:+.6f} | "
                f"{x['positive_seed_count']} / {x['negative_seed_count']} / {x['zero_seed_count']} |"
            )
        lines.append("")
    lines += [
        "## Scope warning",
        "",
        "Exact-match accuracy is nonlinear and path removals can change downstream states. This interaction is a causal-ablation summary under the frozen gate, not proof of an independent additive latent decomposition.",
        "",
    ]
    return "\n".join(lines)


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    artifacts = load(discover(args.inputs))
    report = analyze(artifacts)
    out = args.output_dir or (args.inputs[0] if args.inputs[0].is_dir() else args.inputs[0].parent)
    out.mkdir(parents=True, exist_ok=True)
    j = out / "router_interactions.json"
    m = out / "router_interactions.md"
    j.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    m.write_text(render_md(report), encoding="utf-8")
    print(f"wrote {j}")
    print(f"wrote {m}")
    print(f"status=interaction-audited seeds={report['seed_count']}")
    return report


if __name__ == "__main__":
    main()
