#!/usr/bin/env python3
"""Audit HPM composite step logs for router optimization health.

This script is intentionally descriptive, not a statistical accept-gate. It
summarizes the telemetry added by the router non-collapse prerequisite:

- probability concentration (max softmax weight, entropy/effective paths),
- raw vs effective logit scale,
- softmax Jacobian sensitivity,
- optional tanh-clamp local derivative,
- exact raw-logit -> router-weight Jacobian sensitivity.

It accepts either Modal's flat ``seed<N>_composite_step_log.csv`` layout or
nested local ``*/composite_step_log.csv`` runs. Older logs are accepted but
reported as missing the new health schema unless ``--require-health-schema``
is supplied.

No threshold here is a scientific certificate. Optional warning thresholds
are deliberately labelled HEURISTIC and exist only to triage runs for closer
inspection; causal path necessity remains the frozen-router assay's job.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


HEALTH_COLUMNS = (
    "router_raw_logit_abs_mean",
    "router_logit_abs_mean",
    "router_weight_max_mean",
    "router_entropy_mean",
    "router_effective_paths_mean",
    "router_clamp_jacobian_mean",
    "router_clamp_jacobian_min",
    "router_clamp_jacobian_lt_0p01_frac",
    "router_clamp_jacobian_lt_0p001_frac",
    "router_softmax_jacobian_fro_mean",
    "router_softmax_jacobian_fro_min",
    "router_raw_to_weight_jacobian_fro_mean",
    "router_raw_to_weight_jacobian_fro_min",
    "router_clamp_attenuation_ratio_mean",
    "router_clamp_attenuation_ratio_min",
)

SUBTASKS = ("copy", "mood", "fact", "xref")


def _finite_float(row: Mapping[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def discover(inputs: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            paths.append(p)
            continue
        if not p.is_dir():
            raise FileNotFoundError(p)
        flat = sorted(p.glob("seed*_composite_step_log.csv"))
        nested = sorted(p.rglob("composite_step_log.csv"))
        found = flat or nested
        if not found:
            raise FileNotFoundError(f"no composite step logs under {p}")
        paths.extend(found)
    return paths


def load(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    if "seed" not in rows[0] or "step" not in rows[0]:
        raise ValueError(f"{path}: expected seed and step columns")
    return rows


def summarize_log(path: Path, rows: List[Dict[str, str]], require_health: bool) -> Dict[str, Any]:
    seed_values = {int(float(r["seed"])) for r in rows if r.get("seed", "").strip()}
    if len(seed_values) != 1:
        raise ValueError(f"{path}: expected exactly one seed, got {sorted(seed_values)}")
    seed = next(iter(seed_values))
    rows = sorted(rows, key=lambda r: int(float(r["step"])))
    final = rows[-1]
    present = [c for c in HEALTH_COLUMNS if c in final]
    missing = [c for c in HEALTH_COLUMNS if c not in final]
    if require_health and missing:
        raise ValueError(f"{path}: missing router-health columns: {missing}")

    def series(key: str) -> List[float]:
        vals = [_finite_float(r, key) for r in rows]
        return [v for v in vals if v is not None]

    out: Dict[str, Any] = {
        "seed": seed,
        "source": str(path),
        "steps_logged": len(rows),
        "first_step": int(float(rows[0]["step"])),
        "last_step": int(float(final["step"])),
        "health_schema_complete": not missing,
        "missing_health_columns": missing,
    }

    for task in SUBTASKS:
        value = _finite_float(final, f"eval_{task}_exact")
        if value is not None:
            out[f"final_eval_{task}_exact"] = value

    extrema = {
        "router_raw_logit_abs_mean": max,
        "router_logit_abs_mean": max,
        "router_weight_max_mean": max,
        "router_entropy_mean": min,
        "router_effective_paths_mean": min,
        "router_clamp_jacobian_mean": min,
        "router_clamp_jacobian_min": min,
        "router_clamp_jacobian_lt_0p01_frac": max,
        "router_clamp_jacobian_lt_0p001_frac": max,
        "router_softmax_jacobian_fro_mean": min,
        "router_softmax_jacobian_fro_min": min,
        "router_raw_to_weight_jacobian_fro_mean": min,
        "router_raw_to_weight_jacobian_fro_min": min,
        "router_clamp_attenuation_ratio_mean": min,
        "router_clamp_attenuation_ratio_min": min,
    }
    for key, reducer in extrema.items():
        vals = series(key)
        if vals:
            out[f"trajectory_{'max' if reducer is max else 'min'}_{key}"] = reducer(vals)
            final_value = _finite_float(final, key)
            if final_value is not None:
                out[f"final_{key}"] = final_value

    # Exact invariant for an unclamped telemetry path. This is diagnostic, not
    # an assumption: when both fields exist and match at every logged step,
    # the run behaved as an unclamped router with respect to this transform.
    raw_vals = series("router_raw_logit_abs_mean")
    eff_vals = series("router_logit_abs_mean")
    jac_vals = series("router_clamp_jacobian_mean")
    if raw_vals and len(raw_vals) == len(eff_vals) == len(jac_vals):
        out["telemetry_consistent_with_unclamped"] = all(
            abs(a - b) <= 1.0e-8 and abs(j - 1.0) <= 1.0e-8
            for a, b, j in zip(raw_vals, eff_vals, jac_vals)
        )

    # HEURISTIC triage only. These warnings do not mint claims and should not
    # be fed into ts-assay as if they were pre-registered scientific tests.
    warnings: List[str] = []
    max_weight = out.get("trajectory_max_router_weight_max_mean")
    raw_jac = out.get("trajectory_min_router_raw_to_weight_jacobian_fro_mean")
    clamp_ratio = out.get("trajectory_min_router_clamp_attenuation_ratio_mean")
    if max_weight is not None and max_weight >= 0.99:
        warnings.append("HEURISTIC: mean max router weight reached >=0.99")
    if raw_jac is not None and raw_jac <= 1.0e-3:
        warnings.append("HEURISTIC: mean raw-logit->weight Jacobian Frobenius norm reached <=1e-3")
    if clamp_ratio is not None and clamp_ratio <= 0.1:
        warnings.append("HEURISTIC: clamp reduced raw->weight local sensitivity by >=90%")
    out["warnings"] = warnings
    return out


def aggregate(per_seed: List[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in per_seed
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in {"seed"}
        }
    )
    summary: Dict[str, Any] = {
        "seed_count": len(per_seed),
        "seeds": sorted(int(r["seed"]) for r in per_seed),
        "health_schema_complete_seed_count": sum(bool(r["health_schema_complete"]) for r in per_seed),
    }
    for key in numeric_keys:
        vals = [float(r[key]) for r in per_seed if key in r and math.isfinite(float(r[key]))]
        if vals:
            summary[key] = {
                "mean": statistics.fmean(vals),
                "min": min(vals),
                "max": max(vals),
            }
    return summary


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Router Health Audit",
        "",
        "This report is descriptive telemetry, **not** a ts-assay certificate.",
        "",
        f"Seeds: {report['aggregate']['seed_count']} {report['aggregate']['seeds']}",
        f"Complete new health schema: {report['aggregate']['health_schema_complete_seed_count']}/{report['aggregate']['seed_count']} seeds",
        "",
        "| Seed | Final FACT | Max mean weight | Max raw | Min softmax J | Min raw→weight J | Min clamp ratio | Warnings |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(report["per_seed"], key=lambda r: r["seed"]):
        def fmt(key: str) -> str:
            value = row.get(key)
            return "—" if value is None else f"{float(value):.6g}"
        lines.append(
            "| {seed} | {fact} | {w} | {raw} | {sj} | {rj} | {cr} | {warn} |".format(
                seed=row["seed"],
                fact=fmt("final_eval_fact_exact"),
                w=fmt("trajectory_max_router_weight_max_mean"),
                raw=fmt("trajectory_max_router_raw_logit_abs_mean"),
                sj=fmt("trajectory_min_router_softmax_jacobian_fro_mean"),
                rj=fmt("trajectory_min_router_raw_to_weight_jacobian_fro_mean"),
                cr=fmt("trajectory_min_router_clamp_attenuation_ratio_mean"),
                warn="; ".join(row.get("warnings", [])) or "—",
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `softmax J` becoming small isolates probability-simplex saturation.",
        "- `clamp ratio` below 1 isolates extra attenuation introduced by the tanh clamp.",
        "- `raw→weight J` combines both effects and answers the local question: can a perturbation of the raw router projection still move the routing weights?",
        "- None of these metrics says a path is *functionally necessary*. Use the frozen-router conditional co-ablation + ts-assay result for that claim.",
        "- HEURISTIC warnings are triage thresholds only; do not report them as formal significance tests.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-health-schema", action="store_true")
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    paths = discover(args.inputs)
    per_seed = [summarize_log(path, load(path), args.require_health_schema) for path in paths]
    seeds = [row["seed"] for row in per_seed]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds across input logs: {seeds}")
    report = {"per_seed": per_seed, "aggregate": aggregate(per_seed)}
    output_dir = args.output_dir or (args.inputs[0] if args.inputs[0].is_dir() else args.inputs[0].parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "router_health_audit.json"
    md_path = output_dir / "router_health_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(
        "status=audited "
        f"seeds={report['aggregate']['seed_count']} "
        f"health_schema={report['aggregate']['health_schema_complete_seed_count']}/{report['aggregate']['seed_count']}"
    )
    return report


if __name__ == "__main__":
    main()
