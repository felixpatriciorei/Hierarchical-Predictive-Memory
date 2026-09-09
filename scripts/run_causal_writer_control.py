"""Paired self-writing experiment for the causal finite-capacity writer.

This runner is deliberately narrow.  It compares BCE-only writer training to
BCE plus Top-C set coverage on identical seeds and data, with all teacher
forcing, JEPA writer bias, router mixing, and continuous delta-rule alternatives
excluded.  See docs/engineering/causal_writer_control.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# This file is meant to be run directly from any current directory.  Add the
# repository root before importing the local package instead of requiring a
# user to set PYTHONPATH manually.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hpm.train import build_arg_parser as build_train_arg_parser
from hpm.train import run_training


METRICS = (
    "eval_answer_exact",
    "eval_answer_ce",
    "eval_retrieval_top1",
    "eval_true_fact_written_rate",
    "eval_false_write_rate",
    "eval_missed_fact_rate",
    "eval_writer_required_topc_margin",
    "eval_writer_all_required_topc_rate",
    "eval_writer_topc_margin_samples",
    "train_writer_set_coverage_loss",
    "train_writer_set_coverage_mass_error",
    "train_wall_time_sec",
)

CONDITIONS = ("bce_only", "bce_plus_coverage")


def parse_seeds(value: str) -> List[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one comma-separated seed")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def parse_conditions(value: str) -> List[str]:
    """Parse an ordered, non-duplicated subset of the supported objectives."""
    conditions = [part.strip() for part in value.split(",") if part.strip()]
    if not conditions:
        raise argparse.ArgumentTypeError("provide at least one comma-separated condition")
    unknown = sorted(set(conditions).difference(CONDITIONS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown condition(s) {unknown!r}; choose from {CONDITIONS!r}"
        )
    if len(set(conditions)) != len(conditions):
        raise argparse.ArgumentTypeError("conditions must be unique")
    return conditions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/causal_writer_control"))
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=list(CONDITIONS),
        help=(
            "Comma-separated subset of bce_only,bce_plus_coverage. The default runs "
            "the paired comparison. Use bce_only for the masked-role negative control."
        ),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--writer-required-facts", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, default=2)
    parser.add_argument(
        "--writer-role-mode",
        choices=["visible", "masked"],
        default="visible",
        help=(
            "Whether causal REMEMBER/IRRELEVANT tokens are visible. The masked setting "
            "is a BCE-only negative control and cannot use Top-C coverage supervision."
        ),
    )
    parser.add_argument("--coverage-weight", type=float, default=0.1)
    parser.add_argument("--coverage-eps", type=float, default=0.3)
    parser.add_argument("--coverage-iters", type=int, default=48)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def make_train_args(control: argparse.Namespace, seed: int, coverage_weight: float):
    """Create a complete normal training namespace from locked control values."""
    args = build_train_arg_parser().parse_args([])
    args.model = "hpm_lite_v2"
    args.task = "causal_salience_kv"
    args.write_mode = "learned"
    args.writer_candidate_mode = "fact_pairs"
    args.writer_role_mode = control.writer_role_mode
    args.episodic_only = True
    args.use_jepa_writer_bias = False
    args.use_token_jepa_aux = False
    args.delta_rule_writer = False
    args.learned_writer_teacher_forcing_steps = 0
    args.scheduled_sampling_anneal_steps = 0
    args.sinkhorn_warmup_weight = 0.0
    args.primary_loss_anneal_steps = 0

    args.seed = seed
    args.data_seed = seed
    args.steps = control.steps
    args.eval_every = control.steps
    args.eval_batches = control.eval_batches
    args.batch_size = control.batch_size
    args.seq_len = control.seq_len
    args.window = control.window
    args.d_model = control.d_model
    args.layers = control.layers
    args.heads = control.heads
    args.lr = control.lr
    args.num_facts = control.num_facts
    args.writer_required_facts = control.writer_required_facts
    args.memory_slots = control.memory_slots
    args.lambda_writer_set_coverage = coverage_weight
    args.writer_set_coverage_eps = control.coverage_eps
    args.writer_set_coverage_iters = control.coverage_iters
    args.writer_set_coverage_mass_tolerance = 1.0e-3
    args.device = control.device
    args.out_dir = str(control.out_dir)
    args.save_checkpoint = False
    args.save_step_log = False
    args.record_vram = False
    args.log_every = control.steps
    return args


def paired_summary_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn two metric records per seed into auditable paired deltas."""
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        condition = record["condition"]
        if condition not in {"bce_only", "bce_plus_coverage"}:
            raise ValueError(f"unknown condition {condition!r}")
        seed = int(record["seed"])
        if condition in by_seed.setdefault(seed, {}):
            raise ValueError(f"duplicate record for seed={seed}, condition={condition}")
        by_seed[seed][condition] = record

    rows = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if set(pair) != {"bce_only", "bce_plus_coverage"}:
            raise ValueError(f"seed={seed} is missing one paired condition")
        baseline = pair["bce_only"]
        coverage = pair["bce_plus_coverage"]
        row: Dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            baseline_value = float(baseline[metric])
            coverage_value = float(coverage[metric])
            row[f"bce_{metric}"] = baseline_value
            row[f"coverage_{metric}"] = coverage_value
            row[f"delta_{metric}"] = coverage_value - baseline_value
        rows.append(row)
    return rows


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty paired summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize an unpaired control run for an auditable CSV artifact."""
    rows = []
    for record in records:
        condition = record["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        rows.append({"seed": int(record["seed"]), "condition": condition, **{metric: record[metric] for metric in METRICS}})
    return sorted(rows, key=lambda row: (row["seed"], row["condition"]))


def assert_visible_role_beats_masked_control(
    visible_metrics: Dict[str, Any],
    masked_metrics: Dict[str, Any],
    *,
    min_coverage_gap: float = 0.0,
) -> Dict[str, float]:
    """Validate the causal counterfactual before claiming role use.

    This is deliberately an analysis guard, not a training constraint.  A
    positive gap alone is not a scaling claim, but absent a positive gap the
    visible-role experiment cannot demonstrate that causal utility was used.
    """
    if visible_metrics.get("causal_writer_capacity_contract") != "causal_salience_aligned":
        raise ValueError("visible metrics are not from the aligned causal-salience control")
    if masked_metrics.get("causal_writer_capacity_contract") != "masked_role_negative_control":
        raise ValueError("masked metrics are not from the masked-role negative control")
    visible_coverage = float(visible_metrics["eval_true_fact_written_rate"])
    masked_coverage = float(masked_metrics["eval_true_fact_written_rate"])
    gap = visible_coverage - masked_coverage
    if gap <= min_coverage_gap:
        raise AssertionError(
            "visible-role writer did not exceed its matched masked-role control: "
            f"gap={gap:.6f}, required>{min_coverage_gap:.6f}"
        )
    return {
        "visible_coverage": visible_coverage,
        "masked_coverage": masked_coverage,
        "coverage_gap": gap,
        "visible_answer_exact": float(visible_metrics["eval_answer_exact"]),
        "masked_answer_exact": float(masked_metrics["eval_answer_exact"]),
    }


def main(argv: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    control = build_arg_parser().parse_args(argv)
    if control.steps < 1 or control.eval_batches < 1 or control.batch_size < 1:
        raise ValueError("steps, eval-batches, and batch-size must be positive")
    if not 1 <= control.writer_required_facts <= control.memory_slots < control.num_facts:
        raise ValueError("control requires 1 <= writer-required-facts <= memory-slots < num-facts")
    if "bce_plus_coverage" in control.conditions:
        if control.coverage_weight <= 0.0:
            raise ValueError("coverage-weight must be positive when bce_plus_coverage is selected")
        if control.writer_role_mode != "visible":
            raise ValueError(
                "bce_plus_coverage requires --writer-role-mode visible; the masked-role "
                "negative control is intentionally BCE-only"
            )

    records = []
    for seed in control.seeds:
        for condition in control.conditions:
            weight = 0.0 if condition == "bce_only" else control.coverage_weight
            metrics = run_training(make_train_args(control, seed, weight))
            records.append({"seed": seed, "condition": condition, **{key: metrics[key] for key in METRICS}})

    if set(control.conditions) == set(CONDITIONS):
        rows = paired_summary_rows(records)
        summary_path = control.out_dir / "paired_summary.csv"
        write_rows(summary_path, rows)
        print(f"wrote {summary_path}")
        for row in rows:
            print(
                "seed={seed} delta_coverage={coverage:+.4f} delta_exact={exact:+.4f}".format(
                    seed=row["seed"],
                    coverage=row["delta_eval_true_fact_written_rate"],
                    exact=row["delta_eval_answer_exact"],
                )
            )
        return rows

    rows = condition_rows(records)
    record_path = control.out_dir / "condition_records.csv"
    write_rows(record_path, rows)
    print(f"wrote {record_path}")
    return rows


if __name__ == "__main__":
    main()
