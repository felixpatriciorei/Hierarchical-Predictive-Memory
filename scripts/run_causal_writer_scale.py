"""Resumable scale stress test for the causal finite-capacity writer.

For each candidate-set size M, this runs three matched conditions:

* visible_bce: visible causal roles, independent writer BCE;
* visible_coverage: the same run plus the opt-in Top-C set-coverage loss; and
* masked_bce: roles replaced by an identical filler token, a negative control.

The exact hard episodic budget C and number p of required records stay fixed
while M grows.  Thus this tests ranking under an expanding candidate set, not
an invalid post-hoc-query capacity benchmark.  A JSONL record is fsynced after
each completed condition.  Restarting the exact command skips those records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_causal_writer_control import METRICS, make_train_args, parse_seeds
from hpm.train import run_training


CONDITION_SPECS: Mapping[str, Tuple[str, bool]] = {
    "visible_bce": ("visible", False),
    "visible_coverage": ("visible", True),
    "masked_bce": ("masked", False),
}
DEFAULT_CONDITIONS = tuple(CONDITION_SPECS)
SUMMARY_METRICS = (
    "eval_answer_exact",
    "eval_retrieval_top1",
    "eval_true_fact_written_rate",
    "eval_false_write_rate",
    "eval_missed_fact_rate",
    "eval_writer_required_topc_margin",
    "eval_writer_all_required_topc_rate",
    "eval_writer_topc_margin_samples",
)


def parse_positive_ints(value: str) -> List[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("provide at least one positive comma-separated integer")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def parse_scale_conditions(value: str) -> List[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated condition")
    unknown = sorted(set(values).difference(CONDITION_SPECS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown condition(s) {unknown!r}; choose from {tuple(CONDITION_SPECS)!r}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("conditions must be unique")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/causal_writer_scale"))
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--num-facts-values", type=parse_positive_ints, default=[4, 8, 16])
    parser.add_argument(
        "--conditions",
        type=parse_scale_conditions,
        default=list(DEFAULT_CONDITIONS),
        help="Comma-separated subset of visible_bce,visible_coverage,masked_bce.",
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--writer-required-facts", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, default=2)
    parser.add_argument("--coverage-weight", type=float, default=0.1)
    parser.add_argument("--coverage-eps", type=float, default=0.3)
    parser.add_argument("--coverage-iters", type=int, default=48)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a condition whose exact configuration has a completed JSONL record.",
    )
    return parser


def condition_configuration(
    control: argparse.Namespace,
    *,
    num_facts: int,
    seed: int,
    condition: str,
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Create one locked training namespace and its auditable configuration."""
    try:
        role_mode, uses_coverage = CONDITION_SPECS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown condition {condition!r}") from exc
    if uses_coverage and control.coverage_weight <= 0.0:
        raise ValueError("visible_coverage requires --coverage-weight > 0")

    weight = control.coverage_weight if uses_coverage else 0.0
    # ``make_train_args`` is shared with the fixed-M runner, whose control
    # namespace includes ``num_facts``.  Supply an isolated per-run namespace
    # instead of mutating the scale sweep's parser result between conditions.
    run_control = argparse.Namespace(
        **vars(control), num_facts=num_facts, writer_role_mode=role_mode
    )
    train_args = make_train_args(run_control, seed, weight)
    train_args.out_dir = str(control.out_dir / f"M{num_facts:03d}_{condition}")

    config = {
        "batch_size": control.batch_size,
        "condition": condition,
        "data_seed": seed,
        "d_model": control.d_model,
        "device": control.device,
        "eval_batches": control.eval_batches,
        "heads": control.heads,
        "layers": control.layers,
        "lr": control.lr,
        "memory_slots": control.memory_slots,
        "num_facts": num_facts,
        "seed": seed,
        "seq_len": control.seq_len,
        "steps": control.steps,
        "writer_required_facts": control.writer_required_facts,
        "writer_role_mode": role_mode,
        "writer_set_coverage_eps": control.coverage_eps,
        "writer_set_coverage_iters": control.coverage_iters,
        "lambda_writer_set_coverage": weight,
        "window": control.window,
    }
    return train_args, config


def configuration_signature(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_completed_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                signature = record["configuration_signature"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid completed-record line {line_number} in {path}") from exc
            if signature in records:
                raise ValueError(f"duplicate completed-record signature in {path}: {signature}")
            records[signature] = record
    return records


def append_completed_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    for record in records:
        key = (int(record["num_facts"]), str(record["condition"]))
        grouped.setdefault(key, []).append(record)

    rows = []
    for (num_facts, condition), group in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "num_facts": num_facts,
            "condition": condition,
            "n": len(group),
            "episodic_slots": int(group[0]["memory_slots"]),
            "required_facts": int(group[0]["writer_required_facts"]),
            "masked_analytic_ceiling": (
                int(group[0]["memory_slots"]) / num_facts if condition == "masked_bce" else ""
            ),
        }
        for metric in SUMMARY_METRICS:
            # Older, already-completed scale files predate the margin
            # diagnostic. Keep them readable on resume; a new configuration
            # records every current metric through the shared METRICS tuple.
            if all(metric in record for record in group):
                row[f"mean_{metric}"] = sum(float(record[metric]) for record in group) / len(group)
        rows.append(row)
    return rows


def pairwise_rows(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        key = (int(record["num_facts"]), int(record["seed"]))
        by_key.setdefault(key, {})[str(record["condition"])] = record

    rows = []
    for (num_facts, seed), conditions in sorted(by_key.items()):
        row: Dict[str, Any] = {"num_facts": num_facts, "seed": seed}
        visible_bce = conditions.get("visible_bce")
        coverage = conditions.get("visible_coverage")
        masked = conditions.get("masked_bce")
        if visible_bce is not None and coverage is not None:
            row["delta_coverage_true_fact"] = (
                float(coverage["eval_true_fact_written_rate"])
                - float(visible_bce["eval_true_fact_written_rate"])
            )
            row["delta_coverage_answer_exact"] = (
                float(coverage["eval_answer_exact"]) - float(visible_bce["eval_answer_exact"])
            )
            if (
                "eval_writer_required_topc_margin" in visible_bce
                and "eval_writer_required_topc_margin" in coverage
                and visible_bce.get("eval_writer_topc_margin_samples", 0.0) > 0.0
                and coverage.get("eval_writer_topc_margin_samples", 0.0) > 0.0
            ):
                row["delta_coverage_required_topc_margin"] = (
                    float(coverage["eval_writer_required_topc_margin"])
                    - float(visible_bce["eval_writer_required_topc_margin"])
                )
        if visible_bce is not None and masked is not None:
            row["visible_minus_masked_true_fact"] = (
                float(visible_bce["eval_true_fact_written_rate"])
                - float(masked["eval_true_fact_written_rate"])
            )
        if len(row) > 2:
            rows.append(row)
    return rows


def validate_control(control: argparse.Namespace) -> None:
    if control.steps < 1 or control.eval_batches < 1 or control.batch_size < 1:
        raise ValueError("steps, eval-batches, and batch-size must be positive")
    for num_facts in control.num_facts_values:
        if not 1 <= control.writer_required_facts <= control.memory_slots < num_facts:
            raise ValueError(
                f"M={num_facts} violates 1 <= required <= slots < M; got "
                f"required={control.writer_required_facts}, slots={control.memory_slots}"
            )


def main(argv: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    control = build_arg_parser().parse_args(argv)
    validate_control(control)
    records_path = control.out_dir / "completed_records.jsonl"
    completed = load_completed_records(records_path)
    requested: List[Tuple[str, argparse.Namespace, Dict[str, Any]]] = []
    for num_facts in control.num_facts_values:
        for seed in control.seeds:
            for condition in control.conditions:
                train_args, config = condition_configuration(
                    control, num_facts=num_facts, seed=seed, condition=condition
                )
                signature = configuration_signature(config)
                requested.append((signature, train_args, config))

    active_records: List[Dict[str, Any]] = []
    for signature, train_args, config in requested:
        if control.resume and signature in completed:
            record = completed[signature]
            active_records.append(record)
            print(
                f"resume: M={config['num_facts']} seed={config['seed']} "
                f"condition={config['condition']}"
            )
            continue

        metrics = run_training(train_args)
        record: Dict[str, Any] = {
            "configuration_signature": signature,
            **config,
            "causal_writer_capacity_contract": metrics["causal_writer_capacity_contract"],
            "eval_examples": metrics["eval_examples"],
            **{metric: metrics[metric] for metric in METRICS},
        }
        append_completed_record(records_path, record)
        completed[signature] = record
        active_records.append(record)
        print(
            f"completed: M={config['num_facts']} seed={config['seed']} "
            f"condition={config['condition']}"
        )

        write_csv(control.out_dir / "scale_summary.csv", summary_rows(active_records))
        write_csv(control.out_dir / "scale_pairwise.csv", pairwise_rows(active_records))

    write_csv(control.out_dir / "scale_summary.csv", summary_rows(active_records))
    write_csv(control.out_dir / "scale_pairwise.csv", pairwise_rows(active_records))
    print(f"wrote {control.out_dir / 'scale_summary.csv'}")
    print(f"wrote {control.out_dir / 'scale_pairwise.csv'}")
    return active_records


if __name__ == "__main__":
    main()
