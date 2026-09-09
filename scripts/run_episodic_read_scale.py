"""Resumable candidate-set sweep for the R1 episodic top-k read control.

For every candidate-set size ``M`` and seed, this runs the exact-forward
paired comparison between ``hard_topk`` and ``ste_topk``.  The candidate slots
are oracle-written and fixed, while an aliased query must select one of the
``M`` slots.  Consequently this is a read-gradient scale test, not a
learned-writer, router, or JEPA experiment.

Before either arm is trained, each ``(M, seed)`` receives the same hard-read
anti-saturation check used by the fixed-M control.  A completed JSONL record
is fsynced after each arm, so re-running the exact command resumes safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_episodic_read_control import (
    CONDITIONS,
    METRICS,
    condition_rows,
    make_train_args,
    paired_summary_rows,
    parse_conditions,
    parse_seeds,
    pretrain_retrieval_limit,
    pretraining_metrics,
    write_rows,
)
from hpm.train import run_training


def parse_candidate_counts(value: str) -> List[int]:
    """Parse distinct, supported fixed-slot candidate counts."""

    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated candidate count")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("candidate counts must be unique")
    unsupported = [item for item in values if not 4 <= item <= 50]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"aliased read control supports only 4 <= M <= 50; got {unsupported!r}"
        )
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/episodic_read_scale"))
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--num-facts-values", type=parse_candidate_counts, default=[4, 8, 16])
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=list(CONDITIONS),
        help="Comma-separated subset of hard_topk,ste_topk. Default runs matched pairs.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--max-pretrain-retrieval-top1",
        type=float,
        default=None,
        help=(
            "Override the per-M hard-read anti-saturation ceiling. The safe default is "
            "min(0.50, 1 / M + 0.15)."
        ),
    )
    parser.add_argument("--ste-eps", type=float, default=0.3)
    parser.add_argument("--ste-iters", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-step-log", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run and persist only the untrained anti-saturation checks; never optimize.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip an arm whose exact configuration has a completed JSONL record.",
    )
    return parser


def validate_control(control: argparse.Namespace) -> None:
    if control.steps < 1 or control.eval_batches < 1 or control.batch_size < 1 or control.log_every < 1:
        raise ValueError("steps, eval-batches, batch-size, and log-every must be positive")
    if control.seq_len < 8 or control.window < 1 or control.d_model < 1:
        raise ValueError("seq-len must be >= 8; window and d-model must be positive")
    if control.heads < 1 or control.layers < 1 or control.d_model % control.heads != 0:
        raise ValueError("layers and heads must be positive and d-model must be divisible by heads")
    if control.lr <= 0.0 or control.ste_eps <= 0.0 or control.ste_iters < 1:
        raise ValueError("lr/ste-eps must be positive and ste-iters >= 1")
    if control.max_pretrain_retrieval_top1 is not None and not (
        0.0 < control.max_pretrain_retrieval_top1 < 1.0
    ):
        raise ValueError("max-pretrain-retrieval-top1 must lie strictly between 0 and 1")


def run_configuration(
    control: argparse.Namespace,
    *,
    num_facts: int,
    seed: int,
    condition: str,
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Lock one scale/seed/arm into the reader-only R1 configuration."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition!r}")
    run_control_values = dict(vars(control))
    run_control_values.update(
        num_facts=num_facts,
        out_dir=control.out_dir / f"M{num_facts:03d}",
    )
    run_control = argparse.Namespace(**run_control_values)
    train_args = make_train_args(run_control, seed, condition)
    config = {
        "batch_size": control.batch_size,
        "condition": condition,
        "d_model": control.d_model,
        "data_seed": seed,
        "device": control.device,
        "episodic_read_ste_eps": control.ste_eps,
        "episodic_read_ste_iters": control.ste_iters,
        "eval_batches": control.eval_batches,
        "heads": control.heads,
        "layers": control.layers,
        "lr": control.lr,
        "log_every": control.log_every,
        "max_pretrain_retrieval_top1": pretrain_retrieval_limit(run_control),
        "num_facts": num_facts,
        "save_step_log": control.save_step_log,
        "seed": seed,
        "seq_len": control.seq_len,
        "steps": control.steps,
        "task": "aliased_kv",
        "top_k": 1,
        "window": control.window,
    }
    return train_args, config


def configuration_signature(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_completed_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                signature = str(record["configuration_signature"])
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


def pretraining_row(control: argparse.Namespace, num_facts: int, seed: int) -> Dict[str, Any]:
    """Evaluate the shared hard forward path before either paired arm trains."""

    run_control_values = dict(vars(control))
    run_control_values.update(
        num_facts=num_facts,
        out_dir=control.out_dir / f"M{num_facts:03d}",
    )
    run_control = argparse.Namespace(**run_control_values)
    pretrain = pretraining_metrics(run_control, seed)
    limit = pretrain_retrieval_limit(run_control)
    row = {
        "num_facts": num_facts,
        "seed": seed,
        "task": "aliased_kv",
        "chance_retrieval_top1": 1.0 / float(num_facts),
        "max_allowed_retrieval_top1": limit,
        "pretrain_eval_answer_exact": pretrain["answer_exact"],
        "pretrain_eval_answer_ce": pretrain["answer_ce"],
        "pretrain_eval_retrieval_top1": pretrain["retrieval_top1"],
        "pretrain_eval_retrieval_margin": pretrain["retrieval_margin"],
    }
    if pretrain["retrieval_top1"] > limit:
        raise RuntimeError(
            "refusing saturated read experiment: M={m} seed={seed} has untrained hard-top-1 "
            "retrieval {top1:.4f}, above allowed {limit:.4f}. Do not train this configuration; "
            "inspect the data model for a shortcut."
            .format(m=num_facts, seed=seed, top1=pretrain["retrieval_top1"], limit=limit)
        )
    return row


def active_records_by_m(records: Iterable[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["num_facts"])].append(dict(record))
    return grouped


def scale_pairwise_rows(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Add M to the fixed-M paired result rows, omitting incomplete pairs."""

    rows: List[Dict[str, Any]] = []
    for num_facts, group in sorted(active_records_by_m(records).items()):
        by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for record in group:
            by_seed[int(record["seed"])].append(record)
        complete = [
            item
            for seed in sorted(by_seed)
            if {record["condition"] for record in by_seed[seed]} == set(CONDITIONS)
            for item in paired_summary_rows(by_seed[seed])
        ]
        for row in complete:
            rows.append({"num_facts": num_facts, **row})
    return rows


def scale_summary_rows(pairwise: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in pairwise:
        grouped[int(row["num_facts"])].append(row)
    rows: List[Dict[str, Any]] = []
    for num_facts, group in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "num_facts": num_facts,
            "chance_retrieval_top1": 1.0 / float(num_facts),
            "paired_seed_count": len(group),
        }
        for metric in ("eval_answer_exact", "eval_retrieval_top1", "train_wall_time_sec"):
            delta_key = f"delta_ste_minus_hard_{metric}"
            row[f"mean_{delta_key}"] = sum(float(item[delta_key]) for item in group) / len(group)
        ratios = [
            float(item["ste_train_wall_time_sec"]) / float(item["hard_train_wall_time_sec"])
            for item in group
        ]
        row["mean_ste_over_hard_wall_time_ratio"] = sum(ratios) / len(ratios)
        rows.append(row)
    return rows


def write_progress_artifacts(
    control: argparse.Namespace,
    records: Iterable[Mapping[str, Any]],
    pretraining_rows: Iterable[Mapping[str, Any]],
) -> None:
    record_list = [dict(record) for record in records]
    pretrain_list = [dict(row) for row in pretraining_rows]
    if pretrain_list:
        write_rows(control.out_dir / "pretraining_check.csv", pretrain_list)

    by_m = active_records_by_m(record_list)
    pretraining_by_m: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in pretrain_list:
        pretraining_by_m[int(row["num_facts"])].append(row)
    for num_facts in sorted(set(by_m).union(pretraining_by_m)):
        rows = by_m.get(num_facts, [])
        run_dir = control.out_dir / f"M{num_facts:03d}"
        if pretraining_by_m[num_facts]:
            write_rows(run_dir / "pretraining_check.csv", pretraining_by_m[num_facts])
        if not rows:
            continue
        seeds_to_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for record in rows:
            seeds_to_records[int(record["seed"])].append(record)
        if seeds_to_records and all(
            {record["condition"] for record in seed_records} == set(CONDITIONS)
            for seed_records in seeds_to_records.values()
        ):
            write_rows(run_dir / "paired_summary.csv", paired_summary_rows(rows))
        else:
            write_rows(run_dir / "condition_records.csv", condition_rows(rows))

    pairwise = scale_pairwise_rows(record_list)
    if pairwise:
        write_rows(control.out_dir / "scale_pairwise.csv", pairwise)
        write_rows(control.out_dir / "scale_summary.csv", scale_summary_rows(pairwise))


def main(argv: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    control = build_arg_parser().parse_args(argv)
    validate_control(control)
    records_path = control.out_dir / "completed_records.jsonl"
    completed = load_completed_records(records_path)

    pretraining_rows: List[Dict[str, Any]] = []
    active_records: List[Dict[str, Any]] = []
    requested: List[Tuple[str, argparse.Namespace, Dict[str, Any]]] = []
    for num_facts in control.num_facts_values:
        for seed in control.seeds:
            baseline = pretraining_row(control, num_facts, seed)
            pretraining_rows.append(baseline)
            print(
                "M={m} seed={seed} pretrain_retrieval_top1={top1:.4f} limit={limit:.4f}".format(
                    m=num_facts,
                    seed=seed,
                    top1=baseline["pretrain_eval_retrieval_top1"],
                    limit=baseline["max_allowed_retrieval_top1"],
                )
            )
            for condition in control.conditions:
                train_args, config = run_configuration(
                    control, num_facts=num_facts, seed=seed, condition=condition
                )
                requested.append((configuration_signature(config), train_args, config))
        write_progress_artifacts(control, active_records, pretraining_rows)

    if control.preflight_only:
        print(f"wrote {control.out_dir / 'pretraining_check.csv'}")
        return pretraining_rows

    for signature, train_args, config in requested:
        if control.resume and signature in completed:
            record = completed[signature]
            active_records.append(record)
            print(
                f"resume: M={config['num_facts']} seed={config['seed']} "
                f"condition={config['condition']}"
            )
        else:
            metrics = run_training(train_args)
            record = {
                "configuration_signature": signature,
                **config,
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
        write_progress_artifacts(control, active_records, pretraining_rows)

    if scale_pairwise_rows(active_records):
        print(f"wrote {control.out_dir / 'scale_summary.csv'}")
        print(f"wrote {control.out_dir / 'scale_pairwise.csv'}")
    return active_records


if __name__ == "__main__":
    main()
