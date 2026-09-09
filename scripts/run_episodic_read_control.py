"""Paired aliased-query control for the episodic top-k read gradient.

This deliberately does *not* exercise learned writing, the router, or JEPA.
It holds exact episodic slots fixed, uses top_k=1, and compares normal hard
top-k read to exact-forward ``ste_topk`` on identical model/data seeds.  The
``aliased_kv`` task makes the query a deterministic alias of the stored key,
not the stored key token itself.  A pre-training retrieval gate rejects a
configuration that is already solved before either learning rule can differ.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hpm.data import FactRecallConfig, FactRecallDataset
from hpm.evaluate import evaluate_batches
from hpm.train import build_arg_parser as build_train_arg_parser
from hpm.train import make_model, run_training
from hpm.utils import resolve_device, set_seed


CONDITIONS = ("hard_topk", "ste_topk")
METRICS = (
    "eval_answer_exact",
    "eval_answer_ce",
    "eval_retrieval_top1",
    "eval_retrieval_margin",
    "train_loss",
    "train_retrieval_loss",
    "train_wall_time_sec",
    "parameters",
    "peak_vram_mb",
)


def parse_seeds(value: str) -> List[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one comma-separated seed")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def parse_conditions(value: str) -> List[str]:
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
    parser.add_argument("--out-dir", type=Path, default=Path("runs/episodic_read_aliased_control"))
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=list(CONDITIONS),
        help="Comma-separated subset of hard_topk,ste_topk. Default runs the paired comparison.",
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
    parser.add_argument("--num-facts", type=int, default=8)
    parser.add_argument(
        "--max-pretrain-retrieval-top1",
        type=float,
        default=None,
        help=(
            "Reject a seed if untrained hard-top-1 retrieval exceeds this value. Default is "
            "min(0.50, 1 / num_facts + 0.15): it rejects pre-solved readers while allowing "
            "the observed random-initialization variation in the eight-slot aliased control. "
            "Every seed's actual pre-training result is persisted in pretraining_check.csv."
        ),
    )
    parser.add_argument("--ste-eps", type=float, default=0.3)
    parser.add_argument("--ste-iters", type=int, default=50)
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help=(
            "Training/evaluation progress interval. The STE relaxation is skipped under "
            "evaluation's torch.no_grad() context, so this is a hard-read metric check "
            "rather than extra OT training work."
        ),
    )
    parser.add_argument(
        "--save-step-log",
        action="store_true",
        help="Persist the per-interval metrics for learning-curve analysis, not just final paired deltas.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run only the untrained hard-read anti-saturation check and write "
            "pretraining_check.csv. This never calls the training loop; use it before "
            "a paired run to confirm that aliased_kv begins near chance."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser


def make_train_args(control: argparse.Namespace, seed: int, condition: str) -> argparse.Namespace:
    """Construct the fully specified, reader-only training condition."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    args = build_train_arg_parser().parse_args([])
    args.model = "hpm_lite_v2"
    args.task = "aliased_kv"
    args.write_mode = "oracle"
    args.episodic_only = True
    args.top_k = 1
    args.episodic_read_mode = condition
    args.episodic_read_ste_eps = control.ste_eps
    args.episodic_read_ste_iters = control.ste_iters
    # An explicit retrieval-score cross entropy would train the score head in
    # both arms and confound the answer-loss-through-cutoff question.
    args.lambda_ret = 0.0
    args.lambda_writer = 0.0
    args.lambda_writer_set_coverage = 0.0
    args.lambda_router_entropy = 0.0
    args.lambda_router_z_loss = 0.0
    args.lambda_jepa = 0.0
    args.use_jepa_writer_bias = False
    args.use_token_jepa_aux = False
    args.delta_rule_writer = False
    args.delta_rule_oracle_blend = False
    args.learned_writer_teacher_forcing_steps = 0
    args.scheduled_sampling_anneal_steps = 0
    args.sinkhorn_warmup_weight = 0.0
    args.primary_loss_anneal_steps = 0
    args.memory_slots = None
    args.memory_null_slot = False
    args.memory_control = "normal"
    args.oracle_memory = True

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
    args.device = control.device
    args.out_dir = str(control.out_dir)
    args.save_checkpoint = False
    args.save_step_log = control.save_step_log
    args.record_vram = False
    args.log_every = min(control.log_every, control.steps)
    return args


def pretrain_retrieval_limit(control: argparse.Namespace) -> float:
    """Return the ceiling that identifies a pre-solved untrained reader."""

    if control.max_pretrain_retrieval_top1 is not None:
        return float(control.max_pretrain_retrieval_top1)
    # A random 32- or 64-dimensional embedding initialization can happen to
    # align part of the fixed alias-to-key codebook.  The six-seed CPU
    # preflight for aliased_kv / eight slots ranged 0.107--0.244, despite a
    # nominal 0.125 chance rate.  A chance-plus-small-margin gate would reject
    # such valid but uneven starts. A fixed 0.50 ceiling would be too lax in a
    # later slot-count sweep, however: 0.50 is already near solved at four
    # slots.  Keep a 0.15 absolute allowance above chance (the six-seed M=8
    # maximum was 0.2437 against a 0.275 threshold), capped at 0.50 for very
    # small candidate sets. This rejects the literal-key pathology (1.0) and
    # remains meaningful as slot count grows.
    return min(0.50, 1.0 / float(control.num_facts) + 0.15)


def pretraining_metrics(control: argparse.Namespace, seed: int) -> Dict[str, float]:
    """Measure the hard-read baseline before optimization, on the eval stream.

    ``ste_topk`` has exactly the same forward read as hard top-k.  One
    hard-read evaluation is thus the appropriate pre-training check for both
    arms, and avoids running the OT surrogate just to validate the baseline.
    """

    args = make_train_args(control, seed, "hard_topk")
    set_seed(args.seed)
    device = resolve_device(args.device)
    model = make_model(args, device)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=args.seq_len,
            window=args.window,
            task=args.task,
            num_facts=args.num_facts,
            num_hard_negatives=args.num_hard_negatives,
            seed=data_seed + 100_000,
            oracle_memory=args.oracle_memory,
            repeated_keys=args.repeated_keys,
            similar_values=args.similar_values,
            distractor_fact_spans=args.distractor_fact_spans,
            query_key_noise_only=args.query_key_noise_only,
            fact_order=args.fact_order,
            writer_required_facts=args.writer_required_facts,
            writer_role_mode=args.writer_role_mode,
        )
    )
    return evaluate_batches(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        batches=args.eval_batches,
        device=device,
        task=args.task,
        top_k=args.top_k,
        memory_control=args.memory_control,
        write_mode=args.write_mode,
        use_learned_writer=False,
    )


def paired_summary_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build seed-matched STE-minus-hard deltas without cross-seed pooling."""

    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        seed = int(record["seed"])
        condition = record["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        if condition in by_seed.setdefault(seed, {}):
            raise ValueError(f"duplicate record for seed={seed}, condition={condition}")
        by_seed[seed][condition] = record

    rows = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if set(pair) != set(CONDITIONS):
            raise ValueError(f"seed={seed} is missing one paired condition")
        hard = pair["hard_topk"]
        ste = pair["ste_topk"]
        row: Dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            hard_value = float(hard[metric])
            ste_value = float(ste[metric])
            row[f"hard_{metric}"] = hard_value
            row[f"ste_{metric}"] = ste_value
            row[f"delta_ste_minus_hard_{metric}"] = ste_value - hard_value
        rows.append(row)
    return rows


def condition_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (
            {
                "seed": int(record["seed"]),
                "condition": record["condition"],
                **{metric: record[metric] for metric in METRICS},
            }
            for record in records
        ),
        key=lambda row: (row["seed"], row["condition"]),
    )


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    control = build_arg_parser().parse_args(argv)
    if control.steps < 1 or control.eval_batches < 1 or control.batch_size < 1 or control.log_every < 1:
        raise ValueError("steps, eval-batches, batch-size, and log-every must be positive")
    if control.seq_len < 8 or control.window < 1 or control.d_model < 1:
        raise ValueError("seq-len must be >= 8; window and d-model must be positive")
    if control.heads < 1 or control.layers < 1 or control.d_model % control.heads != 0:
        raise ValueError("layers and heads must be positive and d-model must be divisible by heads")
    if not 4 <= control.num_facts <= 50:
        raise ValueError("aliased control requires 4 <= num-facts <= 50")
    if control.lr <= 0.0 or control.ste_eps <= 0.0 or control.ste_iters < 1:
        raise ValueError("lr/ste-eps must be positive and ste-iters >= 1")
    retrieval_limit = pretrain_retrieval_limit(control)
    if not 0.0 < retrieval_limit < 1.0:
        raise ValueError("max-pretrain-retrieval-top1 must lie strictly between 0 and 1")

    records = []
    pretraining_rows = []
    for seed in control.seeds:
        pretrain = pretraining_metrics(control, seed)
        pretraining_row = {
            "seed": seed,
            "task": "aliased_kv",
            "chance_retrieval_top1": 1.0 / float(control.num_facts),
            "max_allowed_retrieval_top1": retrieval_limit,
            "pretrain_eval_answer_exact": pretrain["answer_exact"],
            "pretrain_eval_answer_ce": pretrain["answer_ce"],
            "pretrain_eval_retrieval_top1": pretrain["retrieval_top1"],
            "pretrain_eval_retrieval_margin": pretrain["retrieval_margin"],
        }
        pretraining_rows.append(pretraining_row)
        write_rows(control.out_dir / "pretraining_check.csv", pretraining_rows)
        if pretrain["retrieval_top1"] > retrieval_limit:
            raise RuntimeError(
                "refusing saturated read experiment: untrained hard-top-1 retrieval is "
                f"{pretrain['retrieval_top1']:.4f}, above the allowed {retrieval_limit:.4f}. "
                "Do not train this configuration; adjust the data model or inspect the task leak."
            )
        print(
            "seed={seed} pretrain_retrieval_top1={top1:.4f} limit={limit:.4f}".format(
                seed=seed, top1=pretrain["retrieval_top1"], limit=retrieval_limit
            )
        )
        if control.preflight_only:
            continue
        for condition in control.conditions:
            metrics = run_training(make_train_args(control, seed, condition))
            records.append(
                {
                    "seed": seed,
                    "condition": condition,
                    **{metric: metrics[metric] for metric in METRICS},
                }
            )

    if control.preflight_only:
        print(f"wrote {control.out_dir / 'pretraining_check.csv'}")
        return pretraining_rows

    if set(control.conditions) == set(CONDITIONS):
        rows = paired_summary_rows(records)
        output_path = control.out_dir / "paired_summary.csv"
        write_rows(output_path, rows)
        print(f"wrote {output_path}")
        for row in rows:
            print(
                "seed={seed} delta_ste_exact={exact:+.4f} delta_ste_retrieval_top1={ret:+.4f}".format(
                    seed=row["seed"],
                    exact=row["delta_ste_minus_hard_eval_answer_exact"],
                    ret=row["delta_ste_minus_hard_eval_retrieval_top1"],
                )
            )
        return rows

    rows = condition_rows(records)
    output_path = control.out_dir / "condition_records.csv"
    write_rows(output_path, rows)
    print(f"wrote {output_path}")
    return rows


if __name__ == "__main__":
    main()
