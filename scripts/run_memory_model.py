from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hpm.train import run_training

VALID_MODELS = {"local", "hpm_lite", "hpm_lite_v2"}

SUMMARY_COLUMNS = [
    "model",
    "task",
    "write_mode",
    "seq_len",
    "window",
    "seed",
    "batch_size",
    "steps",
    "d_model",
    "layers",
    "heads",
    "eval_answer_exact",
    "eval_answer_ce",
    "eval_retrieval_top1",
    "eval_retrieval_topk",
    "eval_avg_written_slots",
    "eval_true_fact_written_rate",
    "eval_false_write_rate",
    "eval_missed_fact_rate",
    "eval_retrieval_margin",
    "parameters",
    "train_wall_time_sec",
    "examples_per_sec_recent",
    "eval_examples_per_sec",
    "peak_vram_mb",
    "run_dir",
    "step_log_path",
]

RAW_SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "task",
    "write_mode",
    "seq_len",
    "window",
    "seed",
    "batch_size",
    "steps",
    "d_model",
    "layers",
    "heads",
    "parameters",
    "device",
    "eval_answer_exact",
    "eval_answer_ce",
    "eval_retrieval_top1",
    "eval_retrieval_topk",
    "eval_true_fact_written_rate",
    "eval_false_write_rate",
    "eval_missed_fact_rate",
    "eval_avg_written_slots",
    "eval_retrieval_margin",
    "train_wall_time_sec",
    "examples_per_sec_recent",
    "eval_examples_per_sec",
    "peak_vram_mb",
    "run_dir",
    "step_log_path",
]

LOCAL_ONLY_BLANK_COLUMNS = [
    "eval_retrieval_top1",
    "eval_retrieval_topk",
    "eval_true_fact_written_rate",
    "eval_false_write_rate",
    "eval_missed_fact_rate",
    "eval_avg_written_slots",
    "eval_retrieval_margin",
]


def parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise argparse.ArgumentTypeError("--models must include at least one model")
    invalid = [model for model in models if model not in VALID_MODELS]
    if invalid:
        valid = ",".join(sorted(VALID_MODELS))
        raise argparse.ArgumentTypeError(f"invalid model(s): {','.join(invalid)}; valid choices: {valid}")

    deduped: list[str] = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped


def str_to_bool_arg(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact HPM-Lite Memory Model comparison from the project brief."
    )
    parser.add_argument("--models", type=parse_models, default=parse_models("local,hpm_lite"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-facts", type=int, default=8)
    parser.add_argument(
        "--memory-slots",
        type=int,
        default=None,
        help="Cap on episodic writer slots, independent of --num-facts. Default (unset) = uncapped. Only takes effect with --write-mode learned and after teacher forcing ends.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/memory_model")
    parser.add_argument("--memory-null-slot", action="store_true")
    parser.add_argument("--null-score-init", type=float, default=0.0)
    parser.add_argument("--write-mode", choices=["oracle", "learned"], default="oracle")
    parser.add_argument("--lambda-writer", type=float, default=0.1)
    parser.add_argument(
        "--episodic-only",
        action="store_true",
        help="hpm_lite_v2 only: drop recurrent/fast_weight/router, answer from episodic read directly.",
    )
    parser.add_argument(
        "--lambda-router-entropy",
        type=float,
        default=0.0,
        help="Weight on the router entropy auxiliary loss (hpm_lite_v2 only). Default 0.0 = off.",
    )
    parser.add_argument(
        "--lambda-router-z-loss",
        type=float,
        default=0.0,
        help=(
            "Weight on the router z-loss auxiliary loss (hpm_lite_v2 only; ST-MoE style, "
            "logsumexp(router_logits)**2, penalizes logit magnitude to prevent softmax "
            "saturation/one-hot lock). Default 0.0 = off. Try 1e-3."
        ),
    )
    parser.add_argument(
        "--router-logit-clamp",
        type=float,
        default=None,
        help=(
            "hpm_lite_v2 only: smoothly bound |router_logit| <= this value via tanh "
            "before softmax -- an architectural guarantee against permanent router "
            "one-hot lock, unlike --lambda-router-z-loss which is a soft incentive. "
            "Default None = off. Try 3.0."
        ),
    )
    parser.add_argument("--router-starvation-guard", type=str_to_bool_arg, default=False)
    parser.add_argument("--router-starvation-a0", type=float, default=4.352636317218953)
    parser.add_argument("--router-starvation-d-max", type=float, default=0.01687028105399182)
    parser.add_argument("--router-discovery-horizon", type=int, default=300)
    parser.add_argument("--router-p-min", type=float, default=0.005850833375006914)
    parser.add_argument(
        "--lambda-jepa",
        type=float,
        default=0.0,
        help="Weight on the JEPA-lite auxiliary loss(es) (hpm_lite_v2 only). Default 0.0 = off.",
    )
    parser.add_argument(
        "--use-jepa-writer-bias",
        type=str_to_bool_arg,
        default=False,
        help="hpm_lite_v2 + --write-mode learned only: bias writer selection with JEPA-lite token surprise. Default False = off.",
    )
    parser.add_argument("--jepa-writer-bias-scale", type=float, default=1.0)
    parser.add_argument("--learned-writer-teacher-forcing-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--save-step-log", action="store_true")
    parser.add_argument("--record-vram", action="store_true")
    parser.add_argument("--summary-csv", type=str, default="results/raw/run_summary.csv")
    parser.add_argument("--save-checkpoint", type=str_to_bool_arg, default=True)
    return parser


def append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def make_training_args(args: argparse.Namespace, model: str) -> SimpleNamespace:
    """Create the single-model training namespace.

    The command-line seed is now the recorded and executed seed for every model.
    Earlier versions added a model-specific offset, which made run IDs like
    ``seed1`` appear when the user asked for ``--seed 0``. That was confusing
    and made processed CSVs need manual repair.
    """
    return SimpleNamespace(
        model=model,
        task="kv",
        seq_len=args.seq_len,
        window=args.window,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_every=args.log_every if args.log_every and args.log_every > 0 else max(1, args.steps),
        eval_batches=10,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        lr=3.0e-4,
        seed=args.seed,
        device=args.device,
        lambda_ret=0.1,
        lambda_writer=args.lambda_writer,
        lambda_router_entropy=getattr(args, "lambda_router_entropy", 0.0),
        lambda_router_z_loss=getattr(args, "lambda_router_z_loss", 0.0),
        router_logit_clamp=getattr(args, "router_logit_clamp", None),
        router_starvation_guard=getattr(args, "router_starvation_guard", False),
        router_starvation_a0=getattr(args, "router_starvation_a0", 4.352636317218953),
        router_starvation_d_max=getattr(args, "router_starvation_d_max", 0.01687028105399182),
        router_discovery_horizon=getattr(args, "router_discovery_horizon", 300),
        router_p_min=getattr(args, "router_p_min", 0.005850833375006914),
        lambda_jepa=getattr(args, "lambda_jepa", 0.0),
        use_jepa_writer_bias=getattr(args, "use_jepa_writer_bias", False),
        jepa_writer_bias_scale=getattr(args, "jepa_writer_bias_scale", 1.0),
        episodic_only=getattr(args, "episodic_only", False),
        learned_writer_teacher_forcing_steps=args.learned_writer_teacher_forcing_steps,
        top_k=1,
        memory_null_slot=args.memory_null_slot,
        null_score_init=args.null_score_init,
        memory_control="normal",
        # Local uses oracle memory selection internally only because the dataset
        # and loss/evaluation helpers expect those tensors to exist. It is not a
        # memory-writing model, so the public summary row is normalized to
        # write_mode="none" below.
        write_mode=args.write_mode if model in {"hpm_lite", "hpm_lite_v2"} else "oracle",
        oracle_memory=True,
        num_facts=args.num_facts,
        # Cap on episodic writer slots, independent of num_facts (hpm_lite_v2 +
        # write_mode=learned only -- see build_model in hpm_lite/train.py). Was
        # accepted by this script's own argparse but silently dropped here,
        # since this SimpleNamespace is built field-by-field rather than by
        # forwarding **vars(args): the flag parsed but never reached the model,
        # so every --memory-slots run through this entry point was actually
        # uncapped regardless of what was passed on the command line.
        memory_slots=getattr(args, "memory_slots", None),
        repeated_keys=False,
        similar_values=False,
        distractor_fact_spans=0,
        query_key_noise_only=False,
        fact_order="random",
        out_dir=args.out_dir,
        save_checkpoint=args.save_checkpoint,
        log_every=args.log_every,
        save_step_log=args.save_step_log,
        record_vram=args.record_vram,
    )


def row_with_run_id(row: dict[str, Any]) -> dict[str, Any]:
    run_dir = str(row.get("run_dir", ""))
    run_id = Path(run_dir).name if run_dir else ""
    return {"run_id": run_id, **row}


def normalize_summary_row(
    row: dict[str, Any],
    *,
    requested_seed: int,
    d_model: int,
    layers: int,
    heads: int,
) -> dict[str, Any]:
    """Repair fields that must be explicit in research CSV outputs.

    ``run_training`` returns metrics from the training loop. This function adds
    the command-level configuration that downstream analysis needs and removes
    misleading local-baseline memory bookkeeping.
    """
    normalized = row_with_run_id(dict(row))
    normalized["seed"] = requested_seed
    normalized["d_model"] = d_model
    normalized["layers"] = layers
    normalized["heads"] = heads

    if normalized.get("model") == "local":
        normalized["write_mode"] = "none"
        for column in LOCAL_ONLY_BLANK_COLUMNS:
            normalized[column] = ""

    return normalized


def main() -> None:
    args = build_arg_parser().parse_args()
    rows: list[dict[str, Any]] = []

    for model in args.models:
        metrics = run_training(make_training_args(args, model))
        rows.append(
            normalize_summary_row(
                metrics,
                requested_seed=args.seed,
                d_model=args.d_model,
                layers=args.layers,
                heads=args.heads,
            )
        )

    out_path = Path(args.out_dir) / "memory_model_summary.csv"
    write_csv_rows(out_path, SUMMARY_COLUMNS, rows)
    if args.summary_csv:
        append_csv_rows(Path(args.summary_csv), RAW_SUMMARY_COLUMNS, rows)

    print(f"wrote {out_path}")
    for row in rows:
        exact = float(row.get("eval_answer_exact", 0.0))
        ce = float(row.get("eval_answer_ce", 0.0))
        print(f"{row['model']} exact={exact:.4f} ce={ce:.4f}")

    by_model = {row["model"]: row for row in rows}
    if "local" in by_model and ("hpm_lite" in by_model or "hpm_lite_v2" in by_model):
        local = by_model["local"]
        hpm = by_model.get("hpm_lite", by_model.get("hpm_lite_v2"))
        gain = float(hpm.get("eval_answer_exact", 0.0)) - float(local.get("eval_answer_exact", 0.0))
        print(f"exact gain={gain:.4f}")


if __name__ == "__main__":
    main()
