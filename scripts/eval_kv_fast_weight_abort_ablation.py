from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from hpm.data import FactRecallConfig, FactRecallDataset
from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model
from hpm.metrics import answer_cross_entropy, answer_span_correct_mask
from hpm.write_modes import apply_write_mode

PATH_NAMES = ("local", "recurrent", "fast_weight", "episodic")
FAST_WEIGHT_INDEX = PATH_NAMES.index("fast_weight")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Paired KV fast-weight necessity check from checkpoint_at_abort.pt: "
            "baseline vs post-gate fast_weight=0 under the exact baseline router weights, "
            "with no renormalization."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval-batches", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False; refusing CPU fallback.")
    return torch.device(device_arg)


def model_forward(model, batch, run_args, *, fixed_weights=None, contribution_mask=None):
    return model(
        batch["input_ids"],
        memory_token_positions=batch["memory_token_positions"],
        memory_mask=batch["memory_mask"],
        answer_positions=batch["answer_positions"],
        query_key_positions=batch["query_key_positions"],
        top_k=int(run_args.get("top_k", 1)),
        task=str(run_args.get("task", "kv")),
        hop_positive_memory_indices=batch["hop_positive_memory_indices"],
        positive_memory_indices=batch.get("positive_memory_indices"),
        positive_memory_mask=batch.get("positive_memory_mask"),
        memory_control=str(run_args.get("memory_control", "normal")),
        use_learned_writer=str(run_args.get("write_mode", "oracle")) == "learned",
        learned_writer_teacher_forcing=False,
        router_fixed_weights=fixed_weights,
        router_contribution_mask=contribution_mask,
    )


def main() -> None:
    cli = parse_args()
    device = require_device(cli.device)
    ckpt = torch.load(cli.checkpoint, map_location=device)

    run_args = dict(ckpt["args"])
    if run_args.get("model") != "hpm_lite_v2":
        raise SystemExit("checkpoint is not hpm_lite_v2")
    if run_args.get("task") != "kv":
        raise SystemExit(f"expected task='kv', found {run_args.get('task')!r}")

    model = HpmLiteV2Model(HpmLiteV2Config(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seed = int(run_args.get("seed", 0))
    data_seed = run_args.get("data_seed", None)
    if data_seed is None:
        data_seed = seed
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=int(run_args.get("seq_len", 512)),
            window=int(run_args.get("window", 64)),
            task="kv",
            num_facts=int(run_args.get("num_facts", 4)),
            num_hard_negatives=int(run_args.get("num_hard_negatives", 0)),
            seed=int(data_seed) + 100_000,
            oracle_memory=bool(run_args.get("oracle_memory", True)),
            repeated_keys=bool(run_args.get("repeated_keys", False)),
            similar_values=bool(run_args.get("similar_values", False)),
            distractor_fact_spans=int(run_args.get("distractor_fact_spans", 0)),
            query_key_noise_only=bool(run_args.get("query_key_noise_only", False)),
            fact_order=str(run_args.get("fact_order", "random")),
            writer_required_facts=int(run_args.get("writer_required_facts", 2)),
            writer_role_mode=str(run_args.get("writer_role_mode", "visible")),
        )
    )

    eval_batches = int(cli.eval_batches if cli.eval_batches is not None else run_args.get("eval_batches", 10))
    batch_size = int(cli.batch_size if cli.batch_size is not None else run_args.get("batch_size", 32))
    if eval_batches < 1 or batch_size < 1:
        raise SystemExit("eval-batches and batch-size must be positive")

    baseline_correct = 0
    ablated_correct = 0
    baseline_ce_sum = 0.0
    ablated_ce_sum = 0.0
    total_examples = 0
    harmed = 0
    helped = 0
    unchanged = 0
    fast_weight_prob_sum = 0.0

    contribution_mask = torch.ones(4, device=device)
    contribution_mask[FAST_WEIGHT_INDEX] = 0.0

    with torch.no_grad():
        for _ in range(eval_batches):
            batch = dataset.sample_batch(batch_size, device=device)
            batch, _ = apply_write_mode(batch, str(run_args.get("write_mode", "oracle")))

            baseline = model_forward(model, batch, run_args)
            weights = baseline["retrieval"].get("router_weights")
            if weights is None:
                raise RuntimeError("checkpoint model did not expose router_weights")

            # Sanity check intervention semantics: holding the baseline weights fixed
            # with an all-ones contribution mask must reproduce the ordinary logits.
            fixed_control = model_forward(
                model,
                batch,
                run_args,
                fixed_weights=weights,
                contribution_mask=torch.ones(4, device=device),
            )
            if not torch.allclose(baseline["logits"], fixed_control["logits"], atol=1e-6, rtol=1e-5):
                raise RuntimeError("fixed-router all-path control does not reproduce baseline logits")

            ablated = model_forward(
                model,
                batch,
                run_args,
                fixed_weights=weights,
                contribution_mask=contribution_mask,
            )

            bmask = answer_span_correct_mask(baseline["logits"], batch["target_ids"], batch["loss_mask"])
            amask = answer_span_correct_mask(ablated["logits"], batch["target_ids"], batch["loss_mask"])
            baseline_correct += int(bmask.sum().item())
            ablated_correct += int(amask.sum().item())
            harmed += int((bmask & ~amask).sum().item())
            helped += int((~bmask & amask).sum().item())
            unchanged += int((bmask == amask).sum().item())
            baseline_ce_sum += float(answer_cross_entropy(baseline["logits"], batch["target_ids"], batch["loss_mask"]).item()) * batch_size
            ablated_ce_sum += float(answer_cross_entropy(ablated["logits"], batch["target_ids"], batch["loss_mask"]).item()) * batch_size
            fast_weight_prob_sum += float(weights[..., FAST_WEIGHT_INDEX].mean().item()) * batch_size
            total_examples += batch_size

    baseline_exact = baseline_correct / total_examples
    ablated_exact = ablated_correct / total_examples
    result = {
        "checkpoint": str(cli.checkpoint),
        "abort_step": ckpt.get("abort_step"),
        "seed": seed,
        "eval_data_seed": int(data_seed) + 100_000,
        "eval_batches": eval_batches,
        "batch_size": batch_size,
        "examples": total_examples,
        "intervention": {
            "router_weights": "held fixed from unablated forward per batch/token",
            "fast_weight_contribution": 0.0,
            "renormalized": False,
        },
        "mean_fast_weight_router_probability": fast_weight_prob_sum / total_examples,
        "baseline_answer_exact": baseline_exact,
        "fast_weight_ablated_answer_exact": ablated_exact,
        "exact_delta_baseline_minus_ablated": baseline_exact - ablated_exact,
        "baseline_answer_ce": baseline_ce_sum / total_examples,
        "fast_weight_ablated_answer_ce": ablated_ce_sum / total_examples,
        "ce_delta_ablated_minus_baseline": (ablated_ce_sum - baseline_ce_sum) / total_examples,
        "paired_examples": {
            "baseline_correct_ablated_wrong": harmed,
            "baseline_wrong_ablated_correct": helped,
            "same_correctness": unchanged,
        },
    }

    output = cli.output or cli.checkpoint.with_name("fast_weight_ablation_at_abort.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
