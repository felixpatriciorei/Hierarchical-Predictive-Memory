from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch

from .data import CONDITIONAL_TASKS, FactRecallConfig, FactRecallDataset, NO_VALUE, VOCAB_SIZE
from .metrics import (
    answer_cross_entropy,
    answer_span_correct_mask,
    answer_span_exact_accuracy,
    count_parameters,
    retrieval_correct_mask,
    retrieval_metrics,
)
from .model import HpmLiteConfig, HpmLiteModel
from .utils import resolve_device, set_seed, str_to_bool
from .write_modes import apply_write_mode, batch_from_memory_selection, writer_metrics
from .writer_objectives import topc_required_margin_diagnostics


@torch.no_grad()
def evaluate_batches(
    model: HpmLiteModel,
    dataset: FactRecallDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    task: str,
    top_k: int,
    memory_control: str = "normal",
    write_mode: str = "oracle",
    use_learned_writer: bool = False,
) -> Dict[str, float]:
    model.eval()
    start_time = time.perf_counter()
    total_examples = 0
    ce_sum = 0.0
    acc_sum = 0.0
    ret_top1_sum = 0.0
    ret_topk_sum = 0.0
    ret_margin_sum = 0.0
    ret_count = 0
    saw_retrieval_output = False
    reasoning_sum = 0.0
    reasoning_count = 0
    cond_positive_correct = 0.0
    cond_positive_total = 0
    cond_negative_correct = 0.0
    cond_negative_total = 0
    cond_no_value_predictions = 0
    cond_target_no_value = 0
    cond_total = 0
    writer_sums = {
        "avg_written_slots": 0.0,
        "true_fact_written_rate": 0.0,
        "false_write_rate": 0.0,
        "missed_fact_rate": 0.0,
    }
    writer_count = 0
    writer_margin_sums = {
        "mean_required_topc_margin": 0.0,
        "all_required_topc_rate": 0.0,
    }
    writer_margin_count = 0
    # HPM-v2 router mixes {local, selective-recurrent, fast-weight, episodic}
    # (see HpmV2PathRouter call order in hpm_v2_model.py). Only present when
    # evaluating an hpm_lite_v2 model; guarded with .get() everywhere so this
    # is a no-op for the v1 model.
    router_weight_names = ("local", "recurrent", "fast_weight", "episodic")
    router_weight_sum = None
    router_weight_count = 0
    # Mean |logit| pre-softmax, tracked separately from router_weight_* above.
    # Weights alone can't tell a router collapsed onto one-hot outputs because
    # it made a confident, earned decision apart from one whose logits simply
    # grew large enough to saturate softmax regardless of path quality; the
    # logits' own scale is what makes that distinguishable (see hpm_v2.py's
    # HpmV2PathRouter.forward).
    router_logit_abs_sum = 0.0
    router_logit_abs_count = 0

    for _ in range(batches):
        batch = dataset.sample_batch(batch_size, device=device)
        batch, write_stats = apply_write_mode(batch, write_mode)
        output = model(
            batch["input_ids"],
            memory_token_positions=batch["memory_token_positions"],
            memory_mask=batch["memory_mask"],
            answer_positions=batch["answer_positions"],
            query_key_positions=batch["query_key_positions"],
            top_k=top_k,
            task=task,
            hop_positive_memory_indices=batch["hop_positive_memory_indices"],
            positive_memory_indices=batch["positive_memory_indices"],
            positive_memory_mask=batch.get("positive_memory_mask"),
            memory_control=memory_control,
            use_learned_writer=use_learned_writer,
            learned_writer_teacher_forcing=False,
        )
        if output.get("retrieval") and "top_indices" in output["retrieval"]:
            saw_retrieval_output = True
        metric_batch = batch
        if use_learned_writer and "writer_memory_token_positions" in output["retrieval"]:
            metric_batch = batch_from_memory_selection(
                batch,
                output["retrieval"]["writer_memory_token_positions"],
                output["retrieval"]["writer_memory_mask"],
            )
            write_stats = writer_metrics(batch, metric_batch)
        if use_learned_writer and {
            "writer_selection_logits",
            "writer_labels",
            "writer_valid_mask",
            "writer_memory_mask",
        }.issubset(output["retrieval"]):
            margin_info = topc_required_margin_diagnostics(
                output["retrieval"]["writer_selection_logits"],
                output["retrieval"]["writer_labels"],
                output["retrieval"]["writer_valid_mask"],
                output["retrieval"]["writer_memory_mask"],
            )
            margin_samples = int(margin_info["samples"])
            if margin_samples:
                for key in writer_margin_sums:
                    writer_margin_sums[key] += margin_info[key] * margin_samples
                writer_margin_count += margin_samples
        logits = output["logits"]
        ce = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        acc = answer_span_exact_accuracy(logits, batch["target_ids"], batch["loss_mask"])
        exact_mask = answer_span_correct_mask(logits, batch["target_ids"], batch["loss_mask"])
        if task in CONDITIONAL_TASKS:
            batch_index = torch.arange(logits.size(0), device=logits.device)
            predictions = logits[batch_index, batch["answer_positions"]].argmax(dim=-1)
            targets = batch["answer_tokens"]
            positive = targets != NO_VALUE
            negative = targets == NO_VALUE
            if positive.any():
                cond_positive_correct += exact_mask[positive].float().sum().item()
                cond_positive_total += int(positive.sum().item())
            if negative.any():
                cond_negative_correct += exact_mask[negative].float().sum().item()
                cond_negative_total += int(negative.sum().item())
            cond_no_value_predictions += int((predictions == NO_VALUE).sum().item())
            cond_target_no_value += int(negative.sum().item())
            cond_total += int(targets.numel())

        total_examples += batch_size
        ce_sum += ce.item() * batch_size
        acc_sum += acc.item() * batch_size
        for key in writer_sums:
            writer_sums[key] += write_stats[key] * batch_size
        writer_count += batch_size

        router_weights = None
        if isinstance(output.get("retrieval"), dict):
            router_weights = output["retrieval"].get("router_weights")
        if router_weights is not None:
            # router_weights: [batch, seq_len, num_paths] -> mean over batch/time.
            batch_mean = router_weights.detach().mean(dim=(0, 1))
            if router_weight_sum is None:
                router_weight_sum = batch_mean.new_zeros(batch_mean.shape)
            router_weight_sum += batch_mean * batch_size
            router_weight_count += batch_size

        router_logits = None
        if isinstance(output.get("retrieval"), dict):
            router_logits = output["retrieval"].get("router_logits")
        if router_logits is not None:
            router_logit_abs_sum += router_logits.detach().abs().mean().item() * batch_size
            router_logit_abs_count += batch_size

        # See the matching comment in train.py: output["retrieval"]["top_indices"]
        # lives in the full-candidate space (not metric_batch's oracle/writer-slot
        # space) whenever the learned writer's full-candidate path is engaged --
        # which is always true here, since eval always runs with
        # learned_writer_teacher_forcing=False. Use the active-space matched
        # labels when the model provides them; falls back to metric_batch's own
        # fields for non-"learned" write modes, where no mismatch exists.
        retrieval_positive_indices = output["retrieval"].get(
            "active_positive_memory_indices", metric_batch["positive_memory_indices"]
        )
        retrieval_positive_mask = output["retrieval"].get(
            "active_positive_memory_mask", metric_batch.get("positive_memory_mask")
        )
        ret = retrieval_metrics(
            output["retrieval"],
            positive_indices=retrieval_positive_indices,
            positive_mask=retrieval_positive_mask,
        )
        if ret:
            ret_top1_sum += ret["retrieval_top1"] * batch_size
            ret_topk_sum += ret["retrieval_topk"] * batch_size
            ret_margin_sum += ret.get("retrieval_margin", 0.0) * batch_size
            ret_count += batch_size
            correct_retrieval = retrieval_correct_mask(
                output["retrieval"],
                positive_indices=retrieval_positive_indices,
                positive_mask=retrieval_positive_mask,
            )
            if correct_retrieval is not None and correct_retrieval.any():
                reasoning_sum += exact_mask[correct_retrieval].float().sum().item()
                reasoning_count += int(correct_retrieval.sum().item())

    metrics = {
        "answer_ce": ce_sum / max(total_examples, 1),
        "answer_exact": acc_sum / max(total_examples, 1),
        "examples": float(total_examples),
        "examples_per_sec": total_examples / max(time.perf_counter() - start_time, 1.0e-9),
    }
    if ret_count:
        metrics.update(
            {
                "retrieval_top1": ret_top1_sum / ret_count,
                "retrieval_topk": ret_topk_sum / ret_count,
                "retrieval_margin": ret_margin_sum / ret_count,
            }
        )
    elif saw_retrieval_output:
        # Learned writers can temporarily write no valid target facts during early
        # training. The retrieval module still ran, so report retrieval as 0
        # instead of dropping the schema key entirely.
        metrics.update({"retrieval_top1": 0.0, "retrieval_topk": 0.0, "retrieval_margin": 0.0})
    if reasoning_count:
        metrics["reasoning_success_given_retrieval"] = reasoning_sum / reasoning_count
    if writer_count:
        metrics.update({key: value / writer_count for key, value in writer_sums.items()})
    # Keep the schema present even for a deliberately non-selective or
    # infeasible setup (for example, a null slot leaves fewer active slots
    # than required labels). A zero margin alone is not interpretable there;
    # ``writer_topc_margin_samples == 0`` explicitly marks it inapplicable.
    metrics.update(
        {
            "writer_required_topc_margin": (
                writer_margin_sums["mean_required_topc_margin"] / writer_margin_count
                if writer_margin_count
                else 0.0
            ),
            "writer_all_required_topc_rate": (
                writer_margin_sums["all_required_topc_rate"] / writer_margin_count
                if writer_margin_count
                else 0.0
            ),
            "writer_topc_margin_samples": float(writer_margin_count),
        }
    )
    if router_weight_count and router_weight_sum is not None:
        avg = (router_weight_sum / router_weight_count).tolist()
        for name, value in zip(router_weight_names, avg):
            metrics[f"router_weight_{name}"] = float(value)
    if router_logit_abs_count:
        metrics["router_logit_abs_mean"] = router_logit_abs_sum / router_logit_abs_count
    if task in CONDITIONAL_TASKS and cond_total:
        metrics.update(
            {
                "positive_condition_accuracy": cond_positive_correct / max(cond_positive_total, 1),
                "negative_condition_accuracy": cond_negative_correct / max(cond_negative_total, 1),
                "memory_required_accuracy": cond_positive_correct / max(cond_positive_total, 1),
                "condition_binding_exact": cond_positive_correct / max(cond_positive_total, 1),
                "no_value_bias_rate": cond_no_value_predictions / cond_total,
                "target_no_value_rate": cond_target_no_value / cond_total,
                "target_value_rate": (cond_total - cond_target_no_value) / cond_total,
                "predicted_value_rate": (cond_total - cond_no_value_predictions) / cond_total,
            }
        )
    return metrics


def parse_seq_lens(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a legacy HPM baseline checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model", choices=["local", "recurrent", "epmem", "hpm_lite", "hebbian"], default="epmem")
    parser.add_argument(
        "--task",
        choices=[
            "kv",
            "twohop",
            "coexisting",
            "conditional",
            "conditional_balanced",
            "conditional_positive_only",
            "conditional_contrastive",
            "longhop",
        ],
        default="kv",
    )
    parser.add_argument("--seq-lens", type=str, default="256,512,1024")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--memory-null-slot", type=str_to_bool, default=False)
    parser.add_argument("--null-score-init", type=float, default=0.0)
    parser.add_argument("--oracle-memory", type=str_to_bool, default=True)
    parser.add_argument(
        "--memory-control",
        choices=["normal", "shuffle_values", "shuffled_values", "random_keys", "corrupt_values", "no_retrieval"],
        default="normal",
    )
    parser.add_argument("--write-mode", choices=["oracle", "fact_token", "random_write", "learned"], default="oracle")
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--repeated-keys", type=str_to_bool, default=False)
    parser.add_argument("--similar-values", type=str_to_bool, default=False)
    parser.add_argument("--distractor-fact-spans", type=int, default=0)
    parser.add_argument("--query-key-noise-only", type=str_to_bool, default=False)
    parser.add_argument("--fact-order", choices=["random", "query_last"], default="random")
    return parser


def config_from_args(args: argparse.Namespace) -> HpmLiteConfig:
    return HpmLiteConfig(
        model_type=args.model,
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        window=args.window,
        max_seq_len=max(2048, max(parse_seq_lens(getattr(args, "seq_lens", "1024")))),
        use_null_slot=args.memory_null_slot,
        null_score_init=args.null_score_init,
        use_learned_writer=args.write_mode == "learned",
    )


def load_model(args: argparse.Namespace, device: torch.device) -> HpmLiteModel:
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        saved_config = checkpoint.get("model_config", {})
        config = HpmLiteConfig(**saved_config)
        model = HpmLiteModel(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return model

    model = HpmLiteModel(config_from_args(args)).to(device)
    return model


def main(argv: Iterable[str] | None = None) -> Dict[str, Dict[str, float]]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model = load_model(args, device)

    results: Dict[str, Dict[str, float]] = {}
    for seq_len in parse_seq_lens(args.seq_lens):
        dataset = FactRecallDataset(
            FactRecallConfig(
                seq_len=seq_len,
                window=args.window,
                task=args.task,
                num_facts=args.num_facts,
                seed=args.seed + seq_len,
                oracle_memory=args.oracle_memory,
                repeated_keys=args.repeated_keys,
                similar_values=args.similar_values,
                distractor_fact_spans=args.distractor_fact_spans,
                query_key_noise_only=args.query_key_noise_only,
                fact_order=args.fact_order,
            )
        )
        metrics = evaluate_batches(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            batches=args.eval_batches,
            device=device,
            task=args.task,
            top_k=args.top_k,
            memory_control=args.memory_control,
            write_mode=args.write_mode,
        )
        metrics["parameters"] = float(count_parameters(model))
        results[str(seq_len)] = metrics

    print(json.dumps(results, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    main()
