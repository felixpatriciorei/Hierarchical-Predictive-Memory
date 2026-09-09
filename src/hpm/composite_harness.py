"""Adapter + runnable harness for CompositeSkillDataset (composite_skill_data.py).

Why this file exists
---------------------
``CompositeSkillDataset.sample_batch`` returns ``input_ids``/``target_ids``
plus four independent ``{subtask}_loss_mask`` / ``{subtask}_answer_position``
/ ``{subtask}_answer_token`` triples. That's enough to *score* four answers.
It is NOT enough to *run* through ``HpmLiteModel``/``HpmLiteV2Model`` as-is:
both require ``memory_token_positions``/``memory_mask`` (where the episodic
path's key/value slots live) and ``query_key_positions`` (which token's
embedding to use as the retrieval query), none of which
``composite_skill_data.py`` produces. See the module docstring discussion in
the chat this file came out of for the full reasoning; short version below.

What this module adds
----------------------
``augment_composite_batch`` derives the missing fields directly from
``CompositeSkillConfig`` (the FACT block's layout is fixed given the config,
so ``memory_token_positions`` is the same for every sample -- no need to
re-derive it per-sample) plus the batch's own already-computed
``fact_answer_position`` values (the identifying token -- COPY_MARKER /
MOOD_QUERY_TOK / the real key / XREF_QUERY_TOK -- always sits exactly one
position before its subtask's ANSWER-marker position, by construction in
``composite_skill_data.py``'s ``_build_tail``).

One forward pass, not four
---------------------------
For ``hpm_lite_v2``: ``HpmLiteV2Model.forward`` computes a *single* episodic
retrieval per sequence and broadcasts it to every position
(``episodic_vector[:, None, :].expand_as(local_state)``), then lets
``HpmV2PathRouter`` mix local/recurrent/fast-weight/episodic *per token*
(``nn.Linear(4*d_model, 4)`` applied at every position, not pooled). That
means logits over the WHOLE sequence come out of one forward call, and all
four sub-tasks' answers can be read off that one tensor at their own
``answer_position`` -- there is no architectural need to call the model once
per sub-task. We point the single ``query_key_positions`` at the FACT
sub-task's key token, since that's the one retrieval XREF also needs
(getting XREF right without a real fact lookup is only possible by chance --
that's the point of that sub-task).

For the older single-retrieval ``hpm_lite`` model type (``model.py``,
``model_type="hpm_lite"``): that architecture only injects the retrieved
vector at ONE ``answer_positions`` index (``episodic_state[batch,
answer_positions] = retrieved``), so it genuinely cannot answer four
sub-tasks in one call. ``forward_composite_all_subtasks`` below falls back to
four separate forward calls for that model type only. ``local``/``recurrent``
model types need no memory args at all and also go through the four-call
path for simplicity (they ignore the extra calls' cost is negligible at
these model sizes).

Verified against the actual model code in this sandbox
--------------------------------------------------------
Ran a real forward + backward pass (see ``__main__`` self-test below and the
short training run this was validated with) confirming: shapes line up,
gradients flow, per-subtask exact-match reads the correct answer token, and
the derived ``memory_token_positions``/``query_key_positions`` actually find
the planted fact (verified by comparing retrieved-slot index against the
independently-recovered positive slot).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .composite_skill_data import (
    COMPOSITE_VOCAB_SIZE,
    SUBTASKS,
    CompositeSkillConfig,
    CompositeSkillDataset,
)
from .hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model
from .metrics import answer_cross_entropy, answer_exact_accuracy, count_parameters, router_health_metrics
from .router_diagnostics import (
    ROUTER_PATH_NAMES,
    all_removed_path_sets,
    removal_key,
    shapley_path_attribution,
)
from .model import HpmLiteConfig, HpmLiteModel
from .train import TinyAdamW, append_csv_row
from .utils import ensure_dir, resolve_device, set_seed, timestamp

SINGLE_PASS_MODEL_TYPES = {"hpm_lite_v2"}


def fact_memory_layout(config: CompositeSkillConfig) -> torch.Tensor:
    """``[num_facts, 2]`` (key_pos, value_pos) token positions of the FACT
    block. Deterministic given the config -- mood block and fact block
    lengths never vary per-sample -- so this is computed once, not per
    sample. Matches ``data.py``'s ``[start + 1, start + 2]`` convention
    (``FACT, key, value, SEP`` -> key at start+1, value at start+2)."""
    mood_end = 1 + config.mood_block_len  # BOS + mood block
    positions = []
    for i in range(config.num_facts):
        start = mood_end + i * 4
        positions.append([start + 1, start + 2])
    return torch.tensor(positions, dtype=torch.long)


def augment_composite_batch(
    batch: Dict[str, torch.Tensor], config: CompositeSkillConfig
) -> Dict[str, torch.Tensor]:
    """Add memory/query fields the model needs but the raw dataset doesn't
    produce. Pure function of ``batch`` + ``config`` -- doesn't mutate
    ``batch``."""
    device = batch["input_ids"].device
    bsz = batch["input_ids"].size(0)

    memory_token_positions = fact_memory_layout(config).to(device)
    memory_token_positions = memory_token_positions[None, :, :].expand(bsz, -1, -1).contiguous()
    memory_mask = torch.ones(bsz, config.num_facts, dtype=torch.bool, device=device)

    # Recover which slot is positive by matching the FACT block's key tokens
    # against fact_query_key -- composite_skill_data.py computes this
    # internally but doesn't export the slot index, only the key VALUE.
    key_positions = memory_token_positions[:, :, 0]  # [B, num_facts]
    keys_at_slots = torch.gather(batch["input_ids"], 1, key_positions)  # [B, num_facts]
    positive_mask = keys_at_slots == batch["fact_query_key"][:, None]
    if not bool(positive_mask.any(dim=1).all()):
        raise AssertionError("fact_query_key not found among FACT block keys -- layout assumption broken")
    positive_index = positive_mask.float().argmax(dim=1)  # [B]

    hop_positive = torch.stack([positive_index, torch.full_like(positive_index, -100)], dim=1)

    # The identifying token for each subtask (COPY_MARKER / MOOD_QUERY_TOK /
    # the real key / XREF_QUERY_TOK) sits exactly one position before that
    # subtask's ANSWER-marker position -- see composite_skill_data.py's
    # _build_tail: [QUERY, <identifier>, ANSWER, <answer_token>, SEP/EOS],
    # and out[f"{name}_answer_position"] already points at the ANSWER-marker
    # token (next-token-prediction convention, same as data.py).
    query_key_positions = {name: batch[f"{name}_answer_position"] - 1 for name in SUBTASKS}

    out = dict(batch)
    out["memory_token_positions"] = memory_token_positions
    out["memory_mask"] = memory_mask
    out["positive_memory_indices"] = positive_index
    out["positive_memory_mask"] = positive_mask
    out["hop_positive_memory_indices"] = hop_positive
    for name in SUBTASKS:
        out[f"{name}_query_key_positions"] = query_key_positions[name]
    # Single retrieval query for the one-forward-pass models: anchor it on
    # FACT's key, since XREF needs that exact same lookup and COPY/MOOD are
    # meant to be solved without episodic help regardless of where the
    # query points (the router has to learn to ignore it for them).
    out["query_key_positions"] = query_key_positions["fact"]
    return out


def build_model(
    model_type: str,
    d_model: int,
    layers: int,
    heads: int,
    window: int,
    seq_len: int,
    device: torch.device,
    *,
    router_logit_clamp: float | None = None,
) -> torch.nn.Module:
    if model_type == "hpm_lite_v2":
        config = HpmLiteV2Config(
            model_type="hpm_lite_v2",
            vocab_size=COMPOSITE_VOCAB_SIZE,
            d_model=d_model,
            layers=layers,
            heads=heads,
            window=window,
            max_seq_len=max(2048, seq_len + 1),
            block_size=window,
            router_logit_clamp=router_logit_clamp,
        )
        return HpmLiteV2Model(config).to(device)
    config = HpmLiteConfig(
        model_type=model_type,
        vocab_size=COMPOSITE_VOCAB_SIZE,
        d_model=d_model,
        layers=layers,
        heads=heads,
        window=window,
        max_seq_len=max(2048, seq_len + 1),
    )
    return HpmLiteModel(config).to(device)


def forward_composite(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    model_type: str,
    top_k: int = 1,
    router_fixed_weights: torch.Tensor | None = None,
    router_contribution_mask: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """Single forward pass covering all four sub-tasks at once. Only valid
    for SINGLE_PASS_MODEL_TYPES (currently hpm_lite_v2) -- see module
    docstring for why. Raises for anything else so a caller can't silently
    get a wrong (single-answer-position) result from the older model type."""
    if model_type not in SINGLE_PASS_MODEL_TYPES:
        raise ValueError(
            f"forward_composite only supports {SINGLE_PASS_MODEL_TYPES}; "
            f"model_type={model_type!r} only injects retrieval at one position "
            "and cannot answer four sub-tasks in a single call."
        )
    return model(
        batch["input_ids"],
        memory_token_positions=batch["memory_token_positions"],
        memory_mask=batch["memory_mask"],
        answer_positions=batch["fact_answer_position"],  # unused by v2 (del'd), kept for signature parity
        query_key_positions=batch["query_key_positions"],
        top_k=top_k,
        task="kv",
        hop_positive_memory_indices=batch["hop_positive_memory_indices"],
        positive_memory_indices=batch["positive_memory_indices"],
        positive_memory_mask=batch["positive_memory_mask"],
        memory_control="normal",
        use_learned_writer=False,
        router_fixed_weights=router_fixed_weights,
        router_contribution_mask=router_contribution_mask,
    )


def per_subtask_exact(logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {
        name: float(
            answer_exact_accuracy(logits, batch[f"{name}_answer_position"], batch[f"{name}_answer_token"]).item()
        )
        for name in SUBTASKS
    }


def _gather_at_positions(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """tensor: [B, T, P] (router weights or logits, per token). positions:
    [B] (this subtask's own answer-position index per example, already
    available as batch[f"{name}_answer_position"]). Returns [B, P]: each
    example's router state at exactly the token where that example's
    answer is scored, not averaged across the whole sequence."""
    idx = positions[:, None, None].expand(-1, 1, tensor.size(-1))
    return torch.gather(tensor, 1, idx).squeeze(1)


def contribution_mask_for_removed_paths(
    removed_paths: tuple[str, ...], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return the post-gate, non-renormalized intervention mask for one arm."""

    return torch.tensor(
        [0.0 if path in removed_paths else 1.0 for path in ROUTER_PATH_NAMES],
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def evaluate_fixed_router_dependencies(
    *,
    model: torch.nn.Module,
    dataset: CompositeSkillDataset,
    config: CompositeSkillConfig,
    model_type: str,
    top_k: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate every conditional path ablation without router self-repair.

    Each batch first gets an ordinary full forward pass.  Its exact per-token
    weights are then held fixed while every non-empty subset of the four path
    *contributions* is set to zero after the gate.  The gate does not see an
    ablated path, is not retrained, and is not renormalized.  This makes the
    result a functional contribution diagnostic rather than an observational
    router-weight plot.
    """

    if batches < 1:
        raise ValueError("router dependence evaluation requires at least one batch")
    removed_sets = all_removed_path_sets(ROUTER_PATH_NAMES)
    score_sums = {
        removed: {subtask: 0.0 for subtask in SUBTASKS}
        for removed in removed_sets
    }
    weight_sums = {subtask: torch.zeros(len(ROUTER_PATH_NAMES), device=device) for subtask in SUBTASKS}

    model.eval()
    for _ in range(batches):
        raw_batch = dataset.sample_batch(batch_size, device=device)
        batch = augment_composite_batch(raw_batch, config)
        full_output = forward_composite(model, batch, model_type, top_k=top_k)
        full_weights = full_output.get("retrieval", {}).get("router_weights")
        if full_weights is None:
            raise RuntimeError("router dependence evaluation requires a model with a live path router")
        for subtask, value in per_subtask_exact(full_output["logits"], batch).items():
            score_sums[()][subtask] += value
        for subtask in SUBTASKS:
            at_answer = _gather_at_positions(full_weights, batch[f"{subtask}_answer_position"])
            weight_sums[subtask] += at_answer.mean(dim=0)

        for removed in removed_sets[1:]:
            output = forward_composite(
                model,
                batch,
                model_type,
                top_k=top_k,
                router_fixed_weights=full_weights,
                router_contribution_mask=contribution_mask_for_removed_paths(
                    removed, device=device, dtype=full_weights.dtype
                ),
            )
            for subtask, value in per_subtask_exact(output["logits"], batch).items():
                score_sums[removed][subtask] += value

    condition_exact = {
        removal_key(removed): {
            subtask: score_sums[removed][subtask] / batches for subtask in SUBTASKS
        }
        for removed in removed_sets
    }
    singleton_deltas = {
        subtask: {
            path: condition_exact["full"][subtask] - condition_exact[removal_key((path,))][subtask]
            for path in ROUTER_PATH_NAMES
        }
        for subtask in SUBTASKS
    }
    shapley = {
        subtask: shapley_path_attribution(
            {removed: condition_exact[removal_key(removed)][subtask] for removed in removed_sets}
        )
        for subtask in SUBTASKS
    }
    answer_weights = {
        subtask: {
            path: float(value)
            for path, value in zip(ROUTER_PATH_NAMES, (weight_sums[subtask] / batches).tolist())
        }
        for subtask in SUBTASKS
    }
    return {
        "diagnostic_kind": "fixed_router_post_gate_conditional_coablation",
        "interpretation": (
            "Router weights were captured from each unablated evaluation batch, then reused exactly "
            "while selected post-gate path contributions were zeroed without renormalization. "
            "Ablation effects are functional evidence under the frozen gate, not proof of independent paths."
        ),
        "batches": batches,
        "batch_size": batch_size,
        "path_names": list(ROUTER_PATH_NAMES),
        "answer_position_router_weights": answer_weights,
        "condition_exact": condition_exact,
        "singleton_exact_drop": singleton_deltas,
        "shapley_exact_attribution": shapley,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    config = CompositeSkillConfig(
        seq_len=args.seq_len,
        window=args.window,
        seed=args.seed,
        copy_span=args.copy_span,
        mood_block_len=args.mood_block_len,
        mood_label_noise=args.mood_label_noise,
        num_facts=args.num_facts,
    )
    eval_config = CompositeSkillConfig(**{**asdict(config), "seed": args.seed + 100_000})
    train_dataset = CompositeSkillDataset(config)
    eval_dataset = CompositeSkillDataset(eval_config)

    model = build_model(
        args.model,
        args.d_model,
        args.layers,
        args.heads,
        args.window,
        args.seq_len,
        device,
        router_logit_clamp=getattr(args, "router_logit_clamp", None),
    )
    optimizer = TinyAdamW(model.parameters(), lr=args.lr)

    run_dir = ensure_dir(Path(args.out_dir) / f"{timestamp()}_composite_{args.model}_seed{args.seed}")
    step_log_path = run_dir / "composite_step_log.csv"
    # Same names/order as evaluate.py's router_weight_* convention (router
    # mixes local/selective-recurrent/fast-weight/episodic, in that order --
    # see HpmV2PathRouter.forward's call signature in hpm_v2_model.py).
    columns = (
        ["step", "model", "seed", "train_loss"]
        + [f"train_{n}_exact" for n in SUBTASKS]
        + [f"eval_{n}_exact" for n in SUBTASKS]
        + [f"router_weight_{p}" for p in ROUTER_PATH_NAMES]  # whole-sequence average (kept for continuity)
        + ["router_logit_abs_mean"]
        + [
            "router_raw_logit_abs_mean",
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
        ]
        + [f"router_weight_{s}_{p}" for s in SUBTASKS for p in ROUTER_PATH_NAMES]  # NEW: at each subtask's own answer position
        + [f"router_logit_abs_mean_{s}" for s in SUBTASKS]
        + [f"router_raw_logit_abs_mean_{s}" for s in SUBTASKS]
        + [f"router_weight_max_mean_{s}" for s in SUBTASKS]
        + [f"router_entropy_mean_{s}" for s in SUBTASKS]
        + [f"router_clamp_jacobian_mean_{s}" for s in SUBTASKS]
        + [f"router_raw_to_weight_jacobian_fro_mean_{s}" for s in SUBTASKS]
    )

    final: Dict[str, Any] = {}
    for step in range(1, args.steps + 1):
        model.train()
        raw_batch = train_dataset.sample_batch(args.batch_size, device=device)
        batch = augment_composite_batch(raw_batch, config)

        output = forward_composite(model, batch, args.model, top_k=args.top_k)
        logits = output["logits"]
        loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            train_scores = per_subtask_exact(logits.detach(), batch)
            model.eval()
            eval_sums = {n: 0.0 for n in SUBTASKS}
            router_weight_sum = None  # running sum, shape [num_paths] -- whole-sequence average
            router_weight_pos_sum = {s: None for s in SUBTASKS}  # running sum per subtask, shape [num_paths]
            router_health_sum: Dict[str, float] = {}
            router_health_pos_sum: Dict[str, Dict[str, float]] = {s: {} for s in SUBTASKS}
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    eval_raw = eval_dataset.sample_batch(args.batch_size, device=device)
                    eval_batch = augment_composite_batch(eval_raw, eval_config)
                    eval_output = forward_composite(model, eval_batch, args.model, top_k=args.top_k)
                    eval_logits = eval_output["logits"]
                    for name, value in per_subtask_exact(eval_logits, eval_batch).items():
                        eval_sums[name] += value

                    # router_weights/router_logits: [batch, seq_len, num_paths]
                    retrieval = eval_output.get("retrieval", {})
                    router_weights = retrieval.get("router_weights")
                    router_logits = retrieval.get("router_logits")
                    router_raw_logits = retrieval.get("router_raw_logits")
                    router_clamp_jacobian = retrieval.get("router_clamp_jacobian")
                    if router_weights is not None:
                        # whole-sequence average -- same reduction evaluate.py
                        # already uses; kept so this run stays comparable to
                        # the previous 8-seed run.
                        batch_mean = router_weights.detach().mean(dim=(0, 1))
                        router_weight_sum = batch_mean if router_weight_sum is None else router_weight_sum + batch_mean
                        # position-conditioned: what is the router actually
                        # doing at the exact token where EACH subtask's
                        # answer is scored, not diluted by the other three
                        # subtasks' regions of the same sequence.
                        for name in SUBTASKS:
                            pos = eval_batch[f"{name}_answer_position"]
                            at_pos = _gather_at_positions(router_weights.detach(), pos).mean(dim=0)  # [num_paths]
                            router_weight_pos_sum[name] = (
                                at_pos if router_weight_pos_sum[name] is None else router_weight_pos_sum[name] + at_pos
                            )

                    if router_weights is not None and router_logits is not None:
                        health = router_health_metrics(
                            router_weights.detach(),
                            router_logits.detach(),
                            router_raw_logits=(router_raw_logits.detach() if router_raw_logits is not None else None),
                            router_clamp_jacobian=(
                                router_clamp_jacobian.detach() if router_clamp_jacobian is not None else None
                            ),
                        )
                        for key, value in health.items():
                            router_health_sum[key] = router_health_sum.get(key, 0.0) + float(value.item())

                        for name in SUBTASKS:
                            pos = eval_batch[f"{name}_answer_position"]
                            pos_weights = _gather_at_positions(router_weights.detach(), pos).unsqueeze(1)
                            pos_logits = _gather_at_positions(router_logits.detach(), pos).unsqueeze(1)
                            pos_raw = (
                                _gather_at_positions(router_raw_logits.detach(), pos).unsqueeze(1)
                                if router_raw_logits is not None
                                else None
                            )
                            pos_clamp_jac = (
                                _gather_at_positions(router_clamp_jacobian.detach(), pos).unsqueeze(1)
                                if router_clamp_jacobian is not None
                                else None
                            )
                            pos_health = router_health_metrics(
                                pos_weights,
                                pos_logits,
                                router_raw_logits=pos_raw,
                                router_clamp_jacobian=pos_clamp_jac,
                            )
                            for key, value in pos_health.items():
                                router_health_pos_sum[name][key] = (
                                    router_health_pos_sum[name].get(key, 0.0) + float(value.item())
                                )
            eval_scores = {n: v / args.eval_batches for n, v in eval_sums.items()}
            router_weight_avg = (
                (router_weight_sum / args.eval_batches).tolist()
                if router_weight_sum is not None
                else [float("nan")] * len(ROUTER_PATH_NAMES)
            )
            router_weight_pos_avg = {
                s: (
                    (router_weight_pos_sum[s] / args.eval_batches).tolist()
                    if router_weight_pos_sum[s] is not None
                    else [float("nan")] * len(ROUTER_PATH_NAMES)
                )
                for s in SUBTASKS
            }

            health_avg = {
                key: value / args.eval_batches
                for key, value in router_health_sum.items()
            }
            health_pos_avg = {
                s: {
                    key: value / args.eval_batches
                    for key, value in router_health_pos_sum[s].items()
                }
                for s in SUBTASKS
            }

            row = {
                "step": step,
                "model": args.model,
                "seed": args.seed,
                "train_loss": float(loss.item()),
                **{f"train_{n}_exact": train_scores[n] for n in SUBTASKS},
                **{f"eval_{n}_exact": eval_scores[n] for n in SUBTASKS},
                **{f"router_weight_{p}": v for p, v in zip(ROUTER_PATH_NAMES, router_weight_avg)},
                "router_logit_abs_mean": health_avg.get("router_logit_abs_mean", float("nan")),
                "router_raw_logit_abs_mean": health_avg.get("router_raw_logit_abs_mean", float("nan")),
                "router_weight_max_mean": health_avg.get("router_weight_max_mean", float("nan")),
                "router_entropy_mean": health_avg.get("router_entropy_mean", float("nan")),
                "router_effective_paths_mean": health_avg.get("router_effective_paths_mean", float("nan")),
                "router_clamp_jacobian_mean": health_avg.get("router_clamp_jacobian_mean", float("nan")),
                "router_clamp_jacobian_min": health_avg.get("router_clamp_jacobian_min", float("nan")),
                "router_clamp_jacobian_lt_0p01_frac": health_avg.get(
                    "router_clamp_jacobian_lt_0p01_frac", float("nan")
                ),
                "router_clamp_jacobian_lt_0p001_frac": health_avg.get(
                    "router_clamp_jacobian_lt_0p001_frac", float("nan")
                ),
                "router_softmax_jacobian_fro_mean": health_avg.get(
                    "router_softmax_jacobian_fro_mean", float("nan")
                ),
                "router_softmax_jacobian_fro_min": health_avg.get(
                    "router_softmax_jacobian_fro_min", float("nan")
                ),
                "router_raw_to_weight_jacobian_fro_mean": health_avg.get(
                    "router_raw_to_weight_jacobian_fro_mean", float("nan")
                ),
                "router_raw_to_weight_jacobian_fro_min": health_avg.get(
                    "router_raw_to_weight_jacobian_fro_min", float("nan")
                ),
                "router_clamp_attenuation_ratio_mean": health_avg.get(
                    "router_clamp_attenuation_ratio_mean", float("nan")
                ),
                "router_clamp_attenuation_ratio_min": health_avg.get(
                    "router_clamp_attenuation_ratio_min", float("nan")
                ),
                **{
                    f"router_weight_{s}_{p}": v
                    for s in SUBTASKS
                    for p, v in zip(ROUTER_PATH_NAMES, router_weight_pos_avg[s])
                },
                **{
                    f"router_logit_abs_mean_{s}": health_pos_avg[s].get("router_logit_abs_mean", float("nan"))
                    for s in SUBTASKS
                },
                **{
                    f"router_raw_logit_abs_mean_{s}": health_pos_avg[s].get(
                        "router_raw_logit_abs_mean", float("nan")
                    )
                    for s in SUBTASKS
                },
                **{
                    f"router_weight_max_mean_{s}": health_pos_avg[s].get("router_weight_max_mean", float("nan"))
                    for s in SUBTASKS
                },
                **{
                    f"router_entropy_mean_{s}": health_pos_avg[s].get("router_entropy_mean", float("nan"))
                    for s in SUBTASKS
                },
                **{
                    f"router_clamp_jacobian_mean_{s}": health_pos_avg[s].get(
                        "router_clamp_jacobian_mean", float("nan")
                    )
                    for s in SUBTASKS
                },
                **{
                    f"router_raw_to_weight_jacobian_fro_mean_{s}": health_pos_avg[s].get(
                        "router_raw_to_weight_jacobian_fro_mean", float("nan")
                    )
                    for s in SUBTASKS
                },
            }
            append_csv_row(step_log_path, columns, row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, sort_keys=True))
            final = row

    final["parameters"] = count_parameters(model)
    final["step_log_path"] = str(step_log_path)
    final["run_dir"] = str(run_dir)
    if getattr(args, "router_dependence_eval", False):
        dependence_config = CompositeSkillConfig(**{**asdict(config), "seed": args.seed + 200_000})
        dependence_dataset = CompositeSkillDataset(dependence_config)
        dependence = evaluate_fixed_router_dependencies(
            model=model,
            dataset=dependence_dataset,
            config=dependence_config,
            model_type=args.model,
            top_k=args.top_k,
            batch_size=args.batch_size,
            batches=getattr(args, "router_dependence_eval_batches", args.eval_batches),
            device=device,
        )
        dependence["seed"] = args.seed
        dependence["model"] = args.model
        dependence["task"] = "composite_four_path"
        dependence_path = run_dir / "router_dependency.json"
        with dependence_path.open("w", encoding="utf-8") as handle:
            json.dump(dependence, handle, indent=2, sort_keys=True)
        final["router_dependency_path"] = str(dependence_path)
        final["router_dependency_diagnostic_kind"] = dependence["diagnostic_kind"]
        print(json.dumps({"router_dependency_path": str(dependence_path)}, sort_keys=True))
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CompositeSkillDataset four-path test.")
    parser.add_argument(
        "--model",
        choices=sorted(SINGLE_PASS_MODEL_TYPES),
        default="hpm_lite_v2",
        help=(
            "Only hpm_lite_v2 is wired up right now -- it's the one model type "
            "that can answer all four sub-tasks in a single forward pass (see "
            "module docstring). local/recurrent/hpm_lite (the older single-"
            "retrieval architecture) would need a four-separate-forward-calls "
            "path that isn't implemented here; add it to forward_composite if "
            "you need those as baselines."
        ),
    )
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--copy-span", type=int, default=8)
    parser.add_argument("--mood-block-len", type=int, default=40)
    parser.add_argument("--mood-label-noise", type=float, default=0.0)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument(
        "--router-dependence-eval",
        action="store_true",
        help=(
            "After training, evaluate all 16 fixed-router post-gate path contribution ablations on a "
            "separate held-out stream and write router_dependency.json. This is a diagnostic only; "
            "it does not train or alter the model."
        ),
    )
    parser.add_argument(
        "--router-dependence-eval-batches",
        type=int,
        default=None,
        help="Held-out batches for --router-dependence-eval. Default: --eval-batches.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument(
        "--router-logit-clamp",
        type=float,
        default=None,
        help=(
            "hpm_lite_v2 only: optional C for effective_logits=C*tanh(raw_logits/C). "
            "Default None preserves the established composite benchmark. When enabled, "
            "the harness logs raw/effective logits, the clamp Jacobian, and the exact "
            "raw-logit-to-softmax local Jacobian norm so forward bounding cannot be "
            "mistaken for optimizer recoverability."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs")
    return parser


def main(argv=None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
