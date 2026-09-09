from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import torch

from .data import FactRecallConfig, FactRecallDataset, VOCAB_SIZE
from .diagnostics import (
    DIAGNOSTIC_CSV_COLUMNS,
    ProbeState,
    candidate_entropy,
    candidate_kl,
    collect_module_grad_norms,
    masked_candidate_distribution,
    topk_jaccard,
)
from .evaluate import evaluate_batches
from .metrics import (
    answer_cross_entropy,
    answer_span_exact_accuracy,
    count_parameters,
    retrieval_metrics,
    router_entropy_loss,
    router_health_metrics,
    router_z_loss,
)
from .model import HpmLiteConfig, HpmLiteModel
from .hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model
from .utils import ensure_dir, resolve_device, set_seed, str_to_bool, timestamp, write_json
from .write_modes import apply_write_mode, batch_from_memory_selection, writer_metrics
from .writer_objectives import topc_required_margin_diagnostics, topc_set_coverage_loss


SUPPORTED_MODEL_IDS = ("local", "recurrent", "epmem", "hpm_lite", "hpm_lite_v2", "hebbian")


def parse_models(value: str) -> list[str]:
    """Parse a comma-separated model list for experiment tooling.

    The historical multi-model launcher was removed during repository cleanup, but the
    validation contract remains useful to external runners and tests.
    """
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise argparse.ArgumentTypeError("model list must include at least one model")
    invalid = [model for model in models if model not in SUPPORTED_MODEL_IDS]
    if invalid:
        valid = ",".join(SUPPORTED_MODEL_IDS)
        raise argparse.ArgumentTypeError(
            f"invalid model(s): {','.join(invalid)}; valid choices: {valid}"
        )
    return list(dict.fromkeys(models))


class TinyAdamW:
    """Small AdamW optimizer to avoid torch._dynamo imports in broken CPU installs."""

    def __init__(
        self,
        parameters,
        lr: float = 3.0e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.01,
    ):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            grad = parameter.grad
            if self.weight_decay:
                parameter.mul_(1.0 - self.lr * self.weight_decay)
            self.m[index].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            self.v[index].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = self.v[index].sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
            step_size = self.lr / bias_correction1
            parameter.addcdiv_(self.m[index], denom, value=-step_size)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal HPM-Lite synthetic recall model.")
    parser.add_argument("--model", choices=SUPPORTED_MODEL_IDS, default="local")
    parser.add_argument(
        "--task",
        choices=[
            "kv",
            "aliased_kv",
            "causal_salience_kv",
            "twohop",
            "coexisting",
            "conditional",
            "conditional_balanced",
            "conditional_positive_only",
            "conditional_contrastive",
            "longhop",
            "noisy_conditional",
        ],
        default="kv",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "Seed for FactRecallDataset (train/eval/probe) ONLY. Default None = use "
            "--seed for data too, byte-identical to all prior behavior/results (--seed "
            "has always doubled as both the model-init/optimizer seed AND the "
            "dataset-sampling seed). Set this to decouple them: keep --seed fixed and "
            "vary --data-seed (or vice versa) to tell apart 'this specific model "
            "initialization is unstable' from 'this specific sampled fact set is hard' "
            "-- with a single shared seed, these two explanations are indistinguishable "
            "from run to run. Added after four consecutive sweeps (flat warmup, "
            "clamp+ramped-warmup, primary-loss-anneal, jepa-bias-clamp) all showed the "
            "exact same condition (jepa_bias_on, seed=0) failing to converge regardless "
            "of which training-dynamics fix was layered on -- consistent with the "
            "instability being tied to what --seed=0 happens to sample or initialize, "
            "not to any of the mechanisms those sweeps targeted. This flag is the direct "
            "way to test that instead of guessing at a fifth fix."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lambda-ret", type=float, default=0.1)
    parser.add_argument("--lambda-writer", type=float, default=0.1)
    parser.add_argument(
        "--lambda-writer-set-coverage",
        type=float,
        default=0.0,
        help=(
            "Additional Top-C set-coverage loss for hpm_lite_v2's hard learned "
            "episodic writer. It directly rewards every causally required candidate "
            "being inside the finite selected set, complementing the independent BCE "
            "writer loss. Default 0.0 means this objective is not computed and all "
            "previous training behavior is unchanged. Requires a finite capacity and "
            "a task with 1 <= required candidates <= capacity < valid candidates."
        ),
    )
    parser.add_argument(
        "--writer-set-coverage-eps",
        type=float,
        default=0.3,
        help="Entropic temperature for the Top-C coverage relaxation. Default 0.3.",
    )
    parser.add_argument(
        "--writer-set-coverage-iters",
        type=int,
        default=48,
        help=(
            "Bisection iterations for --lambda-writer-set-coverage's exact-budget "
            "smooth Top-C relaxation. Default 48; it is independent of the deployed "
            "Sinkhorn straight-through surrogate."
        ),
    )
    parser.add_argument(
        "--writer-set-coverage-mass-tolerance",
        type=float,
        default=1.0e-3,
        help=(
            "Maximum allowed absolute error in the relaxed Top-C membership mass. "
            "The run fails rather than silently training on a non-converged OT plan."
        ),
    )
    parser.add_argument(
        "--episodic-only",
        action="store_true",
        help=(
            "HPM v2 only: drop selective_recurrent, fast_memory, and the path router; "
            "answer directly from the episodic read. See HpmLiteV2Config.episodic_only."
        ),
    )
    parser.add_argument(
        "--lambda-router-entropy",
        type=float,
        default=0.0,
        help=(
            "Weight on the router entropy auxiliary loss (HPM v2 only). Default 0.0 "
            "= off, matches all prior behavior/results exactly. Set > 0 (try 0.01-0.05) to "
            "sharpen the path router's per-token mixing weights instead of letting it hedge "
            "diffuse weight onto paths that don't earn it -- see metrics.router_entropy_loss."
        ),
    )
    parser.add_argument(
        "--lambda-router-z-loss",
        type=float,
        default=0.0,
        help=(
            "Weight on the router z-loss auxiliary loss (hpm_lite_v2 only; ST-MoE style, "
            "logsumexp(router_logits)**2). Default 0.0 = off, matches all prior "
            "behavior/results exactly. Unlike --lambda-router-entropy (which shapes "
            "post-softmax confidence and, at >0, deliberately sharpens toward one-hot), "
            "this penalizes the PRE-softmax logits' own magnitude and is orthogonal to "
            "confidence. Added after a real sweep showed router_logit_abs_mean flat at "
            "~1.2-1.7 through step 200, then jumping 3-7x in one eval window exactly when "
            "the differentiable full-candidate writer path first engaged, coincident with "
            "a loss spike and the router permanently one-hot-locking with no recovery "
            "through step 600. Try 1e-3 (ST-MoE's own default) as a starting point -- see "
            "metrics.router_z_loss for the full mechanism writeup."
        ),
    )
    parser.add_argument(
        "--router-logit-clamp",
        type=float,
        default=None,
        help=(
            "HPM v2 only: bound effective router logits as C*tanh(raw/C) before softmax. "
            "This constrains forward probabilities but can attenuate gradients through tanh. "
            "Default None leaves routing unchanged. Use only as a controlled experiment "
            "with the router-health telemetry enabled."
        ),
    )
    parser.add_argument(
        "--lambda-jepa",
        type=float,
        default=0.0,
        help=(
            "Weight on the JEPA-lite auxiliary loss(es) (HPM v2 only). Default 0.0 = "
            "off, matches all prior behavior/results exactly -- jepa_loss is computed every "
            "forward pass regardless (use_jepa_aux defaults True) but was previously always "
            "discarded. Set > 0 to actually train the block-level predictor/target heads, "
            "and (when --use-jepa-writer-bias is also set) the token-level ones too."
        ),
    )
    parser.add_argument(
        "--use-jepa-writer-bias",
        type=str_to_bool,
        default=False,
        help=(
            "HPM v2 + --write-mode learned only: bias the writer's selection logits "
            "with the JEPA-lite token-level predictive-surprise score, on top of (not "
            "replacing) the oracle-supervised BCE signal. Default False = off, matches all "
            "prior behavior/results exactly. Requires use_jepa_aux (default True)."
        ),
    )
    parser.add_argument(
        "--use-token-jepa-aux",
        type=str_to_bool,
        default=False,
        help=(
            "HPM v2 only: compute the token-level JEPA predictive loss "
            "without feeding its surprise score into the episodic writer. "
            "This is the safe default for developing/evaluating token JEPA: "
            "with --use-jepa-writer-bias=false, the writer's selection and "
            "memory state are unchanged. Default false preserves prior runs."
        ),
    )
    parser.add_argument("--jepa-writer-bias-scale", type=float, default=1.0)
    parser.add_argument(
        "--jepa-bias-clamp",
        type=float,
        default=None,
        help=(
            "HPM v2 + --use-jepa-writer-bias only: bound the JEPA surprise contribution "
            "to writer selection logits with C*tanh(raw/C). Default None leaves the bias "
            "unchanged. This is an experimental containment mechanism, not evidence that "
            "JEPA improves writer selection."
        ),
    )
    parser.add_argument(
        "--delta-rule-writer",
        type=str_to_bool,
        default=False,
        help=(
            "HPM v2 + --write-mode learned only: bypass SupervisedMemoryWriter's "
            "hard torch.topk selection and EpisodicMemory's retrieve_topk gather entirely, "
            "routing every candidate token-position pair through DeltaRuleEpisodicMemory "
            "(hpm.associative_memory) instead -- a continuous delta-rule write gated "
            "by a budget-capped sigmoid(writer_selection_logits), no discrete selection, no straight-through "
            "estimator anywhere in the read path. This is an experimental associative-memory "
            "alternative, not a validated repair of the exact episodic writer; it must be "
            "evaluated under the capacity contract in docs/MATURITY.md. "
            "Because there's a single read "
            "mechanism under this flag, --learned-writer-teacher-forcing-steps and "
            "--scheduled-sampling-anneal-steps stop changing WHAT is read (writer_loss's "
            "oracle BCE supervision is still computed unconditionally either way). Not yet "
            "supported for --task twohop/longhop (raises ValueError rather than silently "
            "answering wrong). See HpmLiteV2Config.use_delta_rule_writer's docstring for the "
            "full list of known limitations (no retrieval_top1/topk/margin diagnostics, "
            "sinkhorn_warmup becomes a no-op, sequential per-candidate write is slow at long "
            "seq_len). Default False = off, matches all prior behavior/results exactly."
        ),
    )
    parser.add_argument(
        "--delta-rule-oracle-blend",
        type=str_to_bool,
        default=False,
        help=(
            "HPM v2 + --delta-rule-writer only: add an oracle-gated curriculum to the "
            "experimental delta-rule writer. DeltaRuleEpisodicMemory's "
            "write_gate becomes `p * oracle_gate + (1 - p) * budgeted_sigmoid(writer_selection_logits)`, "
            "where oracle_gate is a hard {0,1} mask at the true fact position and p is the "
            "same continuous --scheduled-sampling-anneal-steps probability already computed "
            "each step. Pair this with a nonzero "
            "--scheduled-sampling-anneal-steps so p actually decays over training; with "
            "anneal-steps=0, p is a flat 1.0/0.0 step function over the whole run, which "
            "reduces to either the pure-oracle or pure-self-read case throughout instead of "
            "smoothly transitioning between them. Report p=0 evaluation separately: oracle "
            "curriculum performance is not a deployment-time self-writing result. See "
            "HpmLiteV2Config.delta_rule_oracle_blend's docstring for the full mechanism. "
            "Default False = off."
        ),
    )
    parser.add_argument("--learned-writer-teacher-forcing-steps", type=int, default=50)
    parser.add_argument(
        "--scheduled-sampling-anneal-steps",
        type=int,
        default=0,
        help=(
            "HPM v2 + --write-mode learned only: instead of a hard "
            "step<=learned_writer_teacher_forcing_steps switch from the oracle-indexed read "
            "to the differentiable full-candidate Sinkhorn read, anneal a per-step "
            "*probability* of still using the oracle read from 1.0 down to 0.0 over this "
            "many steps immediately after the cutover (classic Bengio et al. 2015 "
            "Scheduled Sampling, applied to the memory-read path instead of the token "
            "path). Default 0 = off, reproduces the exact prior hard-switch behavior "
            "bit-for-bit (p is a step function: 1.0 through cutover, 0.0 after, so the "
            "random draw is deterministic either way). Added after "
            "writer_transition_diagnostics.csv showed the step-250-style collapse is not "
            "a gradual destabilization of any one subsystem -- every grad_norm_* column "
            "sits at noise floor through the last teacher-forced step, then jumps 4-5 "
            "orders of magnitude together in the SAME step the switch flips, because the "
            "self-selected read is exercised with real answer-loss gradient for the first "
            "time ever at that exact step. This anneal gives it graded exposure instead. "
            "Try 100-250. A per-step hard boolean draw was chosen over a continuous "
            "p*oracle + (1-p)*sinkhorn blend deliberately: the blend trains on an "
            "off-manifold interpolated read the model never sees at inference (p=0) or "
            "under pure teacher forcing (p=1), trading the current sharp discontinuity for "
            "a different one. A per-step draw keeps every forward pass a real, coherent "
            "mechanism end-to-end."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling-schedule",
        choices=["cosine", "linear"],
        default="cosine",
        help=(
            "Shape of the oracle-read probability decay over "
            "--scheduled-sampling-anneal-steps. cosine (default) decays slowly at both "
            "ends and fastest through the middle of the window; linear decays at a "
            "constant rate. Only takes effect with --scheduled-sampling-anneal-steps > 0."
        ),
    )
    parser.add_argument(
        "--sinkhorn-warmup-weight",
        type=float,
        default=0.0,
        help=(
            "HPM v2 + --write-mode learned only: weight on an auxiliary answer-CE "
            "loss computed by running the differentiable full-candidate Sinkhorn read "
            "(the writer_full_memory_token_positions/writer_select_bias pair "
            "SupervisedMemoryWriter.forward already computes every single step, teacher "
            "forcing or not) through episodic_memory/router/answer_head IN PARALLEL with "
            "the primary oracle-indexed read, whenever a step IS teacher-forced. Default "
            "0.0 = off, matches all prior behavior/results exactly and adds no extra "
            "compute when disabled. Try 0.1 (matches --lambda-writer's scale).\n"
            "This directly replaces --scheduled-sampling-anneal-steps as the fix for the "
            "tf-release collapse, not a complement to it. The anneal's own real-sweep "
            "data (six seeds, writer_transition_diagnostics.csv from the scheduled- "
            "sampling run) showed WHY it doesn't work: every individual True->False draw "
            "in the anneal reproduced the same 4-5-order-of-magnitude answer_loss shock "
            "as the original hard cutover (18-23 shocks per run, magnitudes 15-150, only "
            "a weak -0.46 correlation between occurrence order and magnitude -- no clean "
            "habituation), AND subsequent teacher-forced steps stopped fully recovering "
            "to the pre-collapse ~0.000 answer_loss floor. The anneal changed WHEN the "
            "Sinkhorn path first got real gradient (spread across many random steps "
            "instead of one fixed step) but never changed THAT it was cold at each of "
            "those first exposures -- calendar frequency of exposure isn't density of "
            "exposure when the intervening steps never touch that path's parameters at "
            "all. This flag targets the actual cold-start directly: give the Sinkhorn "
            "read continuous, small-weight real-task gradient on every teacher-forced "
            "step, so by the time it's asked to carry the full answer task (whether via "
            "a hard cutover or any anneal), it already has task-relevant gradient signal "
            "instead of only auxiliary oracle-matching BCE. Safe to combine with "
            "--scheduled-sampling-anneal-steps > 0 if wanted, but the recommended setup "
            "is warmup weight > 0 with anneal_steps=0 (clean hard cutover) -- the anneal's "
            "own failure mode (repeated shocks, backward contamination of teacher-forced "
            "steps) is avoided entirely if there's no need to sample the collapse "
            "repeatedly in the first place."
        ),
    )
    parser.add_argument(
        "--sinkhorn-warmup-ramp-steps",
        type=int,
        default=0,
        help=(
            "Only takes effect when --sinkhorn-warmup-weight > 0. Default 0 = exact "
            "no-op (flat weight the whole teacher-forcing window, identical to prior "
            "behavior). When > 0, linearly ramps the warmup loss's weight from 0.0 up "
            "to --sinkhorn-warmup-weight over the N steps immediately before "
            "--learned-writer-teacher-forcing-steps, instead of holding it flat from "
            "step 1. Added because the flat-weight sinkhorn-warmup sweep (six seeds, "
            "jepa_bias_{on,off} x 3) showed the warmup loss idling in a 13-32 band for "
            "the whole 140-200 window rather than trending toward the ~0.000 floor the "
            "primary answer_loss reaches over the same window -- 5 of 6 runs had no "
            "statistically real downward trend (|r|<0.28, p>0.07). A flat weight of 0.1 "
            "is apparently sufficient to avoid total cold-start catastrophe but not "
            "sufficient pressure to make the path actually converge before it has to "
            "carry real weight. Ramping the weight up as the cutover approaches asks "
            "little of the path early (stability, matches the flat-weight run's "
            "shock-avoidance result) and more of it right before cutover (pressure to "
            "actually solve the task, not just get exposure to it)."
        ),
    )
    parser.add_argument(
        "--primary-loss-anneal-steps",
        type=int,
        default=0,
        help=(
            "HPM v2 + --write-mode learned only: mirror image of "
            "--sinkhorn-warmup-ramp-steps, applied to the PRIMARY oracle-indexed "
            "answer_loss instead of the warmup loss. Default 0 = exact no-op (the "
            "primary answer_loss keeps its implicit weight of 1.0 for the whole "
            "teacher-forcing window, identical to all prior behavior/results). When "
            "> 0, linearly ramps the primary loss's weight DOWN from 1.0 to "
            "--primary-loss-floor-weight over the N steps immediately before "
            "--learned-writer-teacher-forcing-steps, on teacher-forced steps only "
            "(gated the same way lambda_sinkhorn_warmup already is, on "
            "sinkhorn_warmup_logits being present -- steps where teacher_forcing is "
            "False always keep weight 1.0, since the primary read IS the honest "
            "self-selected read there and needs full weight, not a floor value "
            "computed for a different regime).\n"
            "Added for the reason this whole project's history keeps surfacing under "
            "different names: the router's real path-mixing decision is trained "
            "almost entirely on privileged, oracle-fed episodic content (the primary "
            "read, weight 1.0) for the whole teacher-forcing window, while the "
            "honest self-selected read it will actually have at deployment is capped "
            "at a small residual weight (--sinkhorn-warmup-weight, typically 0.1). "
            "That is the textbook setup the learning-with-privileged-information "
            "literature warns about: a student trained mostly on a teacher signal it "
            "won't have at inference learns a shortcut that collapses the moment "
            "that signal is withdrawn -- which is exactly the cutover cliff this "
            "project has been chasing. The literature's structural fix (asymmetric "
            "actor-critic: let privileged information shape an auxiliary/critic "
            "loss, never let the acting component's dominant training signal "
            "depend on it) is already half-implemented here -- writer_loss already "
            "plays the critic role correctly. This flag is the other half: ramp the "
            "privileged path's dominance DOWN as the sinkhorn-warmup ramp brings the "
            "honest path's weight UP, so by cutover the router has been trained on a "
            "gradually more realistic balance between the two, instead of 1.0 vs 0.1 "
            "for the entire window and then an instant flip to 0.0 vs 1.0 at cutover. "
            "Recommended to try alongside --sinkhorn-warmup-ramp-steps with the same "
            "value, so both ramps span an identical window."
        ),
    )
    parser.add_argument(
        "--primary-loss-floor-weight",
        type=float,
        default=0.1,
        help=(
            "Only takes effect when --primary-loss-anneal-steps > 0. The weight the "
            "primary oracle-indexed answer_loss ramps DOWN to by "
            "--learned-writer-teacher-forcing-steps (from 1.0), on teacher-forced "
            "steps. Default 0.1 matches --sinkhorn-warmup-weight's suggested scale, "
            "so that at cutover the two paths are weighted roughly symmetrically "
            "(0.1 privileged / 0.1 honest) rather than the untouched 1.0 privileged "
            "/ 0.1 honest split. Set to 1.0 to make --primary-loss-anneal-steps > 0 "
            "itself a no-op (ramp with no actual change in weight)."
        ),
    )
    parser.add_argument(
        "--diagnose-writer-transition",
        type=str_to_bool,
        default=False,
        help=(
            "HPM v2 + --write-mode learned only: instrument per-subsystem gradient "
            "norms (local_blocks/selective_recurrent/fast_memory/router/writer/"
            "episodic_memory/jepa) and the writer's candidate-distribution shift "
            "(entropy/KL/top-k Jaccard on a fixed probe batch) at full per-step resolution "
            "in a window around --learned-writer-teacher-forcing-steps. Added after "
            "router_logit_clamp proved the step-250-style collapse survives even an "
            "unconditional bound on the router -- this looks upstream, at the writer/"
            "episodic-memory transition itself, to find which subsystem destabilizes "
            "first. Default False = off, matches all prior behavior/results exactly and "
            "adds no overhead when disabled. Writes "
            "<run_dir>/writer_transition_diagnostics.csv."
        ),
    )
    parser.add_argument(
        "--diagnostic-window",
        type=int,
        default=30,
        help=(
            "Steps before/after --learned-writer-teacher-forcing-steps to instrument at "
            "full per-step resolution when --diagnose-writer-transition is set. Only takes "
            "effect with --diagnose-writer-transition True. Default 30, matching the real "
            "sweep's eval_every=50 granularity being too coarse to see which subsystem "
            "moves first within the step-250 collapse window."
        ),
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--memory-slots",
        type=int,
        default=None,
        help=(
            "Cap on episodic memory capacity (HPM v2 only). Passed through as "
            "HpmLiteV2Config.episodic_capacity. Default None = no hard cap: the learned "
            "writer retains every structurally valid pre-query token pair, not merely the "
            "oracle facts. Set to --num-facts for a capacity-aligned writer-correctness "
            "benchmark. A smaller value requires a task whose write-time salience is "
            "causally observable; the default synthetic post-hoc uniform-query task does not qualify."
        ),
    )
    parser.add_argument(
        "--writer-candidate-mode",
        choices=["all_prequery", "fact_pairs"],
        default="all_prequery",
        help=(
            "hpm_lite_v2 learned writer: candidate proposal before slot selection. "
            "all_prequery (default) scores every adjacent pre-query pair, preserving legacy "
            "behavior. fact_pairs uses the synthetic [FACT, key, value, SEP] grammar to "
            "propose only marked fact pairs, separating proposal from selection for the "
            "capacity-aligned diagnostic. It is a synthetic control, not a general semantic writer."
        ),
    )
    parser.add_argument(
        "--allow-legacy-capacity-mismatch",
        type=str_to_bool,
        default=False,
        help=(
            "Allow the legacy learned-writer benchmark where --memory-slots is "
            "smaller than --num-facts even though the post-hoc query is sampled "
            "uniformly after all facts are written. That task has an intrinsic "
            "upper bound of memory_slots / num_facts for a fixed-slot causal "
            "writer, while teacher forcing exposes all facts; it is unsuitable "
            "as a correctness or scaling test. Default false rejects that setup."
        ),
    )
    parser.add_argument("--memory-null-slot", type=str_to_bool, default=False)
    parser.add_argument("--null-score-init", type=float, default=0.0)
    parser.add_argument(
        "--episodic-read-mode",
        choices=["hard_topk", "ste_topk"],
        default="hard_topk",
        help=(
            "hpm_lite_v2 episodic read operator. hard_topk is the established exact "
            "read and default. ste_topk has exactly the same forward hard top-k result "
            "but uses an entropic-OT straight-through selection gradient during training "
            "so a near-miss candidate is not structurally gradient-disconnected. It is "
            "an experimental, biased gradient estimator; compare it against hard_topk "
            "under an oracle-written read-only control before using it with learned writing."
        ),
    )
    parser.add_argument(
        "--episodic-read-ste-eps",
        type=float,
        default=0.3,
        help="Relative entropic-OT temperature for --episodic-read-mode ste_topk. Default 0.3.",
    )
    parser.add_argument(
        "--episodic-read-ste-iters",
        type=int,
        default=50,
        help="Log-domain Sinkhorn iterations for --episodic-read-mode ste_topk. Default 50.",
    )
    parser.add_argument(
        "--memory-control",
        choices=["normal", "shuffle_values", "shuffled_values", "random_keys", "corrupt_values", "no_retrieval"],
        default="normal",
    )
    parser.add_argument("--write-mode", choices=["oracle", "fact_token", "random_write", "learned"], default="oracle")
    parser.add_argument("--oracle-memory", type=str_to_bool, default=True)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument(
        "--writer-required-facts",
        type=int,
        default=2,
        help=(
            "causal_salience_kv only: number of [REMEMBER, FACT, key, value, SEP] "
            "records later eligible for query. The writer sees REMEMBER before making "
            "its write decision. Must be between 1 and --num-facts - 1."
        ),
    )
    parser.add_argument(
        "--writer-role-mode",
        choices=["visible", "masked"],
        default="visible",
        help=(
            "causal_salience_kv only: visible emits the causal REMEMBER/IRRELEVANT "
            "utility token before each fact candidate. masked replaces both with an "
            "identical filler token as a negative control. masked has a C/M episodic "
            "coverage ceiling and rejects Top-C set-coverage supervision."
        ),
    )
    parser.add_argument("--num-hard-negatives", type=int, default=0)
    parser.add_argument("--repeated-keys", type=str_to_bool, default=False)
    parser.add_argument("--similar-values", type=str_to_bool, default=False)
    parser.add_argument("--distractor-fact-spans", type=int, default=0)
    parser.add_argument("--query-key-noise-only", type=str_to_bool, default=False)
    parser.add_argument("--fact-order", choices=["random", "query_last"], default="random")
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--save-checkpoint", type=str_to_bool, default=True)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--save-step-log", type=str_to_bool, default=False)
    parser.add_argument("--record-vram", type=str_to_bool, default=False)
    return parser


def teacher_forcing_probability(step: int, cutover: int, anneal_steps: int, schedule: str) -> float:
    """Probability of using the oracle-indexed read at this step.

    step <= cutover -> 1.0 (always oracle, identical to the old hard switch).
    step >= cutover + anneal_steps -> 0.0 (always the self-selected Sinkhorn read).
    In between -> decays from 1.0 to 0.0 following `schedule`.

    With anneal_steps=0, cutover+anneal_steps == cutover, so every step is either
    <= cutover (p=1.0) or > cutover (p=0.0) -- a step function identical to the
    pre-scheduled-sampling `step <= learned_writer_teacher_forcing_steps` boolean.
    This is what makes anneal_steps=0 an exact behavioral no-op.
    """
    if step <= cutover:
        return 1.0
    if anneal_steps <= 0 or step >= cutover + anneal_steps:
        return 0.0
    fraction = (step - cutover) / anneal_steps
    if schedule == "linear":
        return 1.0 - fraction
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def validate_causal_writer_capacity_contract(args: argparse.Namespace | SimpleNamespace) -> None:
    """Reject an unlearnable fixed-slot writer benchmark by default.

    In the synthetic KV task, all ``num_facts`` are written before a uniformly
    sampled query identifies one of them. The hard episodic writer is causal
    at each fact position and is trained to mark every fact start positive,
    but it is later allowed to retain only ``memory_slots`` positions. If
    ``memory_slots < num_facts``, no query-agnostic fixed-slot policy can
    retain every possible answer in its episodic slots: its maximum *episodic
    coverage* for a uniformly sampled query is at most memory_slots /
    num_facts. This is not an upper bound on the whole HPM output: another
    path could solve a different task through a separate representation. It
    does make this setup invalid as a correctness test of the hard episodic
    writer. Teacher forcing hides the mismatch by supplying all oracle facts
    during training.

    This guard applies only to the legacy hard-slot path. Delta-rule memory
    has a different continuous-state capacity model and is evaluated through
    its own explicit write budget.
    """
    if getattr(args, "model", None) != "hpm_lite_v2" or getattr(args, "write_mode", None) != "learned":
        return

    task = getattr(args, "task", "kv")
    slots = getattr(args, "memory_slots", None)
    facts = getattr(args, "num_facts", None)
    if task == "causal_salience_kv":
        required = getattr(args, "writer_required_facts", None)
        role_mode = getattr(args, "writer_role_mode", "visible")
        if getattr(args, "delta_rule_writer", False):
            raise ValueError(
                "causal_salience_kv is a hard episodic Top-C writer control; "
                "the continuous delta-rule writer is a different experiment"
            )
        if slots is None:
            raise ValueError(
                "causal_salience_kv requires an explicit --memory-slots value so "
                "the writer's finite allocation decision is testable"
            )
        if required is None or not 1 <= required < facts:
            raise ValueError(
                "causal_salience_kv requires 1 <= --writer-required-facts < --num-facts"
            )
        if slots < required:
            raise ValueError(
                "Invalid causal writer capacity contract: causal_salience_kv has "
                f"{required} causally required records but only {slots} slots. "
                "Set --memory-slots >= --writer-required-facts."
            )
        if slots >= facts:
            raise ValueError(
                "causal_salience_kv must retain fewer slots than its total candidate "
                "records; otherwise there is no Top-C selection problem. Set "
                "--writer-required-facts <= --memory-slots < --num-facts."
            )
        if getattr(args, "writer_candidate_mode", "all_prequery") != "fact_pairs":
            raise ValueError(
                "causal_salience_kv requires --writer-candidate-mode fact_pairs to "
                "isolate fact-span proposal from finite-capacity selection"
            )
        if role_mode not in {"visible", "masked"}:
            raise ValueError("causal_salience_kv writer role mode must be 'visible' or 'masked'")
        return

    if getattr(args, "delta_rule_writer", False):
        return

    if slots is None or facts is None or slots >= facts:
        return
    if getattr(args, "allow_legacy_capacity_mismatch", False):
        return

    raise ValueError(
        "Invalid causal writer capacity contract: the learned hard-slot writer is "
        f"asked to retain {facts} equally possible pre-query facts in only {slots} slots. "
        "Because the query arrives after writing, its episodic coverage is bounded by "
        f"{slots}/{facts} before optimization. Use --memory-slots >= --num-facts "
        "for a hard-writer correctness benchmark; construct an explicitly salience- or "
        "query-conditioned task before evaluating a smaller fixed slot budget. "
        "Set --allow-legacy-capacity-mismatch true only to reproduce historical runs."
    )


def validate_topc_set_coverage_contract(args: argparse.Namespace | SimpleNamespace) -> None:
    """Fail closed before enabling an optional finite-set writer objective."""
    weight = float(getattr(args, "lambda_writer_set_coverage", 0.0))
    if weight < 0.0:
        raise ValueError("--lambda-writer-set-coverage must be non-negative")
    if weight == 0.0:
        return
    if getattr(args, "model", None) != "hpm_lite_v2" or getattr(args, "write_mode", None) != "learned":
        raise ValueError(
            "--lambda-writer-set-coverage requires --model hpm_lite_v2 --write-mode learned"
        )
    if getattr(args, "delta_rule_writer", False):
        raise ValueError(
            "--lambda-writer-set-coverage targets the hard episodic Top-C writer, "
            "not the continuous delta-rule alternative"
        )
    if getattr(args, "memory_slots", None) is None:
        raise ValueError("--lambda-writer-set-coverage requires finite --memory-slots")
    if getattr(args, "use_jepa_writer_bias", False):
        raise ValueError(
            "Top-C set coverage is defined on the raw supervised writer logits; keep "
            "--use-jepa-writer-bias false so a predictive auxiliary cannot override "
            "the tested selection policy"
        )
    if getattr(args, "task", None) != "causal_salience_kv":
        raise ValueError(
            "--lambda-writer-set-coverage is currently restricted to the "
            "causal_salience_kv control; do not apply it to a task whose write-time "
            "utility labels have not been validated"
        )
    if getattr(args, "writer_candidate_mode", None) != "fact_pairs":
        raise ValueError(
            "--lambda-writer-set-coverage requires --writer-candidate-mode fact_pairs"
        )
    if getattr(args, "writer_role_mode", "visible") != "visible":
        raise ValueError(
            "--lambda-writer-set-coverage requires --writer-role-mode visible; "
            "masked role labels are a negative control, not a training target"
        )
    if getattr(args, "writer_set_coverage_eps", 0.3) <= 0.0:
        raise ValueError("--writer-set-coverage-eps must be positive")
    if getattr(args, "writer_set_coverage_iters", 48) < 1:
        raise ValueError("--writer-set-coverage-iters must be positive")
    if getattr(args, "writer_set_coverage_mass_tolerance", 1.0e-3) < 0.0:
        raise ValueError("--writer-set-coverage-mass-tolerance must be non-negative")


def causal_writer_capacity_metadata(args: argparse.Namespace | SimpleNamespace) -> Dict[str, Any]:
    """Describe the writer-capacity contract in machine-readable run metadata.

    ``memory_slots_per_sample`` predates learned writing and means the number
    of *oracle fact annotations* in the data. It is not the configured hard
    episodic capacity. Keep that historical field for compatibility, but
    write distinct fields so dashboards cannot conflate the two again.
    """
    facts = int(getattr(args, "num_facts", 0))
    slots = getattr(args, "memory_slots", None)
    task = getattr(args, "task", "kv")
    is_hard_learned_writer = (
        getattr(args, "model", None) == "hpm_lite_v2"
        and getattr(args, "write_mode", None) == "learned"
        and not getattr(args, "delta_rule_writer", False)
    )
    if not is_hard_learned_writer:
        return {
            "oracle_fact_slots_per_sample": facts,
            "episodic_capacity_configured": slots,
            "causal_writer_capacity_contract": "not_applicable",
            "episodic_coverage_upper_bound": None,
        }
    if task == "causal_salience_kv":
        required = int(getattr(args, "writer_required_facts", 0))
        role_mode = getattr(args, "writer_role_mode", "visible")
        if role_mode == "masked":
            return {
                "oracle_fact_slots_per_sample": facts,
                "episodic_capacity_configured": slots,
                "causal_writer_capacity_contract": "masked_role_negative_control",
                "episodic_coverage_upper_bound": min(float(slots) / max(facts, 1), 1.0),
                "causal_writer_required_facts": required,
                "causal_writer_candidate_records": facts,
            }
        return {
            "oracle_fact_slots_per_sample": facts,
            "episodic_capacity_configured": slots,
            "causal_writer_capacity_contract": "causal_salience_aligned",
            "episodic_coverage_upper_bound": 1.0,
            "causal_writer_required_facts": required,
            "causal_writer_candidate_records": facts,
        }
    if slots is None:
        return {
            "oracle_fact_slots_per_sample": facts,
            "episodic_capacity_configured": None,
            "causal_writer_capacity_contract": "uncapped_prequery_candidates",
            "episodic_coverage_upper_bound": 1.0,
        }

    coverage = min(float(slots) / max(facts, 1), 1.0)
    return {
        "oracle_fact_slots_per_sample": facts,
        "episodic_capacity_configured": int(slots),
        "causal_writer_capacity_contract": "aligned" if slots >= facts else "legacy_mismatch",
        "episodic_coverage_upper_bound": coverage,
    }


def sinkhorn_warmup_weight_at_step(step: int, cutover: int, ramp_steps: int, target_weight: float) -> float:
    """Effective --sinkhorn-warmup-weight at this step.

    ramp_steps <= 0 -> exact no-op, always returns target_weight (flat weight
    for the whole run, identical to every existing sinkhorn-warmup result).
    Otherwise, linearly ramps from 0.0 at step (cutover - ramp_steps) up to
    target_weight at step cutover, so the pressure on the Sinkhorn path to
    actually reduce its own loss increases as the cutover approaches, instead
    of staying flat the whole window. See --sinkhorn-warmup-ramp-steps's
    docstring for why: the flat-weight sweep showed the warmup loss idling
    rather than converging.
    """
    if ramp_steps <= 0 or target_weight <= 0.0:
        return target_weight
    start = cutover - ramp_steps
    if step <= start:
        return 0.0
    if step >= cutover:
        return target_weight
    fraction = (step - start) / ramp_steps
    return target_weight * fraction


def primary_loss_weight_at_step(step: int, cutover: int, ramp_steps: int, floor_weight: float) -> float:
    """Effective weight on the primary oracle-indexed answer_loss, on a
    teacher-forced step.

    This is the mirror image of sinkhorn_warmup_weight_at_step: instead of
    ramping the honest self-selected path's weight UP from 0.0, it ramps the
    privileged oracle-indexed path's weight DOWN from 1.0, over the identical
    N-steps-before-cutover window. Callers are responsible for gating this to
    teacher-forced steps only (exactly like lambda_sinkhorn_warmup is already
    gated on sinkhorn_warmup_logits being present) -- on non-teacher-forced
    steps the primary read IS the honest self-selected read, and needs its
    full weight of 1.0, not a floor value meant for a different regime.

    ramp_steps <= 0 -> exact no-op, always returns 1.0 (identical to every
    existing result, where answer_loss carries an implicit weight of 1.0).
    """
    if ramp_steps <= 0:
        return 1.0
    start = cutover - ramp_steps
    if step <= start:
        return 1.0
    if step >= cutover:
        return floor_weight
    fraction = (step - start) / ramp_steps
    return 1.0 - (1.0 - floor_weight) * fraction


def args_with_defaults(args: argparse.Namespace | SimpleNamespace) -> argparse.Namespace:
    defaults = vars(build_arg_parser().parse_args([]))
    merged = {**defaults, **vars(args)}
    return argparse.Namespace(**merged)




def peak_vram_mb(device: torch.device) -> float:
    """Return peak CUDA memory allocated in MiB, or 0.0 on CPU/non-CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    index = device.index if device.index is not None else torch.cuda.current_device()
    return float(torch.cuda.max_memory_allocated(index) / (1024 ** 2))


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(index)


def append_csv_row(path: Path, fieldnames: list[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


STEP_LOG_COLUMNS = [
    "run_id",
    "step",
    "model",
    "task",
    "write_mode",
    "seq_len",
    "window",
    "seed",
    "batch_size",
    "train_loss",
    "train_answer_ce",
    "train_answer_exact",
    "train_retrieval_loss",
    "train_writer_loss",
    "train_writer_set_coverage_loss",
    "train_writer_set_coverage_soft_required",
    "train_writer_set_coverage_mass_error",
    "train_router_entropy",
    "train_router_z_loss",
    "train_jepa_loss",
    "train_retrieval_top1",
    "train_retrieval_topk",
    "train_retrieval_margin",
    "train_writer_true_fact_written_rate",
    "train_writer_false_write_rate",
    "train_writer_missed_fact_rate",
    "eval_answer_exact",
    "eval_answer_ce",
    "eval_retrieval_top1",
    "eval_retrieval_topk",
    "eval_retrieval_margin",
    "eval_true_fact_written_rate",
    "eval_false_write_rate",
    "eval_missed_fact_rate",
    "eval_avg_written_slots",
    "eval_router_weight_local",
    "eval_router_weight_recurrent",
    "eval_router_weight_fast_weight",
    "eval_router_weight_episodic",
    "eval_router_logit_abs_mean",
    "examples_per_sec_recent",
    "peak_vram_mb",
]

def make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model == "hpm_lite_v2":
        config = HpmLiteV2Config(
            model_type="hpm_lite_v2",
            vocab_size=VOCAB_SIZE,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            window=args.window,
            max_seq_len=max(2048, args.seq_len + 1),
            block_size=args.window,
            use_null_slot=args.memory_null_slot,
            null_score_init=args.null_score_init,
            episodic_read_mode=getattr(args, "episodic_read_mode", "hard_topk"),
            episodic_read_ste_eps=getattr(args, "episodic_read_ste_eps", 0.3),
            episodic_read_ste_iters=getattr(args, "episodic_read_ste_iters", 50),
            use_learned_writer=args.write_mode == "learned",
            episodic_capacity=getattr(args, "memory_slots", None),
            writer_candidate_mode=getattr(args, "writer_candidate_mode", "all_prequery"),
            episodic_only=getattr(args, "episodic_only", False),
            use_token_jepa_aux=getattr(args, "use_token_jepa_aux", False),
            use_jepa_writer_bias=getattr(args, "use_jepa_writer_bias", False),
            jepa_writer_bias_scale=getattr(args, "jepa_writer_bias_scale", 1.0),
            jepa_bias_clamp=getattr(args, "jepa_bias_clamp", None),
            router_logit_clamp=getattr(args, "router_logit_clamp", None),
            sinkhorn_warmup=getattr(args, "sinkhorn_warmup_weight", 0.0) > 0.0,
            use_delta_rule_writer=getattr(args, "delta_rule_writer", False),
            delta_rule_oracle_blend=getattr(args, "delta_rule_oracle_blend", False),
        )
        return HpmLiteV2Model(config).to(device)

    config = HpmLiteConfig(
        model_type=args.model,
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        window=args.window,
        max_seq_len=max(2048, args.seq_len + 1),
        use_null_slot=args.memory_null_slot,
        null_score_init=args.null_score_init,
        use_learned_writer=args.write_mode == "learned",
    )
    return HpmLiteModel(config).to(device)


def forward_batch(model: HpmLiteModel, batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> Dict[str, Any]:
    batch, _ = apply_write_mode(batch, args.write_mode)
    return model(
        batch["input_ids"],
        memory_token_positions=batch["memory_token_positions"],
        memory_mask=batch["memory_mask"],
        answer_positions=batch["answer_positions"],
        query_key_positions=batch["query_key_positions"],
        top_k=args.top_k,
        task=args.task,
        hop_positive_memory_indices=batch["hop_positive_memory_indices"],
        memory_control=args.memory_control,
        use_learned_writer=args.write_mode == "learned",
        learned_writer_teacher_forcing=False,
    )


def run_training(args: argparse.Namespace | SimpleNamespace) -> Dict[str, Any]:
    args = args_with_defaults(args)
    validate_causal_writer_capacity_contract(args)
    validate_topc_set_coverage_contract(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.record_vram:
        reset_peak_vram(device)

    # Default (data_seed=None) reproduces the exact prior behavior: --seed
    # doubles as both the model-init/optimizer seed (set_seed above) and the
    # dataset-sampling seed below. See --data-seed's docstring for why you'd
    # ever want to break that coupling.
    data_seed = getattr(args, "data_seed", None)
    if data_seed is None:
        data_seed = args.seed

    train_dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=args.seq_len,
            window=args.window,
            task=args.task,
            num_facts=args.num_facts,
            num_hard_negatives=args.num_hard_negatives,
            seed=data_seed,
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
    eval_dataset = FactRecallDataset(
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
    model = make_model(args, device)
    optimizer = TinyAdamW(model.parameters(), lr=args.lr)

    # Dedicated RNG for the scheduled-sampling teacher-forcing draw, offset from
    # the dataset seeds (data_seed, +100_000, +200_000) so it can't collide with
    # or perturb any other seeded stream. Always constructed and always advanced
    # once per step (even when scheduled_sampling_anneal_steps=0, where the draw
    # is deterministic regardless) so behavior is identical run-to-run for a
    # given --seed, and adding this flag doesn't change any other RNG's sequence.
    # Deliberately still offset from args.seed (not data_seed) -- this stream is
    # part of the training-dynamics side, not the data-sampling side, of the
    # --data-seed split; keeping it on args.seed means varying --data-seed alone
    # doesn't also perturb which steps get teacher-forced.
    sampling_rng = random.Random(args.seed + 300_000)
    scheduled_sampling_anneal_steps = max(0, getattr(args, "scheduled_sampling_anneal_steps", 0))
    scheduled_sampling_schedule = getattr(args, "scheduled_sampling_schedule", "cosine")

    diagnose_writer_transition = getattr(args, "diagnose_writer_transition", False) and args.write_mode == "learned"
    probe_batch = None
    probe_state = ProbeState()
    diagnostic_rows: list[Dict[str, Any]] = []
    diagnostic_window_steps: set[int] = set()
    if diagnose_writer_transition:
        # A SEPARATE dataset/seed from both train_dataset (data_seed) and
        # eval_dataset (data_seed + 100_000), sampled ONCE, so the exact same
        # input is reused every diagnosed step -- see diagnostics.py's module
        # docstring for why this is required for entropy/KL/Jaccard to be
        # comparable across steps instead of conflated with input variation.
        probe_dataset = FactRecallDataset(
            FactRecallConfig(
                seq_len=args.seq_len,
                window=args.window,
                task=args.task,
                num_facts=args.num_facts,
                num_hard_negatives=args.num_hard_negatives,
                seed=data_seed + 200_000,
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
        probe_batch = probe_dataset.sample_batch(min(8, args.batch_size), device=device)
        probe_batch, _ = apply_write_mode(probe_batch, args.write_mode)
        cutover = args.learned_writer_teacher_forcing_steps
        window = max(0, getattr(args, "diagnostic_window", 30))
        # Extend the instrumented range past the end of the anneal (not just past
        # the old hard cutover) so a scheduled-sampling run doesn't stop
        # instrumenting before the probability has actually reached 0.0.
        diagnostic_window_steps = {
            s
            for s in range(cutover - window, cutover + scheduled_sampling_anneal_steps + window + 1)
            if 1 <= s <= args.steps
        }

    if args.log_every and args.log_every > 0:
        args.eval_every = args.log_every

    run_dir = ensure_dir(Path(args.out_dir) / f"{timestamp()}_{args.model}_{args.task}_seed{args.seed}")
    step_log_path = run_dir / "step_log.csv"
    start_time = time.perf_counter()
    last_time = start_time
    final_metrics: Dict[str, Any] = {}

    for step in range(1, args.steps + 1):
        model.train()
        batch = train_dataset.sample_batch(args.batch_size, device=device)
        batch, write_stats = apply_write_mode(batch, args.write_mode)
        learned_writer = args.write_mode == "learned"
        teacher_forcing_prob = teacher_forcing_probability(
            step,
            args.learned_writer_teacher_forcing_steps,
            scheduled_sampling_anneal_steps,
            scheduled_sampling_schedule,
        )
        # sampling_rng.random() is still drawn even when teacher_forcing_prob is
        # exactly 0.0 or 1.0 (the anneal_steps=0 / pre-cutover / post-anneal
        # cases), so the RNG's sequence -- and therefore every OTHER step's draw
        # -- doesn't shift depending on whether annealing changed anything yet.
        teacher_forcing = learned_writer and sampling_rng.random() < teacher_forcing_prob
        output = model(
            batch["input_ids"],
            memory_token_positions=batch["memory_token_positions"],
            memory_mask=batch["memory_mask"],
            answer_positions=batch["answer_positions"],
            query_key_positions=batch["query_key_positions"],
            top_k=args.top_k,
            task=args.task,
            hop_positive_memory_indices=batch["hop_positive_memory_indices"],
            positive_memory_indices=batch["positive_memory_indices"],
            positive_memory_mask=batch.get("positive_memory_mask"),
            memory_control=args.memory_control,
            use_learned_writer=learned_writer,
            learned_writer_teacher_forcing=teacher_forcing,
            teacher_forcing_prob=teacher_forcing_prob,
        )
        metric_batch = batch
        if learned_writer and "writer_memory_token_positions" in output["retrieval"]:
            learned_write_batch = batch_from_memory_selection(
                batch,
                output["retrieval"]["writer_memory_token_positions"],
                output["retrieval"]["writer_memory_mask"],
            )
            write_stats = writer_metrics(batch, learned_write_batch)
            if not teacher_forcing:
                metric_batch = learned_write_batch
        logits = output["logits"]
        answer_loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        retrieval_loss = output["retrieval"].get("retrieval_loss", logits.new_zeros(()))
        writer_loss = output["retrieval"].get("writer_loss", logits.new_zeros(()))
        lambda_writer_set_coverage = getattr(args, "lambda_writer_set_coverage", 0.0)
        if lambda_writer_set_coverage:
            writer_set_coverage_loss, writer_set_coverage_info = topc_set_coverage_loss(
                output["retrieval"]["writer_logits"],
                output["retrieval"]["writer_labels"],
                output["retrieval"]["writer_valid_mask"],
                capacity=args.memory_slots,
                eps=args.writer_set_coverage_eps,
                n_iters=args.writer_set_coverage_iters,
                mass_tolerance=args.writer_set_coverage_mass_tolerance,
                require_strict_selection=True,
            )
        else:
            writer_set_coverage_loss = logits.new_zeros(())
            writer_set_coverage_info = {
                "samples": 0.0,
                "mean_active_candidates": 0.0,
                "mean_required_candidates": 0.0,
                "mean_soft_positive_coverage": 0.0,
                "max_membership_mass_error": 0.0,
            }
        if learned_writer and {
            "writer_selection_logits",
            "writer_labels",
            "writer_valid_mask",
            "writer_memory_mask",
        }.issubset(output["retrieval"]):
            writer_margin_info = topc_required_margin_diagnostics(
                output["retrieval"]["writer_selection_logits"],
                output["retrieval"]["writer_labels"],
                output["retrieval"]["writer_valid_mask"],
                output["retrieval"]["writer_memory_mask"],
            )
        else:
            writer_margin_info = {
                "samples": 0.0,
                "mean_required_topc_margin": 0.0,
                "all_required_topc_rate": 0.0,
            }
        lambda_router_entropy = getattr(args, "lambda_router_entropy", 0.0)
        router_weights = output["retrieval"].get("router_weights")
        if lambda_router_entropy and router_weights is not None:
            router_entropy = router_entropy_loss(router_weights)
        else:
            router_entropy = logits.new_zeros(())
        lambda_router_z_loss = getattr(args, "lambda_router_z_loss", 0.0)
        router_logits = output["retrieval"].get("router_logits")
        if lambda_router_z_loss and router_logits is not None:
            router_z = router_z_loss(router_logits)
        else:
            router_z = logits.new_zeros(())
        lambda_jepa = getattr(args, "lambda_jepa", 0.0)
        if lambda_jepa:
            # token_jepa_loss is only present when --use-jepa-writer-bias is
            # also set (it's the predictor the writer bias actually reads
            # from); jepa_loss (block-level) is always present when
            # use_jepa_aux is on (the default), whether or not the writer
            # bias is enabled.
            jepa_loss = output["retrieval"].get("jepa_loss", logits.new_zeros(()))
            token_jepa_loss = output["retrieval"].get("token_jepa_loss", logits.new_zeros(()))
            jepa_total = jepa_loss + token_jepa_loss
        else:
            jepa_total = logits.new_zeros(())
        lambda_sinkhorn_warmup = sinkhorn_warmup_weight_at_step(
            step,
            args.learned_writer_teacher_forcing_steps,
            getattr(args, "sinkhorn_warmup_ramp_steps", 0),
            getattr(args, "sinkhorn_warmup_weight", 0.0),
        )
        sinkhorn_warmup_logits = output["retrieval"].get("sinkhorn_warmup_logits")
        if lambda_sinkhorn_warmup and sinkhorn_warmup_logits is not None:
            # Real answer-CE against THIS step's real batch/targets, exactly
            # like answer_loss above, just read through the differentiable
            # full-candidate path instead of the oracle-indexed one. This is
            # what gives that path continuous gradient through the teacher-
            # forcing window instead of only at the cutover -- see
            # --sinkhorn-warmup-weight's docstring.
            sinkhorn_warmup_loss = answer_cross_entropy(sinkhorn_warmup_logits, batch["target_ids"], batch["loss_mask"])
        else:
            sinkhorn_warmup_loss = logits.new_zeros(())
        # Mirror image of the sinkhorn-warmup ramp: only reduce the primary
        # (oracle-indexed) loss's weight on steps that actually used the
        # oracle read this step (teacher_forcing True AND the parallel warmup
        # read was actually computed) -- on every other step the primary read
        # already IS the honest self-selected read and must keep weight 1.0.
        # See --primary-loss-anneal-steps's docstring.
        lambda_primary = 1.0
        primary_loss_anneal_steps = getattr(args, "primary_loss_anneal_steps", 0)
        if primary_loss_anneal_steps > 0 and lambda_sinkhorn_warmup and sinkhorn_warmup_logits is not None:
            lambda_primary = primary_loss_weight_at_step(
                step,
                args.learned_writer_teacher_forcing_steps,
                primary_loss_anneal_steps,
                getattr(args, "primary_loss_floor_weight", 0.1),
            )
        loss = (
            lambda_primary * answer_loss
            + args.lambda_ret * retrieval_loss
            + args.lambda_writer * writer_loss
            + lambda_writer_set_coverage * writer_set_coverage_loss
            + lambda_router_entropy * router_entropy
            + lambda_router_z_loss * router_z
            + lambda_jepa * jepa_total
            + lambda_sinkhorn_warmup * sinkhorn_warmup_loss
        )

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        diag_grad_norms: Optional[Dict[str, Optional[float]]] = None
        if diagnose_writer_transition and step in diagnostic_window_steps:
            # Must be captured BEFORE clip_grad_norm_: clipping rescales
            # every parameter's grad by one global factor, which would erase
            # the relative-magnitude signal this is meant to expose (see
            # diagnostics.py's module docstring).
            diag_grad_norms = collect_module_grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if diagnose_writer_transition and step in diagnostic_window_steps:
            row: Dict[str, Any] = {
                "step": step,
                "teacher_forcing": teacher_forcing,
                "teacher_forcing_prob": teacher_forcing_prob,
                "loss": float(loss.item()),
                "answer_loss": float(answer_loss.item()),
                "writer_loss": float(writer_loss.item()),
                "writer_set_coverage_loss": float(writer_set_coverage_loss.item()),
                "sinkhorn_warmup_loss": (
                    float(sinkhorn_warmup_loss.item())
                    if lambda_sinkhorn_warmup and sinkhorn_warmup_logits is not None
                    else None
                ),
                "sinkhorn_warmup_weight_effective": lambda_sinkhorn_warmup,
                "primary_loss_weight_effective": lambda_primary,
                **(diag_grad_norms or {}),
            }
            model.eval()
            with torch.no_grad():
                probe_output = model(
                    probe_batch["input_ids"],
                    memory_token_positions=probe_batch["memory_token_positions"],
                    memory_mask=probe_batch["memory_mask"],
                    answer_positions=probe_batch["answer_positions"],
                    query_key_positions=probe_batch["query_key_positions"],
                    top_k=args.top_k,
                    task=args.task,
                    hop_positive_memory_indices=probe_batch["hop_positive_memory_indices"],
                    positive_memory_indices=probe_batch.get("positive_memory_indices"),
                    positive_memory_mask=probe_batch.get("positive_memory_mask"),
                    memory_control=args.memory_control,
                    use_learned_writer=True,
                    # Always non-teacher-forced: exercises the differentiable
                    # full-candidate path on every diagnosed step, including
                    # ones still inside the teacher-forcing window, so the
                    # candidate distribution is comparable across the cutover
                    # instead of only existing for steps after it.
                    learned_writer_teacher_forcing=False,
                )
                probe_retrieval = probe_output["retrieval"]
                probe_router_logits = probe_retrieval.get("router_logits")
                row["probe_router_logit_abs_mean"] = (
                    float(probe_router_logits.abs().mean().item()) if probe_router_logits is not None else None
                )
                probe_router_weights = probe_retrieval.get("router_weights")
                if probe_router_weights is not None and probe_router_logits is not None:
                    probe_health = router_health_metrics(
                        probe_router_weights,
                        probe_router_logits,
                        router_raw_logits=probe_retrieval.get("router_raw_logits"),
                        router_clamp_jacobian=probe_retrieval.get("router_clamp_jacobian"),
                    )
                    row["probe_router_raw_logit_abs_mean"] = float(
                        probe_health["router_raw_logit_abs_mean"].item()
                    )
                    row["probe_router_clamp_jacobian_mean"] = float(
                        probe_health["router_clamp_jacobian_mean"].item()
                    )
                    row["probe_router_raw_to_weight_jacobian_fro_mean"] = float(
                        probe_health["router_raw_to_weight_jacobian_fro_mean"].item()
                    )
                else:
                    row["probe_router_raw_logit_abs_mean"] = None
                    row["probe_router_clamp_jacobian_mean"] = None
                    row["probe_router_raw_to_weight_jacobian_fro_mean"] = None
                if "writer_selection_logits" in probe_retrieval and "writer_valid_mask" in probe_retrieval:
                    probs = masked_candidate_distribution(
                        probe_retrieval["writer_selection_logits"], probe_retrieval["writer_valid_mask"]
                    )
                    row["probe_candidate_entropy_mean"] = float(candidate_entropy(probs).mean().item())
                    starts = probe_retrieval["writer_memory_token_positions"][..., 0]
                    mask = probe_retrieval["writer_memory_mask"]
                    if probe_state.probs is not None:
                        row["probe_candidate_kl_vs_prev_mean"] = float(
                            candidate_kl(probs, probe_state.probs).mean().item()
                        )
                        row["probe_topk_jaccard_vs_prev_mean"] = float(
                            topk_jaccard(starts, mask, probe_state.starts, probe_state.mask).mean().item()
                        )
                    else:
                        row["probe_candidate_kl_vs_prev_mean"] = None
                        row["probe_topk_jaccard_vs_prev_mean"] = None
                    probe_state.probs = probs
                    probe_state.starts = starts
                    probe_state.mask = mask
                else:
                    row["probe_candidate_entropy_mean"] = None
                    row["probe_candidate_kl_vs_prev_mean"] = None
                    row["probe_topk_jaccard_vs_prev_mean"] = None
            model.train()
            diagnostic_rows.append(row)

        should_eval = step % args.eval_every == 0 or step == args.steps
        if should_eval:
            now = time.perf_counter()
            elapsed = now - last_time
            examples_per_sec = (args.batch_size * args.eval_every) / max(elapsed, 1.0e-9)
            last_time = now
            train_acc = answer_span_exact_accuracy(logits.detach(), batch["target_ids"], batch["loss_mask"])
            # output["retrieval"]["top_indices"]/["scores"] index into whatever
            # candidate space the model actually retrieved over. That's
            # metric_batch's own memory_token_positions space normally, but
            # when the learned writer's full-candidate differentiable path is
            # engaged (not teacher_forcing), it's a *different*, larger space
            # (see match_positive_slots_into_active_space). Use the matched
            # active-space labels when available; they're None otherwise
            # (teacher forcing, or non-"learned" write modes), in which case
            # metric_batch's own oracle/writer-slot-space fields are already
            # correctly scoped.
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
            eval_metrics = evaluate_batches(
                model=model,
                dataset=eval_dataset,
                batch_size=args.batch_size,
                batches=args.eval_batches,
                device=device,
                task=args.task,
                top_k=args.top_k,
                memory_control=args.memory_control,
                write_mode=args.write_mode,
                use_learned_writer=learned_writer,
            )
            peak_vram = peak_vram_mb(device) if args.record_vram else 0.0
            final_metrics = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_answer_ce": float(answer_loss.item()),
                "train_answer_exact": float(train_acc.item()),
                "train_retrieval_loss": float(retrieval_loss.item()),
                "train_writer_loss": float(writer_loss.item()),
                "train_writer_set_coverage_loss": float(writer_set_coverage_loss.item()),
                "train_writer_set_coverage_soft_required": writer_set_coverage_info[
                    "mean_soft_positive_coverage"
                ],
                "train_writer_set_coverage_mass_error": writer_set_coverage_info[
                    "max_membership_mass_error"
                ],
                "train_writer_required_topc_margin": writer_margin_info[
                    "mean_required_topc_margin"
                ],
                "train_writer_all_required_topc_rate": writer_margin_info[
                    "all_required_topc_rate"
                ],
                "train_writer_topc_margin_samples": writer_margin_info["samples"],
                "train_router_entropy": float(router_entropy.item()),
                "train_router_z_loss": float(router_z.item()),
                "train_jepa_loss": float(jepa_total.item()),
                "examples_per_sec_recent": examples_per_sec,
                "peak_vram_mb": peak_vram,
                **{f"train_writer_{key}": value for key, value in write_stats.items()},
                **{f"train_{key}": value for key, value in ret.items()},
                **{f"eval_{key}": value for key, value in eval_metrics.items()},
            }
            if args.save_step_log:
                step_row = {
                    "run_id": run_dir.name,
                    "model": args.model,
                    "task": args.task,
                    "write_mode": args.write_mode,
                    "seq_len": args.seq_len,
                    "window": args.window,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    **final_metrics,
                }
                append_csv_row(step_log_path, STEP_LOG_COLUMNS, step_row)

            compact = {
                "step": step,
                "loss": round(final_metrics["train_loss"], 4),
                "eval_exact": round(final_metrics["eval_answer_exact"], 4),
                "eval_ce": round(final_metrics["eval_answer_ce"], 4),
            }
            if "eval_retrieval_top1" in final_metrics:
                compact["eval_ret_top1"] = round(final_metrics["eval_retrieval_top1"], 4)
            if learned_writer:
                compact["writer_true_fact_rate"] = round(final_metrics.get("eval_true_fact_written_rate", 0.0), 4)
                compact["writer_false_write_rate"] = round(final_metrics.get("eval_false_write_rate", 0.0), 4)
                compact["writer_missed_fact_rate"] = round(final_metrics.get("eval_missed_fact_rate", 0.0), 4)
            if lambda_writer_set_coverage:
                compact["writer_set_coverage_loss"] = round(final_metrics["train_writer_set_coverage_loss"], 4)
                compact["writer_soft_required_coverage"] = round(
                    writer_set_coverage_info["mean_soft_positive_coverage"], 4
                )
            if "eval_router_weight_local" in final_metrics:
                compact["router_local"] = round(final_metrics["eval_router_weight_local"], 4)
                compact["router_recurrent"] = round(final_metrics["eval_router_weight_recurrent"], 4)
                compact["router_fast_weight"] = round(final_metrics["eval_router_weight_fast_weight"], 4)
                compact["router_episodic"] = round(final_metrics["eval_router_weight_episodic"], 4)
            if "eval_router_logit_abs_mean" in final_metrics:
                compact["router_logit_abs_mean"] = round(final_metrics["eval_router_logit_abs_mean"], 4)
            if lambda_router_entropy:
                compact["router_entropy"] = round(final_metrics["train_router_entropy"], 4)
            if lambda_router_z_loss:
                compact["router_z_loss"] = round(final_metrics["train_router_z_loss"], 4)
            if getattr(args, "router_logit_clamp", None) is not None:
                compact["router_logit_clamp"] = args.router_logit_clamp
            if getattr(args, "jepa_bias_clamp", None) is not None:
                compact["jepa_bias_clamp"] = args.jepa_bias_clamp
            if scheduled_sampling_anneal_steps > 0:
                compact["teacher_forcing_prob"] = round(teacher_forcing_prob, 4)
            if lambda_jepa:
                compact["jepa_loss"] = round(final_metrics["train_jepa_loss"], 4)
            if lambda_sinkhorn_warmup:
                compact["sinkhorn_warmup_loss"] = round(float(sinkhorn_warmup_loss.item()), 4)
            if getattr(args, "sinkhorn_warmup_ramp_steps", 0) > 0:
                compact["sinkhorn_warmup_weight_effective"] = round(lambda_sinkhorn_warmup, 4)
            if getattr(args, "primary_loss_anneal_steps", 0) > 0:
                compact["primary_loss_weight_effective"] = round(lambda_primary, 4)
            print(json.dumps(compact, sort_keys=True))

    diagnostics_csv_path = ""
    if diagnose_writer_transition and diagnostic_rows:
        diagnostics_csv_path = str(run_dir / "writer_transition_diagnostics.csv")
        for row in diagnostic_rows:
            append_csv_row(Path(diagnostics_csv_path), DIAGNOSTIC_CSV_COLUMNS, row)

    total_time = time.perf_counter() - start_time
    capacity_metadata = causal_writer_capacity_metadata(args)
    final_metrics.update(
        {
            "model": args.model,
            "task": args.task,
            "write_mode": args.write_mode,
            "seq_len": args.seq_len,
            "window": args.window,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "seed": args.seed,
            "data_seed": data_seed,
            "device": str(device),
            "parameters": count_parameters(model),
            # Historical name: this is the oracle annotation count, not the
            # configured learned-writer capacity. The explicit fields below
            # are the authoritative capacity record for new runs.
            "memory_slots_per_sample": 0 if args.model == "local" else train_dataset.config.num_facts,
            **capacity_metadata,
            "train_wall_time_sec": total_time,
            "peak_vram_mb": peak_vram_mb(device) if args.record_vram else 0.0,
            "step_log_path": str(step_log_path) if args.save_step_log else "",
            "lambda_writer": args.lambda_writer,
            "lambda_writer_set_coverage": getattr(args, "lambda_writer_set_coverage", 0.0),
            "writer_set_coverage_eps": getattr(args, "writer_set_coverage_eps", 0.3),
            "writer_set_coverage_iters": getattr(args, "writer_set_coverage_iters", 48),
            "writer_set_coverage_mass_tolerance": getattr(args, "writer_set_coverage_mass_tolerance", 1.0e-3),
            "writer_candidate_mode": getattr(args, "writer_candidate_mode", "all_prequery"),
            "writer_required_facts": getattr(args, "writer_required_facts", None),
            "writer_role_mode": getattr(args, "writer_role_mode", "visible"),
            "episodic_only": getattr(args, "episodic_only", False),
            "episodic_read_mode": getattr(args, "episodic_read_mode", "hard_topk"),
            "episodic_read_ste_eps": getattr(args, "episodic_read_ste_eps", 0.3),
            "episodic_read_ste_iters": getattr(args, "episodic_read_ste_iters", 50),
            "lambda_router_entropy": lambda_router_entropy,
            "lambda_router_z_loss": lambda_router_z_loss,
            "router_logit_clamp": getattr(args, "router_logit_clamp", None),
            "lambda_jepa": lambda_jepa,
            "use_token_jepa_aux": getattr(args, "use_token_jepa_aux", False),
            "use_jepa_writer_bias": getattr(args, "use_jepa_writer_bias", False),
            "jepa_writer_bias_scale": getattr(args, "jepa_writer_bias_scale", 1.0),
            "jepa_bias_clamp": getattr(args, "jepa_bias_clamp", None),
            "delta_rule_writer": getattr(args, "delta_rule_writer", False),
            "delta_rule_oracle_blend": getattr(args, "delta_rule_oracle_blend", False),
            "learned_writer_teacher_forcing_steps": args.learned_writer_teacher_forcing_steps,
            "scheduled_sampling_anneal_steps": scheduled_sampling_anneal_steps,
            "scheduled_sampling_schedule": scheduled_sampling_schedule,
            # NOTE: lambda_sinkhorn_warmup here is a per-step local variable that,
            # by the end of the loop, holds whatever the ramp schedule produced on
            # the FINAL step -- not the configured target. Record the actual
            # configured flag value for reproducibility instead.
            "sinkhorn_warmup_weight": getattr(args, "sinkhorn_warmup_weight", 0.0),
            "sinkhorn_warmup_ramp_steps": getattr(args, "sinkhorn_warmup_ramp_steps", 0),
            "primary_loss_anneal_steps": getattr(args, "primary_loss_anneal_steps", 0),
            "primary_loss_floor_weight": getattr(args, "primary_loss_floor_weight", 0.1),
            "diagnose_writer_transition": diagnose_writer_transition,
            "writer_transition_diagnostics_path": diagnostics_csv_path,
            "run_dir": str(run_dir),
        }
    )
    write_json(run_dir / "metrics.json", final_metrics)

    if args.save_checkpoint:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": asdict(model.config),
                "args": vars(args),
                "metrics": final_metrics,
            },
            run_dir / "checkpoint.pt",
        )
    write_run_summary(run_dir / "summary.md", final_metrics, args)
    return final_metrics


def write_run_summary(path: Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# HPM-Lite Run Summary",
        "",
        f"Command model: `{args.model}`",
        f"Task: `{args.task}`",
        f"Sequence length/window: `{args.seq_len}` / `{args.window}`",
        f"Final exact accuracy: `{metrics.get('eval_answer_exact', 0.0):.4f}`",
        f"Final answer CE: `{metrics.get('eval_answer_ce', 0.0):.4f}`",
        f"Wall-clock train time: `{metrics.get('train_wall_time_sec', 0.0):.2f}s`",
        "",
        "This single run is a sanity check, not a claim about the architecture.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    main()
