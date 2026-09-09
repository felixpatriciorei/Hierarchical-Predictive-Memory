from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .data import VOCAB_SIZE
from .associative_memory import DeltaRuleEpisodicMemory, budgeted_sigmoid_write_gate
from .memory import EpisodicMemory
from .model import (
    SupervisedMemoryWriter,
    TransformerBlock,
    match_hop_positive_indices,
    match_positive_slots_into_active_space,
)
from .hpm_v2 import (
    BlockwiseSelectiveRecurrentState,
    FastWeightBlockMemory,
    HpmV2PathRouter,
    JepaLiteAuxiliary,
    block_summaries,
)


@dataclass
class HpmLiteV2Config:
    """Trainable HPM v2 model config.

    v2 keeps the v1 public forward interface so it can plug into the existing
    synthetic KV train/eval pipeline, but replaces the 3-path HPM core with a
    4-path blockwise memory core:

    local attention + selective recurrent state + fast-weight memory + episodic memory.
    """

    model_type: str = "hpm_lite_v2"
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    window: int = 256
    max_seq_len: int = 2048
    dropout: float = 0.0
    block_size: int = 128
    fast_decay_init: float = 0.95
    use_null_slot: bool = False
    null_score_init: float = 0.0
    episodic_read_mode: str = "hard_topk"
    """Episodic read operator.

    ``hard_topk`` is the established exact read and remains the default.
    ``ste_topk`` has identical forward behavior but supplies an entropic-OT
    straight-through gradient across the top-k boundary during training.  It
    is an opt-in experimental estimator, not a replacement for the frozen B1
    writer objective.  ``sparsemax`` remains available on ``EpisodicMemory``
    for isolated experiments but is intentionally not exposed through the v2
    training CLI because its all-slot weights are not a drop-in readout
    contract.
    """
    episodic_read_ste_eps: float = 0.3
    episodic_read_ste_iters: int = 50
    use_learned_writer: bool = False
    use_jepa_aux: bool = True
    jepa_latent_dim: Optional[int] = None
    use_token_jepa_aux: bool = False
    """If True (requires use_jepa_aux=True), train and report the
    token-level JEPA predictive objective independently of the memory
    writer.

    This is deliberately separate from ``use_jepa_writer_bias``.  A
    predictive auxiliary must be measurable on its own before its surprise
    score is allowed to control a load-bearing memory-write decision.  With
    this flag on and ``use_jepa_writer_bias=False``, ``token_jepa_loss`` is
    returned for the training loop but the writer receives no JEPA-derived
    input; consequently its selection logits and memory contents are
    identical to the no-token-JEPA control for identical model weights and
    inputs.  Default False preserves existing runs exactly.
    """
    use_jepa_writer_bias: bool = False
    jepa_writer_bias_scale: float = 1.0
    """If True (requires use_jepa_aux=True and use_learned_writer=True), the
    JEPA-lite token-level predictive-surprise score (JepaLiteAuxiliary.token_surprise)
    is added as an additive bias on the writer's selection logits before
    top-k/Sinkhorn selection -- on top of, not replacing, the oracle-supervised
    BCE signal the scorer is still trained with. Default False = off, matches
    all prior behavior/results exactly.
    """
    episodic_capacity: Optional[int] = None
    writer_candidate_mode: str = "all_prequery"
    """Candidate proposal policy for the learned episodic writer.

    ``all_prequery`` is the historical control: every adjacent causal token
    pair is scored. ``fact_pairs`` is a synthetic-grammar diagnostic that
    proposes only a key/value pair immediately following the explicit FACT
    marker. It separates deterministic span proposal from slot selection;
    it does not make the final HPM writer semantic or autonomous. Default
    preserves legacy behavior.
    """
    writer_diff_select_eps: float = 0.3
    writer_diff_select_iters: int = 50
    jepa_bias_clamp: Optional[float] = None
    """If set (and use_jepa_writer_bias=True), smoothly bounds the JEPA
    surprise term's contribution to the writer's selection logits to
    [-jepa_bias_clamp, +jepa_bias_clamp] via tanh, mirroring
    router_logit_clamp's mechanism -- see
    SupervisedMemoryWriter.forward's surprise_bias_clamp docstring for the
    full rationale. Default None = off, matches all prior behavior/results
    exactly (identical to the unclamped jepa_writer_bias_scale * bias).
    Use only in an explicitly controlled experiment."""
    episodic_only: bool = False
    """Diagnostic isolation mode for episodic writer/read controls.

    When enabled, the recurrent, fast-weight, and routing paths are removed so
    an episodic mechanism can be tested without path compensation. This is not
    a recommended simplification of HPM: later frozen-router evidence shows
    recurrent and fast-weight paths are load-bearing on other task regions.
    """
    router_logit_clamp: Optional[float] = None
    """If set, HpmV2PathRouter smoothly bounds |router_logit| <= this value via
    tanh before softmax. This guarantees a forward probability bound, but it
    does NOT guarantee optimizer recoverability because the tanh derivative
    back to the raw router projection can itself vanish. See
    HpmV2PathRouter.logit_clamp's docstring and the emitted
    ``router_raw_logits``/``router_clamp_jacobian`` diagnostics. Default None
    = off, matches all prior behavior/results exactly. Use only in an explicitly controlled experiment; it is not a proven anti-collapse fix."""
    sinkhorn_warmup: bool = False
    """If True, forward() additionally runs the differentiable full-candidate
    Sinkhorn read (writer_full_memory_token_positions/writer_select_bias --
    computed unconditionally every step regardless of teacher forcing, see
    SupervisedMemoryWriter.forward) through episodic_memory/router/answer_head
    IN PARALLEL with the primary oracle-indexed read, whenever
    learned_writer_teacher_forcing=True, and exposes the result as
    retrieval['sinkhorn_warmup_logits']. This gives the Sinkhorn path real
    answer-loss gradient throughout the teacher-forcing window instead of
    only for the first time at the cutover -- see train.py's
    --sinkhorn-warmup-weight docstring for why this replaces
    --scheduled-sampling-anneal-steps as the fix for the tf-release collapse.
    Default False = off, matches all prior behavior/results exactly and adds
    no extra compute when disabled."""
    use_delta_rule_writer: bool = False
    """If True (requires use_learned_writer=True), completely bypasses the
    discrete torch.topk-based memory path -- both SupervisedMemoryWriter's
    hard-selected memory_token_positions AND EpisodicMemory's retrieve_topk
    gather over the full-candidate/select_bias path used after teacher-
    forcing cutover -- and instead routes EVERY candidate token-position
    pair through DeltaRuleEpisodicMemory (hpm/associative_memory.py):
    a continuous delta-rule write gated by a budget-capped sigmoid of
    ``writer_selection_logits``, with no discrete selection and no
    straight-through estimator anywhere in the read path.

    This is an experimental alternative, not a demonstrated fix for a
    teacher-forcing transition. It changes an exact finite-slot episodic
    store into an associative state, so its accuracy and capacity must be
    evaluated separately from the hard-slot writer. In particular, it does
    not repair an invalid causal capacity benchmark in which fewer slots are
    asked to cover more equally likely future queries. See
    docs/MATURITY.md for the test contract.

    Practical consequence: because there is only ONE read mechanism now --
    not "oracle read during teacher forcing, hard/soft-selected candidate
    read after" -- learned_writer_teacher_forcing no longer changes WHAT is
    read when this flag is on. writer_loss's oracle-supervised BCE is still
    computed unconditionally (it trains the scorer whose sigmoid output
    becomes write_gate); only the *read* path changes. There is
    deliberately no read-space discontinuity at cutover for this mode. That
    is a controlled property of the alternative mechanism, not evidence that
    it solves the original task.

    Known limitations:
    - DeltaRuleEpisodicMemory does not produce top_indices/scores/weights,
      so retrieval_top1/retrieval_topk/retrieval_margin-style metrics
      (hpm/metrics.py) are unavailable when this is on -- they degrade
      to {}/None gracefully (a diagnostics gap, not a crash).
    - sinkhorn_warmup is a no-op here: there's no separate "warm" candidate
      read left to warm up, since the primary read already sees every
      candidate every step. HpmLiteV2Model.forward skips that block
      entirely when this flag is set, regardless of config.sinkhorn_warmup.
    - task in {"twohop", "longhop"} is not yet supported (DeltaRuleEpisodicMemory
      is a single read, no multi-hop query-refinement loop); forward()
      raises ValueError rather than silently returning a wrong answer.
    - The per-candidate write is a sequential Python loop (see
      DeltaRuleEpisodicMemory's module docstring) -- fine for the short
      diagnostic runs this flag is meant to be checked with first, slow at
      the seq_len=2048 lengths the full sweeps use.
    - It has not established competitive exact recall in this repository.
      Oracle blending is a curriculum experiment, not a proof of a valid
      deployment-time write mechanism.

    Default False = off, matches all prior behavior/results exactly.
    """
    delta_rule_oracle_blend: bool = False
    """If True (requires use_delta_rule_writer=True), adds oracle-gated
    curriculum scaffolding to the experimental delta-rule writer.

    Mechanism: DeltaRuleEpisodicMemory.forward's write_gate is normally a
    budget-capped sigmoid(writer_selection_logits). With this on, it becomes a
    continuous per-candidate blend

        write_gate = p * oracle_gate + (1 - p) * budgeted_sigmoid(writer_selection_logits)

    where `p` is `teacher_forcing_prob` (the same continuous
    scheduled-sampling probability train.py already anneals from 1.0 to
    0.0 via --scheduled-sampling-anneal-steps -- see
    teacher_forcing_probability() in train.py) and `oracle_gate` is a hard
    {0, 1} mask, matched into the writer's full-candidate space via
    match_positive_slots_into_active_space, that is 1.0 exactly at the true
    fact position(s) (positive_memory_indices/positive_memory_mask) and 0.0
    everywhere else.

    For a nonzero anneal duration, `p` changes continuously step to step.
    Early training therefore uses oracle-gated writes and later training
    uses learned gates. This is useful for studying a curriculum, but it
    also means the training path has privileged information and must not be
    reported as a self-writing result without the p=0 evaluation.

    If `p` is never annealed (scheduled_sampling_anneal_steps=0, matching
    every delta-rule-writer run tested so far), this is equivalent to a
    flat oracle_gate the entire run -- not useful on its own. Pair this
    flag with a nonzero --scheduled-sampling-anneal-steps so `p` actually
    decays and the model transitions from the easy oracle-scaffolded gate
    to the fully self-selected one continuously over training, rather than
    running the effectively-easier oracle-only mode throughout or the
    effectively-harder self-only mode throughout.

    No effect when use_delta_rule_writer=False (validated in __init__).
    Default False = off, matches use_delta_rule_writer's existing behavior
    exactly -- this is a strict opt-in on top of an opt-in.
    """


class HpmLiteV2Model(nn.Module):
    """HPM v2, wired to the same task API as the v1 model.

    This is intentionally not a from-scratch LLM. It is the next trainable HPM
    research model for the controlled long-range memory benchmark.
    """

    def __init__(self, config: HpmLiteV2Config):
        super().__init__()
        if config.model_type != "hpm_lite_v2":
            raise ValueError(f"HpmLiteV2Model requires model_type='hpm_lite_v2', got {config.model_type!r}")
        if config.use_jepa_writer_bias and not config.use_jepa_aux:
            raise ValueError("use_jepa_writer_bias=True requires use_jepa_aux=True (no JEPA module to source surprise from)")
        if config.use_token_jepa_aux and not config.use_jepa_aux:
            raise ValueError("use_token_jepa_aux=True requires use_jepa_aux=True (no JEPA module to train)")
        if config.use_delta_rule_writer and not config.use_learned_writer:
            raise ValueError("use_delta_rule_writer=True requires use_learned_writer=True (no writer to source selection logits/write_gate from)")
        if config.delta_rule_oracle_blend and not config.use_delta_rule_writer:
            raise ValueError("delta_rule_oracle_blend=True requires use_delta_rule_writer=True (nothing to blend the oracle gate into)")
        if config.use_learned_writer and config.episodic_read_mode == "ste_topk":
            raise ValueError(
                "episodic_read_mode='ste_topk' is currently validated only for a fixed, "
                "oracle-written candidate set and cannot be combined with use_learned_writer=True. "
                "The learned writer's select_bias is itself a straight-through estimator; "
                "their composition requires a separate controlled experiment."
            )
        if config.writer_candidate_mode not in {"all_prequery", "fact_pairs"}:
            raise ValueError("writer_candidate_mode must be 'all_prequery' or 'fact_pairs'")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.local_blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.window, config.dropout) for _ in range(config.layers)]
        )
        if config.episodic_only:
            self.selective_recurrent = None
            self.fast_memory = None
            self.router = None
        else:
            self.selective_recurrent = BlockwiseSelectiveRecurrentState(
                config.d_model, block_size=config.block_size, dropout=config.dropout
            )
            self.fast_memory = FastWeightBlockMemory(
                config.d_model, block_size=config.block_size, decay_init=config.fast_decay_init
            )
            self.router = HpmV2PathRouter(config.d_model, num_paths=4, logit_clamp=config.router_logit_clamp)
        self.episodic_memory = EpisodicMemory(
            config.d_model,
            use_null_slot=config.use_null_slot,
            null_score_init=config.null_score_init,
            read_mode=config.episodic_read_mode,
            read_ste_eps=config.episodic_read_ste_eps,
            read_ste_iters=config.episodic_read_ste_iters,
        )
        self.writer = SupervisedMemoryWriter(config.d_model) if config.use_learned_writer else None
        self.delta_writer = DeltaRuleEpisodicMemory(config.d_model) if config.use_delta_rule_writer else None
        self.jepa = JepaLiteAuxiliary(config.d_model, latent_dim=config.jepa_latent_dim) if config.use_jepa_aux else None

        self.final_ln = nn.LayerNorm(config.d_model)
        self.answer_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.answer_head.weight = self.token_embedding.weight

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]

    def _local_path(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._embed(input_ids)
        for block in self.local_blocks:
            hidden = block(hidden)
        return hidden

    def _maybe_select_writer_memory(
        self,
        *,
        local_state: torch.Tensor,
        input_ids: torch.Tensor,
        query_key_positions: torch.Tensor,
        memory_token_positions: torch.Tensor,
        memory_mask: torch.Tensor,
        hop_positive_memory_indices: Optional[torch.Tensor],
        positive_memory_indices: Optional[torch.Tensor],
        positive_memory_mask: Optional[torch.Tensor],
        use_learned_writer: bool,
        learned_writer_teacher_forcing: bool,
        surprise_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        info: Dict[str, torch.Tensor] = {}
        active_positions = memory_token_positions
        active_mask = memory_mask
        active_positive = hop_positive_memory_indices
        select_bias = None  # only set below when reading through the differentiable path

        if use_learned_writer and self.writer is not None:
            capacity = self.config.episodic_capacity
            max_slots = memory_token_positions.size(1) if capacity is None else min(capacity, memory_token_positions.size(1))
            writer_info = self.writer(
                local_state,
                input_ids,
                query_key_positions,
                memory_token_positions,
                memory_mask,
                max_slots=max_slots,
                writer_eps=self.config.writer_diff_select_eps,
                writer_sinkhorn_iters=self.config.writer_diff_select_iters,
                candidate_mode=self.config.writer_candidate_mode,
                surprise_bias=surprise_bias,
                surprise_bias_scale=self.config.jepa_writer_bias_scale,
                surprise_bias_clamp=self.config.jepa_bias_clamp,
            )
            info.update(writer_info)
            if not learned_writer_teacher_forcing:
                # Read over the FULL candidate array with a differentiable
                # selection bias, instead of the hard `slots`-sized gather.
                # This is the actual fix for the tf-release accuracy cliff:
                # it's the first path where "was this the right candidate to
                # select" gets gradient from the real read/answer loss,
                # instead of only from the auxiliary oracle-matching BCE.
                active_positions = writer_info["writer_full_memory_token_positions"]
                active_mask = writer_info["writer_full_valid_mask"]
                select_bias = writer_info["writer_select_bias"]
                active_positive = match_hop_positive_indices(
                    active_positions,
                    active_mask,
                    memory_token_positions,
                    hop_positive_memory_indices,
                )
                active_positive_index, active_positive_mask = match_positive_slots_into_active_space(
                    active_positions,
                    active_mask,
                    memory_token_positions,
                    positive_memory_indices,
                    positive_memory_mask,
                )
                if active_positive_index is not None:
                    info["active_positive_memory_indices"] = active_positive_index
                if active_positive_mask is not None:
                    info["active_positive_memory_mask"] = active_positive_mask

        return active_positions, active_mask, active_positive, select_bias, info

    def _delta_rule_read(
        self,
        *,
        local_state: torch.Tensor,
        input_ids: torch.Tensor,
        query_key_positions: torch.Tensor,
        memory_token_positions: torch.Tensor,
        memory_mask: torch.Tensor,
        surprise_bias: Optional[torch.Tensor],
        positive_memory_indices: Optional[torch.Tensor] = None,
        positive_memory_mask: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Route every candidate token-position pair through
        DeltaRuleEpisodicMemory instead of SupervisedMemoryWriter's hard
        top-k + EpisodicMemory's retrieve_topk gather. See
        HpmLiteV2Config.use_delta_rule_writer's docstring for why.

        Deliberately reuses SupervisedMemoryWriter.forward unmodified (same
        oracle-supervised BCE scorer, same writer_full_memory_token_positions/
        writer_full_valid_mask fields it already computes for the
        Sinkhorn-warmup path) rather than duplicating that logic -- only
        what happens AFTER the scorer runs changes: no torch.topk gather,
        no Sinkhorn straight-through bias, just a budget-capped sigmoid as
        a continuous write_gate into the delta-rule memory (further blended
        with an oracle gate when delta_rule_oracle_blend is on -- see that
        config field's docstring).
        """
        assert self.writer is not None and self.delta_writer is not None  # guarded by config validation in __init__

        capacity = self.config.episodic_capacity
        max_slots = memory_token_positions.size(1) if capacity is None else min(capacity, memory_token_positions.size(1))
        writer_info = self.writer(
            local_state,
            input_ids,
            query_key_positions,
            memory_token_positions,
            memory_mask,
            max_slots=max_slots,
            writer_eps=self.config.writer_diff_select_eps,
            writer_sinkhorn_iters=self.config.writer_diff_select_iters,
            candidate_mode=self.config.writer_candidate_mode,
            surprise_bias=surprise_bias,
            surprise_bias_scale=self.config.jepa_writer_bias_scale,
            surprise_bias_clamp=self.config.jepa_bias_clamp,
        )
        # A raw independent sigmoid gives every one of ~sequence-length
        # candidates weight about 0.5 at initialization. Its total update
        # strength is then O(sequence length), so even a 0.1% blend of that
        # gate into an oracle write can swamp the oracle state. Cap the
        # learned gate's total mass at the configured episodic capacity (or
        # the oracle fact count when uncapped) while keeping every candidate
        # differentiable.
        write_budget = max_slots
        learned_write_gate = budgeted_sigmoid_write_gate(
            writer_info["writer_selection_logits"],
            writer_info["writer_full_valid_mask"],
            write_budget,
        )
        write_gate = learned_write_gate

        if self.config.delta_rule_oracle_blend and teacher_forcing_prob > 0.0:
            # Match the oracle's true fact position(s) into the writer's
            # full-candidate index space (same helper _maybe_select_writer_
            # memory uses for the old writer's differentiable-selection
            # path), producing a {False, True} mask over candidates that is
            # True exactly at the true fact position(s).
            _, oracle_slot_mask = match_positive_slots_into_active_space(
                writer_info["writer_full_memory_token_positions"],
                writer_info["writer_full_valid_mask"],
                memory_token_positions,
                positive_memory_indices,
                positive_memory_mask,
            )
            if oracle_slot_mask is not None:
                oracle_gate = oracle_slot_mask.to(write_gate.dtype)
                p = float(teacher_forcing_prob)
                write_gate = p * oracle_gate + (1.0 - p) * write_gate

        retrieved, delta_info = self.delta_writer(
            local_state,
            writer_info["writer_full_memory_token_positions"],
            writer_info["writer_full_valid_mask"],
            write_gate,
            query_key_positions,
        )
        info: Dict[str, torch.Tensor] = dict(writer_info)
        info.update(delta_info)
        info["learned_write_gate"] = learned_write_gate
        info["write_gate"] = write_gate
        info["write_gate_budget"] = write_gate.new_full((), float(write_budget))
        return retrieved, info

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_token_positions: torch.Tensor,
        memory_mask: torch.Tensor,
        answer_positions: torch.Tensor,
        query_key_positions: torch.Tensor,
        top_k: int = 1,
        task: str = "kv",
        hop_positive_memory_indices: Optional[torch.Tensor] = None,
        positive_memory_indices: Optional[torch.Tensor] = None,
        positive_memory_mask: Optional[torch.Tensor] = None,
        memory_control: str = "normal",
        use_learned_writer: bool = False,
        learned_writer_teacher_forcing: bool = False,
        teacher_forcing_prob: Optional[float] = None,
        router_fixed_weights: Optional[torch.Tensor] = None,
        router_contribution_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run HPM-v2, optionally with a frozen-router causal intervention.

        ``router_fixed_weights`` and ``router_contribution_mask`` are
        evaluation diagnostics. They retain the unablated model's per-token
        gate while removing selected post-gate path contributions; see
        ``HpmV2PathRouter.forward``. They are invalid for ``episodic_only``
        because that configuration has no router to freeze.
        """
        del answer_positions  # The training loop computes loss from logits/targets.
        if self.config.episodic_only and (
            router_fixed_weights is not None or router_contribution_mask is not None
        ):
            raise ValueError("router interventions require episodic_only=False")
        if self.config.use_delta_rule_writer and task in {"twohop", "longhop"}:
            raise ValueError(
                "use_delta_rule_writer=True does not yet support task="
                f"{task!r}: DeltaRuleEpisodicMemory is a single read with no "
                "multi-hop query-refinement loop."
            )
        # teacher_forcing_prob is the continuous scheduled-sampling
        # probability (see train.py's teacher_forcing_probability()) and is
        # only consulted by delta_rule_oracle_blend. Callers that don't pass
        # it (e.g. existing tests/scripts predating that flag) fall back to
        # the coarse {0.0, 1.0} implied by the boolean, so behavior is
        # unchanged unless a caller opts into the continuous value.
        resolved_teacher_forcing_prob = (
            teacher_forcing_prob if teacher_forcing_prob is not None else (1.0 if learned_writer_teacher_forcing else 0.0)
        )

        local_state = self._local_path(input_ids)
        if not self.config.episodic_only:
            recurrent_state = self.selective_recurrent(local_state)
            fast_state = self.fast_memory(local_state)

        token_jepa_info: Dict[str, torch.Tensor] = {}
        surprise_bias = None
        # Token-level JEPA is an auxiliary objective in its own right.  Do
        # not make learning or measuring it contingent on granting it control
        # over the episodic writer.  The latter remains a separate, explicit
        # intervention for a future causal ablation.
        if (self.config.use_token_jepa_aux or self.config.use_jepa_writer_bias) and self.jepa is not None:
            token_jepa_info = self.jepa.token_surprise(local_state)
        if self.config.use_jepa_writer_bias and use_learned_writer:
            surprise_bias = token_jepa_info["token_surprise"]

        use_delta_rule_writer = self.config.use_delta_rule_writer and use_learned_writer and self.writer is not None
        if use_delta_rule_writer:
            episodic_vector, retrieval_info = self._delta_rule_read(
                local_state=local_state,
                input_ids=input_ids,
                query_key_positions=query_key_positions,
                memory_token_positions=memory_token_positions,
                memory_mask=memory_mask,
                surprise_bias=surprise_bias,
                positive_memory_indices=positive_memory_indices,
                positive_memory_mask=positive_memory_mask,
                teacher_forcing_prob=resolved_teacher_forcing_prob,
            )
            retrieval_info.update(token_jepa_info)
        else:
            active_positions, active_mask, active_positive, select_bias, retrieval_info = self._maybe_select_writer_memory(
                local_state=local_state,
                input_ids=input_ids,
                query_key_positions=query_key_positions,
                memory_token_positions=memory_token_positions,
                memory_mask=memory_mask,
                hop_positive_memory_indices=hop_positive_memory_indices,
                positive_memory_indices=positive_memory_indices,
                positive_memory_mask=positive_memory_mask,
                use_learned_writer=use_learned_writer,
                learned_writer_teacher_forcing=learned_writer_teacher_forcing,
                surprise_bias=surprise_bias,
            )
            retrieval_info.update(token_jepa_info)

            num_hops = 2 if task in {"twohop", "longhop"} else 1
            episodic_vector, episodic_info = self.episodic_memory(
                local_state,
                active_positions,
                active_mask,
                query_key_positions,
                top_k=top_k,
                num_hops=num_hops,
                hop_positive_indices=active_positive,
                memory_control=memory_control,
                select_bias=select_bias,
            )
            retrieval_info.update(episodic_info)

        episodic_state = episodic_vector[:, None, :].expand_as(local_state)
        if self.config.episodic_only:
            mixed = episodic_state
        else:
            mixed, router_weights, router_logits, router_diagnostics = self.router.route_with_diagnostics(
                local_state,
                recurrent_state,
                fast_state,
                episodic_state,
                fixed_weights=router_fixed_weights,
                contribution_mask=router_contribution_mask,
            )
            retrieval_info["router_weights"] = router_weights
            retrieval_info["router_logits"] = router_logits
            # Preserve ``router_logits`` as the historical EFFECTIVE
            # pre-softmax tensor so old metrics/losses remain bit-for-bit
            # compatible. The two new fields expose the hidden state needed
            # to distinguish forward softmax concentration from raw-projection
            # optimization lock behind the optional tanh clamp.
            retrieval_info["router_raw_logits"] = router_diagnostics["raw_logits"]
            retrieval_info["router_clamp_jacobian"] = router_diagnostics["clamp_jacobian"]

        if self.jepa is not None:
            jepa_info = self.jepa(block_summaries(local_state, self.config.block_size))
            retrieval_info.update(jepa_info)

        logits = self.answer_head(self.final_ln(mixed))

        if (
            self.config.sinkhorn_warmup
            and not use_delta_rule_writer
            and use_learned_writer
            and learned_writer_teacher_forcing
            and self.writer is not None
            and "writer_select_bias" in retrieval_info
        ):
            # Same writer forward pass already computed the full-candidate
            # differentiable selection (writer_full_memory_token_positions /
            # writer_full_valid_mask / writer_select_bias) whether or not
            # teacher forcing is on -- see SupervisedMemoryWriter.forward.
            # Re-run ONLY the cheap tail (episodic_memory read -> router mix
            # -> answer_head) through that candidate space, so this path's
            # parameters get real answer-loss gradient on THIS step's real
            # batch, not just on the step it first becomes the primary read.
            warm_positions = retrieval_info["writer_full_memory_token_positions"]
            warm_mask = retrieval_info["writer_full_valid_mask"]
            # Detached deliberately: jepa.token_surprise's output already has
            # one trainer during TF -- token_jepa_loss, the self-supervised
            # objective returned alongside it (see JepaLiteAuxiliary.
            # token_surprise's docstring), wired into training as part of
            # jepa_total regardless of TF state. (NOT writer_loss: model.py's
            # own docstring confirms surprise_bias is added to
            # selection_logits only AFTER writer_loss's BCE is already
            # computed from the unbiased logits, so writer_loss never
            # depended on this output at all.) Before this fix, the warmup
            # answer-loss became a SECOND, competing trainer of the same
            # token_surprise output during TF -- pulling it toward "predicts
            # next-token surprise well" and "makes the warm-path answer come
            # out right" at once, which don't have to agree, especially
            # early when router/answer_head haven't learned to use a warm
            # read yet. That second pull only exists when jepa_bias is on:
            # with it off, self.jepa.token_surprise is never called (see the
            # guard at "if self.config.use_jepa_writer_bias and ..." above),
            # so there's nothing to compete over. This is the reason the
            # jepa_bias_on/off split in the clamped+ramped sweep (off
            # improved on all 3 seeds, on regressed on 2/3, on_seed0
            # specifically failed to converge and posted the worst shock/
            # grad/eval_exact of any run) tracked jepa_bias exactly.
            # Detaching keeps the warm path reading through the SAME
            # selection decision (unchanged positions/magnitude) as an
            # input, while leaving token_jepa_loss as token_surprise's only
            # trainer during TF, same as before this fix existed.
            # router/episodic_memory/answer_head -- the warmup mechanism's
            # actual intended target -- are untouched: detach only cuts the
            # graph upstream of warm_bias, not downstream of it.
            warm_bias = retrieval_info["writer_select_bias"].detach()
            warm_positive = match_hop_positive_indices(
                warm_positions,
                warm_mask,
                memory_token_positions,
                hop_positive_memory_indices,
            )
            warm_vector, warm_info = self.episodic_memory(
                local_state,
                warm_positions,
                warm_mask,
                query_key_positions,
                top_k=top_k,
                num_hops=num_hops,
                hop_positive_indices=warm_positive,
                memory_control=memory_control,
                select_bias=warm_bias,
            )
            warm_state = warm_vector[:, None, :].expand_as(local_state)
            if self.config.episodic_only:
                warm_mixed = warm_state
            else:
                warm_mixed, _, _ = self.router(local_state, recurrent_state, fast_state, warm_state)
            retrieval_info["sinkhorn_warmup_logits"] = self.answer_head(self.final_ln(warm_mixed))
            retrieval_info["sinkhorn_warmup_retrieval_loss"] = warm_info.get("retrieval_loss")

        return {"logits": logits, "retrieval": retrieval_info}
