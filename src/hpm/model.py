from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .data import ANSWER, FACT, QUERY, VOCAB_SIZE
from .memory import EpisodicMemory, HebbianMemory
from .differentiable_topk import differentiable_topk_bias


def make_local_causal_mask(seq_len: int, window: int, device: torch.device | None = None) -> torch.Tensor:
    """Return [T, T] mask where row t can attend only max(0, t-W)..t."""

    idx = torch.arange(seq_len, device=device)
    query = idx[:, None]
    key = idx[None, :]
    return (key <= query) & ((query - key) <= window)


class LocalCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # True sliding-window causal attention.
        #
        # The previous implementation formed a dense [B, H, T, T] score
        # matrix and then masked out non-local positions. That is logically
        # local attention, but its memory cost is still quadratic in T. At
        # T=4096, B=32, H=4, the score tensor alone is about 8 GiB.
        #
        # This chunked gather computes only the last ``window`` keys for each
        # query, so attention memory is O(B * H * T * window), not O(B*H*T*T).
        radius = min(max(int(self.window), 0), max(seq_len - 1, 0))
        chunk_size = 512
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)

        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            chunk_len = end - start
            positions = torch.arange(start, end, device=x.device)
            offsets = torch.arange(radius, -1, -1, device=x.device)
            key_positions = positions[:, None] - offsets[None, :]
            valid = key_positions >= 0
            gather_positions = key_positions.clamp_min(0)

            gather_index = gather_positions[None, None, :, :, None].expand(
                bsz, self.heads, chunk_len, radius + 1, self.head_dim
            )
            k_chunk = torch.gather(
                k.unsqueeze(2).expand(bsz, self.heads, chunk_len, seq_len, self.head_dim),
                dim=3,
                index=gather_index,
            )
            v_chunk = torch.gather(
                v.unsqueeze(2).expand(bsz, self.heads, chunk_len, seq_len, self.head_dim),
                dim=3,
                index=gather_index,
            )

            q_chunk = q[:, :, start:end, :]
            scores = torch.sum(q_chunk.unsqueeze(-2) * k_chunk, dim=-1) * scale
            scores = scores.masked_fill(~valid[None, None, :, :], torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            outputs.append(torch.sum(attn.unsqueeze(-1) * v_chunk, dim=-2))

        y = torch.cat(outputs, dim=2)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LocalCausalSelfAttention(d_model, heads, window, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleRecurrentSummary(nn.Module):
    """EMA summary over completed blocks only, kept for the old recurrent baseline."""

    def __init__(self, d_model: int, block_size: int, decay: float = 0.9):
        super().__init__()
        self.block_size = block_size
        self.decay = decay
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = hidden.shape
        state = hidden.new_zeros(bsz, d_model)
        summaries = hidden.new_zeros(hidden.shape)
        for start in range(0, seq_len, self.block_size):
            end = min(seq_len, start + self.block_size)
            summaries[:, start:end, :] = state[:, None, :]
            block_summary = hidden[:, start:end, :].mean(dim=1)
            state = self.decay * state + (1.0 - self.decay) * block_summary
        return self.proj(summaries)


class GruRecurrentState(nn.Module):
    """Causal recurrent memory path: x_t -> GRU state r_t."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(self.ln(x))
        return self.out(self.dropout(states))



class SupervisedMemoryWriter(nn.Module):
    """Supervised first-stage writer for removing oracle memory writes.

    The writer predicts which pre-query token positions should start a memory
    slot. For the current synthetic KV task, a selected start position ``p``
    writes the pair ``[p, p + 1]`` as key/value token positions. The selection
    itself is hard top-k and therefore not differentiated through; the writer
    is trained with a BCE objective against synthetic support labels. This is
    intentionally a bridge between parser/oracle writes and a later fully
    autonomous write policy.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        query_key_positions: torch.Tensor,
        oracle_memory_token_positions: torch.Tensor,
        oracle_memory_mask: torch.Tensor,
        max_slots: int,
        writer_eps: float = 0.3,
        writer_sinkhorn_iters: int = 50,
        candidate_mode: str = "all_prequery",
        surprise_bias: Optional[torch.Tensor] = None,
        surprise_bias_scale: float = 1.0,
        surprise_bias_clamp: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """``surprise_bias``, if given, is a ``[batch, seq_len]``-or-longer
        per-position score (e.g. ``JepaLiteAuxiliary.token_surprise``) that
        gets added -- scaled by ``surprise_bias_scale`` -- on top of the
        writer's own scorer logits *only* at the selection stage (top-k and
        the differentiable Sinkhorn bias below). It intentionally does not
        touch ``logits`` before the BCE writer_loss is computed: the oracle
        supervision signal stays exactly as it was, and surprise only shifts
        which candidates get chosen, not what the scorer is trained to
        predict.

        ``surprise_bias_clamp``, if set, smoothly bounds the scaled bias term
        to ``[-surprise_bias_clamp, +surprise_bias_clamp]`` via tanh before
        it's added to ``logits`` -- the same architectural pattern as
        ``HpmV2PathRouter``'s ``logit_clamp``. Default None = off, matches all
        prior behavior/results exactly (identical to the unclamped
        ``logits + surprise_bias_scale * bias``). Added because
        ``jepa_bias_on`` vs ``jepa_bias_off`` on an otherwise-identical seed
        showed the JEPA surprise term can, for some seeds, dominate the
        oracle-supervised selection logits enough to destabilize which
        candidates get chosen step to step (elevated candidate-distribution
        entropy/KL, reduced top-k Jaccard overlap vs the immediately prior
        probe step, and a correspondingly non-converging
        ``sinkhorn_warmup_loss``) without improving that seed's writer_loss --
        i.e. the bias term was overriding the scorer's own oracle-trained
        signal rather than refining it. Unlike ``surprise_bias_scale`` (a
        fixed multiplier that still leaves the bias's raw magnitude
        unbounded), this is an unconditional bound: regardless of how large
        ``token_surprise`` gets for a given seed/step, its contribution to
        selection can never exceed ``surprise_bias_clamp`` in magnitude, so
        it can shift close calls but can no longer overwhelm a
        confidently-correct oracle-trained score. Try a value on the order of
        the scorer logits' own typical magnitude (start around 3.0, matching
        --router-logit-clamp's own default suggestion).
        """
        bsz, seq_len, _ = hidden.shape
        device = hidden.device
        candidate_len = max(seq_len - 1, 1)
        logits = self.scorer(hidden[:, :candidate_len, :]).squeeze(-1)

        positions = torch.arange(candidate_len, device=device)[None, :]
        query_positions = query_key_positions[:, None] - 1
        valid = (positions + 1 < query_positions) & (input_ids[:, :candidate_len] != QUERY) & (input_ids[:, :candidate_len] != ANSWER)
        if candidate_mode == "fact_pairs":
            # The clean synthetic grammar provides a deterministic proposal
            # rule: `[FACT, key, value, SEP]` starts a candidate pair at the
            # key token.  Proposal and learned slot selection are different
            # problems.  Restricting the proposal set here lets a benchmark
            # test the latter without silently asking the scorer to discover
            # every possible adjacent pair in a long noise stream.
            preceded_by_fact = torch.zeros_like(valid)
            if candidate_len > 1:
                preceded_by_fact[:, 1:] = input_ids[:, : candidate_len - 1].eq(FACT)
            valid = valid & preceded_by_fact
        elif candidate_mode != "all_prequery":
            raise ValueError(f"unknown writer candidate_mode: {candidate_mode!r}")

        labels = torch.zeros_like(logits)
        starts = oracle_memory_token_positions[:, :, 0]
        safe_starts = starts.clamp(0, candidate_len - 1)
        labels.scatter_(1, safe_starts, oracle_memory_mask.float())
        labels = labels * valid.float()

        if valid.any():
            valid_labels = labels[valid]
            valid_logits = logits[valid]
            pos = valid_labels.sum().clamp_min(1.0)
            neg = (valid_labels.numel() - valid_labels.sum()).clamp_min(1.0)
            pos_weight = (neg / pos).detach().clamp(1.0, 100.0)
            writer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                valid_logits,
                valid_labels,
                pos_weight=pos_weight,
            )
        else:
            writer_loss = logits.new_zeros(())

        selection_logits = logits
        if surprise_bias is not None:
            bias = surprise_bias[:, :candidate_len].to(logits.dtype)
            scaled_bias = surprise_bias_scale * bias
            if surprise_bias_clamp is not None:
                scaled_bias = surprise_bias_clamp * torch.tanh(scaled_bias / surprise_bias_clamp)
            selection_logits = logits + scaled_bias

        masked_logits = selection_logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        slots = min(max_slots, candidate_len)
        top_logits, starts = torch.topk(masked_logits, k=slots, dim=-1)
        memory_mask = top_logits > torch.finfo(logits.dtype).min / 2

        # Differentiable selection bias over the FULL candidate_len array
        # (not the gathered `slots`-sized subset below). This is what lets a
        # downstream read/answer loss shape *which* candidates get selected,
        # instead of only the auxiliary oracle-matching BCE above. Forward
        # value is bit-identical to masking with the hard top-`slots` set
        # (STE); only the backward pass differs. `slots < 1` (degenerate,
        # e.g. candidate_len==0 edge case already guarded by max(seq_len-1,1)
        # upstream) is not expected in practice, but guard anyway.
        if 1 <= slots < candidate_len:
            # IMPORTANT: do not feed `masked_logits` (invalid = torch.finfo.min,
            # ~-3.4e38 for float32) into the Sinkhorn bias. Unlike torch.topk,
            # which only ever compares that sentinel, the OT anchors do real
            # arithmetic on the score range: spread = max - min ~= 3.4e38, then
            # anchor = min - margin*spread overflows past float32's range to
            # -inf, which poisons logsumexp into NaN (-inf - (-inf) = NaN).
            # Use a finite, bounded placeholder instead: safely below any real
            # score by a fixed margin, but not near the dtype's limit. It's
            # discarded anyway!!! `select_bias` is force-set to -1e9 for
            # invalid positions two lines below regardless of what the
            # Sinkhorn step assigns them internally.
            with torch.no_grad():
                real_scores = selection_logits.masked_fill(~valid, float("nan"))
                row_max = torch.nan_to_num(real_scores, nan=-1.0e4).max(dim=-1, keepdim=True).values
            sinkhorn_floor = row_max - 50.0
            sinkhorn_input = torch.where(valid, selection_logits, sinkhorn_floor.expand_as(logits))
            select_bias = differentiable_topk_bias(sinkhorn_input, k=slots, eps=writer_eps, n_iters=writer_sinkhorn_iters)
        else:
            # slots == candidate_len: nothing is actually excluded, bias is all-zero.
            select_bias = torch.zeros_like(masked_logits)
        # Structurally-invalid positions (query/answer tokens, out-of-range)
        # stay hard-excluded regardless of score -- that's a fact about the
        # sequence, not a selection decision, so it must not be differentiable.
        select_bias = torch.where(valid, select_bias, masked_logits.new_full((), -1.0e4))
        full_positions = torch.stack(
            [torch.arange(candidate_len, device=device), (torch.arange(candidate_len, device=device) + 1).clamp_max(seq_len - 1)],
            dim=-1,
        )[None, :, :].expand(bsz, candidate_len, 2)
        if slots < max_slots:
            pad_slots = max_slots - slots
            starts = torch.cat([starts, starts.new_zeros(bsz, pad_slots)], dim=1)
            memory_mask = torch.cat([memory_mask, torch.zeros(bsz, pad_slots, device=device, dtype=torch.bool)], dim=1)
            top_logits = torch.cat([top_logits, top_logits.new_full((bsz, pad_slots), torch.finfo(logits.dtype).min)], dim=1)

        memory_token_positions = torch.stack([starts, (starts + 1).clamp_max(seq_len - 1)], dim=-1)
        write_probs = torch.sigmoid(logits)
        predicted_positive = (write_probs >= 0.5) & valid
        tp = ((predicted_positive & (labels.bool()))).float().sum()
        fp = ((predicted_positive & ~(labels.bool()))).float().sum()
        fn = (((~predicted_positive) & labels.bool())).float().sum()

        return {
            "writer_logits": logits,
            "writer_selection_logits": selection_logits,
            "writer_labels": labels,
            "writer_valid_mask": valid,
            "writer_loss": writer_loss,
            "writer_memory_token_positions": memory_token_positions,
            "writer_memory_mask": memory_mask,
            "writer_top_logits": top_logits,
            "writer_precision": tp / (tp + fp).clamp_min(1.0),
            "writer_recall": tp / (tp + fn).clamp_min(1.0),
            # Full-candidate_len differentiable selection path (new). Unused
            # unless the caller opts in (see hpm_v2_model._maybe_select_writer_memory) --
            # every field above this comment is untouched and behaves exactly
            # as before for any existing caller.
            "writer_full_memory_token_positions": full_positions,
            "writer_full_valid_mask": valid,
            "writer_select_bias": select_bias,
        }


def match_hop_positive_indices(
    active_memory_token_positions: torch.Tensor,
    active_memory_mask: torch.Tensor,
    oracle_memory_token_positions: torch.Tensor,
    oracle_hop_positive_memory_indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Map oracle support slots to slots selected by the learned writer."""

    if oracle_hop_positive_memory_indices is None:
        return None
    matched = torch.full_like(oracle_hop_positive_memory_indices, -100)
    bsz, slots, _ = active_memory_token_positions.shape
    for b in range(bsz):
        for hop in range(oracle_hop_positive_memory_indices.size(1)):
            oracle_slot = int(oracle_hop_positive_memory_indices[b, hop].item())
            if oracle_slot < 0:
                continue
            oracle_pair = oracle_memory_token_positions[b, oracle_slot]
            for slot in range(slots):
                if not bool(active_memory_mask[b, slot].item()):
                    continue
                if torch.equal(active_memory_token_positions[b, slot], oracle_pair):
                    matched[b, hop] = slot
                    break
    return matched


def match_positive_slots_into_active_space(
    active_memory_token_positions: torch.Tensor,
    active_memory_mask: torch.Tensor,
    oracle_memory_token_positions: torch.Tensor,
    oracle_positive_memory_indices: Optional[torch.Tensor],
    oracle_positive_memory_mask: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Re-express the oracle's single-index/multi-slot positive labels in
    whatever candidate space the model actually retrieved over.

    `retrieval_metrics`/`retrieval_correct_mask` compare `positive_indices`/
    `positive_mask` directly against `output["retrieval"]["top_indices"]`.
    That's only valid when both live in the same index space. When the
    learned writer's full-candidate differentiable-selection path is active
    (`learned_writer_teacher_forcing=False`), `top_indices` indexes into
    `writer_full_memory_token_positions` (size ~seq_len), not into the
    oracle's own `memory_token_positions` (size num_facts) that
    `positive_memory_indices`/`positive_memory_mask` were built against.
    Comparing them as-is is an index-space mismatch, not just a shape
    mismatch -- it can silently score wrong, or (as here) throw once
    `top_indices` contains a value past the oracle-sized array's bounds.

    This mirrors `match_hop_positive_indices`'s pair-equality matching, but
    for the plain single-index/multi-slot fields instead of the multi-hop
    ones.
    """

    if oracle_positive_memory_indices is None and oracle_positive_memory_mask is None:
        return None, None

    bsz, active_slots, _ = active_memory_token_positions.shape

    matched_index = None
    if oracle_positive_memory_indices is not None:
        matched_index = torch.full_like(oracle_positive_memory_indices, -1)
        for b in range(bsz):
            oracle_slot = int(oracle_positive_memory_indices[b].item())
            if oracle_slot < 0:
                continue
            oracle_pair = oracle_memory_token_positions[b, oracle_slot]
            for slot in range(active_slots):
                if not bool(active_memory_mask[b, slot].item()):
                    continue
                if torch.equal(active_memory_token_positions[b, slot], oracle_pair):
                    matched_index[b] = slot
                    break

    matched_mask = None
    if oracle_positive_memory_mask is not None:
        matched_mask = torch.zeros(
            bsz, active_slots, dtype=torch.bool, device=active_memory_token_positions.device
        )
        for b in range(bsz):
            for oracle_slot in range(oracle_positive_memory_mask.size(1)):
                if not bool(oracle_positive_memory_mask[b, oracle_slot].item()):
                    continue
                oracle_pair = oracle_memory_token_positions[b, oracle_slot]
                for slot in range(active_slots):
                    if not bool(active_memory_mask[b, slot].item()):
                        continue
                    if torch.equal(active_memory_token_positions[b, slot], oracle_pair):
                        matched_mask[b, slot] = True
                        break

    return matched_index, matched_mask


class MemoryPathRouter(nn.Module):
    """alpha = softmax(W[l_t, r_t, e_t]); m_t = sum_i alpha_i path_i."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(3 * d_model, 3)

    def forward(
        self,
        local_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        episodic_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(torch.cat([local_state, recurrent_state, episodic_state], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        mixed = (
            weights[..., 0:1] * local_state
            + weights[..., 1:2] * recurrent_state
            + weights[..., 2:3] * episodic_state
        )
        return mixed, weights


@dataclass
class HpmLiteConfig:
    model_type: str = "local"
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    window: int = 64
    max_seq_len: int = 2048
    dropout: float = 0.0
    hebbian_decay: float = 0.9
    hebbian_eta: float = 1.0
    use_null_slot: bool = False
    null_score_init: float = 0.0
    use_learned_writer: bool = False


@dataclass
class AnswerTransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    layers: int = 1
    heads: int = 4
    window: int = 64
    max_seq_len: int = 2048
    dropout: float = 0.0


class AnswerTransformerModel(nn.Module):
    """Small causal local-window Transformer used as a no-memory answer baseline."""

    def __init__(self, config: AnswerTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.window, config.dropout) for _ in range(config.layers)]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.answer_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.answer_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.answer_head(self.final_ln(x))


class HpmLiteModel(nn.Module):
    def __init__(self, config: HpmLiteConfig):
        super().__init__()
        if config.model_type not in {"local", "recurrent", "epmem", "hpm_lite", "hebbian"}:
            raise ValueError(f"unknown model_type: {config.model_type}")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.window, config.dropout) for _ in range(config.layers)]
        )

        # Legacy baseline: local mixer + simple recurrent summary.
        self.recurrent = (
            SimpleRecurrentSummary(config.d_model, block_size=config.window)
            if config.model_type == "recurrent"
            else None
        )

        # The actual screenshot model: local mixer, GRU state, episodic retrieval, learned router.
        self.hpm_gru = GruRecurrentState(config.d_model, config.dropout) if config.model_type == "hpm_lite" else None
        self.router = MemoryPathRouter(config.d_model) if config.model_type == "hpm_lite" else None
        self.writer = SupervisedMemoryWriter(config.d_model) if config.use_learned_writer and config.model_type in {"epmem", "hpm_lite"} else None

        if config.model_type in {"epmem", "hpm_lite"}:
            self.memory = EpisodicMemory(
                config.d_model,
                use_null_slot=config.use_null_slot,
                null_score_init=config.null_score_init,
            )
        elif config.model_type == "hebbian":
            self.memory = HebbianMemory(config.d_model, decay=config.hebbian_decay, eta=config.hebbian_eta)
        else:
            self.memory = None

        # Kept for old diagnostic variants; hpm_lite uses router weights instead.
        self.gamma_e = nn.Parameter(torch.tensor(1.0))
        self.gamma_r = nn.Parameter(torch.tensor(0.5))
        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _retrieve_answer_memory(
        self,
        token_emb: torch.Tensor,
        memory_token_positions: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        query_key_positions: Optional[torch.Tensor],
        top_k: int,
        task: str,
        hop_positive_memory_indices: Optional[torch.Tensor],
        memory_control: str,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.memory is None:
            raise RuntimeError("_retrieve_answer_memory called without a memory module")
        if memory_token_positions is None or memory_mask is None or query_key_positions is None:
            raise ValueError("memory models require memory positions and query key positions")
        num_hops = 2 if task in {"twohop", "longhop"} else 1
        return self.memory(
            token_embeddings=token_emb,
            memory_token_positions=memory_token_positions,
            memory_mask=memory_mask,
            query_key_positions=query_key_positions,
            top_k=top_k,
            num_hops=num_hops,
            hop_positive_indices=hop_positive_memory_indices,
            memory_control=memory_control,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_token_positions: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        answer_positions: Optional[torch.Tensor] = None,
        query_key_positions: Optional[torch.Tensor] = None,
        top_k: int = 1,
        task: str = "kv",
        hop_positive_memory_indices: Optional[torch.Tensor] = None,
        memory_control: str = "normal",
        use_learned_writer: bool = False,
        learned_writer_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)
        token_emb = self.token_embedding(input_ids)
        embedded = token_emb + self.position_embedding(positions)[None, :, :]

        local_state = embedded
        for block in self.blocks:
            local_state = block(local_state)

        retrieval_info: Dict[str, torch.Tensor] = {}
        writer_info: Dict[str, torch.Tensor] = {}
        active_memory_token_positions = memory_token_positions
        active_memory_mask = memory_mask
        active_hop_positive_memory_indices = hop_positive_memory_indices

        if use_learned_writer:
            if self.writer is None:
                raise RuntimeError("use_learned_writer=True requires HpmLiteConfig(use_learned_writer=True)")
            if memory_token_positions is None or memory_mask is None or query_key_positions is None:
                raise ValueError("learned writer requires oracle memory positions for supervised labels")
            writer_info = self.writer(
                hidden=local_state,
                input_ids=input_ids,
                query_key_positions=query_key_positions,
                oracle_memory_token_positions=memory_token_positions,
                oracle_memory_mask=memory_mask,
                max_slots=memory_token_positions.size(1),
            )
            if not learned_writer_teacher_forcing:
                active_memory_token_positions = writer_info["writer_memory_token_positions"]
                active_memory_mask = writer_info["writer_memory_mask"]
                active_hop_positive_memory_indices = match_hop_positive_indices(
                    active_memory_token_positions=active_memory_token_positions,
                    active_memory_mask=active_memory_mask,
                    oracle_memory_token_positions=memory_token_positions,
                    oracle_hop_positive_memory_indices=hop_positive_memory_indices,
                )

        state = local_state

        if self.config.model_type == "recurrent" and self.recurrent is not None:
            state = local_state + self.gamma_r * self.recurrent(local_state)

        elif self.config.model_type == "hpm_lite":
            if self.hpm_gru is None or self.router is None or self.memory is None:
                raise RuntimeError("hpm_lite model was not initialized correctly")
            if answer_positions is None:
                raise ValueError("hpm_lite requires answer_positions so episodic retrieval can be routed at the answer token")

            recurrent_state = self.hpm_gru(embedded)
            episodic_state = torch.zeros_like(local_state)
            retrieved, memory_info = self._retrieve_answer_memory(
                token_emb=token_emb,
                memory_token_positions=active_memory_token_positions,
                memory_mask=active_memory_mask,
                query_key_positions=query_key_positions,
                top_k=top_k,
                task=task,
                hop_positive_memory_indices=active_hop_positive_memory_indices,
                memory_control=memory_control,
            )
            batch = torch.arange(bsz, device=input_ids.device)
            episodic_state[batch, answer_positions] = retrieved
            state, router_weights = self.router(local_state, recurrent_state, episodic_state)
            retrieval_info = {**writer_info, **memory_info}
            retrieval_info["router_weights"] = router_weights

        elif self.memory is not None:
            if answer_positions is None:
                raise ValueError("memory models require answer_positions")
            retrieved, memory_info = self._retrieve_answer_memory(
                token_emb=token_emb,
                memory_token_positions=active_memory_token_positions,
                memory_mask=active_memory_mask,
                query_key_positions=query_key_positions,
                top_k=top_k,
                task=task,
                hop_positive_memory_indices=active_hop_positive_memory_indices,
                memory_control=memory_control,
            )
            retrieval_info = {**writer_info, **memory_info}
            batch = torch.arange(bsz, device=input_ids.device)
            state = local_state.clone()
            state[batch, answer_positions] = state[batch, answer_positions] + self.gamma_e * retrieved

        logits = self.lm_head(self.final_ln(state))
        return {"logits": logits, "retrieval": retrieval_info}
