"""Composite multi-skill benchmark for HPM-Lite's four-path ablation.

Purpose
-------
``FactRecallDataset`` (data.py) tests exactly one skill: exact rare-fact
recall under distraction. The existing ``seed_sweep_ablation.csv`` result
(episodic path load-bearing, the other three within noise of zero) was run
entirely on that single-skill task -- it cannot distinguish "the other three
paths are useless" from "this task never asked them to do anything." This
module builds a task where all four proposed HPM paths are independently
necessary, so ablating any one of them predicts a distinct, separable drop
in ITS OWN sub-score, rather than a shared drop in one blended number.

Four sub-tasks, one sequence, four independently-scored answers
-----------------------------------------------------------------
1. COPY -- repeat the last ``copy_span`` tokens verbatim. The source span
   sits inside the local attention window at query time by construction.
   Isolates the LOCAL path: nothing here needs compression or cross-token
   binding, so no other path should be able to solve it better.

2. MOOD -- a block of ``mood_block_len`` tokens, each independently drawn
   as a POS or NEG marker (with configurable label noise), positioned so
   it has aged out of the local window by query time. The question asks
   for the MAJORITY marker across the whole block. No single token
   answers this -- the whole block must be compressed into one running
   summary. Isolates RECURRENT / FAST-WEIGHT: a single exact episodic
   slot cannot answer a majority-vote question over dozens of tokens
   without effectively re-implementing a summary itself.

3. FACT -- identical mechanic to ``FactRecallDataset``'s "kv" task: one
   planted key/value pair plus ``num_facts - 1`` distractor pairs, all
   outside the window at query time. Isolates EPISODIC: this is the one
   sub-task the existing ablation already shows depends on it.

4. XREF -- a question answerable only by combining sub-tasks 2 and 3's
   outputs: "if the mood block's majority was POS, answer with the
   planted fact's value; otherwise answer NO_VALUE." Getting XREF right
   without correctly solving both MOOD and FACT is only possible by
   chance, so XREF score is a direct read on whether the ROUTER is
   actually combining paths rather than leaning on one and ignoring the
   rest.

Grading
-------
Each sub-task gets its own loss mask, its own answer-target position, and
its own answer token, returned as ``{subtask}_loss_mask``,
``{subtask}_answer_position``, ``{subtask}_answer_token`` for
subtask in {"copy", "mood", "fact", "xref"} -- so an eval loop can compute
four independent exact-match scores per batch instead of one blended one.
See ``composite_eval_hint()`` at the bottom of this file for the exact
per-subtask accuracy computation an evaluate.py-style loop should run.

Vocabulary
----------
Reuses data.py's PAD/BOS/EOS/QUERY/ANSWER/SEP/FACT/NO_VALUE tokens and
numeric ranges (NOISE_RANGE, KEY_RANGE, VALUE_RANGE) so this task can share
a model/vocab with the existing FactRecallDataset tasks. Five new special
tokens are added past data.py's VOCAB_SIZE=480 ceiling; COMPOSITE_VOCAB_SIZE
below is what model.vocab_size must be set to when training on this task.
FactRecallDataset tasks remain valid at the smaller VOCAB_SIZE too, since
these new ids are simply unused by them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .data import ANSWER, BOS, EOS, FACT, KEY_RANGE, NO_VALUE, NOISE_RANGE, QUERY, SEP, VALUE_RANGE, VOCAB_SIZE

# New special tokens, past data.py's existing VOCAB_SIZE=480 ceiling.
COPY_MARKER = 480      # delimits the literal span COPY must repeat back
MOOD_QUERY_TOK = 481   # "what was the block's majority mood" query marker
XREF_QUERY_TOK = 482   # "combine mood + fact" query marker
MOOD_POS = 483         # one "positive" mood-block token
MOOD_NEG = 484         # one "negative" mood-block token
COMPOSITE_VOCAB_SIZE = 490

SUBTASKS = ("copy", "mood", "fact", "xref")


@dataclass
class CompositeSkillConfig:
    seq_len: int = 768
    window: int = 64
    seed: int = 0
    copy_span: int = 8
    mood_block_len: int = 40
    mood_label_noise: float = 0.0
    """Fraction of mood-block tokens that are the MINORITY marker, drawn
    independently per token. 0.0 = every token in the block agrees with
    the block's majority label (easiest); higher values make the
    majority-vote question genuinely require aggregating the whole block
    rather than reading any single early token as a shortcut."""
    num_facts: int = 4
    """Total planted key/value pairs (1 positive + num_facts-1 distractors),
    same semantics as FactRecallConfig.num_facts."""


class CompositeSkillDataset:
    """Synthetic four-sub-task batches: COPY, MOOD, FACT, XREF in one
    sequence, each independently scored. See module docstring."""

    def __init__(self, config: CompositeSkillConfig):
        if config.num_facts < 2:
            raise ValueError("num_facts must be >= 2 (>=1 distractor + 1 positive)")
        if not 0.0 <= config.mood_label_noise < 0.5:
            raise ValueError("mood_label_noise must be in [0.0, 0.5)")
        if config.mood_block_len < 4:
            raise ValueError("mood_block_len too short for a meaningful majority-vote question")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.sample_index = 0

    # ------------------------------------------------------------------
    # Batch API, matches FactRecallDataset.sample_batch's shape/dtype
    # conventions so the two datasets are drop-in-compatible in a training
    # loop that branches on task name.
    # ------------------------------------------------------------------
    def sample_batch(self, batch_size: int, device: torch.device | str | None = None) -> Dict[str, torch.Tensor]:
        samples = [self._sample_one() for _ in range(batch_size)]
        batch: Dict[str, torch.Tensor] = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key == "loss_mask" or key.endswith("_loss_mask"):
                tensor = torch.tensor(np.stack(values), dtype=torch.float32)
            elif key == "memory_mask" or key == "positive_memory_mask":
                tensor = torch.tensor(np.stack(values), dtype=torch.bool)
            else:
                tensor = torch.tensor(np.stack(values), dtype=torch.long)
            if device is not None:
                tensor = tensor.to(device)
            batch[key] = tensor
        return batch

    # ------------------------------------------------------------------
    def _sample_one(self) -> Dict[str, np.ndarray]:
        cfg = self.config
        self.sample_index += 1
        rng = self.rng

        tokens: List[int] = [BOS]

        # --- MOOD block -------------------------------------------------
        majority_is_pos = bool(rng.integers(0, 2))
        majority_tok = MOOD_POS if majority_is_pos else MOOD_NEG
        minority_tok = MOOD_NEG if majority_is_pos else MOOD_POS
        mood_labels = rng.random(cfg.mood_block_len) >= cfg.mood_label_noise
        mood_block = [majority_tok if is_majority else minority_tok for is_majority in mood_labels]
        tokens.extend(mood_block)

        # --- FACT block (same mechanic as data.py's "kv" task) ----------
        keys = self._sample_unique(KEY_RANGE, cfg.num_facts)
        values = self._sample_unique(VALUE_RANGE, cfg.num_facts)
        positive_slot = int(rng.integers(0, cfg.num_facts))
        fact_query_key = int(keys[positive_slot])
        fact_answer_value = int(values[positive_slot])
        for i in range(cfg.num_facts):
            tokens.extend([FACT, int(keys[i]), int(values[i]), SEP])

        # --- COPY source span (kept close to the tail so it is inside the
        #     local window at every sub-task's query position) -----------
        copy_literal = self._sample_unique(NOISE_RANGE, cfg.copy_span).tolist()
        tokens.append(COPY_MARKER)
        tokens.extend(copy_literal)
        tokens.append(COPY_MARKER)

        # --- pad with generic noise up to the fixed pre-tail budget -----
        tail_tokens, tail_meta = self._build_tail(fact_query_key, fact_answer_value, copy_literal, majority_is_pos)
        gap_len = cfg.seq_len + 1 - len(tokens) - len(tail_tokens)
        if gap_len < 0:
            raise ValueError("seq_len too short for the requested copy_span/mood_block_len/num_facts")
        gap = rng.integers(NOISE_RANGE[0], NOISE_RANGE[1], size=gap_len).tolist()
        tail_start = len(tokens) + gap_len
        tokens = tokens + gap + tail_tokens

        if len(tokens) != cfg.seq_len + 1:
            raise AssertionError("internal sequence construction bug")

        mood_end = 1 + cfg.mood_block_len
        fact_end = mood_end + cfg.num_facts * 4
        first_query_position = tail_start
        if fact_end >= first_query_position - cfg.window:
            raise ValueError(
                "mood/fact spans fall inside the local window at query time; "
                "increase seq_len or reduce window/mood_block_len/num_facts"
            )

        input_ids = np.asarray(tokens[:-1], dtype=np.int64)
        target_ids = np.asarray(tokens[1:], dtype=np.int64)

        out: Dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }
        loss_mask = np.zeros(cfg.seq_len, dtype=np.float32)
        for name in SUBTASKS:
            # tail_meta stores the tail-relative index of the answer token
            # itself. Convert to a global index in `tokens`, then shift by
            # -1 to land in target_ids/loss_mask space, since target_ids[j]
            # == tokens[j+1] (next-token prediction) -- same convention
            # data.py's FactRecallDataset uses for answer_target_positions.
            pos = tail_start + tail_meta[f"{name}_answer_position"] - 1
            tok = tail_meta[f"{name}_answer_token"]
            m = np.zeros(cfg.seq_len, dtype=np.float32)
            m[pos] = 1.0
            loss_mask[pos] = 1.0
            out[f"{name}_loss_mask"] = m
            out[f"{name}_answer_position"] = np.asarray(pos, dtype=np.int64)
            out[f"{name}_answer_token"] = np.asarray(tok, dtype=np.int64)
        out["loss_mask"] = loss_mask  # combined mask, for code paths that want one number
        out["mood_majority_is_pos"] = np.asarray(int(majority_is_pos), dtype=np.int64)
        out["fact_query_key"] = np.asarray(fact_query_key, dtype=np.int64)
        return out

    # ------------------------------------------------------------------
    def _build_tail(
        self, fact_query_key: int, fact_answer_value: int, copy_literal: List[int], majority_is_pos: bool
    ) -> Tuple[List[int], Dict[str, int]]:
        """Four consecutive QUERY/ANSWER segments, one per sub-task, each
        answered by a single token so exact-match scoring is a plain
        argmax-vs-target comparison, matching data.py's single-token
        answer convention."""
        xref_answer = fact_answer_value if majority_is_pos else NO_VALUE

        tail: List[int] = []
        meta: Dict[str, int] = {}

        # COPY: repeat the first token of the literal span (single-token
        # answer keeps grading identical in shape to the other sub-tasks;
        # the model still must have the exact span in reach to answer).
        tail.extend([QUERY, COPY_MARKER, ANSWER])
        meta["copy_answer_position"] = len(tail)
        tail.append(copy_literal[0])
        meta["copy_answer_token"] = copy_literal[0]
        tail.append(SEP)

        # MOOD
        tail.extend([QUERY, MOOD_QUERY_TOK, ANSWER])
        meta["mood_answer_position"] = len(tail)
        mood_answer = MOOD_POS if majority_is_pos else MOOD_NEG
        tail.append(mood_answer)
        meta["mood_answer_token"] = mood_answer
        tail.append(SEP)

        # FACT
        tail.extend([QUERY, fact_query_key, ANSWER])
        meta["fact_answer_position"] = len(tail)
        tail.append(fact_answer_value)
        meta["fact_answer_token"] = fact_answer_value
        tail.append(SEP)

        # XREF
        tail.extend([QUERY, XREF_QUERY_TOK, ANSWER])
        meta["xref_answer_position"] = len(tail)
        tail.append(xref_answer)
        meta["xref_answer_token"] = xref_answer
        tail.append(EOS)

        return tail, meta

    # ------------------------------------------------------------------
    def _sample_unique(self, value_range: Tuple[int, int], count: int) -> np.ndarray:
        lo, hi = value_range
        return self.rng.choice(np.arange(lo, hi), size=count, replace=False)


def composite_eval_hint() -> str:
    """Not executable -- documents the four-way scoring an evaluate.py-style
    loop should run against a batch from this dataset, analogous to
    evaluate.py's existing answer_span_exact_accuracy call:

        logits = model(batch["input_ids"])              # [B, T, V]
        preds = logits.argmax(dim=-1)                    # [B, T]
        for name in ("copy", "mood", "fact", "xref"):
            pos = batch[f"{name}_answer_position"]        # [B]
            tgt = batch[f"{name}_answer_token"]            # [B]
            pred_at_pos = preds.gather(1, pos.unsqueeze(1)).squeeze(1)
            exact = (pred_at_pos == tgt).float().mean()
            # log as eval_{name}_exact, then run the existing drop_local /
            # drop_recurrent / drop_fast_weight / drop_episodic ablation
            # matrix against all four eval_{name}_exact numbers, not just
            # one blended eval_answer_exact.
    """
    raise NotImplementedError("documentation-only function, see docstring")
