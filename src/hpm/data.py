from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

PAD = 0
BOS = 1
EOS = 2
FACT = 3
QUERY = 4
ANSWER = 5
SEP = 6
IF = 7
NO_VALUE = 8
REMEMBER = 450
HAS = 451
WHEN = 452
UNDER = 453
IS = 454
GIVEN = 455
STORE = 456
CONDITION = 457
NOTE = 458
BELONGS_TO = 459
FOR = 460
MEANS = 461
ARROW = 462
NOISE_WORD = 463
MAYBE = 464
IRRELEVANT = 465
WRONG = 466
TEXT = 467
WITHOUT = 468

NOISE_RANGE = (10, 100)
KEY_RANGE = (100, 200)
VALUE_RANGE = (200, 400)
CONDITION_RANGE = (400, 450)
# The aliased-read control reserves a disjoint, otherwise-unused slice of the
# condition-token range.  It deliberately restricts its fact keys to the
# matching 50-token slice below, so each stored key has one fixed query alias
# without increasing VOCAB_SIZE (which would perturb every legacy task's
# output-softmax normalization).
ALIASED_KEY_RANGE = (100, 150)
ALIAS_RANGE = CONDITION_RANGE
VOCAB_SIZE = 480

CONDITIONAL_TASKS = {
    "conditional",
    "conditional_balanced",
    "conditional_positive_only",
    "conditional_contrastive",
    "conditional_contrastive_stress",
}
STRESS_TASKS = {"conditional_contrastive_stress", "coexisting_stress", "kv_stress"}
NOISY_TASKS = {"noisy_kv", "noisy_coexisting", "noisy_conditional"}
CAUSAL_WRITER_TASKS = {"causal_salience_kv"}


@dataclass
class FactRecallConfig:
    seq_len: int = 512
    window: int = 64
    task: str = "kv"
    num_facts: int = 4
    seed: int = 0
    oracle_memory: bool = True
    repeated_keys: bool = False
    similar_values: bool = False
    distractor_fact_spans: int = 0
    query_key_noise_only: bool = False
    fact_order: str = "random"
    num_positive: int = 2
    num_hard_negatives: int = 0
    similarity_mode: str = "none"
    slot_order: str = "random"
    noise_level: str = "clean"
    marker_rate: float = 1.0
    distractor_count: int = 0
    template_mix: str = "mixed"
    template_augmentation: str = "none"
    writer_required_facts: int = 2
    """Number of explicitly marked records that a finite-capacity writer
    must retain in ``causal_salience_kv``.  These are selected at write time
    by a visible token, before the later query is sampled.  The remaining
    ``num_facts - writer_required_facts`` records are explicit distractors.
    """
    writer_role_mode: str = "visible"
    """``visible`` emits REMEMBER/IRRELEVANT before each candidate in
    ``causal_salience_kv``. ``masked`` replaces both roles with one shared
    filler token while preserving every other distributional detail.  The
    latter is a negative control: it removes the write-time utility signal and
    therefore cannot be used as a learned-writer correctness benchmark.
    """


class FactRecallDataset:
    """Synthetic long-range key-value recall batches.

    ``seq_len`` is the model input length. Internally each generated raw
    sequence has length ``seq_len + 1`` so next-token targets also have
    length ``seq_len``.
    """

    def __init__(self, config: FactRecallConfig):
        if config.task not in {"kv", "aliased_kv", "twohop", "coexisting", "longhop", *CONDITIONAL_TASKS, *STRESS_TASKS, *NOISY_TASKS, *CAUSAL_WRITER_TASKS}:
            raise ValueError(f"unknown task: {config.task}")
        if config.num_facts < (2 if config.task in {"twohop", "longhop", "coexisting", *CONDITIONAL_TASKS, "noisy_coexisting", "noisy_conditional"} else 1):
            raise ValueError("this task needs at least two facts")
        if config.fact_order not in {"random", "query_last"}:
            raise ValueError(f"unknown fact_order: {config.fact_order}")
        if config.similarity_mode not in {"none", "adjacent", "confusable", "mixed"}:
            raise ValueError(f"unknown similarity_mode: {config.similarity_mode}")
        if config.slot_order not in {"original", "random"}:
            raise ValueError(f"unknown slot_order: {config.slot_order}")
        if config.noise_level not in {"clean", "light", "medium", "hard"}:
            raise ValueError(f"unknown noise_level: {config.noise_level}")
        if config.template_mix not in {"simple", "mixed", "paraphrase"}:
            raise ValueError(f"unknown template_mix: {config.template_mix}")
        if config.template_augmentation not in {"none", "light", "heavy", "extreme"}:
            raise ValueError(f"unknown template_augmentation: {config.template_augmentation}")
        if not 0.0 <= config.marker_rate <= 1.0:
            raise ValueError("marker_rate must be in [0, 1]")
        if config.task in CAUSAL_WRITER_TASKS:
            if not 1 <= config.writer_required_facts < config.num_facts:
                raise ValueError(
                    "causal_salience_kv requires "
                    "1 <= writer_required_facts < num_facts"
                )
            if config.fact_order != "random":
                raise ValueError(
                    "causal_salience_kv requires fact_order='random' so record role "
                    "cannot be inferred from ordinal position"
                )
            if not config.oracle_memory:
                raise ValueError(
                    "causal_salience_kv requires oracle_memory=True: the control's "
                    "writer labels are the visible REMEMBER records, not a heuristic extractor"
                )
            if config.distractor_fact_spans:
                raise ValueError(
                    "causal_salience_kv does not accept untagged distractor_fact_spans; "
                    "every candidate's keep/drop role must be causally visible"
                )
            if config.writer_role_mode not in {"visible", "masked"}:
                raise ValueError(
                    "causal_salience_kv writer_role_mode must be 'visible' or 'masked'"
                )
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.sample_index = 0

    def sample_batch(self, batch_size: int, device: torch.device | str | None = None) -> Dict[str, torch.Tensor]:
        samples = [self._sample_one() for _ in range(batch_size)]
        batch: Dict[str, torch.Tensor] = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key == "loss_mask":
                tensor = torch.tensor(np.stack(values), dtype=torch.float32)
            elif key in {"memory_mask", "positive_memory_mask", "answer_target_mask"}:
                tensor = torch.tensor(np.stack(values), dtype=torch.bool)
            else:
                tensor = torch.tensor(np.stack(values), dtype=torch.long)
            if device is not None:
                tensor = tensor.to(device)
            batch[key] = tensor
        return batch

    def _sample_one(self) -> Dict[str, np.ndarray]:
        sample_index = self.sample_index
        self.sample_index += 1
        if self.config.query_key_noise_only:
            tokens, metadata = self._make_no_match_prefix()
        elif self.config.task in NOISY_TASKS:
            tokens, metadata = self._make_noisy_prefix()
        elif self.config.task == "kv_stress":
            tokens, metadata = self._make_kv_stress_prefix()
        elif self.config.task == "causal_salience_kv":
            tokens, metadata = self._make_causal_salience_kv_prefix()
        elif self.config.task == "aliased_kv":
            tokens, metadata = self._make_aliased_kv_prefix()
        elif self.config.task == "kv":
            tokens, metadata = self._make_kv_prefix()
        elif self.config.task == "coexisting_stress":
            tokens, metadata = self._make_coexisting_stress_prefix()
        elif self.config.task == "coexisting":
            tokens, metadata = self._make_coexisting_prefix()
        elif self.config.task == "conditional_contrastive_stress":
            tokens, metadata = self._make_conditional_stress_prefix()
        elif self.config.task in CONDITIONAL_TASKS:
            tokens, metadata = self._make_conditional_prefix(sample_index)
        else:
            tokens, metadata = self._make_twohop_prefix()

        query_tokens = list(metadata["query_tokens"])
        answer_tokens = list(metadata["answer_tokens"])
        tail_len = 1 + len(query_tokens) + 1 + len(answer_tokens) + 1
        gap_len = self.config.seq_len + 1 - len(tokens) - tail_len
        if gap_len < 1:
            raise ValueError("seq_len is too short for requested number of facts")

        gap = self.rng.integers(NOISE_RANGE[0], NOISE_RANGE[1], size=gap_len).tolist()
        if self.config.query_key_noise_only and gap_len > self.config.window + 4:
            gap[gap_len // 2] = metadata["query_key"]
        query_start = len(tokens) + gap_len
        tokens = tokens + gap + [QUERY] + query_tokens + [ANSWER] + answer_tokens + [EOS]

        if len(tokens) != self.config.seq_len + 1:
            raise AssertionError("internal sequence construction bug")

        answer_position = query_start + 1 + len(query_tokens)
        query_key_position = query_start + 1
        max_fact_end = max(end for _, end in metadata["memory_spans"])
        if max_fact_end >= answer_position - self.config.window:
            raise ValueError(
                "facts are inside the local attention window; increase seq_len or reduce window/num_facts"
            )

        input_ids = np.asarray(tokens[:-1], dtype=np.int64)
        target_ids = np.asarray(tokens[1:], dtype=np.int64)
        loss_mask = np.zeros(self.config.seq_len, dtype=np.float32)
        answer_target_positions = np.arange(answer_position, answer_position + len(answer_tokens), dtype=np.int64)
        loss_mask[answer_target_positions] = 1.0

        memory_token_positions = np.asarray(metadata["memory_token_positions"], dtype=np.int64)
        memory_spans = np.asarray(metadata["memory_spans"], dtype=np.int64)
        memory_condition_positions = np.asarray(
            metadata.get(
                "memory_condition_positions",
                infer_condition_positions(tokens, metadata["memory_token_positions"]),
            ),
            dtype=np.int64,
        )
        memory_mask = np.asarray(
            metadata.get("memory_mask", [True] * len(memory_token_positions)),
            dtype=bool,
        )
        if memory_mask.shape != (len(memory_token_positions),):
            raise AssertionError("memory_mask must contain one entry per memory candidate")
        positive_memory_mask = np.asarray(metadata["positive_memory_mask"], dtype=bool)
        stress_slot_types = np.asarray(metadata.get("stress_slot_types", [0] * self.config.num_facts), dtype=np.int64)

        if not self.config.oracle_memory:
            memory_token_positions, memory_spans, memory_condition_positions, memory_mask = extract_fact_memory(
                input_ids=input_ids,
                answer_position=answer_position,
                max_facts=self.config.num_facts,
            )

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "answer_positions": np.asarray(answer_position, dtype=np.int64),
            "answer_target_positions": answer_target_positions,
            "answer_target_mask": np.ones(len(answer_tokens), dtype=bool),
            "answer_token_spans": np.asarray(answer_tokens, dtype=np.int64),
            "query_key_positions": np.asarray(query_key_position, dtype=np.int64),
            "answer_tokens": np.asarray(answer_tokens[0], dtype=np.int64),
            "query_key_tokens": np.asarray(metadata["query_key"], dtype=np.int64),
            "memory_token_positions": memory_token_positions,
            "memory_condition_positions": memory_condition_positions,
            "memory_spans": memory_spans,
            "memory_mask": memory_mask,
            "positive_memory_indices": np.asarray(metadata["positive_memory_index"], dtype=np.int64),
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_memory_indices": np.asarray(metadata["hop_positive_indices"], dtype=np.int64),
            "stress_slot_types": stress_slot_types,
            "stress_slot_count": np.asarray(self.config.num_facts, dtype=np.int64),
            "stress_num_positive": np.asarray(self.config.num_positive, dtype=np.int64),
            "stress_num_hard_negatives": np.asarray(self.config.num_hard_negatives, dtype=np.int64),
        }

    def _make_kv_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        keys = self._key_tokens(self.config.num_facts)
        values = self._value_tokens(self.config.num_facts)
        query_original_index = int(self.rng.integers(0, self.config.num_facts))
        if self.config.repeated_keys:
            keys[query_original_index] = keys[0]
            for i in range(0, self.config.num_facts, 3):
                keys[i] = keys[query_original_index]

        facts = [
            {
                "key": int(keys[i]),
                "value": int(values[i]),
                "is_positive": i == query_original_index,
            }
            for i in range(self.config.num_facts)
        ]
        facts = self._order_facts(facts, positive_roles={"positive"})

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        positive_index = -1
        query_key = -1
        answer_token = -1
        for slot, fact in enumerate(facts):
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 2])
            if fact["is_positive"]:
                positive_index = slot
                query_key = fact["key"]
                answer_token = fact["value"]

        self._append_distractor_fact_like_spans(tokens)
        positive_memory_mask = [False] * self.config.num_facts
        positive_memory_mask[positive_index] = True
        return tokens, {
            "query_key": query_key,
            "query_tokens": [query_key],
            "answer_token": answer_token,
            "answer_tokens": [answer_token],
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": positive_index,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [positive_index, -100],
        }

    @staticmethod
    def _key_to_query_alias(key: int) -> int:
        """Map an aliased-control fact key to its disjoint query token."""

        if not ALIASED_KEY_RANGE[0] <= key < ALIASED_KEY_RANGE[1]:
            raise ValueError(f"aliased control key {key} is outside {ALIASED_KEY_RANGE}")
        return ALIAS_RANGE[0] + (key - ALIASED_KEY_RANGE[0])

    def _make_aliased_kv_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        """Single-hop KV read control without literal query/key matching.

        Facts retain ordinary key/value records, but the query emits a fixed
        *alias* token from a disjoint vocabulary range.  The mapping is shared
        across examples (``alias(key)``) and is therefore learnable from
        selection gradients, while a freshly identity-initialized dot-product
        reader cannot retrieve the right slot merely because the same token
        appears on both sides of the read.  This is a diagnostic for the
        top-1 read-cutoff gradient, not a claim about semantic retrieval.
        """

        keys = self._unique_tokens(ALIASED_KEY_RANGE, self.config.num_facts)
        values = self._value_tokens(self.config.num_facts)
        query_original_index = int(self.rng.integers(0, self.config.num_facts))
        facts = [
            {
                "key": int(keys[i]),
                "value": int(values[i]),
                "is_positive": i == query_original_index,
            }
            for i in range(self.config.num_facts)
        ]
        facts = self._order_facts(facts, positive_roles={"positive"})

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        positive_index = -1
        query_alias = -1
        answer_token = -1
        for slot, fact in enumerate(facts):
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 2])
            if fact["is_positive"]:
                positive_index = slot
                query_alias = self._key_to_query_alias(int(fact["key"]))
                answer_token = int(fact["value"])

        if positive_index < 0:
            raise AssertionError("aliased_kv failed to mark a positive fact")
        positive_memory_mask = [False] * self.config.num_facts
        positive_memory_mask[positive_index] = True
        return tokens, {
            "query_key": query_alias,
            "query_tokens": [query_alias],
            "answer_token": answer_token,
            "answer_tokens": [answer_token],
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": positive_index,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [positive_index, -100],
        }

    def _make_causal_salience_kv_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        """Generate a finite-capacity causal writer control.

        Every candidate is an explicit record
        ``[REMEMBER|IRRELEVANT, FACT, key, value, SEP]``.  The role token is
        visible before the writer reaches the key/value pair.  The later query
        is sampled uniformly from the REMEMBER records only.  Therefore the
        task asks the writer to retain a known-at-write-time subset, rather
        than to predict an unobserved future query.

        ``memory_token_positions`` intentionally contains *all* records while
        ``memory_mask`` labels only the REMEMBER subset.  This is what allows
        a Top-C writer loss and write metrics to distinguish retained facts
        from explicitly tagged candidate distractors.
        """
        required = self.config.writer_required_facts
        keys = self._key_tokens(self.config.num_facts)
        values = self._value_tokens(self.config.num_facts)
        facts = [
            {
                "key": int(keys[index]),
                "value": int(values[index]),
                "required": index < required,
                "role": "required" if index < required else "distractor",
            }
            for index in range(self.config.num_facts)
        ]
        # FactRecallConfig validation forces random order for this task.  Keep
        # _order_facts here so the order-control has one implementation path.
        facts = self._order_facts(facts, positive_roles={"required"})

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        memory_mask: List[bool] = []
        required_indices: List[int] = []
        for slot, fact in enumerate(facts):
            start = len(tokens)
            if self.config.writer_role_mode == "visible":
                role_token = REMEMBER if fact["required"] else IRRELEVANT
            else:
                # Negative control: preserve token count, fact positions, key
                # and value distributions, random order, and later query
                # distribution.  Only the causal utility signal is removed.
                role_token = NOISE_WORD
            tokens.extend([role_token, FACT, fact["key"], fact["value"], SEP])
            memory_token_positions.append([start + 2, start + 3])
            memory_spans.append([start, start + 3])
            is_required = bool(fact["required"])
            memory_mask.append(is_required)
            if is_required:
                required_indices.append(slot)

        query_slot = int(self.rng.choice(required_indices))
        query_fact = facts[query_slot]
        positive_memory_mask = [slot == query_slot for slot in range(self.config.num_facts)]
        return tokens, {
            "query_key": int(query_fact["key"]),
            "query_tokens": [int(query_fact["key"])],
            "answer_token": int(query_fact["value"]),
            "answer_tokens": [int(query_fact["value"])],
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "memory_mask": memory_mask,
            "positive_memory_index": query_slot,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [query_slot, -100],
        }

    def _make_coexisting_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        keys = self._key_tokens(max(self.config.num_facts, 3))
        values = self._value_tokens(self.config.num_facts)
        query_key = int(keys[0])
        positive_values = [int(values[0]), int(values[1])]
        facts = [
            {"key": query_key, "value": positive_values[0], "role": "positive"},
            {"key": query_key, "value": positive_values[1], "role": "positive"},
        ]
        for i in range(self.config.num_facts - 2):
            facts.append({"key": int(keys[(i + 1) % len(keys)]), "value": int(values[i + 2]), "role": "other"})
        facts = self._order_facts(facts, positive_roles={"positive"})

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        positive_indices: List[int] = []
        for slot, fact in enumerate(facts):
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 2])
            if fact["role"] == "positive":
                positive_indices.append(slot)

        positive_memory_mask = [slot in set(positive_indices) for slot in range(self.config.num_facts)]
        return tokens, {
            "query_key": query_key,
            "query_tokens": [query_key],
            "answer_token": min(positive_values),
            "answer_tokens": sorted(positive_values),
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": positive_indices[0],
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [positive_indices[0], -100],
        }

    def _make_conditional_prefix(self, sample_index: int) -> Tuple[List[int], Dict[str, object]]:
        key = int(self._key_tokens(1)[0])
        values = self._value_tokens(max(self.config.num_facts, 2))
        conditions = self._unique_tokens(CONDITION_RANGE, max(self.config.num_facts, 2))
        query_index = int(self.rng.integers(0, 2))
        if self.config.task == "conditional_balanced":
            violates_condition = bool(sample_index % 2)
        elif self.config.task in {"conditional_positive_only", "conditional_contrastive"}:
            violates_condition = False
        else:
            violates_condition = bool(self.rng.integers(0, 2))
        facts = [
            {
                "key": key,
                "value": int(values[0]),
                "condition": int(conditions[0]),
                "role": "other" if violates_condition else "positive" if query_index == 0 else "other",
            },
            {
                "key": key,
                "value": int(values[1]),
                "condition": int(conditions[1]),
                "role": "other" if violates_condition else "positive" if query_index == 1 else "other",
            },
        ]
        for i in range(self.config.num_facts - 2):
            facts.append(
                {
                    "key": int(self._key_tokens(1)[0]),
                    "value": int(values[i + 2]),
                    "condition": int(conditions[i + 2]),
                    "role": "other",
                }
            )
        facts = self._order_facts(facts, positive_roles={"positive"})

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        positive_index = -1
        query_condition = int(conditions[query_index])
        answer_token = NO_VALUE
        if violates_condition:
            used = set(int(condition) for condition in conditions)
            choices = [token for token in range(CONDITION_RANGE[0], CONDITION_RANGE[1]) if token not in used]
            query_condition = int(self.rng.choice(choices))
        for slot, fact in enumerate(facts):
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], IF, fact["condition"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 4])
            if fact["role"] == "positive":
                positive_index = slot
                query_condition = fact["condition"]
                answer_token = fact["value"]

        positive_memory_mask = [False] * self.config.num_facts
        if positive_index >= 0:
            positive_memory_mask[positive_index] = True
        return tokens, {
            "query_key": key,
            "query_tokens": [key, query_condition],
            "answer_token": answer_token,
            "answer_tokens": [answer_token],
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": positive_index,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [positive_index if positive_index >= 0 else -100, -100],
        }

    def _make_twohop_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        keys = self._key_tokens(self.config.num_facts + 1)
        values = self._value_tokens(self.config.num_facts)
        key_a = int(keys[0])
        key_b = int(keys[1])
        value_c = int(values[0])

        facts = [
            {"key": key_a, "value": key_b, "role": "first"},
            {"key": key_b, "value": value_c, "role": "second"},
        ]
        for i in range(self.config.num_facts - 2):
            if self.config.repeated_keys and i % 4 == 0:
                key = key_a
            elif self.config.repeated_keys and i % 4 == 1:
                key = key_b
            else:
                key = int(keys[(i + 2) % len(keys)])
            facts.append({"key": key, "value": int(values[i + 1]), "role": "distractor"})
        facts = self._order_facts(facts, positive_roles={"first", "second"})

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        first_index = -1
        second_index = -1
        for slot, fact in enumerate(facts):
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 2])
            if fact["role"] == "first":
                first_index = slot
            elif fact["role"] == "second":
                second_index = slot

        self._append_distractor_fact_like_spans(tokens)
        positive_memory_mask = [False] * self.config.num_facts
        positive_memory_mask[second_index] = True
        return tokens, {
            "query_key": key_a,
            "query_tokens": [key_a],
            "answer_token": value_c,
            "answer_tokens": [value_c],
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": second_index,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [first_index, second_index],
        }

    def _make_conditional_stress_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        slot_count = self.config.num_facts
        num_positive = max(1, min(self.config.num_positive, slot_count))
        num_hard = max(0, min(self.config.num_hard_negatives, slot_count - num_positive))
        num_key_only = num_hard // 2 + num_hard % 2
        num_condition_only = num_hard // 2
        num_neither = max(slot_count - num_positive - num_hard, 0)

        keys = self._stress_tokens(KEY_RANGE, max(slot_count + 2, 4))
        values = self._stress_tokens(VALUE_RANGE, max(slot_count + 2, 4))
        conditions = self._stress_tokens(CONDITION_RANGE, max(slot_count + 2, 4))
        query_key = int(keys[0])
        query_condition = int(conditions[0])
        answer_value = int(values[0])

        facts: List[Dict[str, object]] = []
        for _ in range(num_positive):
            facts.append(
                {
                    "key": query_key,
                    "value": answer_value,
                    "condition": query_condition,
                    "role": "positive",
                    "stress_type": 1,
                }
            )
        cursor = 1
        for _ in range(num_key_only):
            facts.append(
                {
                    "key": query_key,
                    "value": int(values[cursor % len(values)]),
                    "condition": int(conditions[(cursor % (len(conditions) - 1)) + 1]),
                    "role": "key_only",
                    "stress_type": 2,
                }
            )
            cursor += 1
        for _ in range(num_condition_only):
            facts.append(
                {
                    "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                    "value": int(values[cursor % len(values)]),
                    "condition": query_condition,
                    "role": "condition_only",
                    "stress_type": 3,
                }
            )
            cursor += 1
        for _ in range(num_neither):
            facts.append(
                {
                    "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                    "value": int(values[cursor % len(values)]),
                    "condition": int(conditions[(cursor % (len(conditions) - 1)) + 1]),
                    "role": "neither",
                    "stress_type": 4,
                }
            )
            cursor += 1
        facts = self._order_stress_slots(facts)

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        positive_indices: List[int] = []
        stress_slot_types: List[int] = []
        for slot, fact in enumerate(facts):
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], IF, fact["condition"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 4])
            stress_slot_types.append(int(fact["stress_type"]))
            if fact["role"] == "positive":
                positive_indices.append(slot)

        positive_memory_mask = [slot in set(positive_indices) for slot in range(slot_count)]
        first_positive = positive_indices[0]
        return tokens, {
            "query_key": query_key,
            "query_tokens": [query_key, query_condition],
            "answer_token": answer_value,
            "answer_tokens": [answer_value],
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": first_positive,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [first_positive, -100],
            "stress_slot_types": stress_slot_types,
        }

    def _make_coexisting_stress_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        slot_count = self.config.num_facts
        num_positive = max(1, min(self.config.num_positive, slot_count))
        num_hard = max(0, min(self.config.num_hard_negatives, slot_count - num_positive))
        num_easy = max(slot_count - num_positive - num_hard, 0)

        keys = self._stress_tokens(KEY_RANGE, max(slot_count + 2, 4))
        values = self._stress_tokens(VALUE_RANGE, max(slot_count + num_positive + 2, 4))
        query_key = int(keys[0])

        facts: List[Dict[str, object]] = []
        answer_values = []
        for i in range(num_positive):
            value = int(values[i])
            answer_values.append(value)
            facts.append({"key": query_key, "value": value, "role": "positive", "stress_type": 1})
        cursor = num_positive
        for _ in range(num_hard):
            facts.append(
                {
                    "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                    "value": int(values[cursor % len(values)]),
                    "role": "hard_negative",
                    "stress_type": 2,
                }
            )
            cursor += 1
        for _ in range(num_easy):
            facts.append(
                {
                    "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                    "value": int(values[cursor % len(values)]),
                    "role": "negative",
                    "stress_type": 4,
                }
            )
            cursor += 1
        facts = self._order_stress_slots(facts)

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        positive_indices: List[int] = []
        stress_slot_types: List[int] = []
        for slot, fact in enumerate(facts):
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 2])
            stress_slot_types.append(int(fact["stress_type"]))
            if fact["role"] == "positive":
                positive_indices.append(slot)

        positive_memory_mask = [slot in set(positive_indices) for slot in range(slot_count)]
        sorted_answers = sorted(answer_values)
        first_positive = positive_indices[0]
        return tokens, {
            "query_key": query_key,
            "query_tokens": [query_key],
            "answer_token": sorted_answers[0],
            "answer_tokens": sorted_answers,
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": first_positive,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [first_positive, -100],
            "stress_slot_types": stress_slot_types,
        }

    def _make_kv_stress_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        original = self.config.num_positive
        self.config.num_positive = 1
        try:
            return self._make_coexisting_stress_prefix()
        finally:
            self.config.num_positive = original

    def _make_noisy_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        if self.config.task == "noisy_conditional":
            return self._make_noisy_conditional_prefix()
        if self.config.task == "noisy_coexisting":
            return self._make_noisy_coexisting_prefix()
        return self._make_noisy_kv_prefix()

    def _make_noisy_conditional_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        slot_count = self.config.num_facts
        num_hard = max(0, min(self.config.num_hard_negatives, slot_count - 1))
        keys = self._noisy_tokens(KEY_RANGE, max(slot_count + 2, 4))
        values = self._noisy_tokens(VALUE_RANGE, max(slot_count + 2, 4))
        conditions = self._noisy_tokens(CONDITION_RANGE, max(slot_count + 2, 4))
        query_key = int(keys[0])
        query_condition = int(conditions[0])
        answer_value = int(values[0])

        facts: List[Dict[str, object]] = [
            {"key": query_key, "value": answer_value, "condition": query_condition, "role": "positive"}
        ]
        cursor = 1
        for i in range(num_hard):
            if i % 2 == 0:
                facts.append(
                    {
                        "key": query_key,
                        "value": int(values[cursor % len(values)]),
                        "condition": int(conditions[(cursor % (len(conditions) - 1)) + 1]),
                        "role": "hard",
                    }
                )
            else:
                facts.append(
                    {
                        "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                        "value": int(values[cursor % len(values)]),
                        "condition": query_condition,
                        "role": "hard",
                    }
                )
            cursor += 1
        while len(facts) < slot_count:
            facts.append(
                {
                    "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                    "value": int(values[cursor % len(values)]),
                    "condition": int(conditions[(cursor % (len(conditions) - 1)) + 1]),
                    "role": "other",
                }
            )
            cursor += 1
        return self._build_noisy_sequence(facts, query_key, [query_key, query_condition], [answer_value], conditional=True)

    def _make_noisy_coexisting_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        slot_count = self.config.num_facts
        num_positive = max(1, min(self.config.num_positive, slot_count))
        num_hard = max(0, min(self.config.num_hard_negatives, slot_count - num_positive))
        keys = self._noisy_tokens(KEY_RANGE, max(slot_count + 2, 4))
        values = self._noisy_tokens(VALUE_RANGE, max(slot_count + num_positive + 2, 4))
        query_key = int(keys[0])

        facts: List[Dict[str, object]] = []
        answer_values = []
        for i in range(num_positive):
            value = int(values[i])
            answer_values.append(value)
            facts.append({"key": query_key, "value": value, "role": "positive"})
        cursor = num_positive
        for _ in range(num_hard):
            facts.append(
                {
                    "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                    "value": int(values[cursor % len(values)]),
                    "role": "hard",
                }
            )
            cursor += 1
        while len(facts) < slot_count:
            facts.append(
                {
                    "key": int(keys[(cursor % (len(keys) - 1)) + 1]),
                    "value": int(values[cursor % len(values)]),
                    "role": "other",
                }
            )
            cursor += 1
        return self._build_noisy_sequence(facts, query_key, [query_key], sorted(answer_values), conditional=False)

    def _make_noisy_kv_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        slot_count = self.config.num_facts
        keys = self._noisy_tokens(KEY_RANGE, max(slot_count + 2, 4))
        values = self._noisy_tokens(VALUE_RANGE, max(slot_count + 2, 4))
        query_index = int(self.rng.integers(0, slot_count))
        facts = [
            {
                "key": int(keys[i % len(keys)]),
                "value": int(values[i % len(values)]),
                "role": "positive" if i == query_index else "other",
            }
            for i in range(slot_count)
        ]
        query_key = facts[query_index]["key"]
        answer_value = facts[query_index]["value"]
        return self._build_noisy_sequence(facts, query_key, [query_key], [answer_value], conditional=False)

    def _build_noisy_sequence(
        self,
        facts: List[Dict[str, object]],
        query_key: int,
        query_tokens: List[int],
        answer_tokens: List[int],
        conditional: bool,
    ) -> Tuple[List[int], Dict[str, object]]:
        items = [{"kind": "fact", "fact": fact} for fact in facts]
        items.extend({"kind": "distractor", "conditional": conditional} for _ in range(self.config.distractor_count))
        if self.config.noise_level in {"medium", "hard"} or self.config.slot_order == "random":
            self.rng.shuffle(items)

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_condition_positions: List[int] = []
        memory_spans: List[List[int]] = []
        positive_indices: List[int] = []
        for item in items:
            if item["kind"] == "distractor":
                self._emit_noisy_distractor(tokens, conditional)
                continue
            slot = len(memory_token_positions)
            fact = item["fact"]
            start = len(tokens)
            key_pos, value_pos, condition_pos = self._emit_noisy_fact(tokens, fact, conditional)
            memory_token_positions.append([key_pos, value_pos])
            memory_condition_positions.append(condition_pos)
            memory_spans.append([start, max(key_pos, value_pos, condition_pos)])
            if fact.get("role") == "positive":
                positive_indices.append(slot)

        positive_set = set(positive_indices)
        positive_memory_mask = [slot in positive_set for slot in range(self.config.num_facts)]
        first_positive = positive_indices[0] if positive_indices else -1
        return tokens, {
            "query_key": query_key,
            "query_tokens": query_tokens,
            "answer_token": answer_tokens[0],
            "answer_tokens": answer_tokens,
            "memory_token_positions": memory_token_positions,
            "memory_condition_positions": memory_condition_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": first_positive,
            "positive_memory_mask": positive_memory_mask,
            "hop_positive_indices": [first_positive if first_positive >= 0 else -100, -100],
        }

    def _emit_noisy_fact(self, tokens: List[int], fact: Dict[str, object], conditional: bool) -> Tuple[int, int, int]:
        key = int(fact["key"])
        value = int(fact["value"])
        condition = int(fact.get("condition", -1))
        use_marker = bool(self.rng.random() < self.config.marker_rate)
        templates = self._conditional_templates() if conditional else self._pair_templates()
        template = 0 if use_marker else int(self.rng.choice(templates))
        start = len(tokens)
        if conditional:
            if template == 0:
                seq = [FACT, key, value, IF, condition, SEP]
                key_i, value_i, cond_i = 1, 2, 4
            elif template == 1:
                seq = [REMEMBER, key, HAS, value, WHEN, condition, SEP]
                key_i, value_i, cond_i = 1, 3, 5
            elif template == 2:
                seq = [UNDER, condition, key, IS, value, SEP]
                key_i, value_i, cond_i = 2, 4, 1
            elif template == 3:
                seq = [key, ARROW, value, GIVEN, condition, SEP]
                key_i, value_i, cond_i = 0, 2, 4
            elif template == 4:
                seq = [STORE, key, value, CONDITION, condition, SEP]
                key_i, value_i, cond_i = 1, 2, 4
            elif template == 5:
                seq = [NOTE, value, BELONGS_TO, key, IF, condition, SEP]
                key_i, value_i, cond_i = 3, 1, 5
            elif template == 6:
                seq = [FOR, key, condition, MEANS, value, SEP]
                key_i, value_i, cond_i = 1, 4, 2
            elif template == 7:
                seq = [UNDER, condition, REMEMBER, key, ARROW, value, SEP]
                key_i, value_i, cond_i = 3, 5, 1
            elif template == 8:
                seq = [WHEN, condition, STORE, key, value, SEP]
                key_i, value_i, cond_i = 3, 4, 1
            elif template == 9:
                seq = [GIVEN, condition, key, HAS, value, SEP]
                key_i, value_i, cond_i = 2, 4, 1
            elif template == 10:
                seq = [CONDITION, condition, FOR, key, MEANS, value, SEP]
                key_i, value_i, cond_i = 3, 5, 1
            elif template == 11:
                seq = [TEXT, condition, key, IS, value, SEP]
                key_i, value_i, cond_i = 2, 4, 1
            elif template == 12:
                seq = [WHEN, condition, key, ARROW, value, SEP]
                key_i, value_i, cond_i = 2, 4, 1
            elif template == 13:
                seq = [FOR, key, GIVEN, condition, REMEMBER, value, SEP]
                key_i, value_i, cond_i = 1, 5, 3
            elif template == 14:
                seq = [NOTE, condition, MEANS, value, BELONGS_TO, key, SEP]
                key_i, value_i, cond_i = 5, 3, 1
            elif template == 15:
                seq = [STORE, condition, key, ARROW, value, SEP]
                key_i, value_i, cond_i = 2, 4, 1
            else:
                seq = [GIVEN, condition, REMEMBER, value, FOR, key, SEP]
                key_i, value_i, cond_i = 5, 3, 1
            tokens.extend(seq)
            return start + key_i, start + value_i, start + cond_i

        if template == 0:
            seq = [FACT, key, value, SEP]
            key_i, value_i = 1, 2
        elif template == 1:
            seq = [REMEMBER, key, HAS, value, SEP]
            key_i, value_i = 1, 3
        elif template == 2:
            seq = [key, ARROW, value, SEP]
            key_i, value_i = 0, 2
        elif template == 3:
            seq = [STORE, key, value, SEP]
            key_i, value_i = 1, 2
        elif template == 4:
            seq = [NOTE, value, BELONGS_TO, key, SEP]
            key_i, value_i = 3, 1
        elif template == 5:
            seq = [key, HAS, value, SEP]
            key_i, value_i = 0, 2
        elif template == 6:
            seq = [value, IS, FOR, key, SEP]
            key_i, value_i = 3, 0
        elif template == 7:
            seq = [FOR, key, HAS, value, SEP]
            key_i, value_i = 1, 3
        elif template == 8:
            seq = [value, BELONGS_TO, key, SEP]
            key_i, value_i = 2, 0
        else:
            seq = [REMEMBER, value, FOR, key, SEP]
            key_i, value_i = 3, 1
        tokens.extend(seq)
        return start + key_i, start + value_i, -1

    def _emit_noisy_distractor(self, tokens: List[int], conditional: bool) -> None:
        key = int(self.rng.integers(KEY_RANGE[0], KEY_RANGE[1]))
        value = int(self.rng.integers(VALUE_RANGE[0], VALUE_RANGE[1]))
        condition = int(self.rng.integers(CONDITION_RANGE[0], CONDITION_RANGE[1]))
        if conditional:
            templates = [
                [NOISE_WORD, key, value, condition, SEP],
                [MAYBE, key, value, SEP],
                [IRRELEVANT, key, value, condition, SEP],
                [WRONG, key, value, IF, condition, SEP],
                [TEXT, key, condition, WITHOUT, value, SEP],
                [TEXT, value, WITHOUT, key, SEP],
            ]
        else:
            templates = [
                [NOISE_WORD, key, value, SEP],
                [MAYBE, key, value, SEP],
                [IRRELEVANT, key, value, SEP],
                [WRONG, value, BELONGS_TO, key, SEP],
                [TEXT, value, WITHOUT, key, SEP],
            ]
        tokens.extend(templates[int(self.rng.integers(0, len(templates)))])

    def _conditional_templates(self) -> List[int]:
        if self.config.noise_level == "clean":
            return [3]
        if self.config.template_augmentation == "light":
            return [1, 3, 4, 7, 8]
        if self.config.template_augmentation == "heavy":
            return [1, 3, 4, 7, 8, 9, 10, 11]
        if self.config.template_augmentation == "extreme":
            return [1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        if self.config.template_mix == "simple":
            return [1, 3, 4]
        if self.config.template_mix == "paraphrase":
            return [1, 2, 5, 6]
        return [1, 2, 3, 4, 5, 6]

    def _pair_templates(self) -> List[int]:
        if self.config.noise_level == "clean":
            return [2]
        if self.config.template_augmentation == "light":
            return [1, 2, 3, 4, 5]
        if self.config.template_augmentation == "heavy":
            return [1, 2, 3, 4, 5, 6, 7]
        if self.config.template_augmentation == "extreme":
            return [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.config.template_mix == "simple":
            return [1, 2, 3]
        if self.config.template_mix == "paraphrase":
            return [1, 4, 5, 6, 7, 8, 9]
        return [1, 2, 3, 4]

    def _noisy_tokens(self, token_range: Tuple[int, int], count: int) -> np.ndarray:
        if self.config.noise_level == "hard":
            old = self.config.similarity_mode
            self.config.similarity_mode = "mixed"
            try:
                return self._stress_tokens(token_range, count)
            finally:
                self.config.similarity_mode = old
        if self.config.noise_level == "medium":
            old = self.config.similarity_mode
            self.config.similarity_mode = "adjacent"
            try:
                return self._stress_tokens(token_range, count)
            finally:
                self.config.similarity_mode = old
        return self._unique_tokens(token_range, min(count, token_range[1] - token_range[0]))

    def _make_no_match_prefix(self) -> Tuple[List[int], Dict[str, object]]:
        values = self._value_tokens(self.config.num_facts + 1)
        query_key = int(self._unique_tokens(KEY_RANGE, 1)[0])
        available_keys = np.asarray([token for token in range(KEY_RANGE[0], KEY_RANGE[1]) if token != query_key])
        if self.config.repeated_keys:
            pool_size = max(2, min(self.config.num_facts, max(4, self.config.num_facts // 4)))
            pool = self.rng.choice(available_keys, size=pool_size, replace=False)
            fact_keys = self.rng.choice(pool, size=self.config.num_facts, replace=True)
        else:
            fact_keys = self.rng.choice(available_keys, size=self.config.num_facts, replace=False)
        if self.config.repeated_keys:
            for i in range(0, len(fact_keys), 4):
                fact_keys[i] = fact_keys[0]

        facts = [
            {"key": int(fact_keys[i]), "value": int(values[i]), "role": "distractor"}
            for i in range(self.config.num_facts)
        ]
        facts = self._order_facts(facts, positive_roles=set())

        tokens = [BOS]
        memory_token_positions: List[List[int]] = []
        memory_spans: List[List[int]] = []
        for fact in facts:
            start = len(tokens)
            tokens.extend([FACT, fact["key"], fact["value"], SEP])
            memory_token_positions.append([start + 1, start + 2])
            memory_spans.append([start, start + 2])

        self._append_distractor_fact_like_spans(tokens)
        return tokens, {
            "query_key": query_key,
            "query_tokens": [query_key],
            "answer_token": int(values[-1]),
            "answer_tokens": [int(values[-1])],
            "memory_token_positions": memory_token_positions,
            "memory_spans": memory_spans,
            "positive_memory_index": -1,
            "positive_memory_mask": [False] * self.config.num_facts,
            "hop_positive_indices": [-100, -100],
        }

    def _unique_tokens(self, token_range: Tuple[int, int], count: int) -> np.ndarray:
        low, high = token_range
        if count > high - low:
            raise ValueError("requested more unique tokens than the range contains")
        return self.rng.choice(np.arange(low, high), size=count, replace=False)

    def _key_tokens(self, count: int) -> np.ndarray:
        if not self.config.repeated_keys:
            return self._unique_tokens(KEY_RANGE, count)
        pool_size = max(2, min(count, max(4, self.config.num_facts // 4)))
        pool = self._unique_tokens(KEY_RANGE, pool_size)
        return self.rng.choice(pool, size=count, replace=True)

    def _value_tokens(self, count: int) -> np.ndarray:
        if not self.config.similar_values:
            return self._unique_tokens(VALUE_RANGE, count)
        low, high = VALUE_RANGE
        if count > high - low:
            raise ValueError("requested more similar values than the range contains")
        base = int(self.rng.integers(low, high - count + 1))
        return np.arange(base, base + count, dtype=np.int64)

    def _stress_tokens(self, token_range: Tuple[int, int], count: int) -> np.ndarray:
        low, high = token_range
        count = min(count, high - low)
        if self.config.similarity_mode == "none":
            return self._unique_tokens(token_range, count)
        span = max(count + 2, min(high - low, count + 8))
        base = int(self.rng.integers(low, high - span + 1))
        pool = np.arange(base, base + span)
        if self.config.similarity_mode in {"adjacent", "mixed"}:
            return pool[:count]
        return self.rng.choice(pool, size=count, replace=False)

    def _order_stress_slots(self, facts: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if self.config.slot_order == "random":
            self.rng.shuffle(facts)
        return facts

    def _order_facts(self, facts: List[Dict[str, object]], positive_roles: set[str]) -> List[Dict[str, object]]:
        if self.config.fact_order == "random":
            self.rng.shuffle(facts)
            return facts
        positives = []
        negatives = []
        for fact in facts:
            role = str(fact.get("role", "positive" if fact.get("is_positive") else "distractor"))
            if role in positive_roles:
                positives.append(fact)
            else:
                negatives.append(fact)
        self.rng.shuffle(negatives)
        return negatives + positives

    def _append_distractor_fact_like_spans(self, tokens: List[int]) -> None:
        for _ in range(self.config.distractor_fact_spans):
            key = int(self.rng.integers(KEY_RANGE[0], KEY_RANGE[1]))
            value = int(self.rng.integers(VALUE_RANGE[0], VALUE_RANGE[1]))
            tokens.extend([FACT, key, value, SEP])


def infer_condition_positions(tokens: List[int] | np.ndarray, memory_token_positions: List[List[int]] | np.ndarray) -> List[int]:
    condition_positions: List[int] = []
    for _, value_pos in memory_token_positions:
        if value_pos >= 0 and value_pos + 2 < len(tokens) and tokens[value_pos + 1] == IF:
            condition_positions.append(int(value_pos + 2))
        else:
            condition_positions.append(-1)
    return condition_positions


def extract_fact_memory(
    input_ids: np.ndarray,
    answer_position: int,
    max_facts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heuristic non-oracle memory writes: store key/value after FACT tokens."""

    memory_positions = np.zeros((max_facts, 2), dtype=np.int64)
    memory_spans = np.zeros((max_facts, 2), dtype=np.int64)
    memory_condition_positions = np.full(max_facts, -1, dtype=np.int64)
    memory_mask = np.zeros(max_facts, dtype=bool)

    slot = 0
    for pos, token in enumerate(input_ids[:answer_position]):
        if token != FACT:
            continue
        if pos + 2 >= answer_position:
            continue
        if slot >= max_facts:
            break
        span_end = pos + 4 if pos + 4 < answer_position and input_ids[pos + 3] == IF else pos + 2
        memory_positions[slot] = [pos + 1, pos + 2]
        memory_spans[slot] = [pos, span_end]
        if pos + 4 < answer_position and input_ids[pos + 3] == IF:
            memory_condition_positions[slot] = pos + 4
        memory_mask[slot] = True
        slot += 1

    return memory_positions, memory_spans, memory_condition_positions, memory_mask
