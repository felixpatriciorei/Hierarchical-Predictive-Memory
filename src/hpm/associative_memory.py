"""Experimental continuous delta-rule alternative to hard episodic writes.

``SupervisedMemoryWriter`` normally scores every causal candidate and uses a
hard top-k operation to form an exact finite key/value table. This module
instead writes every candidate into an associative state with a continuous
strength. It is useful for isolating the effects of hard selection and for
testing continuous-write training, but it does *not* preserve the semantics
or exact-recall capacity of an episodic slot store. In particular, it cannot
make a causal writer with fewer slots than equally likely later queries pass
the fixed-slot coverage contract.

This path is deliberately experimental. It is an associative alternative to
exact episodic slots, not a proven replacement for them, and it must be judged
against the capacity and transition contracts in ``docs/MATURITY.md``.

Performance note (read before using at long context)
------------------------------------------------------
The update below is a straightforward SEQUENTIAL scan over candidates
(mirroring `FastWeightBlockMemory`'s per-block loop, but per-candidate
instead of per-block) -- correct, and fine for the unit tests and small/
moderate candidate counts, but O(candidate_len) Python-level iterations.
At the long-context stress-test lengths this repo already reports on
(4096-12288 tokens => thousands of candidates), this will be slow in eager
mode. `FastWeightBlockMemory` sidesteps this by summarizing per BLOCK
(one write per ~128 tokens, not per token); a production version of this
module needs the same block-summarized treatment, or a proper chunked/
parallel delta-rule scan (the "chunk-wise prefix scan" the research brief's
Direction 4.1 section references for DeltaNet/Gated DeltaNet-2), before
it's practical at those lengths. Shipping the correct-but-serial version
first so the mechanism itself -- and its gradient properties -- can be unit
tested and reasoned about, without also debugging a fused/parallel kernel
at the same time.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .memory import gather_token_positions


def budgeted_sigmoid_write_gate(
    selection_logits: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    write_budget: int | None,
) -> torch.Tensor:
    """Return a continuous write gate with bounded total write mass.

    Independent sigmoid gates are unsafe for a long candidate stream: at
    initialization each valid candidate has weight near 0.5, hence the total
    delta-rule update grows as ``O(number_of_candidates)``. A tiny amount of
    such a gate mixed into an oracle write can therefore corrupt the memory
    immediately. This function keeps the ordinary sigmoid ranking and its
    gradients, but caps its aggregate write strength at ``write_budget``.

    When the raw mass exceeds the budget, all valid gates are rescaled by the
    same differentiable factor. It is not a discrete selection: every valid
    candidate remains in the graph and every gate stays in ``[0, 1]``.
    """
    if selection_logits.shape != candidate_valid_mask.shape:
        raise ValueError("selection_logits and candidate_valid_mask must have the same shape")
    if write_budget is not None and write_budget < 0:
        raise ValueError("write_budget must be non-negative or None")

    valid = candidate_valid_mask.to(selection_logits.dtype)
    gate = torch.sigmoid(selection_logits) * valid
    if write_budget is None:
        return gate

    valid_count = valid.sum(dim=-1, keepdim=True)
    budget = valid_count.new_full(valid_count.shape, float(write_budget)).minimum(valid_count)
    raw_mass = gate.sum(dim=-1, keepdim=True)
    # The cap is inactive after the writer has naturally become sparse; this
    # preserves a confident writer's learned calibration instead of forcing it
    # to spend unused write budget on arbitrary candidates.
    scale = (budget / raw_mass.clamp_min(torch.finfo(gate.dtype).eps)).clamp_max(1.0)
    return gate * scale


class DeltaRuleEpisodicMemory(nn.Module):
    """Delta-rule (Widrow-Hoff / KDA-style) associative replacement for the
    discrete-selection episodic pathway.

    No hard top-``k`` selection anywhere in this module, and no straight-
    through estimator: every candidate contributes to the memory state in proportion
    to a continuous ``write_gate`` in ``[0, 1]`` (typically
    ``sigmoid(writer_selection_logits)`` from ``SupervisedMemoryWriter`` --
    the SAME oracle-trained score the old hard top-k thresholded, just used
    as a soft strength instead of a rank cutoff), so the ordinary chain rule
    gives every candidate -- not just the top-``k`` -- a real gradient.

        M <- (I - g_i * k_i k_i^T) @ M + g_i * outer(k_i, v_i)   for i in candidates
        read = q @ M

    ``g_i`` (per-candidate write strength) plays the same role ``beta`` plays
    in ``FastWeightBlockMemory``: both write strength and delta-rule
    correction strength, matching the KDA formulation that module already
    uses successfully.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        candidate_positions: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        write_gate: torch.Tensor,
        query_key_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        token_embeddings: (batch, seq_len, d_model)
        candidate_positions: (batch, C, 2) key/value token-position pairs for
            EVERY candidate -- e.g. ``SupervisedMemoryWriter``'s
            ``writer_full_memory_token_positions``. No pre-selection: this
            module decides how much each one matters, continuously.
        candidate_valid_mask: (batch, C) bool -- structurally valid
            candidates (e.g. ``writer_full_valid_mask``); invalid candidates
            are hard-excluded (write_gate forced to 0) regardless of score,
            same as ``EpisodicMemory``'s ``memory_mask`` semantics.
        write_gate: (batch, C) continuous strength in [0, 1] -- e.g.
            ``sigmoid(writer_selection_logits)``. NOT a hard top-k mask.
        query_key_positions: (batch,) single query token position per batch
            row (matches ``EpisodicMemory.forward``'s interface).
        """

        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [batch, seq_len, d_model]")
        bsz, seq_len, d_model = token_embeddings.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        # write_gate is frequently a non-leaf tensor (e.g. sigmoid(logits)
        # computed upstream by SupervisedMemoryWriter, or -- as in the unit
        # tests -- an affine-transformed torch.rand(...)). PyTorch only
        # populates .grad on leaf tensors by default; a non-leaf tensor's
        # gradient is computed and passed through during backward() but
        # discarded unless retain_grad() was called on it first. Since the
        # entire point of this module is that every candidate's write_gate
        # gets a real gradient (not just a hard-selected subset), make that
        # gradient inspectable for callers/tests regardless of leaf-ness.
        if write_gate.requires_grad:
            write_gate.retain_grad()

        h = self.ln(token_embeddings)
        candidate_repr = gather_token_positions(h, candidate_positions)  # (B, C, 2, d)
        key_source = candidate_repr[:, :, 0, :]
        value_source = candidate_repr[:, :, 1, :]
        keys = F.normalize(self.k_proj(key_source), dim=-1)
        values = self.v_proj(value_source)

        gate = write_gate.to(keys.dtype) * candidate_valid_mask.to(keys.dtype)
        gate = gate.clamp(0.0, 1.0)

        query_source = gather_token_positions(h, query_key_positions[:, None]).squeeze(1)
        query = F.normalize(self.q_proj(query_source), dim=-1)

        num_candidates = keys.size(1)
        memory = h.new_zeros(bsz, d_model, d_model)
        for i in range(num_candidates):
            k_i = keys[:, i, :]
            v_i = values[:, i, :]
            beta_i = gate[:, i].view(bsz, 1, 1)
            existing_for_key = torch.bmm(k_i.unsqueeze(1), memory)  # (B, 1, d)
            correction = beta_i * torch.bmm(k_i.unsqueeze(2), existing_for_key)  # (B, d, d)
            write = beta_i * torch.bmm(k_i.unsqueeze(2), v_i.unsqueeze(1))  # (B, d, d)
            memory = memory - correction + write

        retrieved = torch.bmm(query.unsqueeze(1), memory).squeeze(1)
        retrieved = self.out(retrieved)

        info = {
            "write_gate_mean": gate.sum(dim=-1) / candidate_valid_mask.to(keys.dtype).sum(dim=-1).clamp_min(1.0),
            "memory_frobenius_norm": memory.flatten(1).norm(dim=-1),
        }
        return retrieved, info
