"""Conservative capacity statements and operational delta-memory probes.

The purpose is to define a safe operating envelope before training.  The module
avoids claiming a universal capacity for learned fast weights.  It contains one
exact lower bound (orthogonal keys) plus measurable interference diagnostics.
"""
from __future__ import annotations

import math
from typing import Tuple
import torch
import torch.nn.functional as F


def orthogonal_delta_exact_capacity(key_dim: int) -> int:
    """Zero-interference one-pass capacity lower bound under orthonormal keys.

    With beta=1, no decay, q_i=k_i, and mutually orthonormal unit keys, the
    delta rule stores each association exactly.  At most `key_dim` non-zero
    mutually orthogonal keys exist in R^key_dim.
    """
    d = int(key_dim)
    if d <= 0:
        raise ValueError("key_dim must be positive")
    return d


def mutual_coherence(keys: torch.Tensor) -> float:
    """Maximum absolute off-diagonal cosine similarity for row-wise keys."""
    if keys.ndim != 2 or keys.size(0) < 1:
        raise ValueError("keys must have shape [n, d]")
    k = F.normalize(keys.float(), dim=-1)
    gram = k @ k.T
    gram.fill_diagonal_(0.0)
    return float(gram.abs().max().item())


def delta_rule_state(keys: torch.Tensor, values: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Reference sequential delta-rule state M for row-wise key/value pairs."""
    if keys.ndim != 2 or values.ndim != 2 or keys.size(0) != values.size(0):
        raise ValueError("keys/values must be [n,d_k]/[n,d_v] with matching n")
    b = float(beta)
    if not 0.0 <= b <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    k = F.normalize(keys.float(), dim=-1)
    v = values.float()
    m = torch.zeros(k.size(1), v.size(1), dtype=v.dtype, device=v.device)
    for i in range(k.size(0)):
        ki = k[i]
        vi = v[i]
        existing = ki @ m
        m = m + b * torch.outer(ki, vi - existing)
    return m


def delta_rule_recall(keys: torch.Tensor, values: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    m = delta_rule_state(keys, values, beta=beta)
    q = F.normalize(keys.float(), dim=-1)
    return q @ m


def normalized_recall_cosine(keys: torch.Tensor, values: torch.Tensor, beta: float = 1.0) -> float:
    pred = delta_rule_recall(keys, values, beta=beta)
    target = values.float()
    if target.size(-1) == 1:
        mse = F.mse_loss(pred, target).item()
        return float(1.0 / (1.0 + mse))
    return float(F.cosine_similarity(pred, target, dim=-1).mean().item())


def overwrite_cosine(key: torch.Tensor, old_value: torch.Tensor, new_value: torch.Tensor) -> Tuple[float, float]:
    """Recall similarity to new vs old value after rebinding the same key twice."""
    k = F.normalize(key.reshape(1, -1).float(), dim=-1)
    vals = torch.stack([old_value.float(), new_value.float()], dim=0)
    keys = torch.cat([k, k], dim=0)
    m = delta_rule_state(keys, vals, beta=1.0)
    pred = (k @ m).squeeze(0)
    to_new = float(F.cosine_similarity(pred[None], new_value.float()[None], dim=-1).item())
    to_old = float(F.cosine_similarity(pred[None], old_value.float()[None], dim=-1).item())
    return to_new, to_old
