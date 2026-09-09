"""Architecture-wide HPM maturity contracts.

These helpers intentionally encode *minimum sufficient* pre-training contracts,
not paper-grade claims.  They are side-effect free and can be used by tests,
offline audits, and training guards.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    status: str
    evidence: Dict[str, Any]
    blocker: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def router_probability_floor(
    num_paths: int,
    initial_adverse_gap: float,
    max_adverse_gap_drift_per_step: float,
    horizon_steps: int,
) -> float:
    """Finite-horizon lower bound on a path's softmax probability.

    Assume, for a path k, every competitor satisfies

        z_j - z_k <= A_0 + t d_max

    for t <= T.  Then

        p_k(t) >= 1 / (1 + (K-1) exp(A_0 + T d_max)).

    This does not claim the assumptions hold automatically; it turns a measured
    gap-drift envelope into a concrete survival floor.
    """
    k = int(num_paths)
    if k < 2:
        raise ValueError("num_paths must be >= 2")
    a0 = _finite("initial_adverse_gap", initial_adverse_gap)
    drift = _finite("max_adverse_gap_drift_per_step", max_adverse_gap_drift_per_step)
    t = int(horizon_steps)
    if a0 < 0 or drift < 0 or t < 0:
        raise ValueError("gap, drift, and horizon must be non-negative")
    x = a0 + drift * t
    if x > 700:
        return 0.0
    return 1.0 / (1.0 + (k - 1) * math.exp(x))


def max_router_gap_drift_for_floor(
    num_paths: int,
    initial_adverse_gap: float,
    horizon_steps: int,
    minimum_probability: float,
) -> float:
    """Largest per-step adverse logit-gap drift compatible with a p_min floor."""
    k = int(num_paths)
    t = int(horizon_steps)
    a0 = _finite("initial_adverse_gap", initial_adverse_gap)
    p = _finite("minimum_probability", minimum_probability)
    if k < 2 or t <= 0 or a0 < 0:
        raise ValueError("need num_paths>=2, horizon_steps>0, initial gap>=0")
    if not 0.0 < p < 1.0 / k:
        raise ValueError("minimum_probability must lie in (0, 1/num_paths)")
    gap_budget = math.log((1.0 / p - 1.0) / (k - 1))
    return (gap_budget - a0) / t


def write_temperature_step_bound(jacobian_bound: float, output_change_budget: float) -> float:
    """Maximum |Δtau| allowed by ||dw/dtau|| <= J and ||Δw|| <= eps."""
    j = _finite("jacobian_bound", jacobian_bound)
    eps = _finite("output_change_budget", output_change_budget)
    if j < 0 or eps < 0:
        raise ValueError("bounds must be non-negative")
    if j == 0:
        return math.inf
    return eps / j


def hard_selection_safe(score_margin: float, score_perturbation_linf: float) -> bool:
    """Whether a hard Top-K set is guaranteed unchanged by a score perturbation."""
    gamma = _finite("score_margin", score_margin)
    eps = _finite("score_perturbation_linf", score_perturbation_linf)
    if gamma < 0 or eps < 0:
        raise ValueError("margin and perturbation must be non-negative")
    return 2.0 * eps < gamma


def episodic_capacity_feasible(required_exact_items: int, available_slots: int) -> bool:
    """Hard finite-slot feasibility check for simultaneously required exact items."""
    req, slots = int(required_exact_items), int(available_slots)
    if req < 0 or slots < 0:
        raise ValueError("counts must be non-negative")
    return req <= slots


def jepa_stage_isolation_ok(
    *,
    lambda_jepa: float,
    use_token_jepa_aux: bool,
    use_jepa_writer_bias: bool,
    stage: int,
) -> bool:
    """Check the staged JEPA reintegration contract.

    stage 0: JEPA may be instantiated/measured but contributes no loss and no writer bias.
    stage 1: block-level JEPA loss may train shared representations; token JEPA and writer bias off.
    stage 2: token JEPA may train, but writer bias remains off.
    stage 3: writer bias may be enabled after earlier stages are independently accepted.
    """
    s = int(stage)
    lam = _finite("lambda_jepa", lambda_jepa)
    if s not in {0, 1, 2, 3} or lam < 0:
        raise ValueError("invalid JEPA stage or lambda")
    if s == 0:
        return lam == 0.0 and not use_token_jepa_aux and not use_jepa_writer_bias
    if s == 1:
        return lam > 0.0 and not use_token_jepa_aux and not use_jepa_writer_bias
    if s == 2:
        return lam > 0.0 and bool(use_token_jepa_aux) and not use_jepa_writer_bias
    return lam > 0.0 and bool(use_token_jepa_aux) and bool(use_jepa_writer_bias)


def finite_gradient_ratio(aux_norm: float, primary_norm: float) -> float:
    """Stable auxiliary/primary gradient-norm ratio used by JEPA integration gates."""
    a = _finite("aux_norm", aux_norm)
    p = _finite("primary_norm", primary_norm)
    if a < 0 or p < 0:
        raise ValueError("gradient norms must be non-negative")
    if p == 0:
        return math.inf if a > 0 else 0.0
    return a / p
