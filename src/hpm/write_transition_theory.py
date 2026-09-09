"""Theory-only helpers for HPM's teacher-forcing/write-transition stability track.

The current training loop uses a Bernoulli teacher-forcing branch.  A smooth
probability schedule therefore does *not* make each realized training step a
Lipschitz function of the schedule probability.  This module encodes the
narrow statements that are actually true and useful for tests/notes.
"""

from __future__ import annotations

import math


def schedule_step_lipschitz_bound(anneal_steps: int, schedule: str) -> float:
    """Upper bound on |p(s+1)-p(s)| inside the anneal window.

    For the linear schedule this is exactly 1/A.  For the cosine schedule,
    viewing the schedule as a continuous function of step, the derivative is
    bounded by pi/(2A), which also bounds adjacent-step differences.
    """

    a = int(anneal_steps)
    if a <= 0:
        return math.inf  # the hard cutover is discontinuous in the continuous-step view
    if schedule == "linear":
        return 1.0 / a
    if schedule == "cosine":
        return math.pi / (2.0 * a)
    raise ValueError("schedule must be 'linear' or 'cosine'")


def bernoulli_branch_expected_metric(p: float, oracle_metric: float, self_metric: float) -> float:
    """Expected fixed-model metric under a Bernoulli teacher-forcing branch.

    This applies only when the two branch metrics themselves do not depend on
    p.  Under that assumption E[M|p] = p M_oracle + (1-p) M_self.
    """

    p = float(p)
    a = float(oracle_metric)
    b = float(self_metric)
    if not (math.isfinite(p) and math.isfinite(a) and math.isfinite(b)):
        raise ValueError("inputs must be finite")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return p * a + (1.0 - p) * b


def bernoulli_branch_expected_lipschitz_constant(oracle_metric: float, self_metric: float) -> float:
    """Exact Lipschitz constant of the fixed-branch expectation in p."""

    a = float(oracle_metric)
    b = float(self_metric)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("metrics must be finite")
    return abs(a - b)


def hard_topk_membership_margin(scores: list[float] | tuple[float, ...], k: int) -> float:
    """Return the hard top-k boundary margin score[k-1] - score[k].

    A strictly positive margin gives a local robustness radius: any
    coordinatewise perturbation smaller than half this margin preserves the
    selected top-k set.  Zero margin means an arbitrarily small perturbation
    can change membership, so no local set-valued Lipschitz claim is possible
    at that point.
    """

    vals = [float(v) for v in scores]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("scores must be finite")
    if not 1 <= int(k) < len(vals):
        raise ValueError("need 1 <= k < len(scores)")
    ordered = sorted(vals, reverse=True)
    return ordered[int(k) - 1] - ordered[int(k)]
