"""Small exact helpers for HPM's necessary-path preservation theory.

This module deliberately contains *no training intervention*.  It encodes the
mathematical quantities used by the router non-collapse theory track so tests,
notes, and future experiments refer to one precise convention.

Conventions
-----------
For the two-path starvation helpers, ``deficit`` means

    deficit = z_competitor - z_required >= 0.

Thus the required path probability is ``sigmoid(-deficit)``.  A larger deficit
means the required path is more heavily starved by softmax.
"""

from __future__ import annotations

import math


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def necessary_path_excess_risk_floor(region_probability: float, conditional_excess_risk: float) -> float:
    """Return the population excess-risk floor implied by path necessity.

    If a measurable task region Ω has probability p and removing a path incurs
    conditional expected excess risk at least δ on Ω, while the removal effect
    outside Ω is non-negative, then

        R(model_without_path) - R(full_model) >= p * δ.

    This helper returns that lower bound.  The non-negative-outside-Ω premise
    is a theorem assumption; it is intentionally not smuggled into this
    numerical function.
    """

    p = _finite("region_probability", region_probability)
    delta = _finite("conditional_excess_risk", conditional_excess_risk)
    if not 0.0 <= p <= 1.0:
        raise ValueError("region_probability must lie in [0, 1]")
    if delta < 0.0:
        raise ValueError("conditional_excess_risk must be non-negative")
    return p * delta



def necessary_support_mass_floor(
    conditional_removal_penalty: float,
    loss_lipschitz: float,
    path_norm_bound: float,
) -> float:
    """Lower-bound routed mass on a causally necessary path set.

    Let A be a set of paths and, at a token/example,

        m      = sum_k pi_k h_k
        m_-A   = sum_{k notin A} pi_k h_k
        rho_A  = sum_{k in A} pi_k.

    The frozen-gate post-routing ablation used by HPM has

        ||m - m_-A|| <= H rho_A

    whenever every removed path representation satisfies ||h_k|| <= H.  If
    the downstream loss as a function of the routed representation is
    L-Lipschitz, then

        loss(m_-A) - loss(m) <= L H rho_A.

    Therefore a conditional expected removal penalty at least delta implies

        E[rho_A | Omega] >= delta / (L H).

    This is a *static* non-collapse consequence of functional necessity for a
    fixed trained model.  It does not prove that gradient training will avoid
    starving A before the useful representation is learned.
    """

    delta = _finite("conditional_removal_penalty", conditional_removal_penalty)
    lipschitz = _finite("loss_lipschitz", loss_lipschitz)
    h_bound = _finite("path_norm_bound", path_norm_bound)
    if delta < 0.0:
        raise ValueError("conditional_removal_penalty must be non-negative")
    if lipschitz <= 0.0:
        raise ValueError("loss_lipschitz must be positive")
    if h_bound <= 0.0:
        raise ValueError("path_norm_bound must be positive")
    floor = delta / (lipschitz * h_bound)
    # rho_A is a probability mass and cannot exceed one.  If assumptions plus
    # observed delta demand >1, the assumed L/H bounds are inconsistent with
    # the observation; fail loudly instead of silently clipping the theorem.
    if floor > 1.0 + 1.0e-12:
        raise ValueError(
            "observed removal penalty is incompatible with the supplied "
            "loss/path bounds (delta/(L*H) > 1)"
        )
    return min(1.0, floor)

def required_path_probability(deficit: float) -> float:
    """Stable two-path softmax probability of the required path.

    ``deficit = z_competitor - z_required``.  The result equals
    ``1 / (1 + exp(deficit))`` without overflowing for large deficits.
    """

    d = _finite("deficit", deficit)
    if d >= 0.0:
        e = math.exp(-d)
        return e / (1.0 + e)
    e = math.exp(d)
    return 1.0 / (1.0 + e)


def two_path_softmax_sensitivity(deficit: float) -> float:
    """Return p(1-p), the two-path softmax gap sensitivity."""

    p = required_path_probability(deficit)
    return p * (1.0 - p)


def starvation_gradient_upper_bound(deficit: float, relative_utility_bound: float) -> float:
    """Upper-bound |∂L/∂Δ| for a starved required path in a two-path router.

    Let Δ = z_required - z_competitor = -deficit and suppose

        |<∇_m L, h_required - h_competitor>| <= B.

    Then

        |∂L/∂Δ| <= B p(1-p),

    where p is the required path's softmax probability.  For large positive
    deficits this decays exponentially, approximately B exp(-deficit).
    """

    b = _finite("relative_utility_bound", relative_utility_bound)
    if b < 0.0:
        raise ValueError("relative_utility_bound must be non-negative")
    return b * two_path_softmax_sensitivity(deficit)


def recovery_steps_lower_bound(
    initial_deficit: float,
    target_logit_gain: float,
    learning_rate: float,
    relative_utility_bound: float,
) -> int:
    """Conservative lower bound on GD steps needed to recover logit gap.

    Consider ordinary gradient descent on the two-path logit gap Δ, starting
    at Δ_0 = -A with A = ``initial_deficit`` > 0.  Assume the relative utility
    term is bounded in magnitude by B at every step and ask how many steps are
    required merely to increase Δ by ``g`` (0 < g <= A).

    While Δ remains in [-A, -A+g],

        p(1-p) <= exp(Δ) <= exp(-A+g),

    so one favorable GD step can increase Δ by at most

        η B exp(-A+g).

    Hence at least

        g exp(A-g) / (η B)

    steps are necessary.  This is intentionally a *lower* bound under
    maximally favorable direction; it demonstrates the exponential lock-in
    mechanism rather than predicting actual training time.
    """

    a = _finite("initial_deficit", initial_deficit)
    g = _finite("target_logit_gain", target_logit_gain)
    eta = _finite("learning_rate", learning_rate)
    b = _finite("relative_utility_bound", relative_utility_bound)
    if a <= 0.0:
        raise ValueError("initial_deficit must be positive")
    if g <= 0.0 or g > a:
        raise ValueError("target_logit_gain must satisfy 0 < gain <= initial_deficit")
    if eta <= 0.0:
        raise ValueError("learning_rate must be positive")
    if b <= 0.0:
        raise ValueError("relative_utility_bound must be positive")
    lower = g * math.exp(a - g) / (eta * b)
    return max(1, math.ceil(lower))


def bounded_softmax_probability_interval(num_paths: int, logit_bound: float) -> tuple[float, float]:
    """Probability interval induced by effective logits in [-C, C].

    For K paths with every effective logit z_i in [-C, C], each softmax weight
    lies in

        [1 / (1 + (K-1)e^(2C)),
         1 / (1 + (K-1)e^(-2C))].

    This is a forward probability guarantee only; it says nothing about the
    raw-logit gradient if the bound is implemented by a saturating transform.
    """

    k = int(num_paths)
    c = _finite("logit_bound", logit_bound)
    if k < 2:
        raise ValueError("num_paths must be >= 2")
    if c <= 0.0:
        raise ValueError("logit_bound must be positive")
    # Stable equivalent forms avoid overflow for large C.
    e_neg = math.exp(-2.0 * c)
    p_max = 1.0 / (1.0 + (k - 1) * e_neg)
    p_min = e_neg / (e_neg + (k - 1))
    return p_min, p_max
