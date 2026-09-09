from __future__ import annotations

import math

import pytest

from hpm.router_theory import (
    bounded_softmax_probability_interval,
    necessary_path_excess_risk_floor,
    necessary_support_mass_floor,
    recovery_steps_lower_bound,
    required_path_probability,
    starvation_gradient_upper_bound,
    two_path_softmax_sensitivity,
)


def test_necessary_path_excess_risk_floor_is_product():
    assert necessary_path_excess_risk_floor(0.25, 0.4) == pytest.approx(0.1)
    assert necessary_path_excess_risk_floor(0.0, 0.4) == 0.0


def test_required_path_probability_is_stable_for_large_deficits():
    assert required_path_probability(0.0) == pytest.approx(0.5)
    assert required_path_probability(10.0) == pytest.approx(1.0 / (1.0 + math.exp(10.0)))
    assert required_path_probability(1000.0) == 0.0
    assert required_path_probability(-1000.0) == 1.0


def test_softmax_starvation_sensitivity_decays_exponentially():
    s2 = two_path_softmax_sensitivity(2.0)
    s8 = two_path_softmax_sensitivity(8.0)
    assert s8 < s2 * 0.01
    assert starvation_gradient_upper_bound(8.0, 3.0) == pytest.approx(3.0 * s8)


def test_recovery_lower_bound_grows_exponentially_with_initial_deficit():
    n4 = recovery_steps_lower_bound(4.0, 1.0, learning_rate=1.0e-3, relative_utility_bound=1.0)
    n8 = recovery_steps_lower_bound(8.0, 1.0, learning_rate=1.0e-3, relative_utility_bound=1.0)
    assert n8 > n4 * 40


def test_bounded_softmax_interval_matches_hpm_c3_number():
    p_min, p_max = bounded_softmax_probability_interval(4, 3.0)
    assert p_min == pytest.approx(0.0008255688, rel=1.0e-5)
    assert p_max == pytest.approx(0.9926186, rel=1.0e-5)
    assert 0.0 < p_min < 0.25 < p_max < 1.0


@pytest.mark.parametrize(
    "fn,args",
    [
        (necessary_path_excess_risk_floor, (-0.1, 0.2)),
        (necessary_path_excess_risk_floor, (0.1, -0.2)),
        (recovery_steps_lower_bound, (0.0, 1.0, 0.1, 1.0)),
        (recovery_steps_lower_bound, (2.0, 3.0, 0.1, 1.0)),
        (bounded_softmax_probability_interval, (1, 3.0)),
        (bounded_softmax_probability_interval, (4, 0.0)),
    ],
)
def test_invalid_theory_inputs_are_rejected(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_functional_necessity_implies_static_routing_mass_floor():
    # delta <= L H rho  =>  rho >= delta/(L H)
    assert necessary_support_mass_floor(0.30, loss_lipschitz=2.0, path_norm_bound=3.0) == pytest.approx(0.05)
    assert necessary_support_mass_floor(0.0, loss_lipschitz=2.0, path_norm_bound=3.0) == 0.0


def test_support_mass_floor_rejects_inconsistent_bounds():
    with pytest.raises(ValueError, match="incompatible"):
        necessary_support_mass_floor(1.1, loss_lipschitz=1.0, path_norm_bound=1.0)
