from __future__ import annotations

import math

import pytest

from hpm.write_transition_theory import (
    bernoulli_branch_expected_lipschitz_constant,
    bernoulli_branch_expected_metric,
    hard_topk_membership_margin,
    schedule_step_lipschitz_bound,
)


def test_schedule_bounds():
    assert schedule_step_lipschitz_bound(100, "linear") == pytest.approx(0.01)
    assert schedule_step_lipschitz_bound(100, "cosine") == pytest.approx(math.pi / 200.0)
    assert math.isinf(schedule_step_lipschitz_bound(0, "linear"))


def test_fixed_branch_expectation_is_affine_and_lipschitz():
    assert bernoulli_branch_expected_metric(0.25, 1.0, 0.2) == pytest.approx(0.4)
    assert bernoulli_branch_expected_lipschitz_constant(1.0, 0.2) == pytest.approx(0.8)
    # For an accuracy metric in [0,1], this constant can never exceed 1.
    assert bernoulli_branch_expected_lipschitz_constant(1.0, 0.0) == 1.0


def test_hard_topk_positive_margin_is_locally_stable():
    margin = hard_topk_membership_margin([4.0, 3.0, 1.0, 0.0], 2)
    assert margin == pytest.approx(2.0)
    # Any coordinatewise perturbation < margin/2 cannot swap the boundary pair.
    assert margin / 2.0 == pytest.approx(1.0)


def test_hard_topk_tie_has_zero_margin():
    assert hard_topk_membership_margin([4.0, 2.0, 2.0, 0.0], 2) == 0.0


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_invalid_probability_is_rejected(p):
    with pytest.raises(ValueError):
        bernoulli_branch_expected_metric(p, 1.0, 0.0)
