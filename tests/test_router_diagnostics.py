"""Tests for hpm/router_diagnostics.py -- the pure utilities behind the
frozen-router path-dependence diagnostic (exact Shapley over the 2^P removal
sets, stable condition keys, canonical path ordering).

Written 2026-08-26 before the diagnostic had ever been executed anywhere
(no torch was even installed locally until that day): these pin the
arithmetic and the key format that downstream consumers -- the cross-seed
analyzer and the ts-gate certification CSV -- silently depend on. If
removal_key's join format or the Shapley efficiency identity ever drifts,
these fail before any GPU sweep wastes its time.
"""
import math
import random
from itertools import permutations

import pytest

from hpm.router_diagnostics import (
    ROUTER_PATH_NAMES,
    all_removed_path_sets,
    canonical_removed_paths,
    removal_key,
    shapley_path_attribution,
)


def test_canonical_removed_paths_reorders_into_architecture_order():
    # Input order is irrelevant; output order is the architecture's fixed
    # path order (the same order contribution masks are built in).
    assert canonical_removed_paths(("episodic", "local")) == ("local", "episodic")
    assert canonical_removed_paths(tuple(reversed(ROUTER_PATH_NAMES))) == ROUTER_PATH_NAMES


def test_canonical_removed_paths_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown router path"):
        canonical_removed_paths(("attention",))
    with pytest.raises(ValueError, match="unique"):
        canonical_removed_paths(("local", "local"))


def test_all_removed_path_sets_enumerates_all_subsets_deterministically():
    sets = all_removed_path_sets(ROUTER_PATH_NAMES)
    assert len(sets) == 2 ** len(ROUTER_PATH_NAMES) == 16
    assert len(set(sets)) == 16  # all distinct
    assert sets[0] == ()  # unablated first
    assert sets[-1] == ROUTER_PATH_NAMES  # everything-removed last
    # cardinality-monotone, so downstream "removed_sets[1:] skips full" logic holds
    assert all(len(a) <= len(b) for a, b in zip(sets, sets[1:]))
    with pytest.raises(ValueError):
        all_removed_path_sets(("local", "local"))


def test_removal_key_format_is_stable():
    assert removal_key(()) == "full"
    assert removal_key(("local",)) == "drop_local"
    assert removal_key(("episodic",)) == "drop_episodic"
    # key is order-insensitive because canonicalization runs first
    assert removal_key(("episodic", "local")) == removal_key(("local", "episodic"))
    expected_all = "drop_" + "__".join(canonical_removed_paths(ROUTER_PATH_NAMES))
    assert removal_key(ROUTER_PATH_NAMES) == expected_all


def _random_game(rng: random.Random, names):
    """A deterministic synthetic set function over every removal subset."""
    subsets = all_removed_path_sets(names)
    return {subset: rng.uniform(0.0, 1.0) for subset in subsets}


def _brute_force_shapley(values, names):
    """Independent reference implementation: average marginal contributions
    over all P! player permutations.

    Normalization subtlety this got wrong once: stripping ``path`` out of a
    full permutation leaves the other P-1 players' relative order, and each
    such ordering occurs P times across the P! permutations (the removed
    player's P possible insertion positions). Totals therefore have to be
    divided by P * P!, not P! -- otherwise every attribution comes out
    inflated by exactly P.
    """
    result = {}
    scale = len(names) * math.factorial(len(names))
    for path in names:
        total = 0.0
        for perm in permutations(names):
            others = tuple(p for p in perm if p != path)
            for k in range(len(others) + 1):
                coalition = frozenset(others[:k])
                marginal = (
                    values[tuple(n for n in names if n in coalition)]
                    - values[
                        canonical_removed_paths(
                            (*others[:k], path), names
                        )
                    ]
                )
                total += marginal
        result[path] = total / scale
    return result


def test_shapley_matches_brute_force_permutation_average():
    rng = random.Random(20260826)
    values = _random_game(rng, ROUTER_PATH_NAMES)
    exact = shapley_path_attribution(values)
    brute = _brute_force_shapley({k: v for k, v in values.items()}, ROUTER_PATH_NAMES)
    for path in ROUTER_PATH_NAMES:
        assert exact[path] == pytest.approx(brute[path], abs=1e-9)


def test_shapley_efficiency_identity_sum_equals_full_minus_all_removed():
    rng = random.Random(7)
    values = _random_game(rng, ROUTER_PATH_NAMES)
    attribution = shapley_path_attribution(values)
    total = sum(attribution.values())
    assert total == pytest.approx(values[()] - values[ROUTER_PATH_NAMES], abs=1e-9)


def test_shapley_closed_form_only_episodic_matters():
    # If accuracy depends ONLY on whether episodic is removed, then only
    # episodic can carry attribution, and its value is the full gap: every
    # marginal contribution of episodic is A(no-epi) - A(epi-removed).
    gap = 0.7
    values = {
        subset: (0.3 if "episodic" in subset else 0.3 + gap)
        for subset in all_removed_path_sets(ROUTER_PATH_NAMES)
    }
    attribution = shapley_path_attribution(values)
    assert attribution["episodic"] == pytest.approx(gap, abs=1e-12)
    for path in ("local", "recurrent", "fast_weight"):
        assert attribution[path] == pytest.approx(0.0, abs=1e-12)
    assert sum(attribution.values()) == pytest.approx(gap, abs=1e-12)


def test_shapley_constant_game_attributes_zero_everywhere():
    values = {subset: 0.5 for subset in all_removed_path_sets(ROUTER_PATH_NAMES)}
    attribution = shapley_path_attribution(values)
    assert all(value == pytest.approx(0.0, abs=1e-12) for value in attribution.values())


def test_shapley_rejects_incomplete_extra_or_nonfinite_games():
    complete = {subset: 0.5 for subset in all_removed_path_sets(ROUTER_PATH_NAMES)}

    missing_one = {k: v for k, v in complete.items() if k != ("local",)}
    with pytest.raises(ValueError, match="must provide every removal set"):
        shapley_path_attribution(missing_one)

    extra = dict(complete)
    extra[("bogus",)] = 0.5
    with pytest.raises(ValueError):
        shapley_path_attribution(extra)

    nan_game = dict(complete)
    nan_game[("local",)] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        shapley_path_attribution(nan_game)

    inf_game = dict(complete)
    inf_game[("episodic",)] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        shapley_path_attribution(inf_game)
