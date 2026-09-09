"""Pure utilities for fixed-router path-contribution diagnostics.

Router weights are observational: a large or small softmax mass does not
establish that a path is functionally necessary.  The corresponding causal
intervention for HPM is to retain the full model's per-token router weights
and set selected *path contributions* to zero only in the final mixture.  The
gate consequently cannot re-route around the intervention, either during
training or in the altered forward pass.

For a set R of removed paths, write A_s(R) for exact accuracy on subtask s
under that intervention.  The conditional loss from additionally removing
path i is

    D_{s,i | R} = A_s(R) - A_s(R union {i}).

The Shapley value below averages this quantity over every possible set R of
already-removed alternatives.  It is a descriptive functional attribution;
it is not a claim that the four path states are statistically independent.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial, isfinite
from typing import Dict, Iterable, Mapping, Sequence, Tuple


ROUTER_PATH_NAMES: Tuple[str, ...] = ("local", "recurrent", "fast_weight", "episodic")


def canonical_removed_paths(removed_paths: Iterable[str], path_names: Sequence[str] = ROUTER_PATH_NAMES) -> Tuple[str, ...]:
    """Return a validated removal set in the architecture's fixed path order."""

    requested = tuple(removed_paths)
    known = set(path_names)
    unknown = sorted(set(requested).difference(known))
    if unknown:
        raise ValueError(f"unknown router path(s): {unknown!r}; choose from {tuple(path_names)!r}")
    if len(set(requested)) != len(requested):
        raise ValueError("removed paths must be unique")
    return tuple(path for path in path_names if path in set(requested))


def all_removed_path_sets(path_names: Sequence[str] = ROUTER_PATH_NAMES) -> Tuple[Tuple[str, ...], ...]:
    """Enumerate all 2^P removal sets in deterministic cardinality/order."""

    names = tuple(path_names)
    if len(set(names)) != len(names) or not names:
        raise ValueError("path_names must be a non-empty sequence of unique names")
    return tuple(
        subset
        for count in range(len(names) + 1)
        for subset in combinations(names, count)
    )


def removal_key(removed_paths: Iterable[str], path_names: Sequence[str] = ROUTER_PATH_NAMES) -> str:
    """Stable external key for a frozen-router intervention condition."""

    removed = canonical_removed_paths(removed_paths, path_names)
    return "full" if not removed else "drop_" + "__".join(removed)


def shapley_path_attribution(
    accuracy_by_removed_set: Mapping[Tuple[str, ...], float],
    path_names: Sequence[str] = ROUTER_PATH_NAMES,
) -> Dict[str, float]:
    """Average conditional accuracy loss from each path's removal.

    ``accuracy_by_removed_set`` must contain every removal set, including
    ``()`` for the unablated model.  Its values are usually exact accuracies
    from a shared held-out stream.  The returned values sum to
    ``A(()) - A(all_paths)`` by the Shapley efficiency identity.
    """

    names = tuple(path_names)
    expected = set(all_removed_path_sets(names))
    actual = {canonical_removed_paths(removed, names) for removed in accuracy_by_removed_set}
    if actual != expected:
        raise ValueError(
            "accuracy_by_removed_set must provide every removal set: "
            f"missing={sorted(expected.difference(actual))!r}, extra={sorted(actual.difference(expected))!r}"
        )
    normalized = {
        canonical_removed_paths(removed, names): float(value)
        for removed, value in accuracy_by_removed_set.items()
    }
    if not all(isfinite(value) for value in normalized.values()):
        raise ValueError("all intervention accuracies must be finite")

    count = len(names)
    denominator = float(factorial(count))
    result: Dict[str, float] = {}
    for path in names:
        value = 0.0
        alternatives = tuple(other for other in names if other != path)
        for removed_count in range(len(alternatives) + 1):
            for removed in combinations(alternatives, removed_count):
                base = canonical_removed_paths(removed, names)
                with_path = canonical_removed_paths((*removed, path), names)
                coefficient = factorial(removed_count) * factorial(count - removed_count - 1) / denominator
                value += coefficient * (normalized[base] - normalized[with_path])
        result[path] = value
    return result
