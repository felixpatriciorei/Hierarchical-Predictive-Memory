#!/usr/bin/env python3
"""
CPU-only constructive self-improvement sandbox for HPM's zpoly domain.

This does NOT train HPM or any neural model.

Gate:
  1. Start with only 8 trusted primitive rewrites.
  2. Solve discovery tasks with BFS.
  3. Mine ONLY complete two-action proof traces (fixed binary induction budget).
  4. Anti-unify their before/after states into a candidate macro.
  5. Check candidate conservativity by exact replay + polynomial normalization.
  6. Retain only if it reduces BFS work on a separate validation set.
  7. Use retained macros to generate the next curriculum.
  8. Repeat for three abstraction generations.
  9. Evaluate every library generation on the SAME untouched held-out suite.
 10. Ablate the previous generation and confirm the next generation is no
     longer discoverable under the same binary induction budget.

The desired "glimpse" is an actual causal chain:
  L0 -> L1 -> L2 -> L3
with held-out search cost decreasing each time, and deeper candidates becoming
proposal-reachable only because prior abstractions compressed earlier proofs.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

AST = dict[str, Any]
ALPHA = 0.05

# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

def M(name: str) -> AST:
    return {"meta": name}


def V(name: str) -> AST:
    return {"var": name}


def C(value: int) -> AST:
    return {"const": value}


def Add(a: AST, b: AST) -> AST:
    return {"op": "add", "args": [a, b]}


def Mul(a: AST, b: AST) -> AST:
    return {"op": "mul", "args": [a, b]}


def canonical_json(node: AST) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def same_ast(a: AST, b: AST) -> bool:
    return canonical_json(a) == canonical_json(b)


def is_op(node: AST, op: str) -> bool:
    return (
        node.get("op") == op
        and isinstance(node.get("args"), list)
        and len(node["args"]) == 2
    )


def iter_paths(expr: AST, path: tuple[int, ...] = ()) -> Iterable[tuple[int, ...]]:
    yield path
    if "op" in expr:
        yield from iter_paths(expr["args"][0], path + (0,))
        yield from iter_paths(expr["args"][1], path + (1,))


def get_at_path(expr: AST, path: tuple[int, ...]) -> AST:
    cur = expr
    for idx in path:
        cur = cur["args"][idx]
    return cur


def replace_at_path(expr: AST, path: tuple[int, ...], replacement: AST) -> AST:
    if not path:
        return copy.deepcopy(replacement)
    out = copy.deepcopy(expr)
    cur = out
    for idx in path[:-1]:
        cur = cur["args"][idx]
    cur["args"][path[-1]] = copy.deepcopy(replacement)
    return out


def meta_names(expr: AST) -> set[str]:
    if "meta" in expr:
        return {expr["meta"]}
    if "var" in expr or "const" in expr:
        return set()
    return meta_names(expr["args"][0]) | meta_names(expr["args"][1])


def match_pattern(
    pattern: AST,
    expr: AST,
    env: dict[str, AST] | None = None,
) -> dict[str, AST] | None:
    env = {} if env is None else dict(env)

    if "meta" in pattern:
        name = pattern["meta"]
        if name in env:
            return env if same_ast(env[name], expr) else None
        env[name] = copy.deepcopy(expr)
        return env

    if "var" in pattern or "const" in pattern:
        return env if same_ast(pattern, expr) else None

    if pattern.get("op") != expr.get("op"):
        return None

    env = match_pattern(pattern["args"][0], expr["args"][0], env)
    if env is None:
        return None
    return match_pattern(pattern["args"][1], expr["args"][1], env)


def instantiate(pattern: AST, env: dict[str, AST]) -> AST:
    if "meta" in pattern:
        return copy.deepcopy(env[pattern["meta"]])
    if "var" in pattern or "const" in pattern:
        return copy.deepcopy(pattern)
    return {
        "op": pattern["op"],
        "args": [
            instantiate(pattern["args"][0], env),
            instantiate(pattern["args"][1], env),
        ],
    }


def nary_lgg(exprs: list[AST]) -> AST:
    """N-ary least-general generalization over tree structure."""
    if len(exprs) < 2:
        raise ValueError("LGG requires >=2 expressions")

    counter = [0]
    memo: dict[tuple[str, ...], str] = {}

    def rec(items: list[AST]) -> AST:
        keys = tuple(canonical_json(x) for x in items)
        if all(k == keys[0] for k in keys):
            return copy.deepcopy(items[0])

        first_op = items[0].get("op")
        if (
            first_op in {"add", "mul"}
            and all(x.get("op") == first_op for x in items)
        ):
            return {
                "op": first_op,
                "args": [
                    rec([x["args"][0] for x in items]),
                    rec([x["args"][1] for x in items]),
                ],
            }

        if keys not in memo:
            memo[keys] = f"A{counter[0]}"
            counter[0] += 1
        return M(memo[keys])

    return rec(exprs)


# ---------------------------------------------------------------------------
# Exact polynomial checker
# ---------------------------------------------------------------------------

Poly = dict[tuple[str, ...], int]


def poly_add(a: Poly, b: Poly) -> Poly:
    out = dict(a)
    for mono, coeff in b.items():
        out[mono] = out.get(mono, 0) + coeff
        if out[mono] == 0:
            del out[mono]
    return out


def poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            mono = tuple(sorted(ma + mb))
            out[mono] = out.get(mono, 0) + ca * cb
            if out[mono] == 0:
                del out[mono]
    return out


def polynomial_normal_form(expr: AST) -> Poly:
    if "const" in expr:
        value = int(expr["const"])
        return {} if value == 0 else {(): value}
    if "var" in expr:
        return {(f"var:{expr['var']}",): 1}
    if "meta" in expr:
        return {(f"meta:{expr['meta']}",): 1}
    if is_op(expr, "add"):
        return poly_add(
            polynomial_normal_form(expr["args"][0]),
            polynomial_normal_form(expr["args"][1]),
        )
    if is_op(expr, "mul"):
        return poly_mul(
            polynomial_normal_form(expr["args"][0]),
            polynomial_normal_form(expr["args"][1]),
        )
    raise ValueError(f"bad AST: {expr!r}")


# ---------------------------------------------------------------------------
# Actions and solver
# ---------------------------------------------------------------------------

@dataclass
class Action:
    name: str
    lhs: AST
    rhs: AST
    generation: int
    expansion: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    support_ids: list[str] = field(default_factory=list)
    validation_gain: dict[str, float] = field(default_factory=dict)


def primitive_actions() -> list[Action]:
    return [
        Action("ADD_ZERO_R", Add(M("A"), C(0)), M("A"), 0),
        Action("ADD_ZERO_L", Add(C(0), M("A")), M("A"), 0),
        Action("MUL_ONE_R", Mul(M("A"), C(1)), M("A"), 0),
        Action("MUL_ONE_L", Mul(C(1), M("A")), M("A"), 0),
        Action("MUL_ZERO_R", Mul(M("A"), C(0)), C(0), 0),
        Action("MUL_ZERO_L", Mul(C(0), M("A")), C(0), 0),
        Action(
            "DISTRIB_L",
            Mul(M("A"), Add(M("B"), M("C"))),
            Add(Mul(M("A"), M("B")), Mul(M("A"), M("C"))),
            0,
        ),
        Action(
            "DISTRIB_R",
            Mul(Add(M("A"), M("B")), M("C")),
            Add(Mul(M("A"), M("C")), Mul(M("B"), M("C"))),
            0,
        ),
    ]


def enumerate_action(expr: AST, action: Action):
    for path in iter_paths(expr):
        env = match_pattern(action.lhs, get_at_path(expr, path))
        if env is not None:
            yield path, replace_at_path(
                expr,
                path,
                instantiate(action.rhs, env),
            )


def apply_action_at_path(
    expr: AST,
    action: Action,
    path: tuple[int, ...],
) -> AST | None:
    target = get_at_path(expr, path)
    env = match_pattern(action.lhs, target)
    if env is None:
        return None
    return replace_at_path(expr, path, instantiate(action.rhs, env))


@dataclass
class SearchResult:
    success: bool
    trace: list[tuple[str, tuple[int, ...]]]
    nodes_expanded: int
    successors_generated: int
    wall_seconds: float


def bfs_solve(
    start: AST,
    goal: AST,
    library: list[Action],
    node_budget: int = 50000,
) -> SearchResult:
    started = time.perf_counter()
    queue = collections.deque([(copy.deepcopy(start), [])])
    seen = {canonical_json(start)}
    nodes_expanded = 0
    successors_generated = 0

    while queue and nodes_expanded < node_budget:
        state, trace = queue.popleft()
        nodes_expanded += 1

        if same_ast(state, goal):
            return SearchResult(
                True,
                trace,
                nodes_expanded,
                successors_generated,
                time.perf_counter() - started,
            )

        for action in library:
            for path, nxt in enumerate_action(state, action):
                successors_generated += 1
                new_trace = trace + [(action.name, path)]
                if same_ast(nxt, goal):
                    return SearchResult(
                        True,
                        new_trace,
                        nodes_expanded,
                        successors_generated,
                        time.perf_counter() - started,
                    )
                key = canonical_json(nxt)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((nxt, new_trace))

    return SearchResult(
        False,
        [],
        nodes_expanded,
        successors_generated,
        time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# Seed curriculum
# ---------------------------------------------------------------------------

# Only the cold-start curriculum is externally specified.
# Later curricula are generated from retained actions themselves.
def seed_wrapper_A(term: AST) -> AST:
    return Add(Mul(term, C(1)), C(0))


def seed_wrapper_B(term: AST) -> AST:
    return Add(C(0), Mul(C(1), term))


def seed_wrapper_C(term: AST) -> AST:
    return Mul(Add(term, C(0)), C(1))


SEED_WRAPPERS = (
    ("seed_A", seed_wrapper_A),
    ("seed_B", seed_wrapper_B),
    ("seed_C", seed_wrapper_C),
)


def safe_term_pool() -> list[AST]:
    # No 0/1 and no multiplication with an Add child, so no primitive rewrite
    # fires inside these terms. They behave as irreducible payloads.
    x, y, z = V("x"), V("y"), V("z")
    return [
        x,
        y,
        z,
        Add(x, y),
        Add(y, z),
        Add(x, z),
        Mul(x, y),
        Mul(y, z),
        Mul(z, x),
        Add(x, Mul(y, z)),
        Add(y, Mul(z, x)),
        Add(z, Mul(x, y)),
        Mul(x, Mul(y, z)),
        Mul(y, Mul(z, x)),
        Mul(z, Mul(x, y)),
        Add(Mul(x, y), z),
        Add(Mul(y, z), x),
        Add(Mul(z, x), y),
    ]


def split_terms(seed: int):
    terms = safe_term_pool()
    rng = random.Random(seed)
    rng.shuffle(terms)
    return {
        "discovery": terms[:4],
        "validation": terms[4:5],
        "test": terms[5:7],
    }


def unary_identity(action: Action) -> bool:
    names = meta_names(action.lhs)
    return (
        len(names) == 1
        and action.rhs == {"meta": next(iter(names))}
    )


def wrap_with_action(action: Action, term: AST) -> AST:
    names = meta_names(action.lhs)
    if len(names) != 1:
        raise ValueError(f"{action.name} is not unary")
    name = next(iter(names))
    if action.rhs != {"meta": name}:
        raise ValueError(f"{action.name} is not an identity wrapper")
    return instantiate(action.lhs, {name: term})


def seed_round_tasks(terms: list[AST], prefix: str):
    tasks = []
    for family, wrapper in SEED_WRAPPERS:
        for idx, term in enumerate(terms):
            tasks.append(
                {
                    "id": f"{prefix}:{family}:{idx}",
                    "start": wrapper(term),
                    "goal": term,
                    "family": family,
                }
            )
    return tasks


def compose_action_tasks(
    outer_actions: list[Action],
    inner_actions: list[Action],
    terms: list[AST],
    prefix: str,
):
    tasks = []
    for outer in outer_actions:
        for inner in inner_actions:
            if not unary_identity(outer) or not unary_identity(inner):
                continue
            for idx, term in enumerate(terms):
                start = wrap_with_action(
                    outer,
                    wrap_with_action(inner, term),
                )
                tasks.append(
                    {
                        "id": f"{prefix}:{outer.name}:{inner.name}:{idx}",
                        "start": start,
                        "goal": term,
                        "family": f"{outer.name}>{inner.name}",
                    }
                )
    return tasks


# ---------------------------------------------------------------------------
# Induction and validation
# ---------------------------------------------------------------------------

def action_map(library: list[Action]) -> dict[str, Action]:
    return {a.name: a for a in library}


def replay_trace(
    expr: AST,
    trace: list[tuple[str, tuple[int, ...]]],
    library: list[Action],
) -> AST | None:
    amap = action_map(library)
    state = copy.deepcopy(expr)
    for name, path in trace:
        if name not in amap:
            return None
        state = apply_action_at_path(state, amap[name], path)
        if state is None:
            return None
    return state




def replace_exact_subtree(expr: AST, target: AST, replacement: AST) -> AST:
    if same_ast(expr, target):
        return copy.deepcopy(replacement)
    if "op" not in expr:
        return copy.deepcopy(expr)
    return {
        "op": expr["op"],
        "args": [
            replace_exact_subtree(expr["args"][0], target, replacement),
            replace_exact_subtree(expr["args"][1], target, replacement),
        ],
    }


def parameterize_carried_payload(
    lhs: AST,
    rhs: AST,
    expansion: list[tuple[str, tuple[int, ...]]],
    trusted_library: list[Action],
    generation: int,
) -> tuple[AST, AST]:
    """Generalize an unchanged carried payload when the proof permits it.

    If the entire RHS occurs literally inside the LHS, propose replacing that
    shared subtree with a fresh metavariable. This removes accidental structure
    shared by the discovery examples (e.g. all payloads happened to be Add
    trees). The proposal is accepted only if exact replay and polynomial
    semantics still verify, so this cannot silently widen an invalid rule.
    """
    if "const" in rhs or "meta" in rhs:
        return lhs, rhs

    # Does RHS appear as an exact subtree of LHS?
    if not any(same_ast(get_at_path(lhs, p), rhs) for p in iter_paths(lhs)):
        return lhs, rhs

    fresh = M("P0")
    lhs2 = replace_exact_subtree(lhs, rhs, fresh)
    rhs2 = fresh
    probe = Action(
        name="__probe__",
        lhs=lhs2,
        rhs=rhs2,
        generation=generation,
        expansion=list(expansion),
    )
    if check_candidate(probe, trusted_library)["pass"]:
        return lhs2, rhs2
    return lhs, rhs


def candidate_name(
    generation: int,
    lhs: AST,
    rhs: AST,
    expansion: list[tuple[str, tuple[int, ...]]],
) -> str:
    payload = json.dumps(
        {
            "lhs": lhs,
            "rhs": rhs,
            "expansion": [
                [n, list(p)] for n, p in expansion
            ],
        },
        sort_keys=True,
    ).encode()
    return f"G{generation}_{hashlib.sha256(payload).hexdigest()[:10]}"


def mine_binary_candidates(
    solved: list[tuple[dict[str, Any], SearchResult]],
    trusted_library: list[Action],
    generation: int,
    min_support: int = 2,
) -> list[Action]:
    # Fixed induction budget: only COMPLETE two-action derivations.
    groups: dict[
        tuple[tuple[str, tuple[int, ...]], ...],
        list[tuple[dict[str, Any], SearchResult]],
    ] = collections.defaultdict(list)

    for task, result in solved:
        if result.success and len(result.trace) == 2:
            signature = tuple(result.trace)
            groups[signature].append((task, result))

    candidates: list[Action] = []
    seen = set()

    for signature, support in sorted(
        groups.items(),
        key=lambda kv: repr(kv[0]),
    ):
        if len(support) < min_support:
            continue

        lhs = nary_lgg([task["start"] for task, _ in support])
        rhs = nary_lgg([task["goal"] for task, _ in support])
        lhs, rhs = parameterize_carried_payload(
            lhs,
            rhs,
            list(signature),
            trusted_library,
            generation,
        )

        name = candidate_name(
            generation,
            lhs,
            rhs,
            list(signature),
        )
        if name in seen:
            continue
        seen.add(name)

        candidate = Action(
            name=name,
            lhs=lhs,
            rhs=rhs,
            generation=generation,
            expansion=list(signature),
            support_ids=[task["id"] for task, _ in support],
        )

        checks = check_candidate(candidate, trusted_library)
        if checks["pass"]:
            candidates.append(candidate)

    return candidates


def check_candidate(
    candidate: Action,
    trusted_library: list[Action],
) -> dict[str, Any]:
    amap = action_map(trusted_library)

    trusted_expansion = all(
        name in amap and amap[name].generation < candidate.generation
        for name, _ in candidate.expansion
    )

    replayed = (
        replay_trace(
            candidate.lhs,
            candidate.expansion,
            trusted_library,
        )
        if trusted_expansion
        else None
    )

    replay_exact = (
        replayed is not None
        and same_ast(replayed, candidate.rhs)
    )

    semantic_equal = (
        polynomial_normal_form(candidate.lhs)
        == polynomial_normal_form(candidate.rhs)
    )

    rhs_symbols_subset = (
        meta_names(candidate.rhs)
        <= meta_names(candidate.lhs)
    )

    return {
        "trusted_expansion": trusted_expansion,
        "replay_exact": replay_exact,
        "semantic_equal": semantic_equal,
        "rhs_symbols_subset": rhs_symbols_subset,
        "pass": (
            trusted_expansion
            and replay_exact
            and semantic_equal
            and rhs_symbols_subset
        ),
    }


def solve_tasks(tasks, library):
    return [
        (task, bfs_solve(task["start"], task["goal"], library))
        for task in tasks
    ]


def summarize_search(solved):
    successes = [r for _, r in solved if r.success]
    n = len(solved)
    return {
        "success_rate": len(successes) / n if n else 0.0,
        "mean_action_depth": (
            sum(len(r.trace) for r in successes) / len(successes)
            if successes
            else None
        ),
        "mean_nodes_expanded": (
            sum(r.nodes_expanded for r in successes) / len(successes)
            if successes
            else None
        ),
        "mean_successors_generated": (
            sum(r.successors_generated for r in successes)
            / len(successes)
            if successes
            else None
        ),
        "total_wall_seconds": sum(r.wall_seconds for _, r in solved),
    }


def validate_and_retain(
    candidates: list[Action],
    current_library: list[Action],
    validation_tasks_by_family: dict[str, list[dict[str, Any]]],
):
    retained = []

    for candidate in candidates:
        # Candidate support tasks all came from one structural trace/family.
        # Find validation tasks where the candidate itself matches somewhere.
        eligible = []
        for tasks in validation_tasks_by_family.values():
            for task in tasks:
                if any(
                    match_pattern(
                        candidate.lhs,
                        get_at_path(task["start"], p),
                    )
                    is not None
                    for p in iter_paths(task["start"])
                ):
                    eligible.append(task)

        if not eligible:
            continue

        before = summarize_search(
            solve_tasks(eligible, current_library)
        )
        after = summarize_search(
            solve_tasks(
                eligible,
                current_library + [candidate],
            )
        )

        node_gain = (
            before["mean_nodes_expanded"]
            - after["mean_nodes_expanded"]
        )
        depth_gain = (
            before["mean_action_depth"]
            - after["mean_action_depth"]
        )

        candidate.validation_gain = {
            "node_gain": node_gain,
            "depth_gain": depth_gain,
            "before_nodes": before["mean_nodes_expanded"],
            "after_nodes": after["mean_nodes_expanded"],
            "before_depth": before["mean_action_depth"],
            "after_depth": after["mean_action_depth"],
        }

        if (
            after["success_rate"] >= before["success_rate"]
            and node_gain >= 0.0
            and depth_gain > 0.0
        ):
            retained.append(candidate)

    return retained


# ---------------------------------------------------------------------------
# Fixed final test
# ---------------------------------------------------------------------------

def fixed_test_sequences():
    # Nine depth-3 compositions; fixed before any abstraction is learned.
    return [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (0, 1, 2),
        (2, 1, 0),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
    ]


def make_fixed_test_tasks(test_terms: list[AST]):
    wrappers = [w for _, w in SEED_WRAPPERS]
    tasks = []
    for seq_idx, seq in enumerate(fixed_test_sequences()):
        for term_idx, term in enumerate(test_terms):
            state = copy.deepcopy(term)
            # seq[0] is outermost.
            for wrapper_idx in reversed(seq):
                state = wrappers[wrapper_idx](state)
            tasks.append(
                {
                    "id": f"fixed:{seq_idx}:{term_idx}",
                    "start": state,
                    "goal": term,
                    "sequence": list(seq),
                }
            )
    return tasks


def hoeffding_lb_01(mean: float, n: int) -> float:
    return mean - math.sqrt(
        math.log(1.0 / ALPHA) / (2.0 * n)
    )


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def run_seed(seed: int):
    split = split_terms(seed)
    primitives = primitive_actions()
    libraries = [list(primitives)]

    # Round 0 -> Generation 1.
    d0 = seed_round_tasks(split["discovery"], "D0")
    v0 = seed_round_tasks(split["validation"], "V0")
    s0 = solve_tasks(d0, libraries[-1])
    c1 = mine_binary_candidates(s0, libraries[-1], generation=1)
    r1 = validate_and_retain(
        c1,
        libraries[-1],
        {"seed": v0},
    )
    libraries.append(libraries[-1] + r1)

    # Round 1 curriculum is generated only from retained G1 abstractions.
    g1 = [a for a in r1 if unary_identity(a)]
    d1 = compose_action_tasks(
        g1,
        g1,
        split["discovery"],
        "D1",
    )
    v1 = compose_action_tasks(
        g1,
        g1,
        split["validation"],
        "V1",
    )
    s1 = solve_tasks(d1, libraries[-1])
    c2 = mine_binary_candidates(s1, libraries[-1], generation=2)
    r2 = validate_and_retain(
        c2,
        libraries[-1],
        {"pair": v1},
    )

    # Causal ablation: without G1, pair proofs are 4 primitive actions,
    # so fixed binary induction must discover zero G2 candidates.
    s1_ablate = solve_tasks(d1, primitives)
    c2_ablate = mine_binary_candidates(
        s1_ablate,
        primitives,
        generation=2,
    )

    libraries.append(libraries[-1] + r2)

    # Round 2 curriculum is generated from retained G1 and G2 abstractions.
    g2 = [a for a in r2 if unary_identity(a)]
    d2 = compose_action_tasks(
        g1,
        g2,
        split["discovery"],
        "D2",
    )
    v2 = compose_action_tasks(
        g1,
        g2,
        split["validation"],
        "V2",
    )
    s2 = solve_tasks(d2, libraries[-1])
    c3 = mine_binary_candidates(s2, libraries[-1], generation=3)
    r3 = validate_and_retain(
        c3,
        libraries[-1],
        {"triple": v2},
    )

    # Causal ablation: keep G1 but remove G2. Triple proofs become 3 G1
    # actions, outside the fixed two-action induction budget.
    lib_without_g2 = libraries[1]
    s2_ablate = solve_tasks(d2, lib_without_g2)
    c3_ablate = mine_binary_candidates(
        s2_ablate,
        lib_without_g2,
        generation=3,
    )

    libraries.append(libraries[-1] + r3)

    test_tasks = make_fixed_test_tasks(split["test"])
    test_summaries = [
        summarize_search(solve_tasks(test_tasks, lib))
        for lib in libraries
    ]

    # Remove only the newest generation from each step and verify the
    # corresponding held-out gain disappears.
    generation_ablation = []
    for idx in range(1, len(libraries)):
        full = test_summaries[idx]
        prev = test_summaries[idx - 1]
        generation_ablation.append(
            {
                "generation": idx,
                "full_mean_depth": full["mean_action_depth"],
                "ablated_mean_depth": prev["mean_action_depth"],
                "full_mean_nodes": full["mean_nodes_expanded"],
                "ablated_mean_nodes": prev["mean_nodes_expanded"],
                "depth_gain_removed": (
                    full["mean_action_depth"]
                    < prev["mean_action_depth"]
                ),
                "node_gain_removed": (
                    full["mean_nodes_expanded"]
                    < prev["mean_nodes_expanded"]
                ),
            }
        )

    strict_chain = (
        len(r1) > 0
        and len(r2) > 0
        and len(r3) > 0
        and len(c2_ablate) == 0
        and len(c3_ablate) == 0
        and all(
            test_summaries[i]["mean_action_depth"]
            > test_summaries[i + 1]["mean_action_depth"]
            for i in range(3)
        )
        and all(
            test_summaries[i]["mean_nodes_expanded"]
            > test_summaries[i + 1]["mean_nodes_expanded"]
            for i in range(3)
        )
        and all(x["success_rate"] == 1.0 for x in test_summaries)
    )

    def action_record(a: Action):
        return {
            "name": a.name,
            "generation": a.generation,
            "lhs": a.lhs,
            "rhs": a.rhs,
            "expansion": [
                {"action": n, "path": list(p)}
                for n, p in a.expansion
            ],
            "support_ids": a.support_ids,
            "validation_gain": a.validation_gain,
        }

    return {
        "seed": seed,
        "strict_chain_pass": strict_chain,
        "retained_counts": {
            "G1": len(r1),
            "G2": len(r2),
            "G3": len(r3),
        },
        "proposal_reachability_ablation": {
            "G2_candidates_with_G1": len(c2),
            "G2_candidates_without_G1": len(c2_ablate),
            "G3_candidates_with_G2": len(c3),
            "G3_candidates_without_G2": len(c3_ablate),
        },
        "fixed_test": {
            f"L{i}": summary
            for i, summary in enumerate(test_summaries)
        },
        "generation_ablation": generation_ablation,
        "retained_actions": {
            "G1": [action_record(a) for a in r1],
            "G2": [action_record(a) for a in r2],
            "G3": [action_record(a) for a in r3],
        },
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(results):
    passes = [1.0 if r["strict_chain_pass"] else 0.0 for r in results]
    n = len(results)

    aggregate = {
        "n_seeds": n,
        "seeds": [r["seed"] for r in results],
        "strict_chain_passes": int(sum(passes)),
        "strict_chain_pass_rate": sum(passes) / n,
        "strict_chain_hoeffding_one_sided_lb_95": hoeffding_lb_01(
            sum(passes) / n,
            n,
        ),
        "library_levels": {},
        "proposal_reachability": {
            "all_G2_ablation_zero": all(
                r["proposal_reachability_ablation"][
                    "G2_candidates_without_G1"
                ]
                == 0
                for r in results
            ),
            "all_G3_ablation_zero": all(
                r["proposal_reachability_ablation"][
                    "G3_candidates_without_G2"
                ]
                == 0
                for r in results
            ),
        },
    }

    for level in ("L0", "L1", "L2", "L3"):
        rows = [r["fixed_test"][level] for r in results]
        aggregate["library_levels"][level] = {
            "mean_success_rate": sum(
                x["success_rate"] for x in rows
            )
            / n,
            "mean_action_depth": sum(
                x["mean_action_depth"] for x in rows
            )
            / n,
            "mean_nodes_expanded": sum(
                x["mean_nodes_expanded"] for x in rows
            )
            / n,
            "mean_successors_generated": sum(
                x["mean_successors_generated"] for x in rows
            )
            / n,
            "mean_total_wall_seconds": sum(
                x["total_wall_seconds"] for x in rows
            )
            / n,
        }

    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[260909, 260910, 260911, 260912],
    )
    parser.add_argument(
        "--output",
        default="runs/constructive_rsi/"
        "zpoly_constructive_rsi_report.json",
    )
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        print(f"seed={seed}", flush=True)
        result = run_seed(seed)
        results.append(result)
        print(
            f"  pass={result['strict_chain_pass']} "
            f"retained={result['retained_counts']} "
            f"depths="
            f"{[result['fixed_test'][f'L{i}']['mean_action_depth'] for i in range(4)]}",
            flush=True,
        )

    report = {
        "schema_version": "hpm.constructive_rsi.zpoly.v1",
        "neural_training": False,
        "domain": "zpoly_simplify_v0",
        "induction_budget": (
            "complete solved derivations of exactly two actions"
        ),
        "retention_rule": (
            "candidate must be conservative, preserve validation success, "
            "reduce mean validation action depth, and not increase mean BFS nodes"
        ),
        "per_seed": results,
        "aggregate": aggregate(results),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
