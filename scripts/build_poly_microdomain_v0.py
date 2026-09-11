#!/usr/bin/env python3
"""
Build and verify HPM's first constructive-learning micro-domain.

Domain: zpoly_simplify_v0
Expressions are integer-polynomial ASTs over:
  variables: x, y, z
  constants: 0, 1
  operations: add, mul

The script:
  1. defines the primitive rewrite system;
  2. emits a small set of verified solved derivations;
  3. anti-unifies pairs with identical primitive traces;
  4. verifies each candidate abstraction is a conservative macro:
       - replaying its primitive expansion reaches its RHS exactly;
       - LHS/RHS have identical canonical polynomial semantics;
       - RHS introduces no new symbolic atoms;
  5. writes JSONL artifacts and a compact report.

No third-party dependencies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

AST = dict[str, Any]

DOMAIN = "zpoly_simplify_v0"
DERIVATION_SCHEMA = "hpm.microdomain.derivation.v1"
ABSTRACTION_SCHEMA = "hpm.microdomain.abstraction.v1"

VARIABLES = ("x", "y", "z")
CONSTANTS = (0, 1)
OPERATIONS = ("add", "mul")

PRIMITIVE_RULE_IDS = (
    "ADD_ZERO_R",   # a + 0 -> a
    "ADD_ZERO_L",   # 0 + a -> a
    "MUL_ONE_R",    # a * 1 -> a
    "MUL_ONE_L",    # 1 * a -> a
    "MUL_ZERO_R",   # a * 0 -> 0
    "MUL_ZERO_L",   # 0 * a -> 0
    "DISTRIB_L",    # a * (b + c) -> a*b + a*c
    "DISTRIB_R",    # (a + b) * c -> a*c + b*c
)


def V(name: str) -> AST:
    if name not in VARIABLES:
        raise ValueError(f"unknown domain variable: {name}")
    return {"var": name}


def M(name: str) -> AST:
    return {"meta": name}


def C(value: int) -> AST:
    if value not in CONSTANTS:
        raise ValueError(f"unsupported literal constant: {value}")
    return {"const": value}


def Add(a: AST, b: AST) -> AST:
    return {"op": "add", "args": [a, b]}


def Mul(a: AST, b: AST) -> AST:
    return {"op": "mul", "args": [a, b]}


def canonical_json(node: AST) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def same_ast(a: AST, b: AST) -> bool:
    return canonical_json(a) == canonical_json(b)


def is_const(node: AST, value: int) -> bool:
    return node == {"const": value}


def is_op(node: AST, op: str) -> bool:
    return node.get("op") == op and isinstance(node.get("args"), list) and len(node["args"]) == 2


def rule_add_zero_r(t: AST) -> AST | None:
    if is_op(t, "add") and is_const(t["args"][1], 0):
        return copy.deepcopy(t["args"][0])
    return None


def rule_add_zero_l(t: AST) -> AST | None:
    if is_op(t, "add") and is_const(t["args"][0], 0):
        return copy.deepcopy(t["args"][1])
    return None


def rule_mul_one_r(t: AST) -> AST | None:
    if is_op(t, "mul") and is_const(t["args"][1], 1):
        return copy.deepcopy(t["args"][0])
    return None


def rule_mul_one_l(t: AST) -> AST | None:
    if is_op(t, "mul") and is_const(t["args"][0], 1):
        return copy.deepcopy(t["args"][1])
    return None


def rule_mul_zero_r(t: AST) -> AST | None:
    if is_op(t, "mul") and is_const(t["args"][1], 0):
        return C(0)
    return None


def rule_mul_zero_l(t: AST) -> AST | None:
    if is_op(t, "mul") and is_const(t["args"][0], 0):
        return C(0)
    return None


def rule_distrib_l(t: AST) -> AST | None:
    # a * (b + c) -> a*b + a*c
    if is_op(t, "mul") and is_op(t["args"][1], "add"):
        a = t["args"][0]
        b, c = t["args"][1]["args"]
        return Add(Mul(copy.deepcopy(a), copy.deepcopy(b)),
                   Mul(copy.deepcopy(a), copy.deepcopy(c)))
    return None


def rule_distrib_r(t: AST) -> AST | None:
    # (a + b) * c -> a*c + b*c
    if is_op(t, "mul") and is_op(t["args"][0], "add"):
        a, b = t["args"][0]["args"]
        c = t["args"][1]
        return Add(Mul(copy.deepcopy(a), copy.deepcopy(c)),
                   Mul(copy.deepcopy(b), copy.deepcopy(c)))
    return None


RULES: dict[str, Callable[[AST], AST | None]] = {
    "ADD_ZERO_R": rule_add_zero_r,
    "ADD_ZERO_L": rule_add_zero_l,
    "MUL_ONE_R": rule_mul_one_r,
    "MUL_ONE_L": rule_mul_one_l,
    "MUL_ZERO_R": rule_mul_zero_r,
    "MUL_ZERO_L": rule_mul_zero_l,
    "DISTRIB_L": rule_distrib_l,
    "DISTRIB_R": rule_distrib_r,
}


def get_at_path(expr: AST, path: list[int]) -> AST:
    cur = expr
    for idx in path:
        if not is_op(cur, cur.get("op", "")):
            raise ValueError(f"path {path} descends through a non-binary-op node")
        if idx not in (0, 1):
            raise ValueError(f"invalid child index {idx}")
        cur = cur["args"][idx]
    return cur


def replace_at_path(expr: AST, path: list[int], replacement: AST) -> AST:
    if not path:
        return copy.deepcopy(replacement)
    out = copy.deepcopy(expr)
    cur = out
    for idx in path[:-1]:
        cur = cur["args"][idx]
    cur["args"][path[-1]] = copy.deepcopy(replacement)
    return out


def apply_rule(expr: AST, rule_id: str, path: list[int]) -> AST:
    if rule_id not in RULES:
        raise KeyError(f"unknown rule: {rule_id}")
    target = get_at_path(expr, path)
    replacement = RULES[rule_id](target)
    if replacement is None:
        raise ValueError(
            f"rule {rule_id} does not match at path {path}: {canonical_json(target)}"
        )
    return replace_at_path(expr, path, replacement)


def make_derivation(
    derivation_id: str,
    start: AST,
    trace: list[tuple[str, list[int]]],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    current = copy.deepcopy(start)
    for index, (rule_id, path) in enumerate(trace):
        before = copy.deepcopy(current)
        after = apply_rule(current, rule_id, path)
        steps.append(
            {
                "index": index,
                "rule": rule_id,
                "path": path,
                "expr_before": before,
                "expr_after": copy.deepcopy(after),
            }
        )
        current = after

    record = {
        "schema_version": DERIVATION_SCHEMA,
        "derivation_id": derivation_id,
        "domain": DOMAIN,
        "axiom_set_version": "zpoly-v0.1",
        "start_expr": copy.deepcopy(start),
        "goal_expr": copy.deepcopy(current),
        "steps": steps,
        "cost": {"primitive_steps": len(steps)},
        "verified": True,
    }
    verify_derivation(record)
    return record


def verify_derivation(d: dict[str, Any]) -> None:
    current = copy.deepcopy(d["start_expr"])
    for expected_index, step in enumerate(d["steps"]):
        if step["index"] != expected_index:
            raise ValueError("non-contiguous derivation step index")
        if not same_ast(current, step["expr_before"]):
            raise ValueError(f"step {expected_index}: expr_before mismatch")
        current = apply_rule(current, step["rule"], step["path"])
        if not same_ast(current, step["expr_after"]):
            raise ValueError(f"step {expected_index}: expr_after mismatch")
    if not same_ast(current, d["goal_expr"]):
        raise ValueError("goal_expr does not match replayed final state")


# ---------- Anti-unification (least general generalization) ----------

@dataclass
class LGGState:
    next_meta: int = 0
    pair_to_meta: dict[tuple[str, str], str] | None = None

    def __post_init__(self) -> None:
        if self.pair_to_meta is None:
            self.pair_to_meta = {}

    def fresh_for_pair(self, a: AST, b: AST) -> AST:
        key = (canonical_json(a), canonical_json(b))
        if key not in self.pair_to_meta:
            name = f"A{self.next_meta}"
            self.next_meta += 1
            self.pair_to_meta[key] = name
        return M(self.pair_to_meta[key])


def anti_unify_ast(a: AST, b: AST, state: LGGState) -> AST:
    if same_ast(a, b):
        return copy.deepcopy(a)

    if a.get("op") == b.get("op") and a.get("op") in OPERATIONS:
        aa, ab = a["args"]
        ba, bb = b["args"]
        return {
            "op": a["op"],
            "args": [
                anti_unify_ast(aa, ba, state),
                anti_unify_ast(ab, bb, state),
            ],
        }

    return state.fresh_for_pair(a, b)


def trace_signature(d: dict[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    return [(s["rule"], tuple(s["path"])) for s in d["steps"]]


def anti_unify_derivations(
    d1: dict[str, Any],
    d2: dict[str, Any],
    abstraction_id: str,
) -> dict[str, Any]:
    """
    v0 deliberately only anti-unifies derivations with the same primitive
    rule/path trace. This makes the first abstraction learner conservative:
    it discovers reusable *macros* before attempting arbitrary proof synthesis.
    """
    verify_derivation(d1)
    verify_derivation(d2)

    if d1["domain"] != DOMAIN or d2["domain"] != DOMAIN:
        raise ValueError("wrong domain")
    if trace_signature(d1) != trace_signature(d2):
        raise ValueError("v0 requires identical rule/path traces")

    state = LGGState()
    lhs = anti_unify_ast(d1["start_expr"], d2["start_expr"], state)
    rhs = anti_unify_ast(d1["goal_expr"], d2["goal_expr"], state)

    expansion = [
        {"rule": s["rule"], "path": s["path"]}
        for s in d1["steps"]
    ]

    candidate = {
        "schema_version": ABSTRACTION_SCHEMA,
        "abstraction_id": abstraction_id,
        "domain": DOMAIN,
        "derived_from": [d1["derivation_id"], d2["derivation_id"]],
        "lhs_pattern": lhs,
        "rhs_pattern": rhs,
        "expansion": expansion,
        "cost": {
            "primitive_steps": len(expansion),
            "macro_steps": 1,
            "primitive_steps_saved_per_use": max(0, len(expansion) - 1),
        },
    }
    candidate["checks"] = check_candidate_abstraction(candidate)
    candidate["status"] = "accepted" if candidate["checks"]["pass"] else "rejected"
    return candidate


# ---------- Exact polynomial semantics over Z[symbols] ----------

# Polynomial representation:
#   dict[monomial_tuple, integer_coefficient]
# e.g. 2*x*y + 1 -> {("x","y"): 2, (): 1}
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
        # Each metavariable is treated as an algebraically independent symbol.
        # Equality of the resulting polynomial identity therefore holds after
        # substitution by arbitrary polynomial expressions.
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
    raise ValueError(f"invalid AST: {expr!r}")


def symbolic_atoms(expr: AST) -> set[str]:
    if "var" in expr:
        return {f"var:{expr['var']}"}
    if "meta" in expr:
        return {f"meta:{expr['meta']}"}
    if "const" in expr:
        return set()
    if "op" in expr:
        return symbolic_atoms(expr["args"][0]) | symbolic_atoms(expr["args"][1])
    raise ValueError(f"invalid AST: {expr!r}")


def replay_macro_expansion(candidate: dict[str, Any]) -> AST:
    current = copy.deepcopy(candidate["lhs_pattern"])
    for step in candidate["expansion"]:
        current = apply_rule(current, step["rule"], step["path"])
    return current


def check_candidate_abstraction(candidate: dict[str, Any]) -> dict[str, Any]:
    lhs = candidate["lhs_pattern"]
    rhs = candidate["rhs_pattern"]

    replayed = replay_macro_expansion(candidate)
    replay_exact = same_ast(replayed, rhs)

    lhs_nf = polynomial_normal_form(lhs)
    rhs_nf = polynomial_normal_form(rhs)
    semantic_equal = lhs_nf == rhs_nf

    lhs_atoms = symbolic_atoms(lhs)
    rhs_atoms = symbolic_atoms(rhs)
    rhs_symbols_subset = rhs_atoms <= lhs_atoms

    passed = replay_exact and semantic_equal and rhs_symbols_subset

    return {
        "replay_exact": replay_exact,
        "semantic_equal": semantic_equal,
        "rhs_symbols_subset": rhs_symbols_subset,
        "lhs_normal_form": encode_poly(lhs_nf),
        "rhs_normal_form": encode_poly(rhs_nf),
        "pass": passed,
    }


def encode_poly(poly: Poly) -> list[dict[str, Any]]:
    return [
        {"monomial": list(mono), "coefficient": coeff}
        for mono, coeff in sorted(poly.items())
    ]


def seed_derivations() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Family A: (v * 1) + 0 -> v
    for v in VARIABLES:
        out.append(
            make_derivation(
                f"macro_A_{v}",
                Add(Mul(V(v), C(1)), C(0)),
                [("MUL_ONE_R", [0]), ("ADD_ZERO_R", [])],
            )
        )

    # Family B: 0 + (1 * v) -> v
    for v in VARIABLES:
        out.append(
            make_derivation(
                f"macro_B_{v}",
                Add(C(0), Mul(C(1), V(v))),
                [("MUL_ONE_L", [1]), ("ADD_ZERO_L", [])],
            )
        )

    # Family C: (v + 0) * 1 -> v
    for v in VARIABLES:
        out.append(
            make_derivation(
                f"macro_C_{v}",
                Mul(Add(V(v), C(0)), C(1)),
                [("ADD_ZERO_R", [0]), ("MUL_ONE_R", [])],
            )
        )

    # Family D: (v * 0) + w -> w, for v != w
    pairs = [("x", "y"), ("y", "z"), ("z", "x")]
    for v, w in pairs:
        out.append(
            make_derivation(
                f"macro_D_{v}_{w}",
                Add(Mul(V(v), C(0)), V(w)),
                [("MUL_ZERO_R", [0]), ("ADD_ZERO_L", [])],
            )
        )

    return out


def candidate_pairs(derivations: list[dict[str, Any]]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    # Pair derivations only within the same trace signature.
    groups: dict[tuple[tuple[str, tuple[int, ...]], ...], list[dict[str, Any]]] = {}
    for d in derivations:
        sig = tuple(trace_signature(d))
        groups.setdefault(sig, []).append(d)

    for group in groups.values():
        if len(group) >= 2:
            # One deterministic pair per family is enough for the v0 seed.
            yield group[0], group[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="data/microdomains/zpoly_v0",
        help="output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    derivations = seed_derivations()
    abstractions: list[dict[str, Any]] = []

    for idx, (d1, d2) in enumerate(candidate_pairs(derivations)):
        abstractions.append(
            anti_unify_derivations(d1, d2, f"abs_v0_{idx:03d}")
        )

    deriv_path = out_dir / "derivations.jsonl"
    abs_path = out_dir / "abstractions.jsonl"
    report_path = out_dir / "report.json"

    write_jsonl(deriv_path, derivations)
    write_jsonl(abs_path, abstractions)

    report = {
        "domain": DOMAIN,
        "primitive_rules": list(PRIMITIVE_RULE_IDS),
        "derivations": {
            "count": len(derivations),
            "verified": sum(1 for d in derivations if d["verified"]),
        },
        "abstractions": {
            "count": len(abstractions),
            "accepted": sum(1 for a in abstractions if a["status"] == "accepted"),
            "rejected": sum(1 for a in abstractions if a["status"] == "rejected"),
            "total_primitive_steps_saved_per_use": sum(
                a["cost"]["primitive_steps_saved_per_use"]
                for a in abstractions
                if a["status"] == "accepted"
            ),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        p.name: sha256_file(p)
        for p in (deriv_path, abs_path, report_path)
    }
    (out_dir / "MANIFEST.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {deriv_path}")
    print(f"wrote {abs_path}")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir / 'MANIFEST.sha256.json'}")


if __name__ == "__main__":
    main()
