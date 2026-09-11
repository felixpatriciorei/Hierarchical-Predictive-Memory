#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Reuse the already-frozen zpoly-v0 serialization, primitive engine and HPM
# forward interface. This script only adds the context-shift control.
BASE_PATH = Path(__file__).with_name("eval_hpm_zpoly_v0.py")
spec = importlib.util.spec_from_file_location("_zpoly_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules["_zpoly_base"] = base
spec.loader.exec_module(base)

SEEDS = (260909, 260910, 260911, 260912)
ALPHA = 0.05
SEARCH_BUDGET = 8
CONDITIONS = (
    "primitive_only",
    "primitive_only_shortened_position_locked",
    "macro_augmented",
)


def accepted_macros(path: Path):
    rows = base.load_jsonl(path)
    return sorted(
        [r for r in rows if r.get("status") == "accepted"],
        key=lambda r: r["abstraction_id"],
    )


def build_specs(macros, condition):
    specs = [
        base.ActionSpec(r, base.PRIMITIVE_ACTION_TOKEN[r], "primitive")
        for r in base.BASE_RULE_IDS
    ]
    if condition == "macro_augmented":
        for token_id, macro in zip(base.MACRO_ACTION_TOKENS, macros):
            specs.append(
                base.ActionSpec(
                    macro["abstraction_id"],
                    token_id,
                    "macro",
                    macro=macro,
                )
            )
    elif condition == "primitive_only_shortened_position_locked":
        # Same 2-primitive compression and same token IDs as macro_augmented,
        # but the shortcut may execute only at the root -- the only macro
        # position present in training.
        for idx, (token_id, macro) in enumerate(
            zip(base.SHORTCUT_ACTION_TOKENS, macros)
        ):
            trace = tuple(
                (step["rule"], tuple(step["path"]))
                for step in macro["expansion"]
            )
            specs.append(
                base.ActionSpec(
                    f"LOCKED_{idx:03d}",
                    token_id,
                    "shortcut_locked",
                    primitive_trace=trace,
                )
            )
    elif condition != "primitive_only":
        raise ValueError(condition)
    return specs


def apply_locked_root(expr, trace):
    state = copy.deepcopy(expr)
    for rule_id, rel_path in trace:
        state = base.apply_primitive_at_path(state, rule_id, rel_path)
        if state is None:
            return None
    return state


def apply_action(expr, spec):
    if spec.kind == "primitive":
        return base.apply_primitive_action(expr, spec.name)
    if spec.kind == "macro":
        return base.apply_macro_action(expr, spec.macro)
    if spec.kind == "shortcut_locked":
        return apply_locked_root(expr, spec.primitive_trace or ())
    raise ValueError(spec.kind)


def training_examples(derivations, macros, condition):
    specs = build_specs(macros, condition)

    if condition == "primitive_only":
        return [
            base.TrainingExample(
                step["expr_before"],
                d["goal_expr"],
                step["rule"],
            )
            for d in derivations
            for step in d["steps"]
        ]

    extra_by_trace = {}
    if condition == "primitive_only_shortened_position_locked":
        for sp in specs:
            if sp.kind == "shortcut_locked":
                extra_by_trace[sp.primitive_trace] = sp.name

    rows = []
    for d in derivations:
        goal = d["goal_expr"]
        full_trace = tuple(
            (step["rule"], tuple(step["path"]))
            for step in d["steps"]
        )
        for step_index, step in enumerate(d["steps"]):
            action = step["rule"]
            expr = step["expr_before"]

            if condition == "macro_augmented":
                direct = []
                for sp in specs:
                    if sp.kind != "macro":
                        continue
                    nxt = apply_action(expr, sp)
                    if nxt is not None and base.same_ast(nxt, goal):
                        direct.append(sp.name)
                if len(direct) != (1 if step_index == 0 else 0):
                    raise ValueError(
                        f"unexpected macro targets in {d['derivation_id']} "
                        f"step={step_index}: {direct}"
                    )
                if direct:
                    action = direct[0]

            elif (
                condition == "primitive_only_shortened_position_locked"
                and step_index == 0
            ):
                action = extra_by_trace[full_trace]
                sp = next(s for s in specs if s.name == action)
                nxt = apply_action(expr, sp)
                if nxt is None or not base.same_ast(nxt, goal):
                    raise ValueError(
                        f"locked shortcut failed its root training example: {action}"
                    )

            rows.append(base.TrainingExample(expr, goal, action))

    if len(rows) != 24:
        raise ValueError(f"expected 24 training rows, got {len(rows)}")
    return rows


def context_shift_tasks():
    A = lambda q: base.Add(base.Mul(q, base.C(1)), base.C(0))
    B = lambda q: base.Add(base.C(0), base.Mul(base.C(1), q))
    Cc = lambda q: base.Mul(base.Add(q, base.C(0)), base.C(1))
    D = lambda q, r: base.Add(base.Mul(q, base.C(0)), r)

    return [
        {
            "task_id": "context_A_at_right_child",
            "start": base.Mul(base.C(1), A(base.V("x"))),
            "goal": base.V("x"),
            "intended_macro": "abs_v0_000",
            "intended_locked": "LOCKED_000",
            "macro_path": [1],
            "description": "1*((x*1)+0) -> x; A was trained only at root, now occurs at path [1]",
        },
        {
            "task_id": "context_B_at_left_child",
            "start": base.Mul(B(base.V("y")), base.C(1)),
            "goal": base.V("y"),
            "intended_macro": "abs_v0_001",
            "intended_locked": "LOCKED_001",
            "macro_path": [0],
            "description": "(0+(1*y))*1 -> y; B was trained only at root, now occurs at path [0]",
        },
        {
            "task_id": "context_C_at_right_child",
            "start": base.Mul(base.C(1), Cc(base.V("z"))),
            "goal": base.V("z"),
            "intended_macro": "abs_v0_002",
            "intended_locked": "LOCKED_002",
            "macro_path": [1],
            "description": "1*((z+0)*1) -> z; C was trained only at root, now occurs at path [1]",
        },
        {
            "task_id": "context_D_at_right_child",
            "start": base.Mul(base.C(1), D(base.V("x"), base.V("y"))),
            "goal": base.V("y"),
            "intended_macro": "abs_v0_003",
            "intended_locked": "LOCKED_003",
            "macro_path": [1],
            "description": "1*((x*0)+y) -> y; D was trained only at root, now occurs at path [1]",
        },
    ]


def validate_split(derivations, macros, tasks):
    train_states = {
        base.canonical_json(step["expr_before"])
        for d in derivations
        for step in d["steps"]
    }
    macro_by_name = {m["abstraction_id"]: m for m in macros}
    locked_specs = build_specs(
        macros, "primitive_only_shortened_position_locked"
    )

    for task in tasks:
        if base.canonical_json(task["start"]) in train_states:
            raise AssertionError(f"training leak: {task['task_id']}")

        macro = macro_by_name[task["intended_macro"]]
        matches = [
            list(path)
            for path in base.iter_paths(task["start"])
            if base.match_pattern(
                macro["lhs_pattern"],
                base.get_at_path(task["start"], path),
            )
            is not None
        ]
        if task["macro_path"] not in matches:
            raise AssertionError(
                f"{task['task_id']}: intended macro does not match at "
                f"{task['macro_path']}; matches={matches}"
            )
        if [] in matches:
            raise AssertionError(
                f"{task['task_id']}: macro unexpectedly matches at training root"
            )

        # No position-locked shortcut is allowed to fire on the initial state.
        for sp in locked_specs:
            if sp.kind != "shortcut_locked":
                continue
            if apply_action(task["start"], sp) is not None:
                raise AssertionError(
                    f"{task['task_id']}: locked shortcut unexpectedly fires "
                    f"at root: {sp.name}"
                )


def train(condition, derivations, macros, seed, device):
    torch.manual_seed(seed)
    random.seed(seed)

    specs = build_specs(macros, condition)
    by_name = {s.name: s for s in specs}
    examples = training_examples(derivations, macros, condition)

    model = base.make_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY,
    )
    rng = random.Random(seed)
    losses = []
    started = time.perf_counter()
    model.train()

    for _ in range(base.TRAIN_UPDATES):
        ex = examples[rng.randrange(len(examples))]
        target_spec = by_name[ex.action_name]

        logits = base.forward_action_logits(
            model, ex.expr, ex.goal, specs, device
        )
        logits = base.restricted_logits(logits, specs)
        target_index = specs.index(target_spec)
        target = torch.tensor(
            [target_index], dtype=torch.long, device=device
        )

        loss = F.cross_entropy(logits.unsqueeze(0), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return model, specs, {
        "train_updates": base.TRAIN_UPDATES,
        "training_examples": len(examples),
        "final_train_loss_mean_100": sum(losses[-100:]) / 100.0,
        "training_wall_clock_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate(model, specs, tasks, condition, device):
    model.eval()
    task_rows = []
    started = time.perf_counter()

    for task in tasks:
        state = copy.deepcopy(task["start"])
        goal = task["goal"]
        action_trace = []
        first_action = None
        first_action_effective = False
        solved_at = None
        seen = set()

        intended = None
        if condition == "macro_augmented":
            intended = task["intended_macro"]
        elif condition == "primitive_only_shortened_position_locked":
            intended = task["intended_locked"]

        for step in range(1, SEARCH_BUDGET + 1):
            logits = base.forward_action_logits(
                model, state, goal, specs, device
            )
            action_logits = base.restricted_logits(logits, specs)
            chosen = specs[int(torch.argmax(action_logits).item())]

            if first_action is None:
                first_action = chosen.name

            action_trace.append(chosen.name)
            loop_key = (base.canonical_json(state), chosen.name)
            if loop_key in seen:
                break
            seen.add(loop_key)

            nxt = apply_action(state, chosen)
            if step == 1:
                first_action_effective = nxt is not None
            if nxt is None:
                continue

            state = nxt
            if base.same_ast(state, goal):
                solved_at = step
                break

        task_rows.append(
            {
                **task,
                "success": solved_at is not None,
                "search_steps_to_solution": solved_at,
                "first_action": first_action,
                "intended_condition_action": intended,
                "first_action_is_intended": (
                    None if intended is None else first_action == intended
                ),
                "first_action_effective": first_action_effective,
                "action_trace": action_trace,
                "final_expr": state,
            }
        )

    successes = [r for r in task_rows if r["success"]]
    intended_rows = [
        r for r in task_rows
        if r["first_action_is_intended"] is not None
    ]
    return {
        "exact_match_success_rate": len(successes) / len(task_rows),
        "mean_search_steps_to_solution": (
            sum(r["search_steps_to_solution"] for r in successes)
            / len(successes)
            if successes
            else None
        ),
        "first_action_intended_rate": (
            sum(bool(r["first_action_is_intended"]) for r in intended_rows)
            / len(intended_rows)
            if intended_rows
            else None
        ),
        "first_action_effective_rate": (
            sum(bool(r["first_action_effective"]) for r in task_rows)
            / len(task_rows)
        ),
        "evaluation_wall_clock_seconds": time.perf_counter() - started,
        "tasks": task_rows,
    }


def hoeffding_lb(mean, n, width):
    return mean - width * math.sqrt(
        math.log(1.0 / ALPHA) / (2.0 * n)
    )


def aggregate(per_seed):
    agg = {
        "n_seeds": len(per_seed),
        "seeds": [r["seed"] for r in per_seed],
        "alpha": ALPHA,
        "conditions": {},
        "paired_deltas": {},
    }
    for condition in CONDITIONS:
        blocks = [r["conditions"][condition] for r in per_seed]
        success_rates = [b["exact_match_success_rate"] for b in blocks]
        first_rates = [
            b["first_action_intended_rate"]
            for b in blocks
            if b["first_action_intended_rate"] is not None
        ]
        steps = [
            task["search_steps_to_solution"]
            for b in blocks
            for task in b["tasks"]
            if task["success"]
        ]
        agg["conditions"][condition] = {
            "seed_exact_match_success_rates": success_rates,
            "mean_exact_match_success_rate": sum(success_rates)
            / len(success_rates),
            "hoeffding_exact_match_one_sided_lb_95": hoeffding_lb(
                sum(success_rates) / len(success_rates),
                len(success_rates),
                1.0,
            ),
            "aggregate_successes": sum(
                int(t["success"]) for b in blocks for t in b["tasks"]
            ),
            "aggregate_trials": sum(len(b["tasks"]) for b in blocks),
            "mean_search_steps_to_solution": (
                sum(steps) / len(steps) if steps else None
            ),
            "seed_first_action_intended_rates": first_rates,
            "mean_first_action_intended_rate": (
                sum(first_rates) / len(first_rates)
                if first_rates
                else None
            ),
            "hoeffding_first_action_one_sided_lb_95": (
                hoeffding_lb(
                    sum(first_rates) / len(first_rates),
                    len(first_rates),
                    1.0,
                )
                if first_rates
                else None
            ),
        }

    for metric in (
        "exact_match_success_rate",
        "first_action_intended_rate",
    ):
        macro = [
            r["conditions"]["macro_augmented"][metric]
            for r in per_seed
        ]
        locked = [
            r["conditions"]["primitive_only_shortened_position_locked"][metric]
            for r in per_seed
        ]
        deltas = [a - b for a, b in zip(macro, locked)]
        mean_delta = sum(deltas) / len(deltas)
        agg["paired_deltas"][
            f"macro_minus_position_locked__{metric}"
        ] = {
            "seed_deltas": deltas,
            "mean_delta": mean_delta,
            "hoeffding_one_sided_lb_95": hoeffding_lb(
                mean_delta, len(deltas), 2.0
            ),
            "observation_range": [-1.0, 1.0],
        }

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--derivations",
        default="data/microdomains/zpoly_v0/derivations.jsonl",
    )
    parser.add_argument(
        "--abstractions",
        default="data/microdomains/zpoly_v0/abstractions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="runs/zpoly_v0/"
        "hpm_zpoly_v0_context_shift_report.json",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()

    derivations = base.load_jsonl(Path(args.derivations))
    macros = accepted_macros(Path(args.abstractions))
    tasks = context_shift_tasks()

    if len(derivations) != 12 or len(macros) != 4:
        raise ValueError("expected zpoly-v0 12 derivations / 4 macros")
    validate_split(derivations, macros, tasks)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    report = {
        "schema_version": "hpm.zpoly.context_shift.v1",
        "domain": base.DOMAIN,
        "model": "current_four_path_hpm_v2",
        "device": str(device),
        "protocol": {
            "seeds": list(args.seeds),
            "train_updates_per_condition_per_seed": base.TRAIN_UPDATES,
            "training_data": "same 12 derivations / 24 observed states",
            "new_held_out_category": "context_shifted_macro_position",
            "search_budget": SEARCH_BUDGET,
            "conditions": {
                "primitive_only": "8 base primitive actions",
                "primitive_only_shortened_position_locked": (
                    "8 primitives + 4 two-primitive compressed actions; "
                    "compressed actions execute only at root, matching the "
                    "only macro position present during training"
                ),
                "macro_augmented": (
                    "8 primitives + 4 accepted structural macros; macro "
                    "pattern matcher may apply at any tree position"
                ),
            },
        },
        "held_out_category": tasks,
        "per_seed": [],
    }

    for seed in args.seeds:
        seed_row = {"seed": seed, "conditions": {}}
        for condition in args.conditions:
            print(
                f"seed={seed} condition={condition}",
                flush=True,
            )
            model, specs, train_info = train(
                condition, derivations, macros, seed, device
            )
            eval_info = evaluate(
                model, specs, tasks, condition, device
            )
            seed_row["conditions"][condition] = {
                **train_info,
                **eval_info,
            }
            print(
                f"  exact={eval_info['exact_match_success_rate']:.3f} "
                f"first={eval_info['first_action_intended_rate']} "
                f"steps={eval_info['mean_search_steps_to_solution']}",
                flush=True,
            )
            del model
        report["per_seed"].append(seed_row)

    if tuple(args.conditions) == CONDITIONS:
        report["aggregate"] = aggregate(report["per_seed"])
    else:
        report["aggregate"] = None

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
