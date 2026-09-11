#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

AST = dict[str, Any]
DOMAIN = "zpoly_simplify_v0"

SEED = 260909
TRAIN_UPDATES = 1200
D_MODEL = 48
LAYERS = 1
HEADS = 4
WINDOW = 32
BLOCK_SIZE = 8
MAX_SEQ_LEN = 64
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 1.0e-4
SEARCH_BUDGET = 8

BASE_RULE_IDS = (
    "ADD_ZERO_R",
    "ADD_ZERO_L",
    "MUL_ONE_R",
    "MUL_ONE_L",
    "MUL_ZERO_R",
    "MUL_ZERO_L",
    "DISTRIB_L",
    "DISTRIB_R",
)

# All IDs remain inside HPM v2's existing 480-token vocabulary.
TOK_BOS = 1
TOK_ACTIONS = 300
TOK_EXPR = 301
TOK_GOAL = 302
TOK_QUERY_ACTION = 303
TOK_ADD = 304
TOK_MUL = 305
TOK_X = 306
TOK_Y = 307
TOK_Z = 308
TOK_ZERO = 309
TOK_ONE = 310

PRIMITIVE_ACTION_TOKEN = {
    "ADD_ZERO_R": 430,
    "ADD_ZERO_L": 431,
    "MUL_ONE_R": 432,
    "MUL_ONE_L": 433,
    "MUL_ZERO_R": 434,
    "MUL_ZERO_L": 435,
    "DISTRIB_L": 436,
    "DISTRIB_R": 437,
}
MACRO_ACTION_TOKENS = (438, 439, 440, 441)
VARIABLE_TOKEN = {"x": TOK_X, "y": TOK_Y, "z": TOK_Z}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            out.append(obj)
    return out


def canonical_json(node: AST) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def same_ast(a: AST, b: AST) -> bool:
    return canonical_json(a) == canonical_json(b)


def V(name: str) -> AST:
    return {"var": name}


def C(value: int) -> AST:
    return {"const": value}


def Add(a: AST, b: AST) -> AST:
    return {"op": "add", "args": [a, b]}


def Mul(a: AST, b: AST) -> AST:
    return {"op": "mul", "args": [a, b]}


def is_op(node: AST, op: str) -> bool:
    return node.get("op") == op and isinstance(node.get("args"), list) and len(node["args"]) == 2


def is_const(node: AST, value: int) -> bool:
    return node == {"const": value}


def ast_to_tokens(node: AST) -> list[int]:
    if "var" in node:
        return [VARIABLE_TOKEN[node["var"]]]
    if "const" in node:
        if node["const"] == 0:
            return [TOK_ZERO]
        if node["const"] == 1:
            return [TOK_ONE]
        raise ValueError(f"unsupported constant: {node['const']}")
    if is_op(node, "add"):
        return [TOK_ADD] + ast_to_tokens(node["args"][0]) + ast_to_tokens(node["args"][1])
    if is_op(node, "mul"):
        return [TOK_MUL] + ast_to_tokens(node["args"][0]) + ast_to_tokens(node["args"][1])
    raise ValueError(f"cannot serialize AST: {node!r}")


# ---- primitive rewrites ----------------------------------------------------

def rule_add_zero_r(t):
    return copy.deepcopy(t["args"][0]) if is_op(t, "add") and is_const(t["args"][1], 0) else None


def rule_add_zero_l(t):
    return copy.deepcopy(t["args"][1]) if is_op(t, "add") and is_const(t["args"][0], 0) else None


def rule_mul_one_r(t):
    return copy.deepcopy(t["args"][0]) if is_op(t, "mul") and is_const(t["args"][1], 1) else None


def rule_mul_one_l(t):
    return copy.deepcopy(t["args"][1]) if is_op(t, "mul") and is_const(t["args"][0], 1) else None


def rule_mul_zero_r(t):
    return C(0) if is_op(t, "mul") and is_const(t["args"][1], 0) else None


def rule_mul_zero_l(t):
    return C(0) if is_op(t, "mul") and is_const(t["args"][0], 0) else None


def rule_distrib_l(t):
    if is_op(t, "mul") and is_op(t["args"][1], "add"):
        a = t["args"][0]
        b, c = t["args"][1]["args"]
        return Add(Mul(copy.deepcopy(a), copy.deepcopy(b)),
                   Mul(copy.deepcopy(a), copy.deepcopy(c)))
    return None


def rule_distrib_r(t):
    if is_op(t, "mul") and is_op(t["args"][0], "add"):
        a, b = t["args"][0]["args"]
        c = t["args"][1]
        return Add(Mul(copy.deepcopy(a), copy.deepcopy(c)),
                   Mul(copy.deepcopy(b), copy.deepcopy(c)))
    return None


PRIMITIVE_RULES: dict[str, Callable[[AST], AST | None]] = {
    "ADD_ZERO_R": rule_add_zero_r,
    "ADD_ZERO_L": rule_add_zero_l,
    "MUL_ONE_R": rule_mul_one_r,
    "MUL_ONE_L": rule_mul_one_l,
    "MUL_ZERO_R": rule_mul_zero_r,
    "MUL_ZERO_L": rule_mul_zero_l,
    "DISTRIB_L": rule_distrib_l,
    "DISTRIB_R": rule_distrib_r,
}


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


def canonical_occurrence(paths: list[tuple[int, ...]]) -> tuple[int, ...]:
    # Rule/macro ID is the only action predicted by HPM.
    # Occurrence resolution is fixed: deepest first, then leftmost.
    return sorted(paths, key=lambda p: (-len(p), p))[0]


def apply_primitive_action(expr: AST, rule_id: str) -> AST | None:
    rule = PRIMITIVE_RULES[rule_id]
    replacements = {}
    for path in iter_paths(expr):
        repl = rule(get_at_path(expr, path))
        if repl is not None:
            replacements[path] = repl
    if not replacements:
        return None
    path = canonical_occurrence(list(replacements))
    return replace_at_path(expr, path, replacements[path])


# ---- accepted macro rewrites ----------------------------------------------

def match_pattern(pattern: AST, expr: AST, env: dict[str, AST] | None = None) -> dict[str, AST] | None:
    env = {} if env is None else dict(env)
    if "meta" in pattern:
        name = pattern["meta"]
        if name in env:
            return env if same_ast(env[name], expr) else None
        env[name] = copy.deepcopy(expr)
        return env
    if "var" in pattern or "const" in pattern:
        return env if same_ast(pattern, expr) else None
    if pattern.get("op") != expr.get("op") or pattern.get("op") not in {"add", "mul"}:
        return None
    env = match_pattern(pattern["args"][0], expr["args"][0], env)
    if env is None:
        return None
    return match_pattern(pattern["args"][1], expr["args"][1], env)


def instantiate_pattern(pattern: AST, env: dict[str, AST]) -> AST:
    if "meta" in pattern:
        return copy.deepcopy(env[pattern["meta"]])
    if "var" in pattern or "const" in pattern:
        return copy.deepcopy(pattern)
    return {
        "op": pattern["op"],
        "args": [
            instantiate_pattern(pattern["args"][0], env),
            instantiate_pattern(pattern["args"][1], env),
        ],
    }


def apply_macro_action(expr: AST, macro: dict[str, Any]) -> AST | None:
    replacements = {}
    for path in iter_paths(expr):
        env = match_pattern(macro["lhs_pattern"], get_at_path(expr, path))
        if env is not None:
            replacements[path] = instantiate_pattern(macro["rhs_pattern"], env)
    if not replacements:
        return None
    path = canonical_occurrence(list(replacements))
    return replace_at_path(expr, path, replacements[path])


@dataclass(frozen=True)
class ActionSpec:
    name: str
    token_id: int
    kind: str
    macro: dict[str, Any] | None = None


@dataclass
class TrainingExample:
    expr: AST
    goal: AST
    action_name: str


def build_action_specs(accepted_macros, condition):
    specs = [
        ActionSpec(rule_id, PRIMITIVE_ACTION_TOKEN[rule_id], "primitive")
        for rule_id in BASE_RULE_IDS
    ]
    if condition == "macro_augmented":
        if len(accepted_macros) != 4:
            raise ValueError(f"expected 4 macros, got {len(accepted_macros)}")
        for token_id, macro in zip(MACRO_ACTION_TOKENS, accepted_macros):
            specs.append(ActionSpec(macro["abstraction_id"], token_id, "macro", macro))
    return specs


def training_examples_for_condition(derivations, accepted_macros, condition):
    """Exactly 24 state->action examples in either condition.

    primitive_only labels each observed state with the recorded primitive step.
    macro_augmented replaces that label only when an accepted macro takes the
    SAME observed state directly to the verified derivation goal in one step.
    Thus the target is the shortest available action, never two contradictory
    labels for one state.
    """
    out = []
    for d in derivations:
        goal = d["goal_expr"]
        for step in d["steps"]:
            expr = step["expr_before"]
            action = step["rule"]
            if condition == "macro_augmented":
                direct = []
                for macro in accepted_macros:
                    nxt = apply_macro_action(expr, macro)
                    if nxt is not None and same_ast(nxt, goal):
                        direct.append(macro["abstraction_id"])
                if len(direct) > 1:
                    raise ValueError(f"ambiguous one-step macros for observed training state: {direct}")
                if direct:
                    action = direct[0]
            out.append(TrainingExample(expr, goal, action))
    return out


def held_out_tasks():
    # Witness traces are used only to verify the split is genuinely compositional;
    # they are never shown to either HPM condition.
    return [
        {
            "id": "heldout_01_C_then_primitive",
            "start": Mul(Add(Mul(V("x"), C(1)), C(0)), C(1)),
            "goal": V("x"),
            "description": "((x*1)+0)*1 -> x; outer C macro exposes a remaining primitive MUL_ONE_R",
            "primitive_witness": ["MUL_ONE_R", "ADD_ZERO_R", "MUL_ONE_R"],
            "macro_witness": ["abs_v0_002", "MUL_ONE_R"],
        },
        {
            "id": "heldout_02_A_then_B",
            "start": Add(C(0), Mul(C(1), Add(Mul(V("y"), C(1)), C(0)))),
            "goal": V("y"),
            "description": "0+(1*((y*1)+0)) -> y; compose A inside B",
            "primitive_witness": ["MUL_ONE_R", "ADD_ZERO_R", "MUL_ONE_L", "ADD_ZERO_L"],
            "macro_witness": ["abs_v0_000", "abs_v0_001"],
        },
        {
            "id": "heldout_03_C_then_D",
            "start": Mul(Add(Add(Mul(V("z"), C(0)), V("x")), C(0)), C(1)),
            "goal": V("x"),
            "description": "((((z*0)+x)+0)*1) -> x; outer C exposes a D-pattern",
            "primitive_witness": ["MUL_ZERO_R", "ADD_ZERO_L", "ADD_ZERO_R", "MUL_ONE_R"],
            "macro_witness": ["abs_v0_002", "abs_v0_003"],
        },
        {
            "id": "heldout_04_C_then_A",
            "start": Mul(Add(Add(Mul(V("x"), C(1)), C(0)), C(0)), C(1)),
            "goal": V("x"),
            "description": "((((x*1)+0)+0)*1) -> x; outer C exposes an A-pattern",
            "primitive_witness": ["MUL_ONE_R", "ADD_ZERO_R", "ADD_ZERO_R", "MUL_ONE_R"],
            "macro_witness": ["abs_v0_002", "abs_v0_000"],
        },
        {
            "id": "heldout_05_C_then_B",
            "start": Add(C(0), Mul(C(1), Mul(Add(V("y"), C(0)), C(1)))),
            "goal": V("y"),
            "description": "0+(1*((y+0)*1)) -> y; compose C inside B",
            "primitive_witness": ["ADD_ZERO_R", "MUL_ONE_R", "MUL_ONE_L", "ADD_ZERO_L"],
            "macro_witness": ["abs_v0_002", "abs_v0_001"],
        },
        {
            "id": "heldout_06_A_then_D",
            "start": Add(Mul(Add(Mul(V("z"), C(0)), V("y")), C(1)), C(0)),
            "goal": V("y"),
            "description": "((((z*0)+y)*1)+0) -> y; outer A exposes a D-pattern",
            "primitive_witness": ["MUL_ZERO_R", "ADD_ZERO_L", "MUL_ONE_R", "ADD_ZERO_R"],
            "macro_witness": ["abs_v0_000", "abs_v0_003"],
        },
    ]

def _apply_named_action(expr, action_name, specs):
    by_name = {s.name: s for s in specs}
    return apply_action(expr, by_name[action_name])


def _verify_witness(task, witness, specs):
    state = copy.deepcopy(task["start"])
    for action_name in witness:
        state = _apply_named_action(state, action_name, specs)
        if state is None:
            return False
    return same_ast(state, task["goal"])


def assert_heldout_split(derivations, heldouts, accepted_macros):
    train_states = {
        canonical_json(step["expr_before"])
        for d in derivations
        for step in d["steps"]
    }
    prim_specs = build_action_specs(accepted_macros, "primitive_only")
    macro_specs = build_action_specs(accepted_macros, "macro_augmented")

    for task in heldouts:
        if canonical_json(task["start"]) in train_states:
            raise AssertionError(f"held-out leak: {task['id']}")

        # No condition may solve the held-out initial state in one action.
        for specs, label in ((prim_specs, "primitive"), (macro_specs, "macro")):
            for spec in specs:
                nxt = apply_action(task["start"], spec)
                if nxt is not None and same_ast(nxt, task["goal"]):
                    raise AssertionError(
                        f"held-out {task['id']} is one-step solvable under {label}: {spec.name}"
                    )

        if not _verify_witness(task, task["primitive_witness"], prim_specs):
            raise AssertionError(f"bad primitive witness for {task['id']}")
        if not _verify_witness(task, task["macro_witness"], macro_specs):
            raise AssertionError(f"bad macro witness for {task['id']}")

def build_input_tokens(expr, goal, action_specs):
    tokens = [TOK_BOS, TOK_ACTIONS]
    tokens.extend(spec.token_id for spec in action_specs)
    tokens += [TOK_EXPR] + ast_to_tokens(expr)
    tokens += [TOK_GOAL] + ast_to_tokens(goal)
    tokens += [TOK_QUERY_ACTION]
    if len(tokens) > MAX_SEQ_LEN:
        raise ValueError("serialized task exceeds MAX_SEQ_LEN")
    return tokens


def make_model(device):
    cfg = HpmLiteV2Config(
        vocab_size=480,
        d_model=D_MODEL,
        layers=LAYERS,
        heads=HEADS,
        window=WINDOW,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
        block_size=BLOCK_SIZE,
        use_learned_writer=False,
        use_jepa_aux=False,
        use_token_jepa_aux=False,
        use_jepa_writer_bias=False,
        episodic_read_mode="hard_topk",
        router_logit_clamp=None,
    )
    return HpmLiteV2Model(cfg).to(device)


def forward_action_logits(model, expr, goal, action_specs, device):
    tokens = build_input_tokens(expr, goal, action_specs)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    n = len(tokens)

    # Current episodic API stores key/value token-position pairs.
    pairs = [[i, i + 1] for i in range(max(1, n - 2))]
    memory_token_positions = torch.tensor([pairs], dtype=torch.long, device=device)
    memory_mask = torch.ones((1, len(pairs)), dtype=torch.bool, device=device)

    query_pos = n - 1
    pos = torch.tensor([query_pos], dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        memory_token_positions=memory_token_positions,
        memory_mask=memory_mask,
        answer_positions=pos,
        query_key_positions=pos,
        top_k=1,
        task="kv",
        use_learned_writer=False,
        learned_writer_teacher_forcing=False,
    )
    return out["logits"][0, query_pos, :]


def restricted_logits(logits, action_specs):
    ids = torch.tensor([s.token_id for s in action_specs], dtype=torch.long, device=logits.device)
    return logits.index_select(0, ids)


def apply_action(expr, spec):
    if spec.kind == "primitive":
        return apply_primitive_action(expr, spec.name)
    return apply_macro_action(expr, spec.macro)



def train_condition(condition, derivations, accepted_macros, device):
    torch.manual_seed(SEED)
    random.seed(SEED)

    specs = build_action_specs(accepted_macros, condition)
    by_name = {s.name: s for s in specs}
    examples = training_examples_for_condition(derivations, accepted_macros, condition)
    macro_count = sum(1 for ex in examples if ex.action_name.startswith("abs_v0_"))

    model = make_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    rng = random.Random(SEED)
    losses = []
    started = time.perf_counter()
    model.train()

    for _ in range(TRAIN_UPDATES):
        ex = examples[rng.randrange(len(examples))]
        spec = by_name[ex.action_name]

        logits = forward_action_logits(model, ex.expr, ex.goal, specs, device)
        logits = restricted_logits(logits, specs)
        target_index = specs.index(spec)
        target = torch.tensor([target_index], dtype=torch.long, device=device)

        loss = F.cross_entropy(logits.unsqueeze(0), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return model, specs, {
        "training_examples": len(examples),
        "primitive_labeled_training_examples": len(examples) - macro_count,
        "macro_labeled_training_examples": macro_count,
        "train_updates": TRAIN_UPDATES,
        "final_train_loss_mean_100": sum(losses[-100:]) / min(100, len(losses)),
        "training_wall_clock_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_condition(model, specs, heldouts, device):
    model.eval()
    started = time.perf_counter()
    results = []

    for task in heldouts:
        state = copy.deepcopy(task["start"])
        goal = task["goal"]
        trace = []
        solved_at = 0 if same_ast(state, goal) else None
        seen = set()

        for step in range(1, SEARCH_BUDGET + 1):
            if solved_at is not None:
                break

            logits = forward_action_logits(model, state, goal, specs, device)
            action_logits = restricted_logits(logits, specs)
            spec = specs[int(torch.argmax(action_logits).item())]
            trace.append(spec.name)

            loop_key = (canonical_json(state), spec.name)
            if loop_key in seen:
                break
            seen.add(loop_key)

            nxt = apply_action(state, spec)
            if nxt is None:
                continue
            state = nxt
            if same_ast(state, goal):
                solved_at = step
                break

        results.append({
            "task_id": task["id"],
            "description": task["description"],
            "success": solved_at is not None,
            "search_steps_to_solution": solved_at,
            "action_trace": trace,
            "final_expr": state,
            "goal_expr": goal,
        })

    successes = [r for r in results if r["success"]]
    steps = [r["search_steps_to_solution"] for r in successes]
    return {
        "exact_match_success_rate": len(successes) / len(results),
        "mean_search_steps_to_solution": (
            sum(steps) / len(steps) if steps else None
        ),
        "evaluation_wall_clock_seconds": time.perf_counter() - started,
        "flops": None,
        "tasks": results,
    }


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
    args = parser.parse_args()

    derivations = load_jsonl(Path(args.derivations))
    abstractions = load_jsonl(Path(args.abstractions))
    accepted = sorted(
        [a for a in abstractions if a.get("status") == "accepted"],
        key=lambda a: a["abstraction_id"],
    )

    if len(derivations) != 12:
        raise ValueError(f"expected exactly 12 training derivations, got {len(derivations)}")
    if len(accepted) != 4:
        raise ValueError(f"expected exactly 4 accepted abstractions, got {len(accepted)}")

    heldouts = held_out_tasks()
    assert_heldout_split(derivations, heldouts, accepted)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "schema_version": "hpm.zpoly.eval.v1",
        "domain": DOMAIN,
        "model": "current_four_path_hpm_v2",
        "seed": SEED,
        "device": str(device),
        "inputs": {
            "derivations": args.derivations,
            "abstractions": args.abstractions,
        },
        "condition_definitions": {
            "primitive_only": {
                "allowed_actions": list(BASE_RULE_IDS),
                "training_target": "shortest available primitive action on each of the 24 observed training states",
            },
            "macro_augmented": {
                "allowed_actions": list(BASE_RULE_IDS) + [a["abstraction_id"] for a in accepted],
                "training_target": "shortest available action on the same 24 observed training states; a macro replaces the primitive label only when it reaches the verified goal in one step",
            },
        },
        "held_out_selection_rule": "six initial states absent from all 24 observed training states; each requires >=3 primitive actions and >=2 actions even with macros, with nested/composed arrangements absent from the training derivations",
        "action_application_policy": "deepest-leftmost valid occurrence",
        "search_budget": SEARCH_BUDGET,
        "held_out_split": [
            {
                "task_id": t["id"],
                "description": t["description"],
                "start_expr": t["start"],
                "goal_expr": t["goal"],
            }
            for t in heldouts
        ],
        "conditions": {},
    }

    for condition in ("primitive_only", "macro_augmented"):
        model, specs, train_info = train_condition(
            condition, derivations, accepted, device
        )
        eval_info = evaluate_condition(
            model, specs, heldouts, device
        )
        report["conditions"][condition] = {
            "allowed_actions": [s.name for s in specs],
            **train_info,
            **eval_info,
        }

    output = Path("runs/zpoly_v0/hpm_zpoly_v0_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    headline = {
        name: {
            "exact_match_success_rate": block["exact_match_success_rate"],
            "mean_search_steps_to_solution": block["mean_search_steps_to_solution"],
            "training_wall_clock_seconds": block["training_wall_clock_seconds"],
            "evaluation_wall_clock_seconds": block["evaluation_wall_clock_seconds"],
        }
        for name, block in report["conditions"].items()
    }
    print(json.dumps(headline, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
