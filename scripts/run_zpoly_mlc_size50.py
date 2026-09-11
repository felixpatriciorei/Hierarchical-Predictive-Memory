#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def loadmod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = loadmod('mlc_zbase', ROOT / 'scripts/eval_hpm_zpoly_v0.py')
ctx = loadmod('mlc_zctx', ROOT / 'scripts/eval_hpm_zpoly_context_shift_v0.py')
curve = loadmod('mlc_curve', ROOT / 'scripts/run_zpoly_dataset_curve_point.py')
crmod = loadmod('mlc_zcr', ROOT / 'scripts/run_constructive_rsi_zpoly_v0.py')

from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model
from hpm.train import TinyAdamW

SEED = 260909
EPISODES = 50
TRAIN_UPDATES = 2000
BATCH_SIZE = 32
D_MODEL = 192
LAYERS = 2
HEADS = 4
WINDOW = 256
BLOCK_SIZE = 256
MEMORY_SLOTS = 16
LR = 3e-4
LOG_EVERY = 50
PATH_NAMES = ('local', 'recurrent', 'fast_weight', 'episodic')

# Unused IDs inside the existing 480-token vocabulary.
TOK_STUDY = 320
TOK_DEMO = 321
TOK_DEMO_ANSWER = 322
TOK_QUERY = 323
TOK_EPISODE_END = 324


def accepted_macros():
    return ctx.accepted_macros(ROOT / 'data/microdomains/zpoly_v0/abstractions.jsonl')


def macro_specs():
    return ctx.build_specs(accepted_macros(), 'macro_augmented')


def root_study_bank() -> dict[str, list[dict[str, Any]]]:
    """Checker-verified root examples, deliberately simpler than nested queries."""
    macros = accepted_macros()
    safe = list(crmod.safe_term_pool())
    # A few deeper terms prevent the studies from degenerating to variable-only cases.
    x, y, z = base.V('x'), base.V('y'), base.V('z')
    safe += [base.Add(x, y), base.Mul(y, z), base.Add(base.Mul(x, y), z)]
    uniq = []
    seen = set()
    for t in safe:
        k = base.canonical_json(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for macro in macros:
        names = sorted(crmod.meta_names(macro['lhs_pattern']))
        for i in range(80):
            if len(names) == 1:
                env = {names[0]: uniq[(i * 5 + 1) % len(uniq)]}
            elif len(names) == 2:
                env = {
                    names[0]: uniq[(i * 5 + 1) % len(uniq)],
                    names[1]: uniq[(i * 7 + 3) % len(uniq)],
                }
            else:
                raise ValueError(names)
            expr = base.instantiate_pattern(macro['lhs_pattern'], env)
            goal = base.apply_macro_action(expr, macro)
            if goal is None:
                continue
            ok, reason = curve.checker_verify_macro_instance(expr, goal, macro)
            if not ok:
                raise AssertionError((macro['abstraction_id'], reason))
            key = (base.canonical_json(expr), base.canonical_json(goal))
            if any((base.canonical_json(r['expr']), base.canonical_json(r['goal'])) == key for r in out[macro['abstraction_id']]):
                continue
            out[macro['abstraction_id']].append({
                'expr': expr,
                'goal': goal,
                'action_name': macro['abstraction_id'],
                'checker': {'pass': True, 'root_macro_example': True},
            })
            if len(out[macro['abstraction_id']]) >= 12:
                break
        if len(out[macro['abstraction_id']]) < 8:
            raise RuntimeError(f"too few studies for {macro['abstraction_id']}: {len(out[macro['abstraction_id']])}")
    return dict(out)


def query_pool() -> list[dict[str, Any]]:
    """Nested checker-verified macro queries from the same G1-G3 process as the flat curve.

    The intended macro must occur at a non-root path and must *not* also match
    at root, ruling out the old root-position shortcut.
    """
    macro_rows, _ = curve.generate_pool()
    forbidden = {base.canonical_json(t['start']) for t in ctx.context_shift_tasks()}
    macros = {m['abstraction_id']: m for m in accepted_macros()}
    rows = []
    for r in macro_rows:
        if base.canonical_json(r['expr']) in forbidden:
            continue
        m = macros[r['action_name']]
        paths = [p for p in base.iter_paths(r['expr']) if base.match_pattern(m['lhs_pattern'], base.get_at_path(r['expr'], p)) is not None]
        if not paths or () in paths:
            continue
        rows.append(r)
    if len(rows) < EPISODES:
        raise RuntimeError(f"query pool too small: {len(rows)}")
    return rows


def build_episodes() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    studies = root_study_bank()
    qrows = query_pool()
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in qrows:
        groups[(r['action_name'], int(r['source_generation']))].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: (r['source_action'], r['payload_index']))

    macros = [m['abstraction_id'] for m in accepted_macros()]
    rng = random.Random(SEED)
    episodes = []
    # Macro-balanced, generation-diverse selection over the available verified
    # non-root-only buckets. Some macro/generation pairs have no such rows, so
    # cycle only through buckets that actually exist.
    counters = defaultdict(int)
    gens_by_macro = {m: sorted(g for (mm, g), rows in groups.items() if mm == m and rows) for m in macros}
    macro_seen = defaultdict(int)
    for i in range(EPISODES):
        m = macros[i % len(macros)]
        gens = gens_by_macro[m]
        gen = gens[macro_seen[m] % len(gens)]
        macro_seen[m] += 1
        bucket = (m, gen)
        qi = counters[bucket]
        counters[bucket] += 1
        q = groups[bucket][qi % len(groups[bucket])]
        nstudy = 2 if i % 2 == 0 else 3
        bank = studies[m]
        # Deterministic rotation; no study is the nested query itself by construction.
        start = (i * 3 + macros.index(m)) % len(bank)
        demos = [copy.deepcopy(bank[(start + j) % len(bank)]) for j in range(nstudy)]
        macro_obj = next(mm for mm in accepted_macros() if mm['abstraction_id'] == m)
        match_paths = [p for p in base.iter_paths(q['expr']) if base.match_pattern(macro_obj['lhs_pattern'], base.get_at_path(q['expr'], p)) is not None]
        if not match_paths or () in match_paths:
            raise AssertionError(f'query is not non-root-only for {m}: {match_paths}')
        episodes.append({
            'episode_id': f'mlc_{i:03d}_{m}',
            'macro': m,
            'studies': demos,
            'query': copy.deepcopy(q),
            'study_count': nstudy,
            'checker': {
                'pass': True,
                'query_checker': q.get('checker'),
                'all_studies_checker_pass': all(d['checker']['pass'] for d in demos),
                'query_is_nested_g1_g3': True,
            },
        })
    rng.shuffle(episodes)
    return episodes, studies


def serialize_episode(ep: dict[str, Any], specs) -> list[int]:
    toks = [base.TOK_BOS, base.TOK_ACTIONS]
    toks.extend(s.token_id for s in specs)
    toks.append(TOK_STUDY)
    for demo in ep['studies']:
        toks += [TOK_DEMO, base.TOK_EXPR]
        toks += base.ast_to_tokens(demo['expr'])
        toks += [base.TOK_GOAL]
        toks += base.ast_to_tokens(demo['goal'])
    q = ep['query']
    toks += [TOK_QUERY, base.TOK_EXPR]
    toks += base.ast_to_tokens(q['expr'])
    toks += [base.TOK_GOAL]
    toks += base.ast_to_tokens(q['goal'])
    toks += [base.TOK_QUERY_ACTION]
    return toks


def make_eval_episode(studies, macro_name: str, expr, goal, specs, study_count: int = 3):
    bank = studies[macro_name]
    # Fixed evaluation studies so every checkpoint/task sees identical context.
    demos = [copy.deepcopy(bank[j]) for j in range(study_count)]
    return {
        'episode_id': f'eval_{macro_name}',
        'macro': macro_name,
        'studies': demos,
        'query': {'expr': copy.deepcopy(expr), 'goal': copy.deepcopy(goal), 'action_name': macro_name},
        'study_count': study_count,
    }


def make_model(device):
    cfg = HpmLiteV2Config(
        vocab_size=480,
        d_model=D_MODEL,
        layers=LAYERS,
        heads=HEADS,
        window=WINDOW,
        max_seq_len=512,
        dropout=0.0,
        block_size=BLOCK_SIZE,
        use_learned_writer=True,
        episodic_capacity=MEMORY_SLOTS,
        writer_candidate_mode='all_prequery',
        use_jepa_aux=False,
        use_token_jepa_aux=False,
        use_jepa_writer_bias=False,
        episodic_read_mode='hard_topk',
        router_logit_clamp=None,
    )
    return HpmLiteV2Model(cfg).to(device)


def build_batch(episodes, specs, device):
    seqs = [serialize_episode(ep, specs) for ep in episodes]
    lengths = [len(s) for s in seqs]
    max_n = max(lengths)
    if max_n > 512:
        raise ValueError(f'episode too long: {max_n}')
    ids = torch.zeros((len(seqs), max_n), dtype=torch.long, device=device)
    max_pairs = max(max(1, n - 2) for n in lengths)
    mtp = torch.zeros((len(seqs), max_pairs, 2), dtype=torch.long, device=device)
    mm = torch.zeros((len(seqs), max_pairs), dtype=torch.bool, device=device)
    pos = torch.tensor([n - 1 for n in lengths], dtype=torch.long, device=device)
    for b, (seq, n) in enumerate(zip(seqs, lengths)):
        ids[b, :n] = torch.tensor(seq, dtype=torch.long, device=device)
        pairs = torch.tensor([[i, i + 1] for i in range(max(1, n - 2))], dtype=torch.long, device=device)
        mtp[b, :len(pairs)] = pairs
        mm[b, :len(pairs)] = True
    return ids, mtp, mm, pos, lengths


def forward_batch(model, episodes, specs, device, health=False):
    ids, mtp, mm, pos, lengths = build_batch(episodes, specs, device)
    out = model(
        input_ids=ids,
        memory_token_positions=mtp,
        memory_mask=mm,
        answer_positions=pos,
        query_key_positions=pos,
        top_k=1,
        task='kv',
        use_learned_writer=True,
        learned_writer_teacher_forcing=False,
        teacher_forcing_prob=0.0,
    )
    logits = out['logits'][torch.arange(ids.size(0), device=device), pos, :]
    h = None
    if health:
        ret = out.get('retrieval', {})
        w = ret.get('router_weights')
        z = ret.get('router_logits')
        if w is not None and z is not None:
            idx = torch.arange(ids.size(0), device=device)
            aw = w[idx, pos, :]
            az = z[idx, pos, :]
            h = {
                'router_path_usage': {name: float(aw[:, i].mean().detach().cpu()) for i, name in enumerate(PATH_NAMES)},
                'router_logit_abs_mean': float(az.abs().mean().detach().cpu()),
                'router_max_path_probability_mean': float(aw.max(dim=-1).values.mean().detach().cpu()),
            }
    return logits, h


def train(device):
    torch.manual_seed(SEED)
    random.seed(SEED)
    specs = macro_specs()
    by_name = {s.name: s for s in specs}
    episodes, studies = build_episodes()
    model = make_model(device)
    opt = TinyAdamW(model.parameters(), lr=LR)
    rng = random.Random(SEED)
    losses = []
    health_log = []
    started = time.perf_counter()
    model.train()
    for step in range(1, TRAIN_UPDATES + 1):
        batch = [episodes[rng.randrange(len(episodes))] for _ in range(BATCH_SIZE)]
        logits, h = forward_batch(model, batch, specs, device, health=(step % LOG_EVERY == 0))
        action_logits = torch.stack([base.restricted_logits(logits[i], specs) for i in range(len(batch))])
        targets = torch.tensor([specs.index(by_name[ep['query']['action_name']]) for ep in batch], dtype=torch.long, device=device)
        loss = F.cross_entropy(action_logits, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if h is not None:
            rec = {'step': step, 'train_loss': losses[-1], **h}
            health_log.append(rec)
            print(json.dumps(rec, sort_keys=True), flush=True)
    return model, specs, episodes, studies, {
        'final_train_loss_mean_100': sum(losses[-100:]) / 100,
        'wall_seconds': time.perf_counter() - started,
        'router_health_log': health_log,
    }


@torch.no_grad()
def choose_action(model, specs, studies, macro_name, expr, goal, device):
    ep = make_eval_episode(studies, macro_name, expr, goal, specs, 3)
    logits, h = forward_batch(model, [ep], specs, device, health=True)
    al = base.restricted_logits(logits[0], specs)
    sp = specs[int(torch.argmax(al).item())]
    return sp, h


@torch.no_grad()
def evaluate_structural(model, specs, studies, device):
    model.eval()
    rows = []
    router = []
    for t in ctx.context_shift_tasks():
        state = copy.deepcopy(t['start'])
        goal = copy.deepcopy(t['goal'])
        intended = t['intended_macro']
        trace = []
        first_correct = False
        direct_after_first = False
        solved = False
        for step in range(1, base.SEARCH_BUDGET + 1):
            sp, h = choose_action(model, specs, studies, intended, state, goal, device)
            if h is not None:
                router.append([h['router_path_usage'][n] for n in PATH_NAMES])
            trace.append(sp.name)
            if step == 1:
                first_correct = (sp.name == intended)
            nxt = ctx.apply_action(state, sp)
            if nxt is None:
                continue
            state = nxt
            if step == 1 and first_correct:
                # Records whether the intended macro actually executed, independently
                # of later primitive cleanup required by the legacy structural task.
                direct_after_first = True
            if base.same_ast(state, goal):
                solved = True
                break
        rows.append({
            'task_id': t['task_id'],
            'intended_macro': intended,
            'success': solved,
            'first_action': trace[0] if trace else None,
            'first_action_is_intended': first_correct,
            'intended_macro_executed_on_first_step': direct_after_first,
            'trace': trace,
        })
    selected = [r for r in rows if r['first_action_is_intended']]
    out = {
        'structural_heldout_exact': sum(r['success'] for r in rows) / len(rows),
        'macro_selection_accuracy': sum(r['first_action_is_intended'] for r in rows) / len(rows),
        'completion_conditional_on_correct_selection': (sum(r['success'] for r in selected) / len(selected) if selected else None),
        'completion_conditional_n': len(selected),
        'intended_macro_execution_rate': sum(r['intended_macro_executed_on_first_step'] for r in rows) / len(rows),
        'tasks': rows,
    }
    if router:
        out['router_path_usage'] = {name: sum(v[i] for v in router) / len(router) for i, name in enumerate(PATH_NAMES)}
    return out


@torch.no_grad()
def evaluate_standard(model, specs, studies, device):
    # Reference only. For each standard held-out query, condition on the first
    # macro in its verified macro witness. This is not the primary MLC metric.
    model.eval()
    rows = []
    for t in base.held_out_tasks():
        macro = t['macro_witness'][0]
        state = copy.deepcopy(t['start'])
        goal = copy.deepcopy(t['goal'])
        solved = False
        trace = []
        for _ in range(base.SEARCH_BUDGET):
            sp, _ = choose_action(model, specs, studies, macro, state, goal, device)
            trace.append(sp.name)
            nxt = ctx.apply_action(state, sp)
            if nxt is None:
                continue
            state = nxt
            if base.same_ast(state, goal):
                solved = True
                break
        rows.append({'id': t['id'], 'success': solved, 'trace': trace, 'conditioning_macro': macro})
    return {'eval_exact': sum(r['success'] for r in rows) / len(rows), 'tasks': rows}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output', default=str(ROOT / 'runs/mlc_size50/mlc_size50.json'))
    args = ap.parse_args()
    device = torch.device(args.device)

    episodes, studies = build_episodes()
    lengths = [len(serialize_episode(ep, macro_specs())) for ep in episodes]
    dataset_audit = {
        'episodes': len(episodes),
        'study_counts': {str(k): sum(ep['study_count'] == k for ep in episodes) for k in (2, 3)},
        'macro_counts': {m: sum(ep['macro'] == m for ep in episodes) for m in sorted(studies)},
        'query_generations': {str(g): sum(ep['query']['source_generation'] == g for ep in episodes) for g in (1, 2, 3)},
        'min_tokens': min(lengths),
        'max_tokens': max(lengths),
        'all_checker_verified': all(ep['checker']['pass'] and ep['checker']['all_studies_checker_pass'] for ep in episodes),
    }
    print(json.dumps({'dataset_audit': dataset_audit}, sort_keys=True), flush=True)

    model, specs, episodes, studies, train_meta = train(device)
    structural = evaluate_structural(model, specs, studies, device)
    standard = evaluate_standard(model, specs, studies, device)

    out = {
        'schema_version': 'hpm.zpoly.mlc_size50.v1',
        'seed': SEED,
        'device': str(device),
        'mechanism': 'same-context study_examples_plus_novel_query',
        'dataset_audit': dataset_audit,
        'protocol': {
            'episodes': EPISODES,
            'train_updates': TRAIN_UPDATES,
            'batch_size': BATCH_SIZE,
            'd_model': D_MODEL,
            'layers': LAYERS,
            'heads': HEADS,
            'window': WINDOW,
            'block_size': BLOCK_SIZE,
            'memory_slots': MEMORY_SLOTS,
            'optimizer': 'TinyAdamW',
            'lr': LR,
            'learned_writer': True,
            'teacher_forcing_steps': 0,
            'jepa': False,
            'router_guard': False,
            'router_logit_clamp': None,
            'lambda_router_z_loss': 0.0,
            'study_examples_per_episode': '2-3',
            'query_target': 'macro action',
        },
        'train': train_meta,
        **{k: structural[k] for k in (
            'structural_heldout_exact',
            'macro_selection_accuracy',
            'completion_conditional_on_correct_selection',
            'completion_conditional_n',
            'intended_macro_execution_rate',
            'router_path_usage',
        )},
        'eval_exact': standard['eval_exact'],
        'structural_tasks': structural['tasks'],
        'standard_tasks': standard['tasks'],
        'episodes': episodes,
    }
    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: out[k] for k in (
        'structural_heldout_exact',
        'macro_selection_accuracy',
        'completion_conditional_on_correct_selection',
        'intended_macro_execution_rate',
        'eval_exact',
        'router_path_usage',
    )}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
