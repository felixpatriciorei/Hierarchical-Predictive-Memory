#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, importlib.util, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

HERE=Path(__file__).resolve().parent
P=HERE/'eval_hpm_zpoly_rsi_curriculum_v0.py'
spec=importlib.util.spec_from_file_location('exp',P); exp=importlib.util.module_from_spec(spec); sys.modules['exp']=exp; spec.loader.exec_module(exp)
base=exp.base; ctx=exp.ctx
ALPHA=.05


def direct_tasks(macros):
    old=ctx.context_shift_tasks(); specs=ctx.build_specs(macros,'macro_augmented'); by={s.name:s for s in specs}
    out=[]
    for t in old:
        nxt=ctx.apply_action(t['start'],by[t['intended_macro']])
        if nxt is None: raise AssertionError(t['task_id'])
        out.append({**t,'goal':nxt,'description':t['description']+'; goal is exactly the post-macro state (one-step position test)'})
        # No primitive may solve this direct transition in one step.
        for sp in specs:
            if sp.kind=='primitive':
                p=ctx.apply_action(t['start'],sp)
                if p is not None and base.same_ast(p,nxt):
                    raise AssertionError(f"primitive {sp.name} duplicates direct macro transition for {t['task_id']}")
    return out


def rich_direct_rows(macros,cr,seed,heldouts,derivations):
    oldstates={base.canonical_json(step['expr_before']) for d in derivations for step in d['steps']}
    # Get the verified G1/G2/G3 context candidates, then convert each target
    # into the exact one-step post-macro state. Keep one context per macro per
    # generation: 12 macro-labeled examples total.
    rich=exp.build_rsi_examples(macros,cr,seed,heldouts,oldstates)
    specs=ctx.build_specs(macros,'macro_augmented'); by={s.name:s for s in specs}
    selected=[]; seen=set()
    for r in rich:
        key=(r['action_name'],r['source_generation'])
        if key in seen: continue
        seen.add(key)
        post=ctx.apply_action(r['expr'],by[r['action_name']])
        if post is None: raise AssertionError(r['id'])
        selected.append({**r,'goal':post})
    if len(selected)!=12: raise AssertionError(len(selected))
    return selected


def training_rows(condition,derivations,macros,cr,seed,heldouts):
    root=exp.root_training_rows(derivations,macros)
    if condition=='root_only': return root,[]
    if condition!='rsi_direct_matched': raise ValueError(condition)
    primitive=[r for r in root if r.action_name in base.BASE_RULE_IDS]
    if len(primitive)!=12: raise AssertionError(len(primitive))
    meta=rich_direct_rows(macros,cr,seed,heldouts,derivations)
    rows=primitive+[base.TrainingExample(r['expr'],r['goal'],r['action_name']) for r in meta]
    if len(rows)!=24: raise AssertionError(len(rows))
    return rows,meta


def train(condition,seed,derivations,macros,cr,heldouts,device):
    torch.manual_seed(seed); random.seed(seed)
    specs=ctx.build_specs(macros,'macro_augmented'); by={s.name:s for s in specs}
    examples,meta=training_rows(condition,derivations,macros,cr,seed,heldouts)
    model=base.make_model(device); opt=torch.optim.AdamW(model.parameters(),lr=base.LEARNING_RATE,weight_decay=base.WEIGHT_DECAY); rng=random.Random(seed); losses=[]; start=time.perf_counter(); model.train()
    for _ in range(base.TRAIN_UPDATES):
        ex=examples[rng.randrange(len(examples))]; sp=by[ex.action_name]
        logits=base.restricted_logits(base.forward_action_logits(model,ex.expr,ex.goal,specs,device),specs)
        target=torch.tensor([specs.index(sp)],dtype=torch.long,device=device)
        loss=F.cross_entropy(logits.unsqueeze(0),target); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
    return model,specs,{'training_examples':24,'macro_labeled_examples':12,'primitive_labeled_examples':12,'rsi_curriculum_metadata':meta,'final_train_loss_mean_100':sum(losses[-100:])/100,'training_wall_clock_seconds':time.perf_counter()-start}


@torch.no_grad()
def evaluate_one_step(model,specs,tasks,device):
    model.eval(); rows=[]; start=time.perf_counter()
    for t in tasks:
        logits=base.restricted_logits(base.forward_action_logits(model,t['start'],t['goal'],specs,device),specs)
        sp=specs[int(torch.argmax(logits).item())]
        nxt=ctx.apply_action(t['start'],sp)
        success=nxt is not None and base.same_ast(nxt,t['goal'])
        rows.append({'task_id':t['task_id'],'intended_macro':t['intended_macro'],'chosen_action':sp.name,'chosen_is_intended':sp.name==t['intended_macro'],'success':success,'start':t['start'],'goal':t['goal'],'post_action':nxt})
    rate=sum(r['success'] for r in rows)/len(rows); intended=sum(r['chosen_is_intended'] for r in rows)/len(rows)
    return {'exact_one_step_success_rate':rate,'intended_macro_selection_rate':intended,'evaluation_wall_clock_seconds':time.perf_counter()-start,'tasks':rows}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--condition',choices=['root_only','rsi_direct_matched'],required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    deriv=base.load_jsonl(Path('data/microdomains/zpoly_v0/derivations.jsonl')); mac=ctx.accepted_macros(Path('data/microdomains/zpoly_v0/abstractions.jsonl')); cr=json.loads(Path('evidence/constructive_rsi_report.json').read_text()); held=direct_tasks(mac)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,specs,ti=train(a.condition,a.seed,deriv,mac,cr,held,device); ev=evaluate_one_step(model,specs,held,device); out={'schema_version':'hpm.zpoly.position_direct.v1','seed':a.seed,'condition':a.condition,'device':str(device),**ti,**ev}; Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'seed':a.seed,'condition':a.condition,'loss':ti['final_train_loss_mean_100'],'success':ev['exact_one_step_success_rate'],'intended':ev['intended_macro_selection_rate']},sort_keys=True))
if __name__=='__main__': main()
