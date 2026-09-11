#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, importlib.util, json, random, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

def loadmod(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

base=loadmod('zbase', ROOT/'scripts/eval_hpm_zpoly_v0.py')
ctx=loadmod('zctx', ROOT/'scripts/eval_hpm_zpoly_context_shift_v0.py')
crmod=loadmod('zcr', ROOT/'scripts/run_constructive_rsi_zpoly_v0.py')
from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model
from hpm.train import TinyAdamW

SEED=260909
TRAIN_UPDATES=2000
D_MODEL=192
LAYERS=2
HEADS=4
WINDOW=256
MAX_SEQ_LEN=512
BLOCK_SIZE=256
LR=3e-4
BATCH_SIZE=32
MEMORY_SLOTS=16
LOG_EVERY=50
PATH_NAMES=('local','recurrent','fast_weight','episodic')


def accepted_macros():
    return ctx.accepted_macros(ROOT/'data/microdomains/zpoly_v0/abstractions.jsonl')

def load_constructive_actions():
    r=json.loads((ROOT/'evidence/constructive_rsi_report.json').read_text())
    ps=r['per_seed'][0]
    return [a for g in ('G1','G2','G3') for a in ps['retained_actions'][g]]

def metas(node):
    if 'meta' in node: return {node['meta']}
    if 'op' in node: return metas(node['args'][0]) | metas(node['args'][1])
    return set()

def inst(node, env):
    if 'meta' in node: return copy.deepcopy(env[node['meta']])
    if 'var' in node or 'const' in node: return copy.deepcopy(node)
    return {'op':node['op'],'args':[inst(node['args'][0],env),inst(node['args'][1],env)]}

def payload_pool():
    # Reconstruct the verified RSI-curriculum mechanism: G1-G3 retained
    # action LHS trees provide the OUTER discovered context, while accepted
    # root macros A-D instantiated on safe terms provide payloads that move
    # those known abstractions to unfamiliar internal tree positions.
    safe=list(crmod.safe_term_pool())[:10]
    x,y,z=base.V('x'),base.V('y'),base.V('z')
    safe += [base.Add(base.Add(x,y),z), base.Mul(base.Mul(x,y),z)]
    seen=set(); safe_uniq=[]
    for t in safe:
        k=base.canonical_json(t)
        if k not in seen: seen.add(k); safe_uniq.append(t)

    payloads=list(safe_uniq)
    macros=accepted_macros()
    # Instantiate every accepted abstraction at root, then these whole macro
    # instances are inserted into retained G1-G3 contexts. This is the same
    # context-shift construction used by the prior 12-row RSI curriculum.
    for m in macros:
        names=sorted(base.meta_names(m['lhs_pattern']) if hasattr(base,'meta_names') else crmod.meta_names(m['lhs_pattern']))
        if len(names)==1:
            for t in safe_uniq:
                payloads.append(base.instantiate_pattern(m['lhs_pattern'],{names[0]:t}))
        elif len(names)==2:
            # Deterministic bounded cross-pairing is plenty for D.
            for i,t in enumerate(safe_uniq):
                u=safe_uniq[(i*7+3)%len(safe_uniq)]
                payloads.append(base.instantiate_pattern(m['lhs_pattern'],{names[0]:t,names[1]:u}))
                payloads.append(base.instantiate_pattern(m['lhs_pattern'],{names[0]:u,names[1]:t}))
    out=[]; seen=set()
    for t in payloads:
        k=base.canonical_json(t)
        if k not in seen: seen.add(k); out.append(t)
    return out

def apply_primitive_at_path(expr, rule_id, path):
    target=base.get_at_path(expr, tuple(path))
    repl=base.PRIMITIVE_RULES[rule_id](target)
    if repl is None: return None
    return base.replace_at_path(expr, tuple(path), repl)

def canonical_macro_path(expr, macro):
    paths=[]
    for p in base.iter_paths(expr):
        if base.match_pattern(macro['lhs_pattern'],base.get_at_path(expr,p)) is not None:
            paths.append(p)
    return base.canonical_occurrence(paths) if paths else None

def checker_verify_macro_instance(expr, goal, macro):
    # 1) operational macro application exact
    got=base.apply_macro_action(expr,macro)
    if got is None or not base.same_ast(got,goal): return False, 'macro_apply'
    # 2) exact primitive expansion replay at canonical occurrence
    mp=canonical_macro_path(expr,macro)
    if mp is None: return False,'no_match'
    state=copy.deepcopy(expr)
    for st in macro['expansion']:
        gp=tuple(mp)+tuple(st['path'])
        state=apply_primitive_at_path(state,st['rule'],gp)
        if state is None: return False,'primitive_replay'
    if not base.same_ast(state,goal): return False,'replay_not_exact'
    # 3) exact polynomial semantics
    if crmod.polynomial_normal_form(expr)!=crmod.polynomial_normal_form(goal): return False,'semantic'
    return True,'pass'

def instantiate_macro_root(macro, payloads, idx):
    names=sorted(crmod.meta_names(macro['lhs_pattern']))
    if len(names)==1:
        return base.instantiate_pattern(macro['lhs_pattern'], {names[0]: payloads[idx % len(payloads)]})
    if len(names)==2:
        a=payloads[idx % len(payloads)]
        b=payloads[(idx*7+3) % len(payloads)]
        return base.instantiate_pattern(macro['lhs_pattern'], {names[0]:a, names[1]:b})
    raise ValueError(names)

def generate_pool():
    cache=ROOT/'evidence/scaled_curve_pool.json'
    if cache.exists():
        obj=json.loads(cache.read_text())
        return obj['macro_rows'], obj['primitive_rows']
    macros=accepted_macros(); actions=load_constructive_actions(); payloads=list(crmod.safe_term_pool())
    structural_starts={base.canonical_json(t['start']) for t in ctx.context_shift_tasks()}
    standard_starts={base.canonical_json(t['start']) for t in base.held_out_tasks()}
    forbidden=structural_starts|standard_starts
    # 30 per macro×generation = 360 macro examples, enough for the 250 macro
    # rows required by the size-500 balanced dataset with headroom.
    quota=30
    counts=defaultdict(int); macro_rows=[]; primitive_rows=[]; seen_m=set(); seen_p=set()
    actions=sorted(actions,key=lambda a:(a['generation'],a['name']))
    for macro in macros:
        for a in actions:
            bucket=(macro['abstraction_id'],a['generation'])
            if counts[bucket]>=quota: continue
            names=sorted(metas(a['lhs']))
            if len(names)!=1: continue
            for pi in range(len(payloads)*3):
                if counts[bucket]>=quota: break
                inner=instantiate_macro_root(macro,payloads,pi)
                expr=inst(a['lhs'],{names[0]:inner})
                if base.canonical_json(expr) in forbidden: continue
                goal=base.apply_macro_action(expr,macro)
                if goal is None: continue
                # Same target must not admit a competing macro label.
                producers=[]
                for m2 in macros:
                    g2=base.apply_macro_action(expr,m2)
                    if g2 is not None and base.same_ast(g2,goal): producers.append(m2['abstraction_id'])
                if producers != [macro['abstraction_id']]: continue
                ok,_=checker_verify_macro_instance(expr,goal,macro)
                if not ok: continue
                key=(base.canonical_json(expr),base.canonical_json(goal),macro['abstraction_id'])
                if key in seen_m: continue
                seen_m.add(key); counts[bucket]+=1
                macro_rows.append({'expr':expr,'goal':goal,'action_name':macro['abstraction_id'],'kind':'macro','source_generation':a['generation'],'source_action':a['name'],'payload_index':pi,'checker':{'pass':True,'primitive_replay_exact':True,'semantic_equal':True}})
                # Derive primitive-labeled states from the exact accepted macro
                # expansion on this same verified instance.
                mp=canonical_macro_path(expr,macro); state=copy.deepcopy(expr)
                for st in macro['expansion']:
                    before=copy.deepcopy(state); gp=tuple(mp)+tuple(st['path'])
                    state=apply_primitive_at_path(state,st['rule'],gp)
                    if state is None: break
                    pkey=(base.canonical_json(before),base.canonical_json(goal),st['rule'])
                    if pkey not in seen_p and base.canonical_json(before) not in forbidden:
                        seen_p.add(pkey)
                        primitive_rows.append({'expr':before,'goal':goal,'action_name':st['rule'],'kind':'primitive','source_generation':a['generation'],'source_action':a['name'],'payload_index':pi,'checker':{'pass':True,'derived_from_verified_macro_expansion':macro['abstraction_id']}})
    missing={str(k):quota-v for k,v in counts.items() if v<quota}
    if missing: raise RuntimeError(f'macro bucket quota missing: {missing}')
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps({'macro_rows':macro_rows,'primitive_rows':primitive_rows},indent=2,sort_keys=True)+'\n')
    return macro_rows,primitive_rows

def balanced_prefix(rows, n, keyfn, seed):
    groups=defaultdict(list)
    for r in rows: groups[keyfn(r)].append(r)
    rng=random.Random(seed)
    for g in groups.values(): rng.shuffle(g)
    keys=sorted(groups)
    out=[]; idx={k:0 for k in keys}
    while len(out)<n:
        progressed=False
        for k in keys:
            i=idx[k]
            if i<len(groups[k]):
                out.append(groups[k][i]); idx[k]=i+1; progressed=True
                if len(out)==n: break
        if not progressed: break
    if len(out)<n: raise RuntimeError(f'pool too small: need {n}, got {len(out)}')
    return out

def dataset_for_size(size):
    macro,prim=generate_pool()
    nm=size//2; np=size-nm
    msel=balanced_prefix(macro,nm,lambda r:(r['action_name'],r['source_generation']),SEED+1)
    psel=balanced_prefix(prim,np,lambda r:(r['action_name'],r['source_generation']),SEED+2)
    ds=msel+psel; random.Random(SEED+size).shuffle(ds)
    return ds, {'macro_pool':len(macro),'primitive_pool':len(prim),'macro_rows':len(msel),'primitive_rows':len(psel)}

def make_model(device):
    # Match the frozen TF0 KV substrate where task-independent settings apply.
    cfg=HpmLiteV2Config(
        vocab_size=480,
        d_model=D_MODEL,
        layers=LAYERS,
        heads=HEADS,
        window=WINDOW,
        max_seq_len=2048,
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


def build_batch(rows,specs,device):
    toks=[base.build_input_tokens(r['expr'],r['goal'],specs) for r in rows]
    lengths=[len(t) for t in toks]
    max_n=max(lengths)
    # Right padding is causally after each example's answer position, so it cannot
    # affect the answer-position state of the causal paths.
    ids=torch.zeros((len(rows),max_n),dtype=torch.long,device=device)
    max_pairs=max(max(1,n-2) for n in lengths)
    mtp=torch.zeros((len(rows),max_pairs,2),dtype=torch.long,device=device)
    mm=torch.zeros((len(rows),max_pairs),dtype=torch.bool,device=device)
    pos=torch.tensor([n-1 for n in lengths],dtype=torch.long,device=device)
    for b,(tok,n) in enumerate(zip(toks,lengths)):
        ids[b,:n]=torch.tensor(tok,dtype=torch.long,device=device)
        pairs=[[i,i+1] for i in range(max(1,n-2))]
        pt=torch.tensor(pairs,dtype=torch.long,device=device)
        mtp[b,:len(pairs)]=pt
        mm[b,:len(pairs)]=True
    return ids,mtp,mm,pos


def fwd_batch(model,rows,specs,device,want_router=False):
    ids,mtp,mm,pos=build_batch(rows,specs,device)
    out=model(
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
    logits=out['logits'][torch.arange(ids.size(0),device=device),pos,:]
    health=None
    if want_router:
        ret=out.get('retrieval',{})
        w=ret.get('router_weights')
        z=ret.get('router_logits')
        if w is not None and z is not None:
            # Log answer-position routing only, avoiding padded tail tokens.
            idx=torch.arange(ids.size(0),device=device)
            aw=w[idx,pos,:]
            az=z[idx,pos,:]
            health={
                'router_path_usage': {name: float(aw[:,i].mean().detach().cpu()) for i,name in enumerate(PATH_NAMES)},
                'router_logit_abs_mean': float(az.abs().mean().detach().cpu()),
                'router_max_path_probability_mean': float(aw.max(dim=-1).values.mean().detach().cpu()),
            }
    return logits,health


def fwd(model,expr,goal,specs,device,want_router=False):
    logits,h=fwd_batch(model,[{'expr':expr,'goal':goal}],specs,device,want_router)
    rw=None
    if h is not None:
        rw=[h['router_path_usage'][name] for name in PATH_NAMES]
    return logits[0],rw


def train(size,device):
    torch.manual_seed(SEED); random.seed(SEED)
    macros=accepted_macros(); specs=ctx.build_specs(macros,'macro_augmented'); by={s.name:s for s in specs}
    ds,meta=dataset_for_size(size)
    model=make_model(device)
    # Same optimizer implementation/hyperparameters as the frozen KV baseline.
    opt=TinyAdamW(model.parameters(),lr=LR)
    rng=random.Random(SEED); losses=[]; started=time.perf_counter(); model.train(); router_health=[]
    for step in range(1,TRAIN_UPDATES+1):
        rows=[ds[rng.randrange(len(ds))] for _ in range(BATCH_SIZE)]
        logits,health=fwd_batch(model,rows,specs,device,want_router=(step%LOG_EVERY==0))
        # Restrict each row to the exact same primitive+macro action vocabulary.
        action_logits=torch.stack([base.restricted_logits(logits[i],specs) for i in range(len(rows))],dim=0)
        targets=torch.tensor([specs.index(by[r['action_name']]) for r in rows],dtype=torch.long,device=device)
        loss=F.cross_entropy(action_logits,targets)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        losses.append(float(loss.detach().cpu()))
        if health is not None:
            row={'step':step,**health,'train_loss':losses[-1]}
            router_health.append(row)
            print(json.dumps({'size':size,**row},sort_keys=True), flush=True)
    return model,specs,ds,{**meta,'train_updates':TRAIN_UPDATES,'batch_size':BATCH_SIZE,'memory_slots':MEMORY_SLOTS,'optimizer':'TinyAdamW','lr':LR,'weight_decay':0.01,'teacher_forcing_steps':0,'learned_writer':True,'jepa':False,'router_guard':False,'final_train_loss_mean_100':sum(losses[-100:])/100,'wall_seconds':time.perf_counter()-started,'router_health_log':router_health}

@torch.no_grad()
def eval_tasks(model,specs,tasks,device,structural=False):
    model.eval(); rows=[]; router=[]
    for t in tasks:
        state=copy.deepcopy(t['start']); goal=t['goal']; trace=[]; first=None; first_intended=None; solved=False
        for step in range(1,base.SEARCH_BUDGET+1):
            logits,rw=fwd(model,state,goal,specs,device,True)
            if rw is not None: router.append(rw)
            al=base.restricted_logits(logits,specs); sp=specs[int(torch.argmax(al).item())]
            if first is None:
                first=sp.name
                if structural: first_intended=(first==t['intended_macro'])
            trace.append(sp.name); nxt=ctx.apply_action(state,sp)
            if nxt is None: continue
            state=nxt
            if base.same_ast(state,goal): solved=True; break
        rows.append({'id':t.get('task_id',t.get('id')),'success':solved,'first_action':first,'first_action_is_intended':first_intended,'trace':trace})
    out={'exact':sum(r['success'] for r in rows)/len(rows),'tasks':rows}
    if structural:
        sel=[r for r in rows if r['first_action_is_intended']]
        out['macro_selection_accuracy']=sum(bool(r['first_action_is_intended']) for r in rows)/len(rows)
        out['completion_conditional_on_correct_selection']=(sum(r['success'] for r in sel)/len(sel) if sel else None)
        out['conditional_n']=len(sel)
    if router:
        out['router_path_usage']={name:sum(v[i] for v in router)/len(router) for i,name in enumerate(PATH_NAMES)}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--size',type=int,required=True,choices=[50,200,500]); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cpu'); a=ap.parse_args()
    device=torch.device(a.device)
    model,specs,ds,trainmeta=train(a.size,device)
    structural=eval_tasks(model,specs,ctx.context_shift_tasks(),device,True)
    standard=eval_tasks(model,specs,base.held_out_tasks(),device,False)
    out={
      'schema_version':'hpm.zpoly.dataset_curve.v1','size':a.size,'seed':SEED,'device':str(device),
      'protocol':{'d_model':D_MODEL,'layers':LAYERS,'heads':HEADS,'window':WINDOW,'max_seq_len':2048,'block_size':BLOCK_SIZE,'train_updates':TRAIN_UPDATES,'batch_size':BATCH_SIZE,'memory_slots':MEMORY_SLOTS,'lr':LR,'optimizer':'TinyAdamW','weight_decay':0.01,'learned_writer':True,'learned_writer_teacher_forcing_steps':0,'jepa':False,'router_starvation_guard':False,'router_logit_clamp':None,'lambda_router_z_loss':0.0,'dataset_variable_only':True},
      'checker_verified_dataset':True,'train':trainmeta,
      'structural_heldout_exact':structural['exact'],
      'macro_selection_accuracy':structural['macro_selection_accuracy'],
      'completion_conditional_on_correct_selection':structural['completion_conditional_on_correct_selection'],
      'completion_conditional_n':structural['conditional_n'],
      'eval_exact':standard['exact'],
      'router_path_usage':structural.get('router_path_usage'),
      'structural_tasks':structural['tasks'],'standard_tasks':standard['tasks'],
      'dataset':ds,
    }
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:out[k] for k in ('size','structural_heldout_exact','macro_selection_accuracy','completion_conditional_on_correct_selection','eval_exact','router_path_usage')},sort_keys=True))
if __name__=='__main__': main()
