from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
import torch.nn.functional as F

import run_zpoly_mlc_continuation_size50_milestones as m
from hpm.train import TinyAdamW
base = m.base


def save_opt_state(opt):
    return {
        'step_count': opt.step_count,
        'lr': opt.lr,
        'beta1': opt.beta1,
        'beta2': opt.beta2,
        'eps': opt.eps,
        'weight_decay': opt.weight_decay,
        'm': [x.detach().cpu().clone() for x in opt.m],
        'v': [x.detach().cpu().clone() for x in opt.v],
    }

def load_opt_state(opt, state, device):
    opt.step_count = int(state['step_count'])
    opt.lr = float(state['lr'])
    opt.beta1 = float(state['beta1']); opt.beta2 = float(state['beta2'])
    opt.eps = float(state['eps']); opt.weight_decay = float(state['weight_decay'])
    for dst, src in zip(opt.m, state['m']): dst.copy_(src.to(device))
    for dst, src in zip(opt.v, state['v']): dst.copy_(src.to(device))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--stop', type=int, required=True)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--save', required=True)
    ap.add_argument('--result', required=True)
    args=ap.parse_args()
    device=torch.device(args.device)

    specs=m.macro_specs(); by_name={s.name:s for s in specs}
    episodes, studies=m.build_episodes()
    model=m.make_model(device)
    opt=TinyAdamW(model.parameters(), lr=m.LR)
    rng=random.Random(m.SEED)

    if args.resume:
        ck=torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state'])
        load_opt_state(opt, ck['optimizer_state'], device)
        rng.setstate(ck['sample_rng_state'])
        random.setstate(ck['python_random_state'])
        torch.set_rng_state(ck['torch_rng_state'])
        expected=int(ck['next_step'])
        if expected != args.start:
            raise RuntimeError(f'resume next_step={expected} but --start={args.start}')
    else:
        if args.start != 1: raise RuntimeError('fresh run must start at 1')
        torch.manual_seed(m.SEED); random.seed(m.SEED)
        # Recreate model/optimizer *after* seeding, exactly like the original train().
        model=m.make_model(device)
        opt=TinyAdamW(model.parameters(), lr=m.LR)
        rng=random.Random(m.SEED)

    losses=[]; health=[]; started=time.perf_counter(); model.train()
    for step in range(args.start, args.stop+1):
        batch=[episodes[rng.randrange(len(episodes))] for _ in range(m.BATCH_SIZE)]
        logits,h=m.forward_batch(model,batch,specs,device,health=(step % m.LOG_EVERY == 0))
        action_logits=torch.stack([base.restricted_logits(logits[i],specs) for i in range(len(batch))])
        targets=torch.tensor([specs.index(by_name[ep['query']['action_name']]) for ep in batch],dtype=torch.long,device=device)
        loss=F.cross_entropy(action_logits,targets)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        losses.append(float(loss.detach().cpu()))
        if h is not None:
            rec={'step':step,'train_loss':losses[-1],**h}; health.append(rec); print(json.dumps(rec,sort_keys=True),flush=True)

    structural=m.evaluate_structural(model,specs,studies,device)
    model.train()
    row={
        'step':args.stop,
        'structural_heldout_exact':structural['structural_heldout_exact'],
        'macro_selection_accuracy':structural['macro_selection_accuracy'],
        'completion_conditional_on_correct_selection':structural['completion_conditional_on_correct_selection'],
        'completion_conditional_n':structural['completion_conditional_n'],
        'tasks':structural['tasks'],
    }
    Path(args.result).write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
    ck={
        'next_step':args.stop+1,
        'model_state':model.state_dict(),
        'optimizer_state':save_opt_state(opt),
        'sample_rng_state':rng.getstate(),
        'python_random_state':random.getstate(),
        'torch_rng_state':torch.get_rng_state(),
        'last_chunk_loss_mean':sum(losses)/len(losses),
        'last_chunk_wall_seconds':time.perf_counter()-started,
    }
    torch.save(ck,args.save)
    print(json.dumps({'milestone_structural':row},sort_keys=True),flush=True)

if __name__=='__main__': main()
