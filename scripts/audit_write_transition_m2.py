#!/usr/bin/env python3
"""CPU-only M2 probe for HPM's differentiable Top-K write operator.

Uses the repository's real `differentiable_topk_bias` implementation.  It
samples bounded score vectors, measures exact autograd Jacobian norms, and
checks a declared hard-selection margin/temperature envelope.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch

from hpm.maturity_probes import jacobian_frobenius_norm

try:
    from hpm.differentiable_topk import differentiable_topk_bias
except Exception as exc:
    raise SystemExit("differentiable_topk.py is required in the repo root for the M2 write probe") from exc


def topk_margin(scores, k):
    vals = scores.detach().sort(descending=True).values
    return float((vals[k-1] - vals[k]).item())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates',type=int,default=16)
    ap.add_argument('--k',type=int,default=4)
    ap.add_argument('--temperature',type=float,default=0.3)
    ap.add_argument('--iters',type=int,default=50)
    ap.add_argument('--samples',type=int,default=8)
    ap.add_argument('--score-bound',type=float,default=5.0)
    ap.add_argument('--max-jacobian-fro',type=float,default=100.0)
    ap.add_argument('--min-margin',type=float,default=0.05)
    ap.add_argument('--output-change-budget',type=float,default=0.05)
    ap.add_argument('--output',type=Path,default=Path('runs/write_transition_m2.json'))
    a=ap.parse_args()
    if not 1 <= a.k < a.candidates: raise SystemExit('need 1 <= k < candidates')
    if a.temperature <= 0: raise SystemExit('temperature must be > 0')
    jac, margins=[] ,[]
    for s in range(a.samples):
        g=torch.Generator().manual_seed(7001+s)
        x=(2*torch.rand(a.candidates,generator=g)-1)*a.score_bound
        # Keep the diagnostic away from exact ties while still measuring the
        # implementation on arbitrary bounded score geometry.
        x=x+torch.arange(a.candidates,dtype=x.dtype)*1e-4
        margins.append(topk_margin(x,a.k))
        def fn(z):
            return differentiable_topk_bias(z.unsqueeze(0), k=a.k, eps=a.temperature, n_iters=a.iters).squeeze(0)
        jac.append(jacobian_frobenius_norm(fn,x))
    jmax=max(jac); mmin=min(margins)
    # A local perturbation budget guaranteed not to cross the hard boundary.
    score_eps=0.49*mmin
    passed=math.isfinite(jmax) and jmax <= a.max_jacobian_fro and mmin >= a.min_margin and score_eps>0
    report={
      'passed':passed,'temperature':a.temperature,'iters':a.iters,'score_bound':a.score_bound,
      'jacobian_fro':{'max':jmax,'mean':sum(jac)/len(jac)},
      'hard_topk_margin':{'min':mmin,'mean':sum(margins)/len(margins)},
      'safe_score_perturbation_linf':score_eps,
      'declared_max_jacobian_fro':a.max_jacobian_fro,'declared_min_margin':a.min_margin,
      'note':'This is an operator-level M2 envelope, not a training-trajectory Lipschitz theorem.'
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f'wrote {a.output}')
    print(f"{'PASS' if passed else 'OPEN'} jacobian_max={jmax:.4g} margin_min={mmin:.4g}")

if __name__=='__main__': main()
