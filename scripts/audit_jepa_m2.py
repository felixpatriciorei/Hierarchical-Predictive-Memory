#!/usr/bin/env python3
"""Pre-training JEPA reintegration/isolation audit.

This does not claim JEPA improves representations without training.  It checks
that the architecture has a staged path where JEPA can be enabled without
immediately controlling load-bearing writer selection.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from hpm.maturity_contracts import jepa_stage_isolation_ok


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',type=int,default=0,choices=[0,1,2,3])
    ap.add_argument('--lambda-jepa',type=float,default=0.0)
    ap.add_argument('--use-token-jepa-aux',action='store_true')
    ap.add_argument('--use-jepa-writer-bias',action='store_true')
    ap.add_argument('--max-shared-grad-ratio',type=float,default=0.25)
    ap.add_argument('--output',type=Path,default=Path('runs/jepa_m2.json'))
    a=ap.parse_args()
    structural=jepa_stage_isolation_ok(lambda_jepa=a.lambda_jepa,use_token_jepa_aux=a.use_token_jepa_aux,use_jepa_writer_bias=a.use_jepa_writer_bias,stage=a.stage)
    # Before any training, only structural stages can be accepted.  Stage 0 is
    # the baseline-safe state. Stages 1+ require a future measured shared-trunk
    # gradient ratio and paired outcome test before promotion.
    passed=structural and a.stage==0
    report={
      'passed':passed,'stage':a.stage,'structural_contract_ok':structural,
      'max_shared_grad_ratio_for_future_stage':a.max_shared_grad_ratio,
      'writer_control_enabled':a.use_jepa_writer_bias,
      'interpretation': 'Stage 0 is pre-training mature: JEPA is observational/decoupled. Stages 1-3 are integration-ready but cannot be performance-mature before training; promotion requires bounded auxiliary/primary shared-gradient ratio and paired JEPA-off/on evidence.'
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f'wrote {a.output}')
    print(f"{'PASS' if passed else 'OPEN'} JEPA stage={a.stage} structural={structural}")

if __name__=='__main__':main()
