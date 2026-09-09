#!/usr/bin/env python3
"""Calculate the finite-horizon router starvation contract before training."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from hpm.maturity_contracts import router_probability_floor,max_router_gap_drift_for_floor

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--num-paths',type=int,default=4)
    ap.add_argument('--initial-gap',type=float,required=True,help='worst initial competitor minus required-path logit gap')
    ap.add_argument('--discovery-horizon',type=int,required=True)
    ap.add_argument('--minimum-probability',type=float,default=1e-3)
    ap.add_argument('--observed-max-gap-drift',type=float,required=True,help='measured or conservative max adverse gap growth per step')
    ap.add_argument('--output',type=Path,default=Path('runs/router_dynamic_contract.json'))
    a=ap.parse_args()
    floor=router_probability_floor(a.num_paths,a.initial_gap,a.observed_max_gap_drift,a.discovery_horizon)
    allowed=max_router_gap_drift_for_floor(a.num_paths,a.initial_gap,a.discovery_horizon,a.minimum_probability)
    passed=a.observed_max_gap_drift <= allowed and floor >= a.minimum_probability
    report={'passed':passed,'num_paths':a.num_paths,'initial_gap':a.initial_gap,'discovery_horizon':a.discovery_horizon,'minimum_probability':a.minimum_probability,'observed_max_gap_drift_per_step':a.observed_max_gap_drift,'allowed_max_gap_drift_per_step':allowed,'guaranteed_probability_floor':floor,'monitor_policy':'abort-only: do not auto-balance or clamp; abort if the declared finite-horizon contract is violated'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f'wrote {a.output}')
    print(f"{'PASS' if passed else 'OPEN'} p_floor={floor:.6g} allowed_drift={allowed:.6g}")
if __name__=='__main__':main()
