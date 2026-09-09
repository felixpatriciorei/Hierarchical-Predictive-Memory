#!/usr/bin/env python3
"""Assemble an HPM M2 architecture-maturity board from offline evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(path):
    if path is None: return None
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def gate(name, status, passed, blocker, evidence=None):
    return {"name": name, "status": status, "passed": bool(passed), "blocker": bool(blocker), "evidence": evidence or {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-audit")
    ap.add_argument("--support-audit")
    ap.add_argument("--capacity-audit")
    ap.add_argument("--write-audit")
    ap.add_argument("--jepa-audit")
    ap.add_argument("--b1-regression-ok", action="store_true")
    ap.add_argument("--r1-regression-ok", action="store_true")
    ap.add_argument("--router-dynamic-contract-ok", action="store_true")
    ap.add_argument("--output", default="runs/hpm_m2_maturity.json")
    a = ap.parse_args()
    router, support, capacity, write, jepa = map(load, [a.router_audit,a.support_audit,a.capacity_audit,a.write_audit,a.jepa_audit])
    support_pass = bool(support) and all(any(e.get("mature_under_thresholds") for e in entries) for entries in support.get("tasks",{}).values())
    cap_pass = bool(capacity) and capacity.get("delta_random_key_operational_capacity_at_threshold",0) > 0 and capacity.get("overwrite_test",{}).get("cosine_to_new",-1) > capacity.get("overwrite_test",{}).get("cosine_to_old",1)
    write_pass = bool(write) and bool(write.get("passed", False))
    jepa_pass = bool(jepa) and bool(jepa.get("passed", False))
    gates = [
        gate("B1 episodic writer", "frozen/regression", a.b1_regression_ok, True),
        gate("R1 episodic read", "frozen/regression", a.r1_regression_ok, True),
        gate("router dynamic starvation", "contract+monitor", a.router_dynamic_contract_ok, True, {"static_router_audit_present": bool(router)}),
        gate("write transition", "operator envelope", write_pass, True),
        gate("path support load-bearing", "faithfulness/completeness/minimality", support_pass, True),
        gate("capacity envelope", "episodic+delta substrate", cap_pass, True),
        gate("JEPA reintegration", "staged isolation", jepa_pass, False),
    ]
    report = {"maturity_level":"M2-training-mature", "all_hard_blockers_pass": all(g["passed"] for g in gates if g["blocker"]), "gates":gates}
    out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n")
    print(f"wrote {out}")
    for g in gates: print(f"{'PASS' if g['passed'] else 'OPEN'}  {g['name']}  blocker={g['blocker']}")
    print("M2 READY" if report["all_hard_blockers_pass"] else "M2 NOT READY")

if __name__ == '__main__': main()
