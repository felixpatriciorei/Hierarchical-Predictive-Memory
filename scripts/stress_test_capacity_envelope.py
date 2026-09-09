#!/usr/bin/env python3
"""CPU-only operational capacity envelope for HPM's memory substrates."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F

from hpm.capacity_theory import (
    orthogonal_delta_exact_capacity,
    mutual_coherence,
    normalized_recall_cosine,
    overwrite_cosine,
)


def orthogonal_keys(n, d, g):
    if n > d:
        return None
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q[:, :n].T.contiguous()


def random_keys(n, d, g):
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def run_family(kind, n, d, trials, seed):
    scores, mus = [], []
    for t in range(trials):
        g = torch.Generator().manual_seed(seed + 1009*t + 17*n)
        keys = orthogonal_keys(n, d, g) if kind == "orthogonal" else random_keys(n, d, g)
        if keys is None:
            return None
        values = F.normalize(torch.randn(n, d, generator=g), dim=-1)
        scores.append(normalized_recall_cosine(keys, values))
        mus.append(mutual_coherence(keys))
    return {"mean_recall_cosine": sum(scores)/len(scores), "min_recall_cosine": min(scores), "mean_mutual_coherence": sum(mus)/len(mus)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-dim", type=int, default=128)
    ap.add_argument("--episodic-slots", type=int, default=16)
    ap.add_argument("--max-associations", type=int, default=256)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--recall-threshold", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=Path, default=Path("runs/capacity_envelope.json"))
    args = ap.parse_args()
    ns = sorted(set([1,2,4,8,16,32,64,128,args.max_associations]))
    ns = [n for n in ns if n <= args.max_associations]
    families = {"orthogonal": {}, "random": {}}
    for kind in families:
        for n in ns:
            r = run_family(kind, n, args.key_dim, args.trials, args.seed)
            if r is not None:
                families[kind][str(n)] = r
    empirical = 0
    for n_s, r in families["random"].items():
        if r["min_recall_cosine"] >= args.recall_threshold:
            empirical = max(empirical, int(n_s))

    g = torch.Generator().manual_seed(args.seed + 99991)
    key = torch.randn(args.key_dim, generator=g)
    old = F.normalize(torch.randn(args.key_dim, generator=g), dim=-1)
    new = F.normalize(torch.randn(args.key_dim, generator=g), dim=-1)
    new_cos, old_cos = overwrite_cosine(key, old, new)

    report = {
        "key_dim": args.key_dim,
        "episodic_exact_slot_capacity": args.episodic_slots,
        "delta_zero_interference_exact_lower_bound": orthogonal_delta_exact_capacity(args.key_dim),
        "delta_random_key_operational_capacity_at_threshold": empirical,
        "recall_threshold": args.recall_threshold,
        "overwrite_test": {"cosine_to_new": new_cos, "cosine_to_old": old_cos},
        "families": families,
        "interpretation": "Combined capacity is reported as a substrate envelope, not added into a universal scalar: episodic slots give hard exact capacity; delta memory gives an orthogonal-key lower bound plus an empirical interference curve. End-to-end additive capacity requires successful routing/task decomposition and is not assumed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"episodic exact slots={args.episodic_slots}")
    print(f"delta orthogonal exact lower bound={args.key_dim}")
    print(f"delta random operational capacity>={empirical} at cosine>={args.recall_threshold}")
    print(f"overwrite cosine(new)={new_cos:.4f} cosine(old)={old_cos:.4f}")

if __name__ == "__main__":
    main()
