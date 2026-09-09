"""Run the CompositeSkillDataset four-path test (COPY/MOOD/FACT/XREF) on a
Modal GPU, for hpm_lite_v2 -- the only model type composite_harness.py wires
up (see hpm/composite_harness.py's module docstring for why: it's the
one architecture that answers all four sub-tasks in a single forward pass).

Mirrors scripts/modal_sweep.py's conventions exactly: same image build, same
add_local_dir mount, args passed as a SimpleNamespace straight into the run()
function instead of shelling out to argparse, results starmap'd across seeds
in parallel, one summary table printed at the end.

Setup (one-time, same as modal_sweep.py):
    pip install modal
    modal setup

Run from the repo root:
    modal run scripts/modal_run_composite_harness.py

Output lands locally under runs/composite_four_path/ (gitignored): one
step-log CSV plus one router_dependency.json per seed -- the JSON being the
frozen-router post-gate conditional-coablation diagnostic emitted when
router_dependence_eval is on (see hpm/composite_harness.py).
"""

from __future__ import annotations

import modal

app = modal.App("hpm-composite-four-path-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(".", remote_path="/root/repo")
)


@app.function(image=image, gpu="A100", timeout=60 * 60)
def run_one(seed: int, router_logit_clamp: float | None = None) -> dict:
    import hashlib
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, "/root/repo/src")

    # Ground truth about what's actually executing in THIS container, not
    # what your local disk says -- Modal build/mount caching can make those
    # disagree, and local mtimes can't detect that from your end.
    utils_path = Path("/root/repo/src/hpm/utils.py")
    utils_hash = hashlib.md5(utils_path.read_bytes()).hexdigest()
    print(f"[seed {seed}] utils.py md5 in container: {utils_hash}")

    from hpm.composite_harness import run
    from hpm.utils import set_seed
    import torch

    set_seed(seed)
    print(f"[seed {seed}] cudnn.deterministic after set_seed: {torch.backends.cudnn.deterministic}")

    args = SimpleNamespace(
        model="hpm_lite_v2",
        seq_len=768,
        window=64,
        copy_span=8,
        mood_block_len=40,
        mood_label_noise=0.0,
        num_facts=4,
        batch_size=32,
        steps=2000,
        eval_every=100,
        eval_batches=5,
        # Frozen-router post-gate dependence diagnostic (Stage E evidence).
        # NOTE: router_dependence_eval_batches must be an INT or omitted --
        # run() reads it via getattr(args, ..., default), so an explicit
        # None would slip past the getattr default and crash inside
        # evaluate_fixed_router_dependencies on `batches < 1`.
        router_dependence_eval=True,
        router_dependence_eval_batches=5,
        d_model=128,
        layers=2,
        heads=4,
        top_k=1,
        lr=3.0e-4,
        # None preserves the established benchmark. A finite positive value
        # (e.g. 3.0) enables the optional tanh effective-logit clamp while the
        # patched harness records raw/effective logits and Jacobian health.
        router_logit_clamp=router_logit_clamp,
        seed=seed,
        device="cuda",
        out_dir="/root/repo/runs",
    )
    result = run(args)
    result["seed"] = seed
    result["_utils_py_md5"] = utils_hash
    result["_cudnn_deterministic"] = torch.backends.cudnn.deterministic

    # add_local_dir only mounts INTO the container -- nothing written under
    # /root/repo/runs survives after this function returns. Read the CSV
    # back into memory now so the caller can write it to a real local path.
    step_log_path = result.get("step_log_path")
    if step_log_path:
        from pathlib import Path
        result["step_log_csv_text"] = Path(step_log_path).read_text()
    # Same treatment for the diagnostic artifact: read it into memory while
    # the container still exists, or it dies with the function call.
    dependence_path = result.get("router_dependency_path")
    if dependence_path:
        result["router_dependency_json_text"] = Path(dependence_path).read_text()
    return result


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2,3,4,5,6,7",
    router_logit_clamp: float | None = None,
    out_subdir: str = "",
    allow_overwrite: bool = False,
):
    import datetime as _dt
    import hashlib
    import json
    from pathlib import Path

    from hpm.evidence_provenance import evidence_set_sha256, require_collision_free, sha256_text

    seed_list = [int(s) for s in seeds.split(",")]
    base_out_dir = Path("runs") / "composite_four_path"
    out_dir = base_out_dir / out_subdir if out_subdir else base_out_dir
    # This check intentionally runs BEFORE starmap so a provenance collision
    # cannot burn GPU money and only then discover that old evidence would be
    # replaced.
    require_collision_free(out_dir, seed_list, allow_overwrite=allow_overwrite)
    print(f"Launching {len(seed_list)} composite four-path runs in parallel on Modal A100s...")
    print(f"Local evidence destination: {out_dir}")
    results = list(run_one.starmap((s, router_logit_clamp) for s in seed_list))

    # Write each seed's step log to the LOCAL machine -- this is the part
    # that was missing before: nothing under runs/ inside the container
    # persists past the function call, so we pull the CSV text back through
    # the return value and write it out here instead.
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        csv_text = r.pop("step_log_csv_text", None)
        if csv_text is None:
            print(f"WARNING: seed {r['seed']} returned no step log -- nothing to write for this seed.")
            continue
        dest = out_dir / f"seed{r['seed']}_composite_step_log.csv"
        dest.write_text(csv_text)
        print(f"wrote {dest}")

    for r in results:
        json_text = r.pop("router_dependency_json_text", None)
        if json_text is None:
            print(
                f"WARNING: seed {r['seed']} returned no router_dependency.json -- "
                "the diagnostic did not run in that container."
            )
            continue
        dest = out_dir / f"seed{r['seed']}_router_dependency.json"
        dest.write_text(json_text)
        print(f"wrote {dest}")

    # Content-address the returned evidence so later audits can prove which
    # exact per-seed artifacts a result was computed from.
    artifact_hashes = {}
    dependency_hashes = {}
    for seed in seed_list:
        for suffix in ("composite_step_log.csv", "router_dependency.json"):
            p = out_dir / f"seed{seed}_{suffix}"
            if p.exists():
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                artifact_hashes[p.name] = digest
                if suffix == "router_dependency.json":
                    dependency_hashes[seed] = digest
    manifest = {
        "schema": "hpm_composite_evidence_manifest_v1",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seeds": sorted(seed_list),
        "router_logit_clamp": router_logit_clamp,
        "allow_overwrite": bool(allow_overwrite),
        "output_dir": str(out_dir),
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "router_dependency_evidence_set_sha256": evidence_set_sha256(dependency_hashes) if dependency_hashes else None,
        "container_utils_md5": {str(r["seed"]): r.get("_utils_py_md5") for r in sorted(results, key=lambda x: x["seed"])},
        "cudnn_deterministic": {str(r["seed"]): r.get("_cudnn_deterministic") for r in sorted(results, key=lambda x: x["seed"])},
        "fixed_config": {
            "model": "hpm_lite_v2", "seq_len": 768, "window": 64, "copy_span": 8,
            "mood_block_len": 40, "mood_label_noise": 0.0, "num_facts": 4,
            "batch_size": 32, "steps": 2000, "eval_every": 100, "eval_batches": 5,
            "router_dependence_eval_batches": 5, "d_model": 128, "layers": 2,
            "heads": 4, "top_k": 1, "lr": 3.0e-4,
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"evidence_set_sha256={manifest['router_dependency_evidence_set_sha256']}")

    subtasks = ("copy", "mood", "fact", "xref")
    print("\n=== Container diagnostics (proves what actually ran, not what's on your local disk) ===")
    for r in sorted(results, key=lambda r: r["seed"]):
        print(f"seed {r['seed']}: utils.py md5={r.get('_utils_py_md5')}  cudnn.deterministic={r.get('_cudnn_deterministic')}")

    print("\n=== Summary (final eval_*_exact per seed) ===")
    header = f"{'seed':<6}" + "".join(f"{'eval_' + n + '_exact':<18}" for n in subtasks) + f"{'params':<10}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["seed"]):
        row = f"{r['seed']:<6}"
        for n in subtasks:
            row += f"{r.get(f'eval_{n}_exact', float('nan')):<18.4f}"
        row += f"{r.get('parameters', 0):<10}"
        print(row)
