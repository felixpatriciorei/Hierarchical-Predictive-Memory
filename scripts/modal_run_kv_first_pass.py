"""Run the first HPM KV training pass on one Modal A100.

Launch from the repository root:
    modal run scripts/modal_run_kv_first_pass.py

Runs exactly:
    run_memory_model.py --models hpm_lite_v2 --write-mode learned --steps 2000 \
      --router-starvation-guard false --memory-slots 16 --seed 260909 \
      --device cuda --attention-memory-mode auto --log-every 50 --save-step-log

Text artifacts produced under runs/memory_model/ in the Modal container are copied
back to the same relative paths on the local machine when the run completes.
"""

from __future__ import annotations

import modal

app = modal.App("hpm-kv-first-pass-seed260909")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(".", remote_path="/root/repo")
)


@app.function(image=image, gpu="A100", timeout=3 * 60 * 60)
def run_kv() -> dict:
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path("/root/repo")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    # Reduce allocator fragmentation on long CUDA runs. This does not alter
    # model math; it only lets PyTorch grow CUDA segments more flexibly.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        sys.executable,
        "-u",
        str(repo / "scripts" / "run_memory_model.py"),
        "--models", "hpm_lite_v2",
        "--write-mode", "learned",
        "--steps", "2000",
        "--router-starvation-guard", "false",
        "--memory-slots", "16",
        "--seed", "260909",
        "--device", "cuda",
        "--attention-memory-mode", "auto",
        "--log-every", "50",
        "--save-step-log",
    ]

    def run_streaming(command: list[str]) -> tuple[int, str]:
        print("Launching command:")
        print(" ".join(command))
        proc = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        tail: list[str] = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            if len(tail) > 200:
                del tail[: len(tail) - 200]
        return proc.wait(), "".join(tail)

    returncode, tail = run_streaming(cmd)
    if returncode != 0:
        lower_tail = tail.lower()
        is_cuda_oom = "cuda out of memory" in lower_tail or "cuda driver error: out of memory" in lower_tail
        if is_cuda_oom:
            # Last-resort safety net: restart cleanly from the same seed/config
            # with the proven low-memory execution path instead of terminating
            # the Modal app and requiring another paid container launch.
            fallback = list(cmd)
            mode_index = fallback.index("--attention-memory-mode") + 1
            fallback[mode_index] = "memory_saver"
            print(
                "[attention-memory] auto run hit CUDA OOM; restarting the exact "
                "training seed/config in memory_saver mode inside the SAME Modal container."
            )
            returncode, tail = run_streaming(fallback)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

    root = repo / "runs" / "memory_model"
    artifacts: dict[str, str] = {}
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
                rel = path.relative_to(repo).as_posix()
                artifacts[rel] = path.read_text(encoding="utf-8", errors="replace")

    return {"artifacts": artifacts}


@app.local_entrypoint()
def main():
    from pathlib import Path

    result = run_kv.remote()
    artifacts = result.get("artifacts", {})
    if not artifacts:
        print("WARNING: training completed but no text artifacts were returned.")
        return

    for rel, text in sorted(artifacts.items()):
        dest = Path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(f"wrote {dest}")
