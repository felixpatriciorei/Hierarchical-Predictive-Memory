# Reproducibility

## Re-foundation baseline

Before cleanup, the uploaded laboratory snapshot compiled and passed:

```text
248 passed
Python 3.13.5
PyTorch 2.10.0+cpu
NumPy 2.3.5
```

The new repository is re-tested after package/layout cleanup; the final retained-test count is
recorded in `CLEANUP_REPORT.md`.

## What is intentionally not in Git

- `.git` from the old repository;
- checkpoints (`*.pt`, `*.pth`, `*.ckpt`);
- raw `runs/` directories;
- live TorchScope feeds;
- generated Plotly/Matplotlib atlases;
- cloud stdout/stderr logs;
- Python caches;
- old backup copies;
- abandoned/failed intervention sweep scripts;
- the former LLM/RAG symbolic-memory demo branch;
- the former structured-readout/noisy-extraction side project.

These were useful during exploration but are not part of the maintained HPM architecture.

## Evidence

Compact evidence under `evidence/` is hashed by `evidence/MANIFEST.sha256`. Large original run
storage should be archived separately if long-term byte-level reproduction is required.
