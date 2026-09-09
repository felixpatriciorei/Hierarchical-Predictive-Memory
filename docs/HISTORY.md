# Project history and retired material

## Former repository

The project began as `HPM-Lite-Memory-Model`, a small exact-recall research testbed. Over time it
accumulated the writer, read-side controls, four-path router, fast-weight memory, maturity theory,
TorchScope evidence integration, many failed interventions, visualization code, and unrelated
side experiments.

By September 2026 the old name and repository structure no longer described the maintained work.
The new repository is therefore a clean re-foundation rather than a direct copy of Git history.

## Retired from the maintained tree

- large historical long-context result/figure system;
- LLM/RAG symbolic-slot benchmark;
- structured-readout and noisy-extraction branch;
- dozens of one-off Modal sweep scripts (router z-loss, episodic-only simplification, scheduled
  sampling, primary-loss annealing, old Sinkhorn combinations, etc.);
- generated patch/test-report files;
- obsolete roadmap/repo-cleanup documents;
- raw checkpoints, debug feeds, caches, and backup folders.

Retirement does not mean every historical experiment was worthless. It means those files are not
needed to understand, test, or advance the current HPM architecture.

## Important retained lessons

- Exact episodic capacity must be aligned with the task; an impossible slot/query contract cannot
  be repaired by optimization tricks.
- Writer, reader, router, and predictive auxiliary claims must be isolated before composition.
- Sharp softmax routing can be useful specialization; causal path dependence matters more than
  cosmetic load balance.
- Human-readable summaries are not evidence provenance.
