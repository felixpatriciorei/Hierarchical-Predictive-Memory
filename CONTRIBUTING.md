# Contributing

HPM is an experimental research codebase. Changes that affect model behavior should include:

1. a narrowly stated hypothesis;
2. a regression or causal control when possible;
3. a reproducible configuration;
4. evidence provenance (seed/config/artifact hashes); and
5. an explicit update to `docs/STATUS.md` if a frozen claim changes.

Run `python -m pytest -q` before submitting changes. Do not commit checkpoints, raw run folders,
or generated debug feeds.
