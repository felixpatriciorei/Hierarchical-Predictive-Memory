# Dependency policy

Fact-check date: 2026-09-09.

The project deliberately avoids a fake `pip freeze` lock from a developer workstation. The
install contract is in `pyproject.toml`; release tags should record a tested environment.

## Current ecosystem snapshot

At the cleanup date:

- PyTorch 2.14.0 was the current stable release (2026-09-02).
- NumPy 2.5.3 was current (2026-09-06); NumPy 2.5 supports Python 3.12-3.14.
- pytest 9.1.1 was current stable; 9.2 was still an unreleased draft.
- SciPy 1.18.1 was current stable.
- pandas 3.0.5 was current stable, but pandas is not required by the maintained core package.

The repository's re-foundation test run itself used Python 3.13.5, PyTorch 2.10.0+cpu, and
NumPy 2.3.5. That environment is a tested baseline, not a claim that those were latest.

## Why Python 3.12-3.14

This range matches the current NumPy 2.5 stable support window while remaining inside current
PyTorch support. Python 3.11 support was intentionally dropped from the new package metadata.
