# Hierarchical Predictive Memory (HPM)

HPM is a research prototype for studying whether a compact neural system can combine
**local computation, recurrent state, fast-weight associative memory, and exact episodic
memory** without collapsing those mechanisms into one undifferentiated path.

This repository is the clean successor to `HPM-Lite-Memory-Model`. It keeps the evidence-
compatible model identifier `hpm_lite_v2` inside historical configs, but the maintained
Python package and repository are now simply **HPM**.

> **Status:** research prototype. Several component-level claims are frozen; the whole
> architecture is not yet claimed to be generally superior to a Transformer, a language
> model, or an AGI system.

## Architecture

```text
input
  |
  +-- local sliding-window path
  +-- selective recurrent path
  +-- delta-rule / fast-weight path
  +-- exact episodic key-value path
  |          ^
  |          +-- learned writer (B1)
  |          +-- hard-forward / relaxed-backward read option (R1)
  |
  +-- softmax router
             |
          prediction
```

The recurrent path is **Mamba-inspired**, not the official Mamba/Mamba-2 implementation.
The fast-weight path is **DeltaNet-family-inspired**, not an exact implementation of KDA
or Gated DeltaNet-2. See [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## What is currently established

| Track | Status | Narrow claim |
|---|---|---|
| B1 writer | **frozen** | A finite-capacity episodic writer can learn a causally visible allocation rule under the controlled salience task, including self-writing at `p=0`. |
| R1 read | **frozen empirical mechanism** | The hard-forward / relaxed-backward Top-K read improves the aliased-key read control across all tested paired seeds and candidate counts. |
| R2 router dependence | **partly assay-certified** | Recurrent/COPY and fast-weight/MOOD are fixed-sample assay accepts; episodic/FACT was also accepted, but its original seed-0 raw source was later overwritten, so full raw replay provenance is incomplete. |
| Router non-collapse | **active** | Static necessary-support results and router-health instrumentation exist; dynamic pre-specialization starvation remains open. |
| Write-transition stability | **active** | Schedule regularity, expected branch continuity, and positive-margin hard-selection stability are established; a full smooth-operator/trajectory bound is not. |
| Capacity | **active** | Episodic hard capacity is explicit; delta-memory operational capacity is stress-tested rather than claimed as a universal closed-form law. |
| JEPA | **staged / off by default for load-bearing control** | Predictive auxiliaries are kept separate from writer control until paired evidence justifies promotion. |

The authoritative board is [`docs/STATUS.md`](docs/STATUS.md). The architecture-wide
promotion contract is [`docs/MATURITY.md`](docs/MATURITY.md).

## Install

Python 3.12–3.14:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest -q
```

The repository cleanup baseline passed **all retained tests** before release; see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact environment and what
was intentionally removed from version control.

## Minimal usage

The historical CLI model identifier remains `hpm_lite_v2` for experiment compatibility:

```bash
python -m hpm.train --model hpm_lite_v2 --task kv --steps 100 --device cpu
```

For architecture-maturity checks:

```bash
python scripts/audit_hpm_maturity.py --help
python scripts/stress_test_capacity_envelope.py --help
python scripts/router_dynamic_contract.py --help
```

For the frozen-router composite diagnostic:

```bash
python scripts/analyze_router_dependency.py --help
python scripts/analyze_path_support.py --help
python scripts/analyze_router_interactions.py --help
```

## Evidence policy

Large raw runs, checkpoints, live feeds, generated figures, and cloud logs are **not tracked**.
Compact claim-facing evidence is under [`evidence/`](evidence/), with hashes in
`evidence/MANIFEST.sha256`.

The statistical accept/reject records in `evidence/assay/` were produced by the separate
TorchScope fixed-sample metric-delta gate. HPM does not treat a human-readable summary as a
replacement for those records.

## Repository map

```text
src/hpm/              maintained architecture and theory helpers
scripts/              current evidence runners, analyzers, and maturity audits
tests/                retained regression/causal-contract tests
evidence/             compact claim-facing evidence only
docs/                  canonical architecture, status, maturity, and research notes
```

## Scope

HPM is deliberately small and auditable. The current evidence is mostly synthetic and
mechanism-oriented. It does **not** establish:

- general language-model quality;
- universal router non-collapse;
- universal memory-capacity scaling laws;
- a benefit from JEPA in the load-bearing writer/router path;
- superiority to matched modern Transformer/SSM baselines on broad tasks; or
- recursive/self-improving intelligence.

Those are future empirical questions, not implied conclusions.

## License

MIT. See [`LICENSE`](LICENSE).
