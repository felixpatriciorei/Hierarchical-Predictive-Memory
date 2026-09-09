# Current project status

_Last reconciled: 2026-09-09 during repository re-foundation._

This file is the canonical status source. Date-stamped patch reports and competing
`CURRENT_STATE` files from the old repository were deliberately removed.

## B1 — episodic writer: FROZEN

The narrow B1 claim is a controlled finite-allocation result: when write-time utility is
causally visible and `required_facts <= episodic_slots < candidate_facts`, the learned writer
can outperform a matched masked-role control and self-write without oracle support.

Important boundaries:

- this is not natural-language semantic salience;
- the role token is intentionally visible in the positive control;
- candidate-growth makes the ranking problem harder;
- the Top-C auxiliary is an experimental training surrogate, not a universal writer objective.

Compact B1 summaries are under `evidence/b1/`.

## R1 — episodic read: FROZEN EMPIRICAL MECHANISM

The aliased-key control isolates the hard Top-K gradient cutoff. Across the vectorized
three-seed replication, STE minus hard read improved:

- answer exact: mean `+0.679167`, minimum `+0.634375`;
- retrieval top-1: mean `+0.652083`, minimum `+0.614063`.

A candidate-set sweep at `M = 4, 8, 16` also had positive paired effects for all nine
seed/size combinations.

**Statistical scope:** the old TorchScope fixed-sample gate remained inconclusive at `n=3` for
the replication metrics. R1 is therefore frozen as a replicated mechanism result, not as a
TorchScope-certified population-level claim.

## R2 — heterogeneous router dependence

Frozen-gate post-softmax/no-renormalization co-ablation established heterogeneous functional
dependence on the composite benchmark.

### Accepted fixed-sample assay claims

- recurrent retained for COPY: `n=9`, mean `+0.979167`, lower bound `+0.571209`;
- fast-weight retained for MOOD: `n=9`, mean `+0.597917`, lower bound `+0.189959`;
- episodic retained for FACT: `n=9`, mean `+0.420139`, lower bound `+0.012181` at certification time.

### FACT provenance caveat

A later seed-0 observational rerun overwrote the original seed-0 raw dependency JSON. The
minted FACT certificate and audit record survive, and the original submitted trial values can
be reconstructed, but the current flat nine-seed raw directory is not byte-replayable evidence
for that certificate.

The current raw nine-seed set still has episodic removal hurting FACT in 9/9 seeds, but its
mean singleton drop is `+0.366667`, which would not clear the same conservative n=9 Hoeffding
lower bound. Do not silently substitute the current directory for the original certified set.

## Necessary-support theory: ACTIVE

The static result is intentionally support-set based rather than “one task = one path”. Under
bounded path outputs and Lipschitz downstream loss, a support set with a positive frozen-gate
removal penalty cannot carry arbitrarily tiny routed mass in the trained model.

Open dynamic question: can co-training starve a not-yet-useful path before it has learned a
representation that exposes its eventual necessity?

## Write-transition stability: ACTIVE

Global pointwise Lipschitz continuity is false for the current mechanism because Bernoulli
teacher forcing and hard Top-K introduce discontinuities. Established pieces are:

- schedule regularity;
- fixed-model expected branch continuity; and
- positive-margin local stability of hard selection.

A concrete smooth-operator Jacobian envelope and trajectory-level guarantee remain open.

## Capacity: ACTIVE

- Episodic exact capacity is a hard slot budget.
- Delta/fast-weight capacity is treated operationally through interference/overwrite stress
  tests, not as a single universal scalar.
- Do not add “episodic slots + delta associations” into one capacity number without a task /
  routing decomposition theorem.

## JEPA: STAGED

JEPA remains optional. Stage 0 is isolation-safe. Predictive loss may be studied before any
JEPA-derived signal is allowed to affect writer selection. No load-bearing JEPA benefit is
currently frozen.

## Whole-architecture release target

The next architecture milestone is **M2 training-mature**: every cardinal component has a
specified contract, observable failure mode, operating envelope, fail/abort condition, and
provenance/regression guard. See `docs/MATURITY.md`.
