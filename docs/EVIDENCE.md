# Evidence and claim boundaries

## Why the repository tracks only compact evidence

The former repository mixed code with hundreds of MB of checkpoints, raw step logs, live feeds,
generated dashboards, and experiment scratch space. That makes review harder and makes evidence
provenance easier to accidentally overwrite.

This repository tracks only compact claim-facing artifacts. Large raw run directories belong in
versioned external experiment storage, not Git.

## B1 writer evidence

`evidence/b1/` contains compact candidate-growth and learning-horizon summaries. The key result is
not “the writer solved salience”; it is that a controlled causal allocation task is learnable and
distinguishable from its matched masked-role ceiling.

At `M=16`, `p=C=2`, 1000 steps, the three-seed means were:

- visible BCE exact: `0.49375`;
- visible Top-C auxiliary exact: `0.62031`;
- visible BCE required-record write rate: `0.49688`;
- visible Top-C required-record write rate: `0.61810`.

The earlier 250-step `M=16` masked control remained near the analytic `C/M = 0.125` coverage ceiling.

## R1 read evidence

`evidence/r1/paired_summary.csv` is the vectorized three-seed paired replication. All three seeds
improve under the hard-forward / relaxed-backward read estimator.

The slot-scale audit independently checks `M=4,8,16` and records all nine paired improvements.
This is a proof-of-mechanism-style empirical record, not a fixed-sample assay acceptance at n=3.

## R2 router evidence

`evidence/r2/current_nine_seed/` contains the **current** nine frozen-router dependency JSONs and
aggregate audits. These are useful current evidence, but the seed-0 file is not the original seed-0
artifact used by the accepted episodic/FACT assay.

The three accepted assay audit records are preserved under `evidence/assay/audit/`.

### Accepted records

- `3f5e685b...` recurrent / COPY
- `8e3ae6ec...` fast-weight / MOOD
- `7c3b1d1f...` episodic / FACT (raw-source provenance caveat)

`evidence/r2/reconstructed_fact_cert_trials_7c3b1d1f7cc9.csv` reproduces the numerical FACT trials
submitted at certification time, but is explicitly reconstructed rather than a recovered raw JSON.

## Statistical interpretation

TorchScope's fixed-sample metric-delta checker uses paired trial deltas and a conservative
Hoeffding bound. A certificate establishes only the stated metric comparison under the submitted
trial set and checker assumptions. It is not a universal architecture theorem.
