# Evidence provenance policy

## 2026-09-09 FACT incident

A later N0 seed-0 run reused the flat output names in the composite experiment directory and
silently replaced the original seed-0 router dependency JSON used by an earlier accepted FACT
certificate.

The certificate itself is still an immutable record of the submitted numerical trials, but the
current raw directory cannot be advertised as the original byte-replayable evidence set.

## Repository policy after re-foundation

1. Raw experiment outputs are ignored by Git.
2. Every promoted evidence bundle must have artifact SHA-256 hashes.
3. Experiment runners should refuse collisions by default.
4. Human summaries never replace cert/audit files.
5. Reconstructed artifacts must say `reconstructed` in the filename and documentation.
6. Changes to frozen model semantics create a new evidence epoch rather than silently updating old claims.
