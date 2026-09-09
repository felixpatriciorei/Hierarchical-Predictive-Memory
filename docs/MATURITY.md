# HPM M2 architecture-maturity contract

M2 means “mature enough that training validates learned behavior rather than discovering a
basic uninstrumented architectural defect.” It is below theorem-complete or publication-final
maturity.

Every lane must have:

1. specification;
2. observability;
3. an analytic or conservative empirical operating envelope;
4. a failure/abort condition; and
5. regression + evidence provenance.

## B1 writer

Frozen settings and current regression suite must remain green. Changes to writer objectives
re-open B1 and require new controls.

## R1 read

Hard-forward/relaxed-backward semantics and read-mode invariants must remain green. New
retrieval operators are new experiments, not silent replacements.

## Router dynamic starvation

Declare a finite discovery horizon `T`, initial adverse gap `A0`, worst observed/conservative
adverse gap drift `d_max`, and minimum survival mass `p_min`. The guard is abort-only; it must
not silently clamp/balance the router.

For K paths:

`p_k(t) >= 1 / (1 + (K-1) exp(A0 + T d_max))`.

The measured envelope must keep this above `p_min` for the declared discovery window.

## Write-transition stability

For smooth differentiable selection, use the real operator at `tau >= tau_min > 0` and record
a finite local Jacobian envelope over the intended score range. For deployed hard selection,
require a positive Top-K boundary margin and perturbations below half that margin.

No global pointwise Lipschitz claim is made.

## Path load-bearing

For a proposed support set S, report from the frozen-gate intervention cube:

- faithfulness: `full - drop(S)`;
- completeness: `full - keep_only(S)`;
- minimality: `keep_only(S) - keep_only(S\\{k})` for each `k in S`.

Thresholds must be declared before a support set is promoted to a frozen claim.

## Capacity

- Episodic workload must fit its exact slot contract.
- Delta/fast-weight memory must pass an intended-load interference and overwrite stress test.
- The first training workload must be recorded inside that measured envelope.

## JEPA

Stage 0: no JEPA loss/control influence.
Stage 1: block-level predictive loss only, writer control disabled.
Stage 2: token-level predictive loss, still no writer control.
Stage 3: writer bias only after paired evidence supports it.

## Release board

`scripts/audit_hpm_maturity.py` is the machine-readable board. A future training run should
consume artifacts from the exact checkout/config being promoted rather than relying on an old
human summary.
