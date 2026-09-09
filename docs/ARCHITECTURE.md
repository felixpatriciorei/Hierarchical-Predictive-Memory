# Architecture

## Maintained model

The current maintained architecture is the four-path HPM v2 implementation. Historical
artifacts use the identifier `hpm_lite_v2`; that string is retained for evidence replay.

### 1. Local path

A sliding-window Transformer-style path for short-range token interactions. It is not a
full-context baseline and should not be described as one.

### 2. Selective recurrent path

`BlockwiseSelectiveRecurrentState` uses a compact learned state with input-dependent step
sizes and state-space-style parameters. It is **Mamba-inspired**, but it is not the official
Mamba selective scan or Mamba-2 SSD implementation.

### 3. Fast-weight path

`FastWeightBlockMemory` stores block-level associations in a recurrent matrix and combines:

- normalized query/key projections;
- input-dependent channel-wise decay; and
- a delta-rule correction before writing a new value.

Its erase and write/correction strength are still coupled through one scalar `beta`. That is
simpler than Gated DeltaNet-2, which explicitly decouples erase and write gates.

### 4. Episodic path

An exact finite key-value memory with a learned candidate scorer/writer and hard selection.
B1 concerns the writer; R1 concerns the read operator. These are separate claims and should
remain separately testable.

### Router

The router softmax-mixes the four path states. Router-health telemetry distinguishes:

1. probability concentration;
2. optimization lock / low raw-logit-to-weight sensitivity; and
3. functional collapse, measured by frozen-gate path removal.

Sharp routing is not automatically failure: R2 shows task-conditioned sharp routing can be
causally load-bearing.

### JEPA auxiliary

JEPA-style predictive losses are auxiliary and staged. The safe baseline keeps predictive
losses from controlling writer selection. Writer bias is an experimental later stage, not a
default architectural requirement.

## Isolation mode

`episodic_only=True` is retained only as a **diagnostic isolation mode** for writer/read
controls. It must not be interpreted as evidence that the other paths are useless.
