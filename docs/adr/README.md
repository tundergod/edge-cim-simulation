# Architecture Decision Records

Each ADR records **one** significant design decision: `context` (problem + forces) → `decision` → `consequences`, plus a `status`. ADRs are the "why we decided X" log; `OVERALL.md` is the "what to build" brief.

## Convention

- **Numbering:** monotonic (`0008-*.md`, `0009-*.md`, …). One decision per file.
- **Status lifecycle:** `Proposed` → `Accepted` → `Superseded by ADR-XXXX` / `Deprecated`.
- **Immutable once acted on:** to reverse or change an *Accepted* decision you've started building on, **write a new ADR that supersedes it** and mark the old one `Superseded by ADR-XXXX`. Don't rewrite history. (Typo/clarity fixes in place are fine; substantive reversals are not.)
- **When to write one:** any decision with real alternatives + consequences (not trivia). The per-phase workflow naturally produces them — e.g. op-vs-tensor finalization after Phase 0.1, threshold recalibration after the first end-to-end sim (these will *supersede* the relevant parts of ADR-0003 / ADR-0006).
- **Consumption:** per `docs/agents/domain.md`, read the ADRs touching an area before working in it; if your work contradicts one, surface it explicitly rather than silently overriding.

## Index

| # | Decision |
|---|---|
| [0001](0001-engine-fidelity.md) | Engine = lightweight event-driven + bandwidth contention |
| [0002](0002-memory-model.md) | Memory = Ramulator2, representative-iteration, swappable |
| [0003](0003-scheduler-mapper.md) | Scheduler = static-first, op/tensor granularity, validation-first |
| [0004](0004-mixed-precision.md) | Mixed-precision boundary modeling |
| [0005](0005-energy-model.md) | Energy = spec-based + opportunistic anchor |
| [0006](0006-validation-bridging-extrapolation.md) | Validation contract + L4 bridging + extrapolation |
| [0007](0007-op-inventory-extraction.md) | Op inventory = PyTorch runtime tracer |
