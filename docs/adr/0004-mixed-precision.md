# ADR-0004 — Mixed-precision boundary modeling

Status: Accepted (2026-06-03)

## Context
Units have different native precisions (CIM INT8, Mali FP16, NPU INT8/INT16/FP16, CPU any). Crossing a precision boundary (e.g. CIM-INT8 FFN → GPU-FP16 attention) needs dequant/requant, costing time, memory traffic, and potentially output quality. Mixed precision is the stated main research surface, but its mechanism was unspecified.

## Decision
- **(a) Boundary = explicit conversion op.** The scheduler inserts a conversion op wherever two adjacent ops land on units with incompatible native precision; its cost (time/energy/memory) is calibrated in Phase 0.2 (dequant/requant are cheap elementwise ops, measurable on CPU/NPU). **Where the conversion op is placed (which unit pays its cost and contention) is itself a scheduler decision (ADR-0003).** Precision boundaries become visible, computable trace elements.
- **(b) Performance+energy is the output; quality is measured once, not simulated.** We do not simulate numerics. Output quality of one chosen INT8×FP16 split is **measured once on real hardware** (perplexity / a small benchmark) as a sanity check that mixed precision is valid.
- **(c) Precision mostly determined by unit; NPU is the free knob.** CIM=INT8, Mali=FP16 fix precision by placement; the NPU (INT8/INT16/FP16) is the one genuine precision lever. Scheduler precision decisions therefore occur at NPU placement and at CIM↔GPU boundaries — a bounded, solvable, verifiable problem.

## Consequences
No numerical/quantization-error modeling in scope. One mixed-precision quality measurement is required (Phase 0.2). Conversion-op costs become part of the Phase 0.2 micro-benchmark set.
