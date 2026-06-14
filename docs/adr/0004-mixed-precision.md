# ADR-0004 — Mixed-precision boundary modeling

Status: Accepted (2026-06-03)

## Context
Units have different native precisions (CIM INT8, Mali FP16, NPU INT8/INT16/FP16, CPU any). Crossing a precision boundary (e.g. CIM-INT8 FFN → GPU-FP16 attention) needs dequant/requant, costing time, memory traffic, and potentially output quality. Mixed precision is the stated main research surface, but its mechanism was unspecified.

## Decision
- **(a) Boundary = explicit conversion op.** The scheduler inserts a conversion op wherever two adjacent ops land on units with incompatible native precision; its cost (time/energy/memory) is **modeled analytically in Phase 2** — a dequant/requant is a deterministic memory-bound cast (read int8, write fp16, × scale), priced by the existing M2/M4 per-op (memory/elementwise) models × the number of boundary crossings the scheduler inserts. **No dedicated measurement campaign is required** (precisions are fixed per unit; see (c)). **Where the conversion op is placed (which unit pays its cost and contention) is itself a scheduler decision (ADR-0003).** Precision boundaries become visible, computable trace elements.
- **(b) Performance+energy is the output; quality is measured once, not simulated.** We do not simulate numerics. Output quality of one chosen INT8×FP16 split is **measured once on real hardware** (perplexity / a small benchmark) as a sanity check that mixed precision is valid.
- **(c) Precision mostly determined by unit; NPU is the free knob.** CIM=INT8, Mali=FP16 fix precision by placement; the NPU (INT8/INT16/FP16) is the one genuine precision lever. Scheduler precision decisions therefore occur at NPU placement and at CIM↔GPU boundaries — a bounded, solvable, verifiable problem.

## Consequences
No numerical/quantization-error modeling in scope. One mixed-precision quality measurement is required (a single INT8×FP16 quality sanity check). Conversion-op cost is **modeled analytically in Phase 2** (memory-bound cast via the existing M2/M4 per-op models × the scheduler's boundary-crossing count) — it is **not** a measurement target.

## Revision (2026-06-11)
Originally (a)/Consequences obliged a Phase-0.2 measurement of the conversion-op cost. Superseded: because precision is fixed per unit (CIM=INT8, GPU=FP16; see (c)), a precision-boundary dequant/requant is a deterministic memory-bound cast whose cost the existing M2/M4 per-op models already capture; Phase 2 inserts the cast op into the op stream and prices it analytically. No conversion-op measurement campaign is needed. This removes the prior "headline measurement gap" framing wherever it appeared.

## Revision (2026-06-14, Phase 2.2b — implemented)
(a) is realized: `simulator/runtime/precision.py::insert_conversions` inserts a `convert` OpNode on each value-flow edge that crosses the **GPU int8↔fp16 boundary** (the int8 KV-cache / CIM-matmul value feeding the FP16 GPU attention, and the FP16 attention output requantised for the next INT8 CIM matmul). Cost = a memory-bound cast (`out_elems × (read+write precision bytes)`) metered by the M3 memory pool; `Platform.price` returns compute 0 (no double-charge), no new parameter. Boundaries are **scheduler-declared by physical unit-pair** (the GPU island), so the AllCim baseline inserts **zero** conversions and its L4 is byte-identical. (b) holds: **mixed-precision OUTPUT QUALITY is NOT modeled (D3 limitation)** — `validation/report_mixed_precision.py` reports only the conversion COST + the modeled heterogeneous decode, labelled `simulated` (no concurrent-unit silicon).
