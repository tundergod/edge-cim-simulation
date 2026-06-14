# ADR-0003 — Scheduler / mapper (M6, the contribution)

Status: Accepted (2026-06-03)

## Context
M6 decides per-op unit, precision, placement, pipeline — this is the paper's contribution. Three sub-decisions: static vs dynamic; partition granularity; how the "characterization-driven split" is computed and validated.

## Decision
- **Static first, swappable.** Offline assignment (reproducible, easy to validate, easy to compare to silicon), behind a swappable `Scheduler` interface so a dynamic strategy can replace it later if the paper needs it.
- **op-level default + tensor-level for big matmuls.** Assign each op to one unit by default; enable **tensor-level (intra-op weight-centric split, à la HeteroInfer)** only for the big matmuls (QKV/O/FFN) where overlap pays off. Neither granularity affects simulation speed (event count is ~10²–10³/token; the runtime driver is Ramulator requests). **The exact op-vs-tensor list is finalized after Phase 0.1 op inventory.**
- **Validation-first.** M6 first implements SOTA strategies (HeteroInfer weight-centric, HPIM split, all-CIM baseline) as plugins. Goal at this stage: validate (1) simulator correctness and (2) that applying a SOTA strategy reproduces that SOTA's performance — both by configuring our platform to match the SOTA platform (external validation, do this for HeteroInfer — note this config exercises GPU+NPU+contention only, with CIM off; CIM paths are validated separately via L1/L4) and by applying the strategy on the Metis platform (self-consistency). The *novel* contribution scheduler comes after.

## Consequences
The `Scheduler` plugin interface must be expressive enough to encode SOTA strategies. External validation needs HeteroInfer's platform params (some estimated). op-vs-tensor finalization is a Phase-0.1-gated follow-up.

## Revision (2026-06-14, Phase 2.2b)
The `Scheduler(ABC).assign(dag, cfg) -> dag` interface is implemented (`simulator/runtime/scheduler.py`): a pure, idempotent op→unit + memory-domain annotator with a `SCHEDULERS` registry. Two plugins ship: **`AllCimScheduler`** (the L4-gated all-AIPU INT8 baseline; `pipeline=False`, single-accelerator serial) and **`CimHeteroScheduler`** (the project's own CIM-INT8 matmul × GPU-FP16 attention config; `pipeline=True`, SIMULATED — no concurrent-unit silicon, Aetina out). op-level placement is realized; **tensor-level weight-centric split is deferred** (post-Phase-2). The **"validation-first" SOTA reproduction (HeteroInfer/HPIM) is deferred to a later wave** (NOT cancelled) — 2.2a/2.2b prioritised the M6 ABC + the project's own CimHetero config + the conversion-op cost.
