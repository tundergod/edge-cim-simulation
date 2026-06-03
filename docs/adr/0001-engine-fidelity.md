# ADR-0001 — Engine fidelity: lightweight discrete-event + bandwidth contention

Status: Accepted (2026-06-03)

## Context
The timing engine (M3) needs a fidelity level. Options: (a) analytical critical-path sum, (b) lightweight discrete-event with resource + memory-bandwidth contention, (c) cycle-level. The CIM-centric thesis depends on modeling heterogeneous *overlap* (CIM ∥ GPU ∥ NPU) and the *decode memory wall* (shared-bandwidth saturation — e.g. HeteroInfer measured GPU+NPU≈60 GB/s on *its* mobile SoC while neither alone saturates that platform's 68 GB/s peak; our own saturation point is calibrated from Aetina, ADR-0002). (a) cannot represent either, making the M6 contribution invisible. (c) is rejected as the wrong *altitude* for a system-orchestration contribution (see below), not merely a tooling choice.

## Decision
**(b)** — a lightweight, **non-cycle-accurate** discrete-event engine: compute units run concurrently; **shared memory bandwidth is a contended resource**; per-op latencies come from silicon-calibrated fitted equations.

Verified acceptable at top arch/sys venues (websearch + subagent): LLMCompass (ISCA'24, analytical, 4.1% e2e / 10.9% per-op vs A100), GenZ (5.82% geomean), LLMServingSim (IISWC'24, ≈14.7% avg). Cycle accuracy is demanded only when the *contribution itself* is a memory/command-scheduling mechanism (IANUS, Duplex) — ours is system-level orchestration, so a contention model is the right altitude. Silicon calibration makes this *more* credible than several cycle-accurate-but-uncalibrated competitors (HPIM, CENT, PAPI report no silicon validation).

## Consequences
Incurs validation obligations (see ADR-0006): per-unit fit error as median+p95+max; end-to-end validation; validate the contention model specifically (reproduce the ~60 GB/s knee); ±20% sensitivity; ablations (concurrency-off, contention-off); name Ramulator2/DRAMsim3 in related work.
