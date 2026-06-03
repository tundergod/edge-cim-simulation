# ADR-0005 — Energy model (M7) under no power telemetry

Status: Accepted (2026-06-03)

## Context
Neither board has usable power telemetry: Aetina Metis Alpha = vendor TDP estimates only; production Metis Card = PCIe Rev1, no telemetry; only M.2 Max (not owned) has INA236. So per-op (and even per-board) power cannot be measured directly.

## Decision
- **Spec-based per-component estimation** is the main model: **CIM** = vendor 15 TOPS/W × utilization, **cross-checked against the *form* of NeuroSim's CIM energy model** (NeuroSim validates to <1% chip-level error *after calibration* on a 40 nm **analog RRAM** macro; Metis is **digital SRAM-CIM**, so this is a model-form cross-check, *not* a silicon-grade source for our coefficient); **DRAM** = JEDEC pJ/bit × Ramulator-reported bytes; **PCIe** = spec pJ/bit × bytes; **NPU/GPU** = tech-node/datasheet; **CPU** = ARM datasheet × activity factor.
- **Opportunistic aggregate-power anchor (optional, not on the Phase 2 critical path).** When SSH to the boards is available, first try **option ① (free)**: read Intel RAPL on Machine 2's x86 host + check for `hwmon`/INA sensors. **Caveat:** RAPL reports only the x86 *host package* — not the Metis card (no telemetry, PCIe Rev1) nor the Aetina ARM SoC — so the anchor is a whole-host / wall-plug **aggregate** and cannot isolate CIM/NPU/GPU energy. If needed, defer to **② a wall-plug meter** (idle-vs-load delta against a fixed workload loop) or **③ a DC inline meter** (discuss later). Used only to sanity-check the spec-based **total**, not per-component.
- **Transparency.** Report energy with an explicit **uncertainty band** and **±20% coefficient sensitivity**; ensure qualitative conclusions ("which strategy is more energy-efficient") do not flip. State in limitations that energy is estimated, not measured.

The closest competitor (HPIM) reports no energy at all; CENT/PAPI use datasheet/analytical energy — so a transparent spec-based model with sensitivity is within (indeed ahead of) norms.

## Consequences
Energy is estimated, not measured (stated limitation). Conclusions must be robust to ±20%. The energy model is buildable in Phase 2 without any measurement; the anchor is a Phase-0.2/validation add-on.
