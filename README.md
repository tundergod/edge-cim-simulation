# edge-cim-simulation

A **real-silicon-calibrated simulator** of LLM inference on a **CIM-enabled heterogeneous mobile SoC** — Compute-in-Memory (CIM) as a peer compute unit alongside NPU + GPU + CPU over unified memory. Calibrated against real Axelera Metis AIPU silicon.

## What this is

No silicon exists for "discrete CIM on a heterogeneous mobile SoC running LLM," and the real Metis cards can't be the study object directly (Alpha can't compute LLM; the production card's LLM path is closed/precompiled-only). So we **simulate** that SoC and **calibrate** it against measurements from two real Metis boards. The research surface is **CIM-centric mixed-precision scheduling** (which op runs on which unit at which precision), decided by measured per-unit characteristic curves rather than pre-committed.

## Repo map

| Path | What |
| --- | --- |
| [overall.md](overall.md) | **The project brief** — goal, problem, position, Phase 0 characterization plan, simulator architecture (6 boxes / M1–M7), risks, scope. *Preliminary — revise as needed.* |
| [voyager-sdk.md](voyager-sdk.md) | **SDK characterization reference for all agents** — how to extract every measurement the simulator needs from the Voyager SDK / Metis silicon. Tagged `[DOC]`/`[FORUM]`/`[MEASURED]`/`[GAP]`. |
| [papers/](papers/) | Literature notes + real-silicon investigation reports. See [papers/README.md](papers/README.md). |

## Where to start

1. Read [overall.md](overall.md) for the goal and plan.
2. Read [voyager-sdk.md](voyager-sdk.md) before designing any measurement.
3. Skim [papers/metis-silicon/](papers/metis-silicon/) for the real-silicon ground truth (the calibration anchors L4 + L6).

## Status

Bootstrap phase. The literature corpus, SDK reference, and project brief are in place. Next: Phase 0 real-board characterization (see [overall.md](overall.md) § Phase 0) → then the M1–M7 simulator build. The `simulator/`, `measurements/`, `characterization/`, `validation/`, `tools/`, `docs/` directories described in `overall.md` are the planned layout, created as work begins.

## Key external references

- Voyager SDK: <https://github.com/axelera-ai-hub/voyager-sdk>
- Axelera community forum (Metis M.2): <https://community.axelera.ai/metis-m-2-3>
