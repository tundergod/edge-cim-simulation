---
type: source
title: "The Configuration Wall: Characterization and Elimination of Accelerator Configuration Overhead"
created: 2026-06-05
updated: 2026-06-05
raw_path: raw/papers/configuration-wall-asplos2026.pdf
source_kind: paper
ingest_level: full
authors: [Josse Van Delm, Anton Lydike, Joren Dumoulin, Jonas Crols, Xiaoling Yi, Ryan Antonio, Jackson Woodruff, Tobias Grosser, Marian Verhelst]
venue: ASPLOS
year: 2026
arxiv: 2511.10397
doi: 10.1145/3760250.3762225
tags: [accelerator-configuration-overhead, configuration-roofline, per-call-overhead, host-control-overhead, compiler-optimization, mlir, gemmini, opengemm, round-trip-tax]
---

# The Configuration Wall: Characterization and Elimination of Accelerator Configuration Overhead

## TL;DR

As accelerators get faster, the *fixed per-call cost the host pays to set them up* — writing config registers, packing parameter bits, launching, synchronizing — stops being negligible and becomes the binding constraint. The paper names this the **configuration wall** and formalizes it with a **configuration roofline** (new axes: *operation-to-configuration intensity* `I_OC` in ops/byte-of-config, and *configuration bandwidth* `BW_Config` in config-bytes/time). The model predicts that **faster accelerators are hit harder** (more peak ops stranded behind the same fixed setup). On Gemmini only **26.78 %** of peak is attainable un-optimized; two compiler passes — **configuration deduplication** (skip re-writing unchanged config registers across calls) and **configuration–computation overlap** (software-pipeline setup behind the prior call's execution) — recover up to **2.71×** (OpenGeMM) / **+11 %** geomean (Gemmini). Implemented as an MLIR dialect (`accfg`: `setup`/`launch`/`await`). **Explicit scope caveat: the paper does *not* count DMA / data movement as "configuration"** — it isolates the control/setup half of per-call overhead.

## Key claims

- **Configuration wall exists and grows with accelerator speed** (§1, §3): the host's setup/control/sync time per offload "is not spent productively by either host or accelerator"; for fast accelerators the system becomes *configuration-bound*, not compute-bound.
- **Configuration roofline** (§3): two new first-class quantities —
  - `I_OC` = accelerator ops executed per byte of configuration (ops/byte);
  - `BW_Config` = config bytes the host can set per unit time;
  - **concurrent** setup: `P_attainable = min(P_peak, BW_Config · I_OC)`;
  - **sequential** setup: `P_attainable = 1 / (1/P_peak + 1/(BW_Config · I_OC))`;
  - **effective** config BW folds in runtime parameter *calculation*: `BW_Config,Eff = N_config_bytes / (T_calc + T_set)`.
  - A *knee* separates configuration-bound from compute-bound regions, exactly mirroring the classic compute/memory roofline knee.
- **Quantified strand** (§5): Gemmini (16×16 systolic, RISC-V Rocket, RoCC custom instrs) — peak 512 ops/cycle, eff. config BW ≈ 0.913 byte/cycle, `I_OC` ≈ 205 ops/byte → only **26.78 %** of peak attainable un-optimized. OpenGeMM (Snitch-core concurrent-config matrix unit) — peak 1024 ops/cycle.
- **Two compiler optimizations** (§4): (a) **configuration deduplication** — SSA value-tracking proves a config register already holds the needed value across invocations and elides the rewrite; (b) **configuration–computation overlap** — for concurrent-config HW, hoist `setup` before the prior `await` and software-pipeline inside loops.
- **Results** (§6): OpenGeMM **2× geomean, up to 2.71×**; Gemmini **+11 % geomean**; optimizations move workloads from the configuration-bound region into the compute-bound region.
- **Scope boundary (critical for us)** (§3): "we do not consider such data movement part of the configuration" — configuration = register writes / parameter setup / bit-packing, **excluding DMA transfers and descriptor programming**.

## Motivation

Per-Watt performance pushes compute off the CPU onto ever-more-complex accelerators, but complexity multiplies the number of configuration knobs the host must set per call. Each added option increases the accelerator's usefulness yet *directly* lowers achievable performance unless the setup is optimized away. The faster the datapath, the larger the fraction of its peak that a fixed setup cost strands — an overlooked, accelerator-class-wide tax that the roofline makes visible and the compiler can attack.

## Method

- **Formal model**: derive the configuration roofline (above) for both *sequential* (setup serializes with compute, e.g. Gemmini RoCC) and *concurrent* (setup overlaps compute, e.g. OpenGeMM) configuration disciplines.
- **`accfg` MLIR dialect** makes configuration semantics explicit instead of opaque `volatile` asm:
  - `accfg.setup` → writes config, yields a state SSA value;
  - `accfg.launch` → consumes state, starts the accelerator, yields a token;
  - `accfg.await` → blocks on the token.
  - Explicit state/token edges are what let the compiler prove dedup-safety and legal overlap.
- **Evaluation platforms** (tightly-coupled host+accelerator, *not* PCIe/discrete): Gemmini in the Spike simulator; OpenGeMM in a Verilator cycle-accurate model.

## Results

- OpenGeMM: **2× geomean speedup (up to 2.71×)** from moving configuration-bound GEMMs into the compute-bound region.
- Gemmini: **+11 % geomean**; the headline analytical result is that the un-optimized ceiling is **26.78 %** of peak — i.e. ~3/4 of the systolic array is stranded by sequential RoCC configuration before optimization.
- The roofline correctly predicts which matrix sizes are configuration-bound vs compute-bound and the size at which the knee is crossed.

## Contributions

1. Names and formalizes the **configuration wall**; gives the first **configuration roofline** with closed-form attainable-performance for sequential and concurrent setup disciplines.
2. **`accfg` MLIR dialect** that exposes configuration as explicit dataflow (state/token), enabling provably-correct compiler optimization.
3. Two retargetable passes — **configuration deduplication** and **configuration–computation overlap** — with measured 2×-class gains on a concurrent-config accelerator.

## Limitations / open questions

- **Excludes data movement / DMA by construction** — models only the control/setup half of per-call overhead; a discrete PCIe accelerator's per-call floor is dominated by the *other* (data-transfer) half.
- **Tightly-coupled substrates only** (RoCC custom instr / Snitch core): per-call setup is single-to-tens of cycles, ~10³× smaller than a PCIe round trip — the *magnitudes* do not transfer, only the *modeling form*.
- Two accelerators, two simulators; no measured silicon and no discrete/offload (PCIe, MMIO) configuration path.
- Dedup relies on static SSA equivalence; dynamic / data-dependent config values limit its reach.

## D1–D9 review lens

| # | Dimension | Reading |
|---|---|---|
| D1 | Baselines | Un-optimized config-bound execution vs the two passes, on each accelerator. |
| D2 | Novelty | Configuration roofline as a third roofline axis (beyond compute/memory) is the clear delta. |
| D3 | Evaluation | Two accelerators in two simulators — narrow; no PCIe/discrete, no silicon. |
| D4 | Platform | Spike + Verilator cycle-accurate; credible for tightly-coupled, silent on discrete. |
| D5 | Motivation | Strong — "faster accelerator, bigger stranded fraction" is a real, general trend. |
| D6 | Mechanism cost | Compiler passes; runtime cost is the saved config writes, quantified. |
| D7 | Venue | ASPLOS-natural (architecture × compiler co-design). |
| D8 | Consistency | Coherent; scope boundary (excludes DMA) stated up front. |
| D9 | Significance | High for tightly-coupled accelerators; for us, a transferable *framing + model*, not a drop-in. |

## Connections

**Phase 2 observations (relevance to this simulator)**

- **Directly relevant as a framing + modeling formalism for our round-trip-tax thesis — with one essential caveat.** Our measured per-call host↔device floor on the Aetina Metis Alpha (~911 µs / p95 1112 µs on the matmul proxy; ~5–8 ms on CNN, [[metis-step1-cnn-characterization-2026-05-23]]) makes the *exact* point this paper generalizes: a 214-TOPS AIPU **starved by fixed per-call overhead**, with faster compute stranding *more* of itself behind the same floor. The configuration roofline is the principled version of our "round-trip tax."
- **Caveat (do not over-claim):** this paper *explicitly excludes DMA / data movement* from "configuration" — it models the control/setup half. Our Alpha floor is dominated by the *data-transfer* half (PCIe Gen3 ×4 ~3.5 GB/s + `dma_poll=1` completion polling). So the configuration roofline models **one stage** of our per-call floor; the A3 "per-call DMA stage breakdown" sub-experiment is precisely what separates config-overhead from data-transfer, and only the config-overhead stage maps onto this model. The two rooflines are *complementary*, not interchangeable.
- **What it drives:**
  - **M2 / M3** — adopt the *form* `T_call = max/Σ(compute, data-move, setup) + fixed_overhead`; the sequential-config equation `1/P = 1/P_peak + 1/(BW·I_OC)` is the clean way to fold a fixed per-call setup term into the timing model and is a candidate fit form in Phase 1.
  - **Roofline validation (L1/L3/L6)** — a *configuration/round-trip roofline* alongside our compute and memory rooflines gives a third 2-D consistency check; the knee location is a falsifiable prediction.
  - **M6 scheduler levers** — our empirically-winning mitigations are this paper's optimizations under different names: Mode-3 "one DMA carrying 4 frames" = amortizing fixed overhead = the *overlap/batching* lever; weight-stationary reuse / not re-streaming = *configuration deduplication*; double-buffering hiding streaming behind compute = *configuration–computation overlap*.
- **Positioning value:** lets us cite a 2026 ASPLOS paper that independently formalizes "fast accelerators are configuration/per-call-overhead bound," strengthening the round-trip-tax narrative beyond our own silicon.

**Assessed-but-not-ingested siblings (2026-06-05 DMA-bottleneck triage)** — same theme, but substrate-mismatched against our PCIe-host↔CIM-DMA bottleneck and with no module to drive (same bar that removed cent / cxl-pnm):
- **QuCo (HPCA'26, U. Murcia + W&M + NVIDIA)** — a hardware "Queue Configurator" that auto-configures *Automatic Tile Transfer (ATT)* descriptors on NVIDIA GPU **TMA** (Hopper Tensor Memory Accelerator). The GPU-DMA analogue of this paper, but on the NVIDIA-GPU substrate that is explicitly *future work* for us (Mali is our SoC GPU; NVIDIA = the deferred Accel-Sim `gpu_backend` plug-in). No v1 module to drive.
- **COMET (HPCA'26, NUDT + PKU)** — communication + memory co-design for fine-grained AI inference across **multi-chiplet-module (MCM)** accelerators (on-package chiplet interconnect / NoP). Our topology is host-SoC + a single CIM accelerator on PCIe, not a multi-chiplet accelerator — the comms model does not map.
- **Fastmove (FAST'23, USTC + SmartX)** — revitalizes the **on-chip DMA (Intel I/OAT-class)** to move data DRAM↔NVM in storage systems, with CPU+DMA load-splitting and small-transfer-overhead study. Adjacent theme (per-transfer DMA overhead, CPU/DMA coordination) but a different DMA domain (storage, not accelerator offload); no transferable model for our PCIe round trip.

**Concepts / entities / projects / ideas**

- [[metis-step1-cnn-characterization-2026-05-23]] — our first-hand round-trip-tax evidence (7.9× same-HW gap; per-call DMA floor; Mode-3 amortization); this paper is its formal generalization.
- [[metis-llm-investigation-desktop-2026-05-19]] · [[system-axelera-metis-card]] — the LLM-decode weight-streaming wall (24.23 GB/s); the *memory* roofline that the configuration roofline sits beside.
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — HeteroInfer; per-unit characterization methodology this complements with a per-call-overhead axis.
- [[papi-asplos2025]] — arithmetic-intensity-threshold mapping for M6; the configuration roofline adds a second threshold (when per-call overhead, not AI, decides the unit).
- [[hpim-arxiv2025]] — competitor; its intra-token pipeline is a configuration–computation-overlap instance.
- **[[cim-centric-llm-mobile-soc]]** — project spec; this note feeds M2/M3 timing-model form, the M6 dedup/overlap levers, and a third (configuration/round-trip) roofline for L1/L3/L6 validation.
