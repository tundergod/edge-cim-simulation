# Metis AIPU — ISSCC 2024 (11.3) — ingest notes

**Paper:** Hager et al., "Metis AIPU: A 12nm 15TOPS/W 209.6TOPS SoC for Cost- and Energy-Efficient Inference at the Edge," ISSCC 2024, Session 11.3, Axelera AI. PDF: `metis-aipu-isscc2024.pdf` (this folder).

**Why it matters here:** this is the silicon spec for the CIM unit we calibrate M1 against. Reading it surfaced **several errors in the Phase-1 M1 report** (see §Implications). These notes are the authoritative architecture reference; M1 must be reconciled to them.

## Architecture (the load-bearing facts)

- **Quad-core SoC.** 4 homogeneous AI-Cores. **52.4 TOPS/core → 209.6 TOPS compound.** RISC-V system controller, PCIe Gen3, LPDDR4x (**optional**), 32 MiB shared L2 SRAM, NoC.
- **Per AI-Core:** a D-IMC MVM engine + DPU (element-wise/activations) + DWPU (depthwise/pool/upsample) + **4 MiB L1 SRAM** + RISC-V control. On-chip SRAM total = 52 MiB (32 L2 + 4×4 L1 + 4×1 D-IMC).
- **MVM engine = a 512×512 INT-8/8/26 D-IMC crossbar PER CORE.** Segmented into **16 IMC banks**, each = **512 input × 32 output × 4 weight sets** (16×32 = 512 output). Bit-serial: each cycle processes 512 single-bit activations, accumulated over 8 cycles; 26-bit accumulators; the integer ALU **returns a vector of 64 INT32 per cycle** to the DPU.
- **Output granularity:** bank clock-gating granularity = **32 output channels**; the output is organized as **512 = 8 × 64** (Fig 11.3.4). → the "channel-64" behaviour we see has a real physical basis (64 outputs/cycle, 8×64 organization).
- **Precision = INT8 weights, INT8 activations, full-precision (26b) accumulation.** It is an **integer** engine → throughput is **OP/s (GOP/s / TOPS)**, *not* FLOP/s.
- **Cores are flexible:** can (a) jointly tackle one network for **higher throughput**, (b) work simultaneously on one network to **cut latency** (this is the multi-core single-instance / "Mode 1" mapping), or (c) run different networks independently (multi-instance).
- **Energy efficiency** = 15 TOPS/W @0.68V (random uniform); up to 82 TOPS/W under high sparsity. Peak 57.3 TOPS @0.7V/875MHz; nominal 54.2 TOPS @800MHz.
- **D-IMC (digital), not analog** → noise-immune, deterministic/repeatable MVMs (contrast NeuroSim's analog-RRAM model — ADR-0005's "model-form cross-check only" caveat is correct).

## Implications for our Phase-1 M1 model (corrections needed)

1. **The crossbar is 512×512 PER CORE, 4 cores — NOT a single "2048×2048 array".** Our empirical tiling boundary at **N≈2048 = 4 cores × 512 output channels**. So our fitted `T_tile`/2048 boundary describes the **4-core combined engine** under the default compile (most likely the multi-core single-instance "Mode 1" mapping), not one physical 2048-wide array. The report's "2048×2048 crossbar tile" wording is architecturally wrong and must be reframed as "4 × (512×512), effective 2048 output width."
2. **Core count of our measurements was never recorded.** `run_metis_cim.py` used the default `compile`/`axrunmodel` (no `--aipu-cores`); per voyager-sdk §3 the cores-spanned = the `<N>` in the `.../<N>/model.json` path, which we did not capture. The 2048=4×512 coincidence strongly implies all 4 cores, but this should be **confirmed on the board** (issue: the fit is per-default-compile, presumed 4-core).
3. **GFLOP/s → GOP/s.** INT8 integer ops; our "GFLOP/s" labels are wrong (issue #18). The peak is 209.6 TOPS = 209,600 GOP/s; our measured decode-GEMV ~204 GOP/s is ~0.1% of peak — decode is memory/latency-bound and never approaches the compute ceiling (issue #16: M1 only models the memory-bound regime).
4. **The "64" has a physical origin:** 64 INT32 outputs/cycle, 512=8×64 output organization, 32-channel gating. Explains the channel-64 staircase risers (was unexplained in the report — issue #10).
5. **Device "envelope ~6M params":** the on-chip weight memory (L1/L2/D-IMC SRAM, since Alpha has no LPDDR4x populated). Probed empirically (6.3M OK, 8.4M fail `zeMemAllocDevice`). NOT host memory.
6. **N=3072 staircase point is `tiled_extrapolated` (=2×T_tile), not native** — must not be plotted as "measured" (issues #11/#17).
