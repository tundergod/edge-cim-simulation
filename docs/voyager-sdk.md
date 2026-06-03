# Voyager SDK — Hardware Characterization Reference

**Audience: every agent working on this CIM simulator.** Read this before designing any measurement, characterization, or calibration step against Metis silicon.

This doc answers one question: **what can the Voyager SDK tell us about the Metis AIPU, and how do we extract the numbers the simulator needs to be calibrated against?** It merges (a) the official Voyager SDK docs (GitHub `axelera-ai-hub/voyager-sdk`), (b) the Axelera community forum (`community.axelera.ai`), and (c) this team's own on-silicon measurements on the two real boards.

**Provenance tags used throughout:**
- `[DOC]` — stated in the official SDK documentation
- `[FORUM]` — from a community.axelera.ai thread (staff answer unless noted "community")
- `[MEASURED]` — empirically verified by us on real silicon (Aetina Metis Alpha or the production Metis Card)
- `[GAP]` — the docs/forum are **silent**; the simulator must either measure it empirically or we must ask Axelera. Gaps are first-class: they define the measurement campaign.

> Two SDK versions are in play. **Aetina RKC-A02 (Metis Alpha)** runs **Voyager v1.3.1**; the **production Metis Card** runs **v1.6**. Op coverage, knob names, and tooling differ between them — always note which version produced an artifact or number. The GitHub default branch is `latest` (not `main`); release branches `release/v1.4…v1.6` exist.

---

## 0. The two target boards (what we are characterizing)

| | **Aetina RKC-A02** (heterogeneous SoC) | **Production Metis Card** (CIM-runs-LLM anchor) |
|---|---|---|
| Role in project | Phase 0 Machine 1 — characterize 4 compute units + PCIe boundary | Phase 0 Machine 2 — real CIM running LLM (L4 anchor) |
| Host | Rockchip RK3588 (4×A76 + 4×A55, Mali-G610 MP4, RKNPU2 6 TOPS), 16 GB LPDDR4 | i9-class x86, Ubuntu 24.04, RTX 3090 as GPU reference |
| Metis silicon | **Alpha** M.2 — PCIe Gen3 ×4, **no on-card DRAM**, 32 MB L2 SRAM only, 1 GB PCIe-IOMMU window into host LPDDR4 | **Production** quad-core AIPU, on-card LPDDR4x (holds multi-GB LLM weights) |
| Voyager SDK | **v1.3.1** | **v1.6** (`axelera-llm==1.6.0`) |
| LLM compute | **Impossible** — `-1301` closed-firmware wall + no on-card DDR. Vision only. | **Works** (precompiled-only); batch-1 decode is a ~24 GB/s on-card-DDR memory wall |
| Use for simulator | CIM tile timing, PCIe/DMA, RKNPU2/Mali/CPU op micro-benchmarks | End-to-end INT8 LLM tok/s (L4); bridging-assumption anchor |

Full board notes: [papers/platforms/system-aetina-rkc-a02.md](papers/platforms/system-aetina-rkc-a02.md) and [papers/platforms/system-axelera-metis-card.md](papers/platforms/system-axelera-metis-card.md). Detailed silicon investigations: [papers/metis-silicon/](papers/metis-silicon/).

---

## 1. Metis AIPU architecture — the constants the simulator hard-codes

Pulled from the forum technical deepdive `product-updates/the-metis-ai-platform-a-technical-deepdive-125` `[FORUM]`, the vendor spec page, and our `axdevice` readouts `[MEASURED]`:

| Quantity | Value | Source |
|---|---|---|
| AI cores | **4** homogeneous, proprietary RISC-V + Digital In-Memory Compute (D-IMC) | `[FORUM]`/`[MEASURED]` |
| Per-core throughput | ~53.5 TOPS INT8 → **~214 TOPS INT8** total | `[FORUM]`/vendor |
| Native datatype | **INT8** weights+activations, **INT32** accumulate | `[FORUM]` |
| Efficiency | **15 TOPS/W INT8** (silicon-only marketing); **~270 FPS/W end-to-end** measured on ResNet-50 (vs 614 FPS/W silicon-only) | vendor / `[FORUM]` thread 1355 |
| L1 SPM | **4 MiB per core**, software-managed scratchpad (NOT a cache) | `[FORUM]`/`[MEASURED]` |
| In-core MVM memory | ~1 MiB per core | `[FORUM]` |
| L2 SRAM | **32 MiB shared** (4×8 MB banks), RISC-V-firmware-managed | `[FORUM]`/`[MEASURED]` |
| On-chip total | >52 MiB | `[FORUM]` |
| NoC bandwidth | >1 Tb/s | `[FORUM]` |
| Clock | 800 MHz default (DVFS 20–800 MHz, see §6/§8) | `[MEASURED]` |
| Channel tiling granularity | **64** (inferred from op constraints — `Pad`/`Slice` require channel multiple of 64) | `[DOC]` inferred |
| Control core | application-class RISC-V running a **closed RTOS** — not user-modifiable | `[FORUM]` |

**Memory-tier sizes for the *production* card's on-card LPDDR are NOT documented** `[GAP]` — board SKUs ship 1 GB / 4 GB / 16 GB (M.2 Max) / up to 64 GB (4-chip PCIe). Measured effective decode bandwidth on the production card is **~24.2 GB/s** `[MEASURED]` (see §9).

> **L1/L2 are software-/compiler-managed, not caches** `[MEASURED]`. There is **no runtime evict/prefetch/realloc API**; placement is fixed at compile time via `dpu_constants_home`/`l2_constraint` (§6). The simulator's memory model (box ④) should treat L1/L2 as statically-allocated scratchpads, not demand-paged caches.

---

## 2. Supported operations — what runs on the AIPU vs the host

Authoritative op matrix: `docs/reference/onnx-opset17-support.md` (v1.6) `[DOC]`. Targets **ONNX opset 13–17**. ~31 ops, each **Supported** (all configs) or **Constrained** (conditional). On v1.3.1 (Aetina) the supported set is **opset-14**, ~31 vision ops `[FORUM]`/`[MEASURED]`.

**Fully supported:** `BatchNormalization`, `GlobalAveragePool`, `GlobalMaxPool`, `HardSwish`, `LeakyRelu`, `Relu`, `Sigmoid`, `Tanh`.

**Constrained — the ones that matter for the op cost model:**
- `Conv` — explicit padding only; symmetric kernel/stride/dilation; grouped conv needs `kernel² < 128`. Depthwise conv **is hardware-supported** (a block-sparse-diagonal unit) `[FORUM]` — contrary to "depthwise is a problem op". (But RKNPU2 *does* collapse on depthwise+Swish — see §10.)
- `MatMul`, `Softmax`, `Transpose[0,1,3,2]`, `Reshape` — **"YOLO11 attention blocks only"** `[DOC]`. **This is the critical transformer limitation:** general MatMul / attention is NOT a first-class supported op via the public compiler — it is whitelisted only for specific YOLO11 patterns.
- `Add`/`Mul`/`Sub` — same-shape or broadcast against `[1,C,1,1]`/scalar only; `Sub` can't do same-shape; `Mul` operands can't be the identical node.
- `Pad` — `constant` mode only, channel padding **multiple of 64**. `Slice` — single axis; non-channel slices need channel multiple of 64. `Concat` — 4D only. `Gemm` — `transA=0` only.
- `Clip` — only ReLU6 and HardTanh. `Resize` — nearest/linear.

**Native layout is 4D NCHW feature maps** `[DOC]`. Ops outside the matrix do **not** silently fall back to host at runtime — they cause **compile-time rejection** `[FORUM]`. The `deploy.py` flow auto-splits the graph into `preamble.onnx` (host preprocess) + `model.axm` (AIPU) + `postamble.onnx` (host postprocess) `[DOC]`; anything pushed into pre/postamble is a **host CPU cost**, not an AIPU cost — the simulator must account for it on the CPU unit.

**Not supported as custom models** `[FORUM]`: Gemma3, Whisper. **No LayerNorm/RMSNorm, RoPE, generic Attention, Gather/Embedding, GELU, or dynamic shapes in the public compiler** `[MEASURED]` — this is exactly why LLMs can't be self-compiled (§9).

---

## 3. Quantization & precision

- **INT8 is the execution precision** `[DOC]`/`[FORUM]`. `compiler.quantize()` does INT8 PTQ with calibration data.
- Quantization schemes (`quantization_scheme`, default `per_tensor_histogram`): `per_tensor_histogram` / `per_tensor_min_max` / `hybrid_per_tensor_per_channel` `[DOC]`.
- **AxMO** (Axelera Model Optimizer): separate FP32→INT8 PTQ tool, **ViT-only** support, only knob `smooth_quant_alpha` (default 0.5) `[DOC]`.
- **INT4: `[GAP]`** — not documented anywhere (op matrix, AxMO, compiler configs all silent); no community confirmation of INT4 on current silicon `[FORUM]`. LLM zoo models are tagged `*-static` with **no stated bit-width** — likely INT8, possibly mixed. **For the simulator: model INT8 as native; treat INT4 as out-of-scope unless a vendor artifact appears** (matches `overall.md` Out-of-scope). Our int4 decode projection (~2.04×) is a *linear extrapolation, not measured* `[MEASURED]`.

---

## 4. Measurement & profiling toolchain — the core of this doc

This is the surface the Phase 0 agents drive to produce ground-truth files. Tools ranked by usefulness for the simulator:

### 4.1 `axrunmodel` — cleanest raw-device micro-benchmark `[DOC]`
Runs a deployed `model.json` on synthetic/repeated input (no decode/pre/post). **Best tool for isolated CIM/op characterization.** Ref: `docs/reference/axrunmodel.md`.
- `-d/--devices 0,1`; `--seconds N` (default 10); `--aipu-cores 1-4`; `--throttle-fps N`.
- `--double-buffer / --no-double-buffer` (default **on**).
- `--input-dmabuf / --no-input-dmabuf`, `--output-dmabuf / --no-output-dmabuf` (default **on**) — **toggle these to isolate the PCIe/DMA transfer contribution** (L2 measurement).
- `--show-bar-chart`, `--show-histogram` — FPS over time / distribution (variance characterization).
- Outputs three throughput numbers:
  - **Device FPS** = 1/device-execution-time, **includes data transfers**
  - **Host FPS**
  - **System FPS** = total frames / wall-clock (the headline)
- Multicore: the batch number in the model path (`.../<N>/model.json`) = cores a single instance spans; `--aipu-cores` caps cores and it spins up enough instances to fill them.

> **Critical DMA-opts gotcha** `[MEASURED]`: a direct AxRuntime app defaults to `double_buffer=False` + dmabuf off; `axrunmodel` defaults them **on**. The difference is **30–90% throughput**. A prior measurement of "1147 FPS mode-2" was actually **2181 FPS** with proper opts — always enable `double_buffer=True, input_dmabuf=True, output_dmabuf=True` for apples-to-apples numbers.

### 4.2 `inference.py` — end-to-end pipeline measurement `[DOC]`
Ref: `docs/reference/inference.md`, `docs/tutorials/benchmarking.md`. Drives the full GStreamer pipeline (source→pre→AIPU→post→app).
- `--pipe {gst, torch, torch-aipu}`: `gst` = full end-to-end (perf path); `torch` = CPU FP32 accuracy reference; `torch-aipu` = host preproc + AIPU (isolates quantization accuracy loss).
- `--aipu-cores 1-4`, `--frames N` (0=all).
- `--show-host-fps` / `--show-system-fps` — dispatch-point vs end-to-end throughput.
- `--show-stream-timing` — **per-frame latency + jitter** (use for variance/CoV).
- `--show-stats` — **per-pipeline-element timing breakdown** (element name, µs, effective FPS per element). This is how we found the Aetina colorconvert bottleneck. *Note:* this is pipeline-stage granularity, **not** per-NN-op `[FORUM]`.
- `--show-cpu-usage` — host CPU %.
- Accuracy run: `./inference.py <model> <dataset> --no-display` → mAP / Top-1.

Example (Aetina, in the detached SDK container):
```bash
docker exec axelera-sdk bash -c 'cd /home/ubuntu/voyager-sdk && source venv/bin/activate && \
  python3 inference.py efficientnet_b0-imagenet fakevideo --no-display --show-stats'
```

### 4.3 `axmonitor` — live telemetry dashboard `[DOC]`/`[FORUM]`
The telemetry surface the simulator's energy/utilization model should mirror. Shows:
- **Per-core utilization %**, per-core temperature (Sys-core + AI-core 0–3), clock frequencies.
- **DDR usage & bandwidth** with a plot (new in v1.6) — the **only documented DDR-bandwidth readout**; use it to classify memory- vs compute-bound ops.
- **PCIe DMA info**.
- **Power** min/max/avg over a 1-s window — **M.2 Max only** (see §8).

### 4.4 `AxInferenceNet` C++ API — the richest per-op timing hook `[DOC]`
Ref: `docs/reference/axinferencenet.md`. The async inference engine exposes a **`LatencyCallback`**:
```cpp
void(const std::string& opname, uint64_t throughput_ns, uint64_t latency_ns)
```
**Per-operator latency and throughput in nanoseconds** — the best-documented path to per-op AIPU timing for calibrating the simulator's op cost model. `properties_from_string()` parses the `.axnet` config files `inference.py` emits, so you can dump-and-replay a configured pipeline. `num_children` (0–4) sets model instances per device (throughput↔latency tradeoff).

### 4.5 `axtrace` / `device_profiling` — lower-level, undocumented `[GAP]`
- The Python tracers (`axelera/app/inf_tracers.py`) shell out to an **`axtrace`** subprocess that reads **megakernel cycle counts**; `AipuTracer` logs **per-core latencies in DEBUG mode**. `axtrace` itself is **not in the public reference docs** — discover its CLI on the box (`axtrace --help`) or ask on the forum. This is the closest thing to per-op/per-core cycle data.
- Both runtimes accept `device_profiling` (0/1) and `host_profiling` (0/1) load properties — **but their output content/format is undocumented** `[GAP]`.

### 4.6 Runtime-API timing — you must time it yourself `[DOC]`
**No latency/timing metric is queryable through the documented runtime API.** `instance.run()` returns `None` (Python) / only `axrResult` (C). For the AxRuntime harness (Phase 1b pattern), wrap `instance.run()` in `clock_gettime` / wall-clock yourself, or fall back to `axrunmodel` / `AxInferenceNet::LatencyCallback`. `instance.run()` is **opaque** — it bundles host→device DMA + AIPU compute + device→host DMA into one synchronous call; sub-stage timing is not exposed at this level `[MEASURED]`. Per-call latency on the Aetina board (polling-mode DMA) has a few-ms fixed floor irrespective of model compute.

### 4.7 LLM measurement — `axllm --show-stats` `[DOC]`
`axllm <model> --show-stats` prints **tokenization time, prefill time, time-to-first-token (TTFT), generation tokens/sec, and hardware metrics** — the LLM measurement hook for the L4 anchor (§9).

---

## 5. Runtime APIs (for custom measurement harnesses)

### 5.1 Python `axelera.runtime` `[DOC]`
`docs/reference/axelera.runtime.md`. Hierarchy `Object → {Context, Connection, Model, ModelInstance}`. Canonical flow:
```python
import axelera.runtime as axr
with axr.Context() as ctx:
    model = ctx.load_model(path)                              # -> Model
    conn  = ctx.device_connect(device, num_sub_devices=1)
    inst  = conn.load_model_instance(model, aipu_cores=0, num_sub_devices=1)
    inst.run(inputs, outputs)                                 # -> None  (time it yourself)
```
- `Context.list_devices()`, `configure_device(dev, **kwargs)`, `device_ready(dev)`, `read_device_configuration(dev) -> dict`.
- `Model.inputs()/outputs() -> list[TensorInfo]` (shape, dtype, **quant scale/zero_point**, padding).
- Multi-core/instance: `aipu_cores`, `num_sub_devices` kwargs; per-core config via `configure_device(... mvm_utilisation_core_0..3 ...)`.
- Profiling kwargs: `device_profiling`, `host_profiling` (default 0; output format undocumented).

The canonical example is `examples/axruntime/axruntime_example.py`. Phase 1b harness skeleton (fill all 4 cores):
```python
batch  = in_infos[0].shape[0]
n_inst = max(1, 4 // batch)            # cap so n_inst*batch <= 4 cores
conns  = [ctx.device_connect(None, batch) for _ in range(n_inst)]
insts  = [c.load_model_instance(model, num_sub_devices=batch, aipu_cores=batch) for c in conns]
# warmup, then time instance.run(inputs[i], outputs[i]) across threads
```

### 5.2 C/C++ `axruntime` `[DOC]`
Mirrors Python: `axr_create_context`, `axr_load_model`, `axr_device_connect`, `axr_load_model_instance`, `axr_run_model_instance`, `axr_read_device_configuration`, `axr_get_model_properties`.
- **`axrArgument`** carries `void* ptr`, **`int fd` (DMA-buf fd; 0/-1 ⇒ host memory)**, `offset` (must be 0), `size` — the zero-copy hook.
- **`axrTensorInfo`**: `dims[]`, `bits`, `type`, `name`, `padding[][2]`, `double scale`, `int zero_point`.
- **`axrDeviceInfo`**: `name`, **`subdevice_count` (4 for Metis)**, `board_type`, `firmware_version`. Board types include `AXR_BOARD_METIS_OMEGA_PCIE`, `AXR_BOARD_METIS_OMEGA_M2`.
- Device-config keys: `clock_profile:int` (MHz), `aipu_cores`, `num_sub_devices`, `input_dmabuf`/`output_dmabuf`, `double_buffer`.

---

## 6. Compiler config — the sweep surface

Two paths: high-level `deploy.py` (YAML in → deployed model + pipeline) and low-level `compile`/`axcompile` (ONNX in → artifact). Most characterization uses `deploy.py`. The knob bible is `docs/reference/compiler_configs_full.md` (~100 fields); short curated version `compiler_configs.md`.

**All compile-time overrides go inside `extra_kwargs.compilation_config:` in the model YAML** (placing them directly under `models.<name>` is rejected) `[MEASURED]`.

### 6.1 Knobs the simulator should sweep
| Knob | Default | What it controls / why sweep |
|---|---|---|
| `aipu_cores` (`aipu_cores_used`) | 1 | cores compiled-for (1–4) — throughput scaling |
| `multicore_mode` | `multiprocess` | `multiprocess`/`multithread`/`batch`/`cooperative`/`pipeline` — see §6.2 |
| `resources` (`resources_used`) | 1.0 | mem fraction; coupling rule `resources ≥ aipu_cores/aipu_cores_max` |
| `dpu_constants_home` | `global.l2` | **weights in L2 vs DDR** — the key L2-vs-spill latency/energy sweep |
| `elf_in_ddr` | true | ELF in DDR (frees L2) vs L2 |
| `io/constant/workspace_memory_pool` | `global.ddr` | initial buffer tier (promoted upward during opt) |
| `l1/l2/ddr_constraint` | null | per-tier byte caps → force tiling |
| `tiling_depth` | 1 | depth-first fusion depth (`6` recommended when on) |
| `pipeline_spatial_tiles` / `pipeline_channel_tiles` | true | SW pipeline over H-tiles / out-channel-tiles |
| `quantization_scheme` (`ptq_scheme`) | `per_tensor_histogram` | PTQ strategy |
| `frequency` | 800 MHz | **DVFS** clock (20 MHz–800 MHz) — power/perf curve |
| `mvm_utilization_limit` | 1.0 | MVM array usage ceiling (0.125–1.0) — **directly impacts power** |
| `enable_icr` / `enable_swicr` | true | in-core weight replication for IMC utilization |

`deploy.py` flags: `--mode {QUANTIZE, QUANTCOMPILE, PREQUANTIZED(default)}`, `--pipe {gst,torch,torch-aipu}`, `--aipu-cores 1-4` (**runtime-side, NOT baked into the artifact** `[MEASURED]`), `--num-cal-images` (default 200), `--metis {auto,none,pcie,m2}`, `--models-only`, `--export`.

### 6.2 `multicore_mode` semantics (verified) `[MEASURED]`/`[FORUM]`
| Mode | aipu_cores=1 | aipu_cores=4 (+`resources:1.0`) | Meaning |
|---|---|---|---|
| `batch` | OK | **✅ batched `[4,…]` input** — all 4 cores cooperate on one inference with the shared mem budget | the only working compile-time multi-core mode in v1.3.1 |
| `multiprocess` | OK (default) | ❌ "requires exactly 1 core" | single-core compile; host replicates via OS processes |
| `multithread` | OK | ❌ "requires exactly 1 core" | same, threads |
| `cooperative` | ❌/model-dependent | ❌ "Unsupported" | **declared but NOT implemented** in v1.3.1 |
| `pipeline` | ❌ "Unsupported" | ❌ | **declared but NOT implemented** in v1.3.1 |

**Consequence:** in v1.3.1, **no intra-frame parallelism is exposed** — `cooperative`/`pipeline` are walled off at validation. Matches `overall.md` Out-of-scope. The working 4-core trio is exactly `{aipu_cores: 4, resources: 1.0, multicore_mode: batch}`:
```yaml
extra_kwargs:
  compilation_config:
    aipu_cores: 4
    resources: 1.0
    multicore_mode: batch
```
All 5 Phase-0 CNN models batch cleanly via this path (it forces the `singlecore_to_batched_multicore` compiler pass). ResNet-50/YOLOv8n hit `aipu_cores:4` by Voyager default; MobileNetV2/EfficientNet-B0/ResNet-152 default to `1` and need the explicit override.

### 6.3 Batch > 1 — hard constraint `[FORUM]`/`[MEASURED]`
`axcompile` **rejects `dim[0] > 1`** — "the SDK splits work across cores, not across batches." Setting YAML `input_tensor_shape:[N>1,…]` hits a TVM frontend bug (`Only batch sizes of 1 supported` / quantized-mul Relay assertion). The *only* batched-compile path is the 3-knob trio with `input_tensor_shape:[1,…]` preserved, which constrains output batch to `aipu_cores` (≤4). **Batches >4 and `1 < batch < cores` are both unreachable.** `axrunmodel --aipu-cores M` requires M to match the compiled core count.

### 6.4 Verify a knob landed `[MEASURED]`
Always inspect after deploy, before benching:
```bash
MJ=$(find <build>/<network>/ -name model.json | head -1)
CC=$(find <build>/<network>/ -name compile_config.json | head -1)
python3 -c "import json;m=json.load(open('$MJ'));c=json.load(open('$CC'));
print('input shape:',m['inputs'][0]['shape']);
[print(f'{k}:',c.get(k,'MISSING')) for k in ['aipu_cores_used','multicore_mode','dpu_constants_home','elf_in_ddr','tiling_depth']]"
```
Frozen fields (e.g. `aipu_cores_max`) reject overrides at validation (~7–10 s fail, before compile). Build artifacts land under `<build>/<name>/<name>/<aipu_cores_used>/model.json` (+ `compile_config.json`, `pool_l2_const.bin` weights). `quantized/` subdir is root-owned.

---

## 7. Memory hierarchy & data movement (simulator box ④ inputs)

- **Tiers exposed by the compiler:** `global.l2` and `global.ddr` pools; L1 implied by `l1_constraint`. Weights home = `dpu_constants_home` (L2 default, auto-spills to DDR when weights > 32 MB L2) `[DOC]`/`[MEASURED]`.
- **dma-buf / zero-copy:** toggled everywhere (`input_dmabuf`/`output_dmabuf`, `dmabuf_inputs/outputs`, `axrArgument.fd`). v1.6 adds "DMA-buf passthrough for ARM without memory copies" — relevant to the Aetina ARM host `[DOC]`. Voyager also uses OpenCL `cl_khr_external_memory_dma_buf` for copy-free buffer sharing.
- **`double_buffer`** overlaps host↔device transfer with compute (+10–40% throughput, latency tradeoff).
- **PCIe / IOMMU (the L2 measurement target):**
  - Aetina link: **PCIe Gen3 ×4 ≈ 3.9 GB/s** usable host↔card ceiling `[MEASURED]`.
  - Metis Alpha exposes a **1 GB PCIe-IOMMU window** mapping host LPDDR4 into device address space — NOT real on-card DRAM `[MEASURED]`. (BAR2 = 32 MB L2 SRAM at `0x900000000`; BAR0 = 4 KB.)
  - Forum SMMU-v3 thread (1330): default PCIe window 14 MB → expanded to 128 MB via device-tree; SMMU translation faults need `CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y`; runtime needs **DMA-buf heaps (kernel ≥ 5.6)**, ION (4.19) incompatible `[FORUM]`.
  - **No documented analytical PCIe bandwidth/latency model** `[GAP]` — isolate empirically by toggling dmabuf/double-buffer in `axrunmodel` (Device FPS includes transfers; subtract host-only path).
- **No "unified host-device memory" feature on current Metis** `[FORUM]` — this is a forward-looking platform *assumption* of the simulator (Europa-anchored), not a present capability. The Alpha's 1 GB IOMMU window is the closest descriptive evidence.
- **Driver tunables (Aetina, DMA/latency benchmarking)** `[MEASURED]`: `dma_poll` (**must be 1** on this board — interrupt mode times out at 2 s), `dma_timeout`, `irq_timeout`, `single_msi`, `enable_dmabuf_sync`.

---

## 8. Power & energy measurement

- **On-board sensor: INA236** current/power monitor on the board power rail, sampled **~200 Hz**; the power limiter reads it to throttle `[DOC]`. **Board-rail aggregate only — no per-core / per-tier breakdown** `[GAP]`.
- **Reading power is M.2 Max-only** in practice `[DOC]`/`[FORUM]`:
  - `axdevice --set-power-limit <W>` (closed-loop, M.2 Max only; ~23 W budget), `axdevice -v` to view.
  - `axmonitor` power min/max/avg — M.2 Max only. **Standard single-chip M.2/PCIe cards do NOT show power in `axmonitor`** `[FORUM]`. The Aetina Alpha and the production test card (PCIe Rev1) have **no usable power telemetry** `[MEASURED]` → all power figures are nameplate/TDP estimates.
- **Temperature: 5 sensors** (board + 4 core). Read via `inference.py` (auto-shows max), `axlogdevice --slog`, or app tracer `core_temp` `[DOC]`.
- **DVFS knobs:** `axdevice --set-clock {100,200,400,600,700,800 MHz}`, `--set-core-clock`, `--set-mvm-limitation <1-100%>`; compiler `frequency` + `mvm_utilization_limit` `[DOC]`.
- **Specs:** 15 TOPS/W INT8; Metis 1-chip card ~8–15 W typical; M.2 Max ~6.5 W avg; 4-chip PCIe 30–58 W `[DOC]`/vendor.

**Implication for the simulator's energy layer (box ⑤):** energy must be **spec-based + activity-factor estimated**, not measured — this matches `overall.md` M7 and Out-of-scope ("Energy as measurement" excluded). The only trustworthy on-silicon energy lever is whole-board INA delta on M.2 Max, or a wall-plug/rail delta against a fixed workload loop elsewhere.

---

## 9. LLM support status

**Experimental, precompiled-only, memory-wall-bound** `[DOC]`/`[FORUM]`/`[MEASURED]`.

- **Component:** AxLLM / `axllm` CLI (`--prompt`, interactive, `--rich-cli`, `--ui` Gradio). Pipelines `transformers-aipu` (default) vs `transformers` (CPU/GPU dev fallback). Aux: `axextractembeddings`.
- **Precompiled zoo (7, all `*-static`):** `phi3-mini-{512,1024,2048}-static`, `llama-3-2-1b-1024-4core-static`, `llama-3-2-3b-1024-4core-static`, `llama-3-1-8b-1024-4core-static`, `velvet-2b-1024-4core-static`. Slug = context length + core count. Min card RAM: 4 GB (≤3B), 16 GB (8B / phi3-2048). `[MEASURED 2026-06-04]` the production metiscard `build/` carries **both** 1-core (`…-1024-static`) and 4-core (`…-1024-4core-static`) artifacts for llama-3.2-1b/3b and llama-3.1-8b (the 1c slug is runnable — verified via `axllm llama-3-2-1b-1024-static --show-stats`). **llama context is fixed at 1024** (baked into the slug); only phi3 ships 512/1024/2048 — so a context sweep is phi3-only.
- **Cannot self-compile an LLM** `[FORUM]`/`[MEASURED]`: no public LLM compiler; you reference Axelera-hosted `precompiled_url`, not arbitrary HF URLs. Public `axcompile` is a vision/CNN ONNX→INT8 compiler lacking RMSNorm/RoPE/Attention/Embedding/GELU/dynamic-shapes. The prefill+gen dual-megakernel artifact comes only from Axelera's internal toolchain.
- **Measured LLM behavior (production card, Llama-3.2-1B)** `[MEASURED]`:
  - Throughput: GPU RTX 3090 (fp16) **187 tok/s** · AIPU **15.0 tok/s** · CPU 0.67 tok/s.
  - **Batch-1 decode is a pure on-card-LPDDR weight-streaming memory wall**: decode time ∝ weight bytes, r²=0.997, **eff bandwidth ≈ 24.2 GB/s**; MAC array ~99% idle (≈0.02% of 214 TOPS peak).
  - Prefill is compute-region (~8.1 TFLOP/s effective); prefill device time flat vs real prompt length (static padded graph).
  - More cores barely help decode: 4c/1c speedup 1.31× (1B) → 1.12× (8B) — bigger model = more bandwidth-bound. `network.py` asserts `batch==cores`; only 1c & 4c artifacts ship.
  - Staff confirm: Llama-3.2-3B runs **6+ tok/s single core** fully offline `[FORUM]`.
- **Alpha M.2 cannot run any LLM** `[MEASURED]`: `-1301` closed-firmware compute wall (not a timeout) + no on-card DDR (weights can't fit the 1 GB IOMMU window). Load is beatable from user space (`ze_shim2` LD_PRELOAD); compute is not. **The `-1301` code and the 1 GB-IOMMU-blocks-LLM mechanism are OUR findings — the forum is silent on both** `[GAP]`; treat as our own characterization, not vendor-stated.
- **Forward anchor — Europa (2026):** 128 MB L2 SRAM, LPDDR5 ~200 GB/s, ~45 W, "unified architecture", targets up to Llama-3-70B `[FORUM]`/vendor. No shipping silicon / measured data exists.

---

## 10. Per-unit characterization recipes → Phase 0 output files

Maps the toolchain above to the `measurements/` files `overall.md` Phase 0 needs. **All ground-truth lives in `measurements/`; this table is the "how".**

| Unit / target | Tool & method | Sweep | Output |
|---|---|---|---|
| **CIM tile (Metis Alpha)** | `axrunmodel` on single-op / small ONNX (differential method); `dpu_constants_home: l2` vs `ddr`; AxInferenceNet `LatencyCallback` for per-op ns | conv/matmul shapes: in/out ch, H/W, seq; ch ∈ {64…1024}; INT8; Mode 1 | `measurements/aetina/metis_alpha_{cnn_proxy,matmul,pcie}.json` |
| **PCIe / DMA** | `axrunmodel` with dmabuf/double-buffer toggled; eBPF + LD_PRELOAD on `instance.run()` stages | dmabuf on/off, DB on/off | `metis_alpha_pcie.json` |
| **RKNPU2** | RKNN-Toolkit2 matmul micro-bench | hidden 2048/4096/8192, seq 1/256/2048; INT8/INT16/FP16; batch 1/4/16/32 | `measurements/aetina/rknpu2_matmul.json` |
| **Mali-G610** | self-written OpenCL matmul kernel (avoid framework noise) | same shapes; FP16 primary + FP32 ref | `measurements/aetina/mali_matmul.json` |
| **CPU (A76)** | on-target micro-bench of LLM support ops (sampling, RoPE, KV append/evict); `taskset -c 4-7 chrt -f 50`; `clock_gettime` + `perf stat` | per-op | `measurements/aetina/cpu_ops.json` |
| **End-to-end LLM (production card)** | `axllm --show-stats` | Llama-3.2-1B/3B/8B; ctx 512/1024/2048 | `measurements/metis_card/vendor_llm_int8.json` |
| **Variance (Stage 0)** | repeat representative ops, cold-starts × iterations, compute CoV | per unit | `measurements/{unit}/variance_profile.json` |

Known per-unit pathologies to expect `[MEASURED]` (from Step-1, see [papers/metis-silicon/metis-step1-cnn-characterization-2026-05-23.md](papers/metis-silicon/metis-step1-cnn-characterization-2026-05-23.md)):
- **Metis Alpha:** vendor pipeline is host-pipeline-bound (AIPU starved); 7.9× gap between vendor-pipe and direct-harness on the same silicon; Mode 3 (SDK batched multi-core) best; **1378 FPS MobileNetV2** peak.
- **RKNPU2:** latency-bound at b=1, scales 5–10× with batching; **EfficientNet-B0 collapses on depthwise+Swish**.
- **Mali:** compute/bandwidth-bound, 10–25× slower than Metis, batching doesn't help; realistic role = preprocessing offload only.

**Phase 0.2 capability verification** `[MEASURED 2026-06-04]` (smoke-tested on the real boards):
- **CIM matmul micro-bench = 1×1 conv proxy.** Raw standalone `MatMul`/`Gemm` ONNX **fails to compile** (`ONNXGraphCleanerError` on the MatMul node — confirms general MatMul is not a clean compile path). A **1×1 `Conv2d` (mathematically = matmul)** compiles cleanly: `compile --input x.onnx --input-shape 1,C,1,1 --output DIR` (auto-calibrates with 100 random samples, no imageset needed, ~7 s) → `axrunmodel DIR/compiled_model/model.json --seconds N` returns dev/host/system FPS. **Use 1×1 conv as the GEMM/GEMV primitive** for the sweep-matrix shapes.
- **End-to-end LLM (metiscard):** `axllm <model> --prompt "…" --show-stats` verified — emits Tokenization / Prefill / TTFT / Gen tok/s (llama-3.2-1b ≈ 11 tok/s on the 16 GB card, fw v1.6.0).
- **RKNPU2:** `rknn-toolkit-lite2` (on-board inference runtime) installs on aarch64 ✓; **`rknn-toolkit2` (ONNX→`.rknn` converter) fails to build on aarch64** (`onnxoptimizer` wheel) → convert on an x86 host (metiscard) then run on-board via rknnlite, or install the C/C++ build deps and retry.
- `compile --log-level` requires UPPERCASE (`WARNING`, not `warning`).

---

## 11. Gotchas that corrupt measurements (check before trusting a number)

- **Metis Alpha can drop off the PCIe bus** `[MEASURED 2026-06-03]` — the card may vanish from `lspci` (slot `0000:01:00.0` re-enumerates as garbage `16c3:abcd`, no `/dev/metis` node, `metis.ko` auto-unloaded); `axdevice` then errors `No target device found in lspci`. Recover from the host: `echo 1 | sudo tee /sys/bus/pci/devices/0000:01:00.0/remove` → `echo 1 | sudo tee /sys/bus/pci/rescan` (re-enumerates as `1f9d:1100`) → `sudo modprobe metis` (→ `/dev/metis-0:1:0` returns). **Then recreate the SDK docker container** (`docker rm -f axelera-sdk; ~/start-sdk-bg.sh`) — it maps `/dev/metis` at creation, so a container started while the card was absent won't see it. `~/reset_device.sh` automates the rescan. **Always run `axdevice` to confirm presence before a measurement session.**
- **DMA opts off by default in direct AxRuntime** (§4.1) — 30–90% under-report. Enable double_buffer + dmabuf.
- **`dma_poll=1` mandatory on Aetina** — interrupt mode 2 s-timeouts (community-confirmed) `[MEASURED]`.
- **`--mode PREQUANTIZED` does NOT skip calibration** — a 3-knob compile still ran ~60 min of 200-image calibration `[MEASURED]`.
- **Clock-profile name mismatch (v1.6.0)** — `libaxruntime` sends `clock_profile_core_0`, firmware expects `aicore0` → "Failed to set clock frequency"; workaround `AXELERA_CONFIGURE_BOARD=0` **but that can leave cores at non-default clocks → corrupts FPS/W & latency** `[FORUM]`.
- **Thermal throttling** — uncooled cores hit 106–110 °C; control thermal state for repeatable runs `[FORUM]`.
- **PCIe bus-enumeration freezes** on some RK3588/Rock-5B/RPi5 hosts (Above-4G/IOMMU/x1-link) silently invalidate runs `[FORUM]`.
- **`>1024` token LLM input** silently yields 0 tokens (no exception) — SDK lacks input validation `[MEASURED]`.
- **Build collision** — changing only the YAML filename (not the `name:` field) overwrites a build `[MEASURED]`.
- **`curl`/`wget` are chmod 000 on Aetina** — use Python `urllib` for downloads `[MEASURED]`.

---

## 12. Documented gaps → the measure/ask list

These are silent in docs+forum and define what the simulator must establish empirically (or what to ask Axelera):

1. **INT4 / LLM bit-width** — confirm actual precision of `*-static` LLMs.
2. **L1/L2/on-card-DDR exact byte sizes & bandwidths** per board SKU.
3. **`device_profiling`/`host_profiling` output format**, and **`axtrace` CLI + per-op/per-core cycle output**.
4. **Power granularity** — no per-core/per-tier energy; no energy-per-op/token. Only board-rail INA (M.2 Max).
5. **Analytical PCIe/DMA bandwidth-latency model** — measure via dmabuf/DB toggles.
6. **Attention / KV-cache mapping** onto the AIPU for the precompiled LLMs (separate from the public compiler).
7. **Standalone GEMM benchmark** — none in the zoo; general MatMul isn't a supported public-compiler op (YOLO11 whitelist only). May need the LLM path or won't compile via `deploy.py`.
8. **Aetina (Alpha) vs production-card parity** — power API, op coverage, knob equivalence (v1.3.1 vs v1.6) all unverified.
9. **`-1301` / 1 GB-IOMMU LLM block** — our finding only; no vendor corroboration.

---

## 13. Source pointers

**SDK (GitHub `axelera-ai-hub/voyager-sdk`, branch `latest`):**
- Op coverage: `docs/reference/onnx-opset17-support.md`
- Device micro-bench: `docs/reference/axrunmodel.md` · pipeline: `docs/reference/inference.md`, `docs/tutorials/benchmarking.md`
- Runtime APIs: `docs/reference/axelera.runtime.md`, `docs/reference/axruntime.md`, **`docs/reference/axinferencenet.md`** (LatencyCallback)
- Compiler: **`docs/reference/compiler_configs_full.md`** (knob bible), `compiler_configs.md`, `compiler_cli.md`, `compiler_api.md`, `deploy.md`
- Power/thermal: `docs/reference/thermal_and_power_guide.md`, `docs/reference/axdevice.md`
- LLM: `docs/tutorials/llm.md` · Zoo: `docs/reference/model_zoo.md`
- Tracers source: `axelera/app/inf_tracers.py` · AxRuntime example: `examples/axruntime/axruntime_example.py`
- On the Aetina box: `docker exec axelera-sdk cat /home/ubuntu/voyager-sdk/docs/reference/<file>`

**Forum (community.axelera.ai) — keep open while characterizing:**
- `metis-m-2-3/reproducing-the-fps-w-benchmark-claim-1355` — silicon-only vs end-to-end FPS/W (614 vs 270)
- `voyager-sdk-2/multi-core-mode-1350` — core-allocation latency/throughput tradeoff
- `metis-m-2-3/how-to-use-batch-size-1-for-input-during-model-compilation-1332` — batch=1 hard constraint
- `metis-m-2-3/metis-m-2-bus-error-...-kernel-6-18-smmu-v3-...-1330` — DMA/IOMMU/PCIe-window/BAR internals
- `product-updates/the-metis-ai-platform-a-technical-deepdive-125` — architecture constants
- `metis-m-2-3/metis-m-2-temperature-1335` — `axmonitor` telemetry surface
- `voyager-sdk-2/how-to-configure-custom-llm-models-in-yaml-1180` — LLM precompiled-only
- `voyager-sdk-2/import-custom-models-235` — supported ops; Gemma3/Whisper unsupported

**Our own silicon notes (in this repo):** [papers/metis-silicon/](papers/metis-silicon/) and [papers/platforms/](papers/platforms/).

---

*Maintenance: when the SDK is upgraded or a knob/behavior is verified on silicon, update THIS file first, then re-cite from experiment notes. Tag every new line `[DOC]`/`[FORUM]`/`[MEASURED]`/`[GAP]`.*
