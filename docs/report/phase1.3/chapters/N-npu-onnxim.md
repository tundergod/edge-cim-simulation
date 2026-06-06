# N (1.3) — ONNXim：NPU 的「重型 sim 交叉驗證」（全模擬，ONNXim ≠ #13）

Phase 1.2 的 NPU 引擎是**解析 systolic-roofline**，**全模擬**（無 RKNPU2 silicon，issue #13）。Phase 1.3 把 **ONNXim**（POSTECH 的 cycle-level NPU 模擬器）配成 **RKNPU2-approx**，當 `engine='onnxim'` drop-in，對解析的趨勢做交叉驗證。**這是 sim-vs-sim**：兩邊都模擬、都不是 silicon，所以 #13 維持 *superseded-not-satisfied*、ONNXim ≠ #13。

## 怎麼跑（Docker on metiscard）

ONNXim 只在 Ubuntu-20.04/gcc-10/conan-1.57 build（Mac 無 container runtime、conan 1.57 也裝不上 Py3.13），所以在 **metiscard**（x86 Ubuntu、Docker）用它自己的 `ubuntu:20.04` Dockerfile build（釘 commit `a1e86296`），driver `npu_onnxim_trace.py` 一個 `docker run` 內逐 shape 生 matmul ONNX → `Simulator` → parse `Simulation Finished at … us`，結果寫回 `simulated/onnxim/rknpu2_sim_matmul.json`。

**RKNPU2-approx config**（`tools/onnxim/rknpu2_approx.json`）：3 核 × **32×32** systolic（借 Hexagon 32×32，與解析同維）、INT8（`precision:1`）、`core_freq` 1000 MHz → **6.14 TOPS**（≈ datasheet 6）、DRAM 25 GB/s（ramulator2 DDR4）、icnt simple。

> **兩個誠實踩雷**（執行中發現、已記進 config doc）：(1) 8×8 模板的 `spad_size:64` 對 32×32 陣列**太小**，會在 `Mapping.cc` tiling 除零 → 把 spad/accum 放大成 2048/512。(2) 即使如此,**ONNXim 對 N≤64 的 GEMM 仍 SIGFPE crash**（退化 tiling）→ staircase 從 **N=128** 起跑（仍看得到通道趨勢）。

## 結果：趨勢一致，絕對值差 ~4×（一致的系統性 offset）

![N3](../../../figures/phase1.3/N3.png)

15 個 shape（投影 q/o/kv/gate/down 在 M=1,256 + K=2048 通道 staircase N≥128）：

- **趨勢一致**：staircase **單調遞增、≈∝N**（ONNXim 56→1487 µs for N=128→3072），和解析 roofline 同形狀（HeteroInfer Fig3 的通道 staircase）。`staircase_monotone = True`。
- **絕對值**：ONNXim **一致地比解析高 ~4×**（median |delta| **318%**、max 493%，且各 shape delta 很穩定 ~315–330%）。例：(1,4096,4096) ONNXim 3121 µs vs 解析 835 µs。

**這不是打架，是抽象層級不同**：ONNXim 是 cycle-level，把 systolic fill/drain、NoC、DRAM scheduling 的開銷都算進去；解析 roofline 是 `max(compute, memory)+factor`,把這些開銷抽象掉。兩者**都不是 silicon**,所以我們不能說誰「對」——但**趨勢吻合**是這個交叉驗證的價值,而那個 ~4× 的系統性 offset 被誠實記錄(它告訴我們:解析 roofline 相對 cycle-level 模型是偏樂觀的下界)。

## 定位 + 誠實標註

- `engine='onnxim'` 回 ONNXim 的 cycle-level latency,標 `simulated (ONNXim generic-systolic, RKNPU2-approx)、NOT silicon`。**解析 NpuModel 維持 Phase-1.2 主交付**。
- **ONNXim ≠ issue #13**:ONNXim 是更重的**模擬器**,不是 RKNPU2 silicon;它**沒有**達成 #13 的 silicon gate(#13 維持 superseded-not-satisfied、獨立)。這是 sim-vs-sim 趨勢交叉驗證,不是 silicon 校準。
- 報告 `validation/reports/phase1.3/m4_npu_onnxim.json`(逐 shape delta + 每 shape 斷言為 ONNXim hit)、圖 `N3`、本章。可重產:`tools/onnxim/README.md`(build runbook,釘 `a1e86296`)→ `npu_onnxim_trace.py` → `build_m4_npu_onnxim.py` → `npu_n3_fig.py`。
