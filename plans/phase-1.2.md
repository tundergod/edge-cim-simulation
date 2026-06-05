# Plan: Phase 1.2 — 模組化「engine + 可換 spec」全保真 component 模擬層

範圍：把每個非-micro-benchmark 單元建成 **模型引擎 + 可換 spec 模組（換型號/參數只改 spec 檔）**：**CPU**（instruction-count, SimpleSSD 式）、**NPU**（ONNXim）、**記憶體**（DRAM analytic 全-spec + Ramulator2 LPDDR5 + SRAM CACTI tier）、**GPU**（micro-benchmark + analytic roofline 槽）。CIM 與 GPU-micro-benchmark 維持 1.1 實測為主。分支 `phase-1.2`。所有 1.2 模型標 `simulated, not silicon-validated`（CPU 對 cpu_ops.json 校準者另標 calibrated）。執行：**序列前置 → 平行 component fan-out（每 component 一 subagent）→ 序列合併**。

## 決策（已定）

- **D1 CPU**：instruction-count **roofline** 引擎 `latency = max(compute_us, memory_us) + overhead`，`compute_us = op_count/(W·IPC·freq)·η_c`（NEON 利用率）、`memory_us = bytes/(BW·η_bw)`（達成/峰值 BW），bytes 由 elements×dtype×passes、**BW 取自記憶體 spec 模組（A76 共用 SoC LPDDR）**。reduction op（softmax/rmsnorm）落 memory 分支→由**可換的記憶體 BW spec 驅動**（非塞進不可轉移的單一 η）；element-wise（swiglu/residual）落 compute 分支。`η_c`/`η_bw` 對 **fp32** `cpu_ops.json` 校準（calibrated-not-guessed）。swappable by {freq,W,IPC} + 記憶體 spec（換 LPDDR4→5 則 reduction op 自動跟著變、免重校）。**多核/big.LITTLE**：spec 以**多叢集**描述（big 4×A76 + LITTLE 4×A55，各自 freq/W/IPC）+ 指派策略；`compute_us` 用「指派到的核的吞吐 Σ[W·IPC·freq]」（異質核相加），**`memory_us` 不隨核數變（共享 LPDDR、BW 飽和）**→ compute-bound op 加核變快、memory-bound reduction 加核幾乎無效。**僅 A76-實測配置 silicon-calibrated；A55 + 多核 scaling = analytic（無量測）→ 標 simulated、平行效率與 A55 spec=assumption；upgrade=量 A55+多核**。非 cycle-level（逐核 pipeline/cache-coherence/DVFS/OS 排程超出範圍，需 gem5）。殘留誠實警告：reduction 串行相依 + 小-op `η_bw` 達成率仍有晶片特性；fp16=numpy 模擬上界、只用 fp32 校準。
- **D2 NPU**：ONNXim（generic-systolic 配 RKNPU2-approx）為模擬資料源 → 封閉式 `m4_npu.py` 對其擬合；HeteroInfer trend 交叉驗證。build 失敗 → OVERALL.md risk #7 fallback（解析 systolic / lookup-override）+ 回報 user。
- **D3 記憶體**：`MemoryModel` swappable 介面（ADR-0002）。**analytic 引擎涵蓋全 spec**（LPDDR4/4X/5，= 頻寬+timing 數字）為預設/fast-DSE；**Ramulator2 引擎接其內建 LPDDR5**（subprocess + 代表性迭代 → 有效 BW/latency → 解析外推）為高保真選項；LPDDR4 的 Ramulator2 C++ port 列**後續升級**（Ramulator2 無 LPDDR4）。**SRAM（Metis AIPU L1/L2 SPM）≠ Ramulator2** → CACTI 推得固定 (latency, BW) tier；residency 為 `architecture-only`（batch=1 INT8 decode 權重 ≫ 32MiB → 永不命中、恆走 DRAM）。
- **D4 GPU**：Mali 維持 1.1 micro-benchmark（實測校準）為主模型；1.2 加可換 **analytic roofline 模型** `time=max(FLOPs/peak_compute, bytes/peak_BW)+launch`（換 GPU 丟 spec）。無 Mali cycle-sim 存在（全 NVIDIA/AMD）→ 不假裝有。
- **D5 swappable spec**：所有單元參數抽成 `simulator/specs/*.json`（CPU/NPU/GPU/記憶體×3/SRAM）；統一 loader；換型號=換 spec 檔。各 spec 欄位標 `measured|datasheet|assumption`。

---

## 0. 序列前置（平行前必做）

1. 建骨架：`docs/figures/phase1.2/`、`validation/reports/phase1.2/`、`docs/report/phase1.2/chapters/`、`measurements/onnxim/`、`simulator/specs/`。 → verify：目錄存在。
2. **共用 spec 介面**：`simulator/specs/loader.py`（`load_spec(name)->dict`；registry）+ spec 檔格式約定（`{device, fields..., provenance:{field: measured|datasheet|assumption|source}}`）。 → verify：`load_spec` 讀任一 spec 回 dict + provenance。
3. **寫入已知 spec 模組**（用 Phase-1.2 研究查得的數字）：
   - `cpu_rk3588.json`：**多叢集** {big: 4×A76 @2.3GHz, little: 4×A55 @1.8GHz}，各 {NEON 128-bit (fp32 W=4/fp16 W=8), IPC 2, L1/L2}、L3 3MB shared、ops_per_elem 初值、指派策略（預設 big-only=量測基準）；CPU 可達記憶體 BW 取自記憶體 spec。A55 spec=datasheet（無量測）。
   - `npu_rknpu2.json`：3-core、6 TOPS INT8、INT4/8/16/FP16/BF16/TF32、~1GHz、systolic 維=`assumption`（未公開→借 Hexagon 32×32、標 borrowed）。
   - `gpu_mali_g610.json`：4-core Valhall、~1GHz、peak FP32 ~512 GFLOP/s、FP16 ~1 TFLOP/s、INT8=`assumption`、L2 1MB。
   - `mem_lpddr4.json` / `mem_lpddr4x.json` / `mem_lpddr5.json`：data_rate {3200|(板 4224)、4266|(4224)、6400}、bus 64-bit(4ch×16)、peak {25.6|33.8、34.1|33.8、51.2} GB/s、eff_factor（1.1 量得 0.71）、timing 概要。
   - `sram_metis_aipu.json`：L1 4MiB/core×4、L2 32MiB、D-IMC 1MiB/core×4（容量 source=ISSCC）；bw/latency=CACTI 推得或 `assumption`。
   → verify：6+ spec 檔存在、每欄位帶 provenance；峰值 BW 數學一致（MT/s×8B）。

---

## 平行 fan-out（每個 work-package 一 subagent；皆依賴步驟 0、彼此獨立）

> 每 WP 交付：**引擎(.py) + 用到的 spec + 擬合/驗證 script + report JSON + 圖 + 章節草稿**；皆標 honesty。各 WP 做完由其 subagent 自驗，合併階段再交叉驗。

### WP-CPU（instruction-count CPU 模型）
4. 引擎 `simulator/models/m4_cpu.py`（擴/取代）：`CpuModel(spec)`，`op_us = max(compute_us, memory_us) + overhead`；`compute_us = op_count/(Σ_assigned_cores[W·IPC·freq])·η_c`（異質核吞吐相加）、`memory_us = bytes/(BW·η_bw)`（BW 讀記憶體 spec、共享不隨核數變）；op_count/bytes 由 `characterization/aetina/run_cpu_ops.py` 的 element count；指派策略決定用哪些核。 → verify：回正/單調；compute-bound op（swiglu/residual）隨核數加速、memory-bound reduction（softmax/rmsnorm）不隨核數變、memory 分支隨記憶體 spec 變。
5. 校準 `tools/analysis/fit_m4_cpu_instrcount.py`：**先確認 `cpu_ops.json` 量測的核數/thread**（校準基準）；對 fp32 解 `η_c`（compute 分支，element-wise op）+ `η_bw`（memory 分支，reduction op）（softmax kv sweep→memory 斜率；其餘每模型一點→跨 1B/3B/7B/8B 單一因子交叉檢查 op_count/bytes 標度）。寫 `validation/reports/phase1.2/m4_cpu.json`（每 op：分支、η_c/η_bw、對量測誤差；reduction=memory-bound 註、**A55+多核 scaling=simulated/未量**、fp16=上界註、量測基準核數）。 → verify：JSON 每 op 分支+因子+誤差+誠實註；A76-基準校準誤差記錄、A55/多核標 simulated。
6. 圖 `C1`（`tools/plotting/phase1_2_figs.py`）：子圖分 op；**X**：op size（H/kv/F/V）；**Y**：latency(µs)；量測點 vs instruction-count 模型線；標哪些 op 是 reduction(記憶體受限)。 → verify：`docs/figures/phase1.2/C1_cpu_instrcount.{png,svg,pdf}`。
7. 章節草稿 `docs/report/phase1.2/chapters/C-cpu.md`（架構→spec→原理(工作量÷速度)→圖→白話→sim-vs-measured + 可換性說明）。 → verify：含 sim-vs-measured 段 + 可換性。

### WP-NPU（ONNXim + 封閉式）
8. ONNXim 取得+建置 `tools/onnxim/`（vendored/submodule）+ RKNPU2-approx config（讀 `npu_rknpu2.json`；systolic 維借 Hexagon 32×32 標 borrowed）。 → verify：build 成功 + smoke matmul；**失敗→記 risk#7 fallback + 回報 user（不靜默改路徑）**。
9. `tools/analysis/npu_onnxim_trace.py`：由 `measurements/op_inventory/` NPU-bound shapes export ONNX→ONNXim 輸入（ADR-0007：export 次要、fallback=由 traced graph 直建）。 → verify：對 NPU shapes 產出輸入。
10. 跑 ONNXim → `measurements/onnxim/rknpu2_sim_matmul.json`（#13 silicon 的模擬替身，標 `simulated, NOT silicon`）。 → verify：每 shape 有延遲 + simulated 標。
11. 引擎 `simulator/models/m4_npu.py`（取代 stub）：`NpuModel(spec)`，封閉式（FLOPs/G_eff + native attn bmm）對 ONNXim 表擬合；roofline。 → verify：復現 ONNXim 延遲（fit median/p95）、正/單調/knee 合理。
12. `tools/analysis/build_m4_npu.py`：封閉式-vs-ONNXim fit 誤差 + HeteroInfer trend 量化勾稽（staircase 週期 vs Fig3、order/shape ≤6× vs Fig4、單-proc BW frac 59–66% vs Fig5）。寫 `validation/reports/phase1.2/m4_npu.json`（全標 simulated + `upgrade_path:#13`）。 → verify：fit 誤差 + trend + simulated + upgrade_path。
13. 圖 `N1`(ONNXim staircase；caption 標對照 Fig3、非單位對單位)、`N2`(offload 三方：CIM/Mali 實線 silicon、NPU 虛線 sim)、`N3`(封閉式 vs ONNXim 點/線)。 → verify：`docs/figures/phase1.2/{N1_npu_staircase,N2_npu_offload,N3_npu_fit}.*`。
14. 章節 `docs/report/phase1.2/chapters/N-npu.md`。 → verify：含 sim-vs-reference + ONNXim 定位（非 silicon）。

### WP-MEM（DRAM analytic 全-spec + Ramulator2 LPDDR5 + SRAM CACTI）
15. 引擎 `simulator/models/m2_memory.py`（擴）：`MemoryModel(spec, engine='analytic'|'ramulator2')` swappable；analytic：`stream_us=bytes/(peak_BW·eff)`，吃 `mem_lpddr4/4x/5.json`（全 spec 即時可換）；kv_append 沿用。 → verify：三 spec 各回正/單調；LPDDR5 eff 對 1.1 的 33.3、LPDDR4x 對 L4 anchor 24.2 一致。
16. Ramulator2 引擎 `tools/onnxim/`→`tools/ramulator2/`（vendored/submodule build）+ `tools/analysis/mem_ramulator2.py`：subprocess + 代表性迭代（一個 decode/prefill 迭代 trace → 有效 BW/latency）接內建 **LPDDR5**；標 LPDDR4 port 為後續。 → verify：Ramulator2 build + 對一個 LPDDR5 trace 回有效 BW；**失敗→analytic 為主、Ramulator2 列升級 + 回報 user**。
17. SRAM tier `simulator/models/m1_cim_spm.py`：`SramTier(spec)` 讀 `sram_metis_aipu.json`（CACTI/assumption 的 latency+BW）；`residency(working_set)`→(層, BW)，docstring 標 `Metis AIPU SPM, software-managed, architecture-only`。 → verify：8B 權重 working_set 回 DRAM 層（證 never-binds）。
18. `tools/analysis/build_mem_report.py`：(a) 三 DRAM spec 對照 + 對 anchor 勾稽；(b) Ramulator2-vs-analytic（LPDDR5）差；(c) SRAM what-if（全放 L2 反事實）。寫 `validation/reports/phase1.2/m2_memory.json`（analytic=calibrated、Ramulator2=高保真、SRAM=architecture-only/simulated、kv=unvalidated）。 → verify：JSON 三項 + 誠實標。
19. 圖 `M1`(三 spec BW vs bytes)、`M2`(Ramulator2 vs analytic LPDDR5)、`M3`(SRAM tier 階梯 + 標 in-scope 落 DRAM)。 → verify：`docs/figures/phase1.2/{M1_mem_specs,M2_ramulator_vs_analytic,M3_sram_tiers}.*`。
20. 章節 `docs/report/phase1.2/chapters/M-memory.md`（DRAM 可換 spec + Ramulator2 + SRAM；LPDDR4 不在 Ramulator2 的誠實說明）。 → verify：含 sim-vs-reference + swappable 說明。

### WP-GPU（analytic roofline 槽）
21. 引擎 `simulator/models/m4_gpu_roofline.py`（新，與既有 micro-benchmark `m4_gpu.py` 並存）：`GpuRooflineModel(spec)` 讀 `gpu_mali_g610.json`；`time=max(FLOPs/peak_compute_eff, bytes/peak_BW_eff)+launch`；efficiency 對 1.1 mali_matmul 量測校準。 → verify：對 1.1 Mali GEMM 量測點誤差記錄（roofline eff 校準）。
22. `tools/analysis/fit_gpu_roofline.py`：擬 efficiency/launch（對 mali_matmul.json）。寫 `validation/reports/phase1.2/m4_gpu_roofline.json`（標 roofline=可換槽；Mali 絕對值仍以 micro-benchmark 為準）。 → verify：JSON 校準誤差 + 「micro-benchmark 為主、roofline 為換型號槽」定位。
23. 圖 `G1`(roofline：arithmetic intensity vs 吞吐，標 compute/BW-bound 與 decode/prefill 點)。 → verify：`docs/figures/phase1.2/G1_gpu_roofline.*`。
24. 章節 `docs/report/phase1.2/chapters/G-gpu.md`（為何無 Mali cycle-sim、roofline 槽的用途與侷限）。 → verify：含無-Mali-sim 誠實說明 + 可換性。

---

## 合併（序列；所有 WP 完成後）

25. 交叉驗證所有 WP：spec loader 能載全部 6+ spec 並餵各引擎；各 report JSON 的 honesty 標一致；no fake gate（CPU=calibrated 數值、NPU/GPU-roofline/SRAM=simulated/trend）。 → verify：一支 `tools/analysis/check_phase1_2.py` 跑通、列各引擎×spec 的 sanity。
26. contracts：`m4_cpu.yaml`(instr-count+η 校準)、`m4_npu.yaml`(BLOCKED→SIMULATED ONNXim-based)、`m2.yaml`(swappable engine + 全 spec + SRAM 屬 M1-SPM + ADR-0002 偏離註)、`m4_gpu.yaml`(+roofline 槽)、新增 `specs.yaml`(spec 模組清單 + provenance)。 → verify：各 contract 反映引擎/spec/gate。
27. 報告：`tools/report/build_phase1_2_report.py`（複製 `build_phase1_report.py`，ORDER=[intro,C-cpu,N-npu,M-memory,G-gpu]、CH/FIG/OUT 指 phase1.2）；beginner-friendly 中文，開頭講「engine+可換 spec」總圖 + 「哪些實測/哪些模擬」。→ HTML→PDF。 → verify：`docs/report/phase1.2/index.html`+`phase1.2-report.pdf` 產出、圖內嵌、每章 sim-vs-reference。
28. `docs/phase1.2-findings.md`：逐 component {引擎 + spec + 驗證 + gate/trend + honesty + 可換性 + upgrade path}；總表「engine + spec 模組」。 → verify：每 component 一節 + 總表。
29. OVERALL.md（Phase 1.2→已完成 ✅ + 模組化架構 + 報告路徑）、LOG.md（1.2 完成 + ONNXim/Ramulator2/CACTI/instr-count/roofline 納入 + LPDDR4-not-in-Ramulator2 釐清 + ADR-0002 偏離）。 → verify：反映完成。
30. secret-scan（`grep -rnI "hf_[A-Za-z0-9]\{20\}"` 乾淨）+ vendored sim 確認 license/體積 + commit（不動 `papers/`）+ `gh pr create` `phase-1.2`→`main`。 → verify：grep 無命中；PR 開出；通知 user 最後 review。

Outputs:
- spec：`simulator/specs/loader.py` + `{cpu_rk3588（多叢集 A76+A55）, npu_rknpu2, gpu_mali_g610, mem_lpddr4, mem_lpddr4x, mem_lpddr5, sram_metis_aipu}.json`
- 引擎：`m4_cpu.py`(instr-count)、`m4_npu.py`(ONNXim-fit)、`m2_memory.py`(swappable+analytic+Ramulator2)、`m1_cim_spm.py`(CACTI tier)、`m4_gpu_roofline.py`(新)
- 外部 sim：`tools/onnxim/`、`tools/ramulator2/`、`measurements/onnxim/rknpu2_sim_matmul.json`
- script：`tools/analysis/{fit_m4_cpu_instrcount,npu_onnxim_trace,build_m4_npu,mem_ramulator2,build_mem_report,fit_gpu_roofline,check_phase1_2}.py`、`tools/plotting/phase1_2_figs.py`、`tools/report/build_phase1_2_report.py`
- 驗證：`validation/reports/phase1.2/{m4_cpu,m4_npu,m2_memory,m4_gpu_roofline}.json`；contracts `m4_cpu/m4_npu/m2/m4_gpu.yaml` + 新 `specs.yaml`
- 圖：`docs/figures/phase1.2/{C1,N1,N2,N3,M1,M2,M3,G1}.{png,svg,pdf}`
- 報告：`docs/report/phase1.2/index.html`+`phase1.2-report.pdf`+`docs/phase1.2-findings.md`；OVERALL.md、LOG.md；PR
