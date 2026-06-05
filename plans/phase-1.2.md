# Plan: Phase 1.2 — Wave 1：模組化「engine + 可換 spec」校準-analytic 元件層

範圍：把每個非-micro-benchmark 單元建成 **模型引擎 + 可換 spec 模組**，全用**校準-analytic**（對手上的 silicon 校準），交付一個**完整、可換型號的模擬器**。重型 sim（ONNXim / Ramulator2）= **Phase 1.3**（插進本期就緒的同一介面）。分支 `phase-1.2`。執行：**序列前置 → 平行 component fan-out → 序列合併**。非-測得項標 `simulated`；CPU 對 cpu_ops.json 校準者標 `calibrated`。

## 決策（grill 2026-06-06 收斂）

- **D1 CPU**：instruction-count **roofline** `latency=max(compute_us, memory_us)+overhead_op`；`compute_us=op_count/(Σ_assigned[W·IPC·freq])·η_c`、`memory_us=bytes/(BW_tier·η_bw)`。**BW_tier 依工作集選 cache 層、非一律 LPDDR**（decode op 駐 L1/L2/L3→不碰 LPDDR；「換 LPDDR4→5 重算」只對 prefill）。成本主因 `exp()`（softmax/swiglu）用較高 ops_per_elem、不靠 reduction/elementwise 二分；`overhead_op` 為 per-op 固定開銷（rmsnorm/rope/residual 近常數）。校準對 **fp32** cpu_ops.json、報 per-op 殘差。**量測基準=單 A76 核單緒**→多核外推。多核/big.LITTLE：spec 多叢集（A76 IPC=2 / A55 IPC=1）+ 指派；A55+多核=simulated。
- **D2 NPU（本期=analytic）**：解析 systolic-roofline，參數來自 datasheet（6 TOPS）+ HeteroInfer **borrowed trend**（systolic 維借 Hexagon 32×32、標 borrowed；BW 對 RK3588 ~34×59–66%≈20–22）。**ONNXim 移 Phase 1.3**（當更好的資料源、再 fit/cross-check）。全標 `simulated`、`upgrade=#13 板/ONNXim`。
- **D3 記憶體（本期=analytic 全-spec + SRAM）**：`MemoryModel(spec, engine='analytic')` 涵蓋 LPDDR4/4X/5（頻寬+timing）；SRAM=CACTI 推得固定 (latency,BW) tier、residency `architecture-only`（8B 權重≫32MiB→永不命中）。**Ramulator2 移 Phase 1.3**。
- **D4 GPU**：Mali micro-benchmark 為主 + 可換 analytic roofline 槽。誠實：1.1 `20.12`=FP16（INT8 零資料）；Mali 峰值 512 GFLOP/s=assumption（可能低 2-4×）；roofline=形狀趨勢+下界（不可轉移校準）。
- **D5 swappable spec + 共用引擎介面**：`simulator/specs/*.json`（CPU/NPU/GPU/記憶體×3/SRAM/CIM-拓樸×2）+ loader；**共用介面 = `Engine(spec, engine='analytic'|…).predict(workload) -> {latency_us, …}`**（spec **建構時綁**、`predict` 只吃 workload；輕/重同簽名、heavy 引擎內部封裝 adapter+per-shape 快取）→ Phase 1.3 重型插進來不用改 API。
- **D6 CIM 雙變體 + Card 重驗（grill：CIM kernel 未凍結）**：CIM 計算引擎(M1)拓樸無關、共用；兩拓樸 spec（A=Alpha 無 on-card DRAM、B=Card 有）。**Metis Card 上同一顆 AIPU 活著、`axrunmodel` 可用（已 SSH 驗證可行性）**→ CIM kernel **可重新量測/驗證**（見下「CIM-Card 任務」）。Card e2e tok/s 驗 memory-wall+拓樸+弱 n_cores。

## 稽核修正清單（2026-06-06，建 spec/引擎時逐項套用）

- **CIM**：`alloc_envelope_param_count`(6M 參數量)≠`native_max_kn`(4.19M K·N)；envelope「~14MB」是假設(vs 1GB BAR)；G_eff p95 註百分位法。
- **CPU**：rope=**heads·hd**(+stack ≥2-pass)；softmax=**heads·(kv+1)**；每 op 記 bytes/passes；量測基準單核單緒(記入 spec)；A55 IPC=1；swiglu fp16=混精；無 CPU mem-BW micro-benchmark→cache/DRAM BW 達成率 gap。
- **記憶體**：34.1/4224 標 `assumption`(in-repo 無 data-rate 出處)；sim eff 0.65 vs 量測 0.71 需一句說明；PCIe 3.5 vs 3.9 殘差註。
- **GPU**：INT8 零資料；`20.12`=FP16；峰值 512=assumption；roofline=趨勢+下界；`ksweep_saturation_M` dead param 標註不刪。
- **NPU**：dtypes 僅 INT4/8/16/FP16；BW 交叉驗證引 **Fig5**；59–66% 分母=68 峰值需標明；無 RKNPU2 power→能耗不可判。

---

## 0. 序列前置（平行前必做）

1. 骨架：`docs/figures/phase1.2/`、`validation/reports/phase1.2/`、`docs/report/phase1.2/chapters/`、`simulator/specs/`。 → verify：目錄存在。
2. **共用 spec 介面** `simulator/specs/loader.py`（`load_spec(name)->dict`+provenance）。 → verify：讀任一 spec 回 dict+provenance。
2b. **共用引擎介面（具體、可在 fan-out 前驗）** `simulator/models/engine.py`：`class UnitEngine(ABC)` + `__init__(self, spec, engine='analytic')` + `@abstractmethod predict(self, wl: Workload) -> dict`；`@dataclass Workload`（op, M, K, N, kv, dtype, bytes…）；**凍結 return-dict keys = {latency_us, bound('compute'|'memory'|'floor'), provenance}**；有輕/重的單元（記憶體/NPU）預留 `engine=`（重型 Phase 1.3 插）。附 `tests/test_engine_iface.py`：dummy engine `predict(sample_wl)` 回凍結 keys。 → verify：**conformance test 通過**（dummy engine 回正確 keys）——可在平行 fan-out 前跑。各 WP 引擎只需 fill-in 同簽名。
3. **寫入已知 spec**（用研究查得的數字、每欄位 provenance）：`cpu_rk3588`（多叢集 A76@2.3G IPC2 + A55@1.8G IPC1、NEON、L1/L2/L3 容量 **+ 各層 BW（A76 cache BW，標 `assumption`：ARM A76 TRM 推估，note「≠ Metis AIPU SRAM tier，WP-CPU 只依本 spec、不依 WP-MEM」）**）、`npu_rknpu2`（3核6TOPS、INT4/8/16/FP16、systolic 維=borrowed 32×32）、`gpu_mali_g610`（4核~1G、FP32 512=assumption、FP16 1T、INT8 未量）、`mem_lpddr4/4x/5`（{3200|板4224, 4266|4224, 6400} MT/s × 64-bit = {25.6|33.8, 34.1|33.8, 51.2}、峰值標 assumption、eff 0.71）、`sram_metis_aipu`（L1 4MiB×4+L2 32MiB+D-IMC 1MiB×4、bw=CACTI/assumption）、`cim_topo_alpha`（911µs floor/PCIe3.9/~14MB/LLM-incapable）、`cim_topo_card`（on-card LPDDR4x/無 floor/24.2/多GB/LLM-capable）。 → verify：8 spec 存在、峰值 BW 數學一致、記憶體峰值標 assumption。

## 平行 fan-out（每 WP 一 subagent；依賴步驟 0、彼此獨立）

### WP-CPU（instruction-count roofline）
4. 引擎 `m4_cpu.py`：`CpuModel(spec)` 依 D1。 → verify：正/單調；decode op 走 cache 分支、加核行為正確。
5. 校準 `tools/analysis/fit_m4_cpu_instrcount.py`（fp32、確認基準核數、η_c/η_bw/overhead_op 解、per-op 殘差）→ `validation/reports/phase1.2/m4_cpu.json`。 → verify：每 op 分支+因子+誤差+誠實註。
6. 圖 `C1`（子圖分 op、X=size 變數、Y=µs、量測 vs 模型、標 reduction）。 → verify：圖產出。
7. 章節 `chapters/C-cpu.md`（架構→spec→原理→圖→白話→sim-vs-measured+可換性）。 → verify：含對照+可換性。

### WP-NPU（analytic systolic-roofline；ONNXim 在 1.3）
8. 引擎 `m4_npu.py`（取代 stub）：`NpuModel(spec)` roofline（6 TOPS ceiling + 對齊 padding + order/shape factor）+ native attn bmm。 → verify：正/單調/knee 合理/對齊維較快。
9. `tools/analysis/build_m4_npu.py`：HeteroInfer trend 量化勾稽（staircase vs Fig3、order/shape ≤6× vs Fig4、BW frac 59–66%(分母68) vs Fig5）→ `validation/reports/phase1.2/m4_npu.json`（全標 simulated；`upgrade=#13(silicon, **未滿足、僅取代-非達成**) + ONNXim(1.3, 模擬非 silicon)`）。`m4_npu.yaml`: BLOCKED→**SIMULATED**，**明寫 SIMULATED 驗收 = trend-shape 勾稽（staircase 單調+knee 落 borrowed 32×32 對齊、order/shape ≤6×、BW frac 59–66%）、無 per-op 數值 gate；#13 的 median/p95 silicon gate 標 superseded-not-satisfied**（記為 ADR-0006 gate 修訂）。 → verify：JSON 三 trend 量化條件 + SIMULATED 驗收字句 + #13/ONNXim 區分。
10. 圖 `N1`(staircase 對照 Fig3 形狀)、`N2`(offload：CIM/Mali 實線 silicon、NPU 虛線 sim)。章節 `chapters/N-npu.md`。 → verify：圖產出、NPU 標 simulated、章節含 sim-vs-reference。

### WP-MEM（analytic 全-spec + SRAM CACTI；Ramulator2 在 1.3）
11. 引擎 `m2_memory.py`：`MemoryModel(spec, engine='analytic')` 吃 mem_lpddr4/4x/5 + CIM 拓樸 spec(選 floor/envelope)；kv_append 沿用。 → verify：三 spec 正/單調、LPDDR5→33.3、LPDDR4x→24.2、拓樸A付floor/B不付。
12. SRAM `m1_cim_spm.py`：`SramTier(spec)` CACTI/assumption tier + `residency` 標 architecture-only。 → verify：8B 權重回 DRAM 層。
13. `tools/analysis/build_mem_report.py`（三 spec 對 anchor、SRAM what-if）→ `validation/reports/phase1.2/m2_memory.json`。`m2.yaml`(L1/L2 屬 M1-SPM、ADR-0002 偏離註)。 → verify：JSON+誠實標。
14. 圖 `M1`(三 spec BW)、`M3`(SRAM tier)。章節 `chapters/M-memory.md`（含 LPDDR4-not-in-Ramulator2、Ramulator2 在 1.3 的說明）。 → verify：圖+章節。

### WP-GPU（analytic roofline 槽）
15. 引擎 `m4_gpu_roofline.py`（與既有 micro-benchmark 並存）：`GpuRooflineModel(spec)`，efficiency 對 mali_matmul 校準、標下界。 → verify：對 1.1 量測點誤差記錄。
16. `tools/analysis/fit_gpu_roofline.py` → `validation/reports/phase1.2/m4_gpu_roofline.json`（micro-benchmark 為主、roofline 為換型號槽）。圖 `G1`(roofline)。章節 `chapters/G-gpu.md`（無 Mali sim 誠實說明）。 → verify：JSON+圖+章節。

### CIM-Card 重驗（量測任務；grill 確認可行）
16.5 **SPIKE（先跑、~30 分，決定 17 可行性）**：SSH Card，確認 (a) v1.6 是否仍有 Alpha 用的**低階 `compile`/`axcompile`**（Alpha `run_metis_cim.py:65` 用 `compile`、非 `deploy.py`；voyager-sdk:339 記「general MatMul 非 deploy.py 支援 op、YOLO11 whitelist」）；(b) 1×1-conv proxy 能否編出 model.json。 → verify：回報「low-level compile 在/不在 + conv proxy 編得出/編不出」——**決定步驟 17 走 compile（首選，1:1 Alpha）或退 fallback**。
17. **`characterization/metis_card/run_metis_cim_v16.py`**：移植 Alpha `run_metis_cim.py`（1×1-conv proxy）到 Card——SDK 路徑 `~/tundergod/voyager-sdk`；**編譯首選低階 `compile`（與 Alpha 同路徑）**；`compile` 已移除才退 `deploy.py <yaml> --mode QUANTCOMPILE`（+ INT8 校準資料）；`axrunmodel` 同。先跑 `aspect`+`staircase64` 幾 shape。 → verify：Card 編出 model.json + axrunmodel 回 dev fps。
18. `tools/analysis/validate_cim_card.py`：Card `dev_lat_us`/`dev_gflops`（**= dev/system split 已隔離計算，非合成差分**）**對 Alpha 13 點交叉驗證**；**兩板同為 800MHz（machines.md/voyager-sdk §1）→ 直接比、無需 rescale**（Card DVFS 非 800 則先固定 clock）；補 **prefill/compute-bound** shape。寫 `validation/reports/phase1.2/cim_card_revalidate.json`。 → verify：Card vs Alpha 同顆 AIPU 一致性數字 + prefill compute-bound 點。
> 註：需上 Card（已 SSH 驗證 AIPU 活、axrunmodel/onnx 在）。**失敗（low-level compile 不在 且 deploy.py 不支援 MatMul，或量化卡關）→ 退「Alpha 13 點 calibrated（非凍結、待板）+ Card e2e 驗 memory-wall」並回報 user**。

## 合併（序列）
19. 交叉驗證：loader 載全 spec 餵各引擎；honesty 標一致；no fake gate（CPU=calibrated、NPU/GPU-roofline/SRAM=simulated、**CIM=Alpha 13pts calibrated；+ Card-revalidated 若步驟 17/18 成功、否則 + Card-e2e-only**）。一支 `tools/analysis/check_phase1_2.py`。 → verify：跑通+列各引擎×spec sanity。
20. contracts（m4_cpu/m4_npu/m2/m4_gpu.yaml + 新 specs.yaml）。 → verify：反映引擎/spec/gate。
21. 報告 `tools/report/build_phase1_2_report.py`（複製 build_phase1_report.py、ORDER=[intro,C-cpu,N-npu,M-memory,G-gpu,CIM-card]、指 phase1.2）→ HTML→PDF。`docs/phase1.2-findings.md`（逐 component+CIM-Card 重驗+「engine+spec」總表+honesty 強度）。 → verify：HTML/PDF 產出、每章 sim-vs-reference。
22. OVERALL.md（Phase 1.2 完成 + Phase 1.3 列）、LOG.md。secret-scan + commit（不動 papers/）+ `gh pr create` phase-1.2→main + 通知 user。 → verify：grep 無命中、PR 開出。

Outputs：spec×8 + `specs/loader.py` + `models/engine.py`（UnitEngine ABC + Workload）+ `tests/test_engine_iface.py`；引擎 m4_cpu/m4_npu/m2_memory/m1_cim_spm/m4_gpu_roofline；CIM-Card `run_metis_cim_v16.py`+`validate_cim_card.py`；scripts（fit/build/check）；validation/reports/phase1.2/*.json；圖 C1/N1/N2/M1/M3/G1；報告 index.html+pdf+findings；contracts；OVERALL/LOG；PR。
