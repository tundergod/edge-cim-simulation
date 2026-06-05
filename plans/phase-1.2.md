# Plan: Phase 1.2 — 模擬補完 component（NPU+ONNXim / 完整 CPU / 記憶體子系統）

範圍：建模 + 驗證 **完整 CPU、NPU（含 ONNXim）、記憶體子系統**。執行序：CPU → NPU → 記憶體。每 component 做完開 subagent 驗證、討論至無問題才進下一個。分支 `phase-1.2`。所有 1.2 模型標 `simulated, not silicon-validated`。

## 決策（approval 時定，先不執行）

- **D1（NPU 路徑）— 已定（user）：ONNXim 在 1.2 建**。ONNXim（generic systolic-NPU 模擬器）配成 RKNPU2-approx，當 NPU 的**模擬資料來源**（補 #13 缺的 `rknpu2_matmul.json` silicon）；封閉式 `m4_npu.py` 對 ONNXim 表擬合（比照 1.1 m4_gpu 對 mali 量測）；HeteroInfer 為 trend 交叉驗證。ONNXim 非對 RKNPU2 silicon 驗證（systolic 維等為 assumption）→ 仍標 `simulated`。若環境無法 build ONNXim → OVERALL.md risk #7（per ADR-0007）fallback（解析 systolic / lookup-override）並回報 user。
- **D2（記憶體子系統範圍）— 已定（user）：建**。LPDDR5 SoC streaming 在 1.1 已建（eff 33.3）；本期新增 = **Metis AIPU SPM（L1 4MiB/core + L2 32MiB；屬 CIM/M1 的 software-managed scratchpad、非 SoC DRAM、非 cache）residency 旋鈕**，標 `architecture-only`（batch=1 INT8 decode 權重 8GB ≫ 32MiB → in-scope 永不命中、恆走 LPDDR5）+ `simulated` + ADR-0002 偏離註（階層原列 Phase 2、本期僅建旋鈕、Ramulator2 後端仍 Phase 2）。
- **D3（A/B topology sensitivity）**：系統級，延後 Phase 2（ADR-0006 validate-then-swap 在整合層才成立）；不在 1.2。

## 0. 共用前置

1. 建輸出骨架：`docs/figures/phase1.2/`、`validation/reports/phase1.2/`、`docs/report/phase1.2/chapters/`、`measurements/onnxim/`。 → verify：目錄存在。

---

## A. 完整 CPU 計算模型（M4-CPU 擴建）

2. `tools/analysis/fit_m4_cpu_full.py`：讀 `measurements/aetina/cpu_ops.json`，每 op 用其物理 size 變數建模：rmsnorm/rope_apply/residual=`a+b·H`、softmax=`a+b·kv`、swiglu=`a+b·F`（F=FFN intermediate dim）、sampling_argmax=`a+b·V`（V=vocab）；近常數者標 `const`。寫 `simulator/models/params/m4_cpu_full.json`（每 op：`{size_var, form: linear|const, params, n_points, source}`）。 → verify：JSON 每 op 有 size_var + form；swiglu size_var=F、sampling size_var=V。
3. fit script 內加 **leave-one-out 弱 sanity**（非 1.1 silicon gate）：每 linear op fit 3 模型→預測第 4，輸出全 4 個原始相對誤差（不報 p95）；swiglu 僅 3 個相異 F、sampling 近常數 → 預設 `const-fallback`。寫 `validation/reports/phase1.2/m4_cpu_full.json`，每 op 標 `{form, 4_raw_errors|const, label: "weak-sanity n=4, not 1.1 silicon gate"}`。 → verify：JSON 每 op 列 4 原始誤差或 const 標記 + weak-sanity 標籤；swiglu/sampling=const-fallback。
4. 擴 `simulator/models/m4_cpu.py`：`op_us()` linear op 接受其 size 變數、const op 回常數、超量測範圍回值帶 `extrapolated=True`；per-op form 讀自 `m4_cpu_full.json`。 → verify：各 op 回正數、對量測點誤差=步驟 3、swiglu/sampling 走 const。
5. 更新 `validation/contracts/m4_cpu.yaml`：每 op 標 size_var + `form: linear|const-fallback`；softmax 維持 median≤10%/p95≤20% gate；加註「const op 的跨模型 shared-slope 是比 1.1 per-model 弱的假設」。 → verify：contract 列每 op size_var/form + shared-slope 註記。
6. **圖 C1**（`/nature-figure`，`tools/plotting/phase1_2_figs.py`）：子圖分 op；**X**：該 op size 變數（H/kv/F/V）；**Y**：latency（µs，fp16=上界）；量測點(dots, n=4) vs 模型(line)，const-fallback op 標「查表、不擬合」。 → verify：`docs/figures/phase1.2/C1_cpu_full.{png,svg,pdf}` 產出。
7. subagent code-review + 精準度確認（CPU）；無 blocking 才進 B。 → verify：subagent 回報無 blocking。

---

## B. NPU（M4-NPU 解封；ONNXim 模擬資料 + 封閉式模型）

8. ONNXim 取得+建置到 `tools/onnxim/`（vendored / submodule）+ RKNPU2-approx config（systolic 維=`assumption`，RKNPU2 未公開→預設借 Hexagon 32×32、標 `borrowed`；6 TOPS、1.0GHz、INT8；mem BW=RK3588 LPDDR4-4266×64 理論峰值 ~34 GB/s × 單-proc 59–66%）。 → verify：ONNXim build 成功 + smoke matmul 跑出延遲；config 記 RKNPU2-approx 參數帶 `assumption`/`borrowed` 標。**若無法 build** → 記 OVERALL.md risk #7（per ADR-0007）fallback（解析 systolic / lookup-override）+ 回報 user（不靜默改路徑）。
9. `tools/analysis/npu_onnxim_trace.py`：由 `measurements/op_inventory/` 的 NPU-bound shapes（proj matmul + attention bmm）export ONNX 子圖 → ONNXim 輸入（ADR-0007：ONNX export 次要、fallback=由 traced graph 直建輸入）。 → verify：對 NPU shapes 產出 ONNXim 輸入；fallback 路徑註記。
10. 跑 ONNXim → per-shape NPU 延遲表 `measurements/onnxim/rknpu2_sim_matmul.json`（#13 缺的 silicon 的模擬替身）。 → verify：JSON 每 NPU shape 有延遲；標 `simulated (ONNXim generic-systolic, RKNPU2-approx config), NOT silicon`。
11. `simulator/models/params/m4_npu.json` + 實作 `simulator/models/m4_npu.py`（取代 `NotImplementedError`）：對 ONNXim 表**擬合封閉式**（FLOPs/G_eff + native attn bmm，比照 m4_gpu）；roofline 形式（compute ceiling 6 TOPS + 對齊 padding + memory）。 → verify：模型復現 ONNXim 延遲（fit median/p95 記錄）；正值/單調/roofline-knee 合理；attn bmm 存在。
12. `tools/analysis/build_m4_npu.py`：(a) 封閉式 vs ONNXim fit 誤差；(b) **HeteroInfer trend 勾稽**（量化）——ONNXim staircase 對齊週期 vs Fig3、order/shape factor ≤6× vs Fig4、單-proc BW frac 59–66% vs Fig5。寫 `validation/reports/phase1.2/m4_npu.json`（fit 誤差 + 三 trend + 全標 `simulated, not silicon-validated` + `upgrade_path: #13 → rknpu2_matmul.json 取代 ONNXim 表、再 fit + 驗 ONNXim`）。 → verify：JSON 列 fit 誤差 + trend 量化勾稽 + simulated + upgrade_path。
13. 更新 `validation/contracts/m4_npu.yaml`：`status: BLOCKED`→`SIMULATED (ONNXim-based)`；acceptance=封閉式-vs-ONNXim fit 誤差 + HeteroInfer trend（標明**非** silicon median/p95、數值 silicon gate 待 #13）；保留 `#13` upgrade。 → verify：contract 反映 ONNXim-based simulated + silicon gate 待 #13。
14. **圖 N1+N2+N3**：N1 ONNXim staircase — **X**：輸出維（對齊到 config systolic 維）；**Y**：eff 吞吐（GOP/s），caption 標「staircase 形狀對照 HeteroInfer Fig3、非單位對單位」。N2 offload 三方 — **X**：kv 長度；**Y**：每 token attn latency（ms,log）；CIM/Mali=實線(silicon)、NPU=虛線(simulated/ONNXim)。N3 封閉式 vs ONNXim — **X**：shape；**Y**：latency（µs），點(ONNXim) vs 線(封閉式)。 → verify：`docs/figures/phase1.2/{N1_npu_staircase,N2_npu_offload,N3_npu_fit}.*` 產出；NPU/ONNXim 曲線視覺標 simulated。
15. subagent code-review + 精準度確認（NPU，重點查 ONNXim 的 RKNPU2-approx 假設誠實標、generic-systolic 侷限揭露、無 silicon 卻假裝有 gate、Hexagon trend 借用標明）；無 blocking 才進 C。 → verify：subagent 回報無 blocking。

---

## C. 記憶體子系統（LPDDR5 整理 + Metis SPM residency）

16. `tools/analysis/fit_m2_lpddr5.py`：整理 1.1 LPDDR5 SoC streaming（eff 33.3=51.2×0.65），對 L4 anchor（LPDDR4x 24.2=71% peak）+ datasheet/HeteroInfer（單-proc 59–66%）勾稽；kv_append 沿用 1.1 解析式。寫 `validation/reports/phase1.2/m2_lpddr5.json`（LPDDR5 vs L4/datasheet 一致、kv 標 UNVALIDATED）。 → verify：JSON 列 LPDDR5 一致性 + kv unvalidated。
17. `simulator/models/params/m1_cim_spm.json`：L1 4MiB/core、L2 32MiB（`capacity source=ISSCC-2024`）；`bw_GBs` 標 `assumption`（note：ISSCC 未列 GB/s，以 on-chip SRAM 通則估 + 數值依據）。 → verify：JSON 各層 capacity(source) + bw(assumption flag)；無「論文為 BW 來源」字樣。
18. `simulator/models/m1_cim_spm.py`：`spm_residency(working_set_bytes)`→(層, eff BW)；docstring 標 `Metis AIPU SPM, software-managed, not a cache, M1/CIM, architecture-only`。 → verify：函式單調、正值；8B 權重 working_set 回 LPDDR5 層。
19. `tools/analysis/build_m1_spm_whatif.py`：架構 what-if（用 SRAM BW 重算 decode tok/s）。寫 `validation/reports/phase1.2/m1_cim_spm.json`（標 `architecture-only, never binds for in-scope batch=1 INT8 decode, simulated` + what-if delta）。 → verify：JSON 標 never-binds + simulated + delta。
20. contracts：`m2.yaml` 註 L1/L2 屬 M1-SPM（非 M2）、LPDDR5 標已驗(對 L4)；`m1.yaml` 加 `cim_spm_residency: built 1.2, simulated, architecture-only` + ADR-0002 偏離註（階層原列 Phase 2；本期僅建 architecture 旋鈕、Ramulator2 後端仍 Phase 2）。 → verify：contract 反映正確歸屬 + simulated + ADR 偏離註。
21. **圖 D1 + D2**：D1 SPM 階層 — **X**：working-set bytes(log)；**Y**：eff BW(GB/s)，標 L1/L2/LPDDR5 階梯 + 「in-scope 權重落 LPDDR5」。D2 what-if — **X**：記憶體配置(LPDDR5 實際 / 全放 L2 反事實)；**Y**：8B decode tok/s，標反事實/simulated。 → verify：`docs/figures/phase1.2/{D1_spm_tiers,D2_l2_whatif}.*` 產出且標反事實。
22. subagent code-review + 精準度確認（記憶體，重點查 M1/M2 歸屬、never-binds 揭露）；無 blocking 才進 D。 → verify：subagent 回報無 blocking。

---

## D. 報告 + findings + 收尾

23. `docs/phase1.2-findings.md`：逐 component {模型形式 + 參數 + 驗證方式 + 結果 + `simulated` 標 + upgrade path}；明寫驗證強度差異（CPU=4 點 leave-one-out 弱 sanity + shared-slope 比 1.1 弱；NPU=ONNXim 模擬資料(generic-systolic, RKNPU2-approx 假設)+ HeteroInfer trend、無 silicon；SPM=architecture-only never-binds）。 → verify：每 component 一節含數字/trend + 誠實標 + 強度說明。
24. `docs/report/phase1.2/chapters/`：beginner-friendly 中文，每 component 獨立章（架構→參數→原理→圖→白話→sim-vs-reference），開頭說明「為何是模擬、gate 較弱、與 1.1 量測級之別；NPU 用 ONNXim 模擬資料而非 silicon」。`tools/report/build_phase1_2_report.py`=**複製** `build_phase1_report.py`（CH/FIG/OUT/ORDER 指 phase1.2）。 → verify：`docs/report/phase1.2/index.html` 產出、圖內嵌、每章有 sim-vs-reference 段。
25. HTML→PDF（headless Chrome → `docs/report/phase1.2/phase1.2-report.pdf`）。 → verify：PDF 非空、頁數>0、圖在。
26. OVERALL.md（Phase 1.2 → 已完成 ✅ + 報告路徑 + NPU 用 ONNXim 模擬資料）、LOG.md（1.2 完成 + ONNXim 納入 + L1/L2 屬 M1-SPM 釐清 + ADR-0002 偏離）。 → verify：OVERALL/LOG 反映完成。
27. secret-scan（`grep -rnI "hf_[A-Za-z0-9]\{20\}"` 乾淨）+ commit（不動 `papers/`；ONNXim 若 vendored 確認 license/體積）+ `gh pr create` `phase-1.2`→`main`，摘要含 component gate/trend + simulated 標 + ONNXim 用途 + ADR-0002 偏離。 → verify：grep 無命中；PR 開出；通知 user 最後 review。

Outputs:
- 模型：`m4_cpu.py`(擴)、`m4_npu.py`(實作)、`m1_cim_spm.py`(新) + params `m4_cpu_full.json`、`m4_npu.json`、`m1_cim_spm.json`
- ONNXim：`tools/onnxim/`(vendored + RKNPU2-approx config)、`tools/analysis/npu_onnxim_trace.py`、`measurements/onnxim/rknpu2_sim_matmul.json`
- 擬合/建構：`tools/analysis/{fit_m4_cpu_full,build_m4_npu,fit_m2_lpddr5,build_m1_spm_whatif}.py`、`tools/plotting/phase1_2_figs.py`、`tools/report/build_phase1_2_report.py`
- 驗證：`validation/reports/phase1.2/{m4_cpu_full,m4_npu,m2_lpddr5,m1_cim_spm}.json`；contracts `m4_cpu.yaml`、`m4_npu.yaml`、`m2.yaml`、`m1.yaml` 更新
- 圖：`docs/figures/phase1.2/{C1,N1,N2,N3,D1,D2}.{png,svg,pdf}`（NPU/ONNXim 標 simulated）
- 報告：`docs/report/phase1.2/index.html` + `phase1.2-report.pdf` + `docs/phase1.2-findings.md`；OVERALL.md、LOG.md；PR `phase-1.2`→`main`
