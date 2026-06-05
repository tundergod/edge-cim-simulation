# Plan: Phase 1.2 — 模擬補完 component（NPU / 完整 CPU / 記憶體子系統）

範圍：建模 + 驗證 **完整 CPU、NPU、記憶體子系統**。執行序：CPU → NPU →（記憶體，視 D2）。每 component 做完開 subagent 驗證、討論至無問題才進下一個。分支 `phase-1.2`。所有 1.2 模型標 `simulated, not silicon-validated`。

## 決策（approval 時定，先不執行）

- **D1（NPU 路徑）**：解析 systolic-roofline 模型為 v1；ONNXim 延後 Phase 2。步驟 C 不含 ONNXim build。
- **D2（記憶體子系統範圍）**：LPDDR5 SoC streaming 在 1.1 已建（eff 33.3）；本期記憶體新增工作 = **Metis AIPU SPM（L1 4MiB/core + L2 32MiB；屬 CIM/M1 的 software-managed scratchpad、非 SoC DRAM、非 cache）residency 旋鈕**。對 batch=1 INT8 dense decode，權重 8GB ≫ 32MiB → **residency 永不命中、恆走 LPDDR5**（純架構研究旋鈕）；ADR-0002 將記憶體階層列 Phase 2。**選項 (a)** 1.2 建此 SPM 旋鈕（步驟 B2–B7）；**選項 (b)** 延後 Phase 2（與 ADR-0002 一致；刪步驟 B2–B7，記憶體component 僅留 LPDDR5 整理 B1）。**reviewer 建議 (b)**（never-binds + ADR-0002 + 簡化）；user 先前明示要建 L1/L2 → 於 approval 定。註：選 (b) 則 1.2 實質 = NPU + CPU。
- **D3（A/B topology sensitivity）**：系統級，延後 Phase 2（ADR-0006 validate-then-swap 在整合層才成立）；不在 1.2。

## 0. 共用前置

1. 建輸出骨架：`docs/figures/phase1.2/`、`validation/reports/phase1.2/`、`docs/report/phase1.2/chapters/`。 → verify：目錄存在。

---

## A. 完整 CPU 計算模型（M4-CPU 擴建）

2. `tools/analysis/fit_m4_cpu_full.py`：讀 `measurements/aetina/cpu_ops.json`，每 op 用其物理 size 變數建模：rmsnorm/rope_apply/residual=`a+b·H`、softmax=`a+b·kv`、swiglu=`a+b·F`（F=FFN intermediate dim）、sampling_argmax=`a+b·V`（V=vocab）；近常數者標 `const`。寫 `simulator/models/params/m4_cpu_full.json`（每 op：`{size_var, form: linear|const, params, n_points, source}`）。 → verify：JSON 每 op 有 size_var + form；swiglu size_var=F、sampling size_var=V。
3. fit script 內加 **leave-one-out 弱 sanity**（非 1.1 silicon gate）：每 linear op fit 3 模型→預測第 4，輸出全 4 個原始相對誤差（不報 p95）；swiglu 僅 3 個相異 F、sampling 近常數 → 預設 `const-fallback`。寫 `validation/reports/phase1.2/m4_cpu_full.json`，每 op 標 `{form, 4_raw_errors|const, label: "weak-sanity n=4, not 1.1 silicon gate"}`。 → verify：JSON 每 op 列 4 原始誤差或 const 標記 + weak-sanity 標籤；swiglu/sampling=const-fallback。
4. 擴 `simulator/models/m4_cpu.py`：`op_us()` linear op 接受其 size 變數、const op 回常數、超量測範圍回值帶 `extrapolated=True`；per-op form 讀自 `m4_cpu_full.json`。 → verify：各 op 回正數、對量測點誤差=步驟 3、swiglu/sampling 走 const。
5. 更新 `validation/contracts/m4_cpu.yaml`：每 op 標 size_var + `form: linear|const-fallback`；softmax 維持 median≤10%/p95≤20% gate；加註「const op 的跨模型 shared-slope 是比 1.1 per-model 弱的假設」。 → verify：contract 列每 op size_var/form + shared-slope 註記。
6. **圖 C1**（`/nature-figure`，`tools/plotting/phase1_2_figs.py`）：子圖分 op；**X**：該 op size 變數（H / kv / F / V）；**Y**：latency（µs，fp16=上界）；量測點(dots, n=4) vs 模型(line)，const-fallback op 標「查表、不擬合」。 → verify：`docs/figures/phase1.2/C1_cpu_full.{png,svg,pdf}` 產出。
7. subagent code-review + 精準度確認（CPU）；無 blocking 才進 B。 → verify：subagent 回報無 blocking。

---

## B. NPU（M4-NPU 解封；解析 systolic 模型，無 silicon）

8. `simulator/models/params/m4_npu.json`：RKNPU2=3-core 6 TOPS INT8 1.0GHz systolic（source=datasheet）；roofline 參數 — compute ceiling 6 TOPS；systolic 維（`assumption`，note 記 RKNPU2 未公開、借 Hexagon）；對齊 padding staircase（`borrowed: HeteroInfer Fig3`）；order/shape factor ≤6×（`borrowed: Fig4`）；memory BW = RK3588 LPDDR4-4266×64 理論峰值 ~34 GB/s × 單-proc 59–66% ≈ 20–22 GB/s（note：理論峰值非 RK3588 實測；分數借 HeteroInfer Fig5）。 → verify：JSON 各參數帶 `source|assumption|borrowed` 標；BW 由 ~34 推得、非 40–45。
9. 實作 `simulator/models/m4_npu.py`（取代 `NotImplementedError`）：`NpuModel.matmul_us=max(compute_systolic, memory)`（roofline + 對齊 padding + order/shape factor）；native attention bmm（offload 第二候選）。 → verify：回正數、roofline knee 合理、對齊維比未對齊快、order factor 生效。
10. `tools/analysis/build_m4_npu.py`：NPU 預測表 + HeteroInfer 定性 trend 勾稽（量化條件）：(a) 對齊 staircase 週期=假設 systolic 維；(b) order factor ≤6×；(c) 單-proc BW 落 20–22。寫 `validation/reports/phase1.2/m4_npu.json`（三 trend 條件 + 全標 `simulated, not silicon-validated` + `upgrade_path: #13 → rknpu2_matmul.json 數值 fit`）。 → verify：JSON 三 trend 量化勾稽 + simulated 標 + upgrade_path。
11. 更新 `validation/contracts/m4_npu.yaml`：`status: BLOCKED`→`SIMULATED`；保留 `#13` 為 upgrade path；acceptance=trend-cross-check（標明非 median/p95、因無 silicon）。 → verify：contract 反映 simulated + 數值 gate 待 #13。
12. **圖 N1 + N2**：N1 staircase — **X**：輸出維（對齊到假設 systolic 維）；**Y**：eff 吞吐（GOP/s），標 padding penalty + `borrowed: Fig3`。N2 offload 三方 — **X**：kv 長度；**Y**：每 token attention latency（ms，log）；CIM/Mali=實線(silicon)、NPU=虛線(simulated)。 → verify：`docs/figures/phase1.2/N1_npu_staircase.*`、`N2_npu_offload.*` 產出；NPU 曲線視覺標 simulated。
13. subagent code-review + 精準度確認（NPU，重點查「無 silicon 卻假裝有 gate」「Hexagon→RKNPU2 借用是否誠實標」）；無 blocking 才進 C/D。 → verify：subagent 回報無 blocking。

---

## C. 記憶體子系統（視 D2；含 LPDDR5 整理 + [選項 a] Metis SPM residency）

14. `tools/analysis/fit_m2_lpddr5.py`：整理 1.1 LPDDR5 SoC streaming（eff 33.3=51.2×0.65），對 L4 anchor（LPDDR4x 24.2=71% peak）+ datasheet/HeteroInfer（單-proc 59–66%）勾稽；kv_append 沿用 1.1 解析式。寫 `validation/reports/phase1.2/m2_lpddr5.json`（LPDDR5 vs L4/datasheet 一致、kv 標 UNVALIDATED）。 → verify：JSON 列 LPDDR5 一致性 + kv unvalidated。
15. **[D2=a]** `simulator/models/params/m1_cim_spm.json`：L1 4MiB/core、L2 32MiB（`capacity source=ISSCC-2024`）；`bw_GBs` 標 `assumption`（note：ISSCC 未列 GB/s，以 on-chip SRAM 通則估 + 數值依據）。 → verify：JSON 各層 capacity(source) + bw(assumption flag)；無「論文為 BW 來源」字樣。
16. **[D2=a]** `simulator/models/m1_cim_spm.py`：`spm_residency(working_set_bytes)`→(層, eff BW)；docstring 標 `Metis AIPU SPM, software-managed, not a cache, M1/CIM, architecture-only`。 → verify：函式單調、正值；8B 權重 working_set 回 LPDDR5 層。
17. **[D2=a]** `tools/analysis/build_m1_spm_whatif.py`：架構 what-if（用 SRAM BW 重算 decode tok/s）。寫 `validation/reports/phase1.2/m1_cim_spm.json`（標 `architecture-only, never binds for in-scope batch=1 INT8 decode, simulated` + what-if delta）。 → verify：JSON 標 never-binds + simulated + delta。
18. contracts：`m2.yaml` 註 L1/L2 屬 M1-SPM（非 M2）、LPDDR5 標已驗(對 L4)；**[D2=a]** `m1.yaml` 加 `cim_spm_residency: built 1.2, simulated, architecture-only` + ADR-0002 偏離註（階層原列 Phase 2；本期僅建 architecture 旋鈕、Ramulator2 後端仍 Phase 2）。 → verify：contract 反映正確歸屬 + simulated +（選 a 時）ADR 偏離註。
19. **[D2=a] 圖 D1 + D2**：D1 SPM 階層 — **X**：working-set bytes(log)；**Y**：eff BW(GB/s)，標 L1/L2/LPDDR5 階梯 + 「in-scope 權重落 LPDDR5」。D2 what-if — **X**：記憶體配置(LPDDR5 實際 / 全放 L2 反事實)；**Y**：8B decode tok/s，標反事實/simulated。 → verify：`docs/figures/phase1.2/D1_spm_tiers.*`、`D2_l2_whatif.*` 產出且標反事實。
20. subagent code-review + 精準度確認（記憶體，重點查 M1/M2 歸屬、never-binds 揭露）；無 blocking 才進 D。 → verify：subagent 回報無 blocking。

---

## D. 報告 + findings + 收尾

21. `docs/phase1.2-findings.md`：逐 component {模型形式 + 參數 + 驗證方式 + 結果 + `simulated` 標 + upgrade path}；明寫驗證強度差異（CPU=4 點 leave-one-out 弱 sanity + shared-slope 比 1.1 弱；NPU=借 Hexagon trend、無 silicon；[選 a] SPM=architecture-only never-binds）。 → verify：每 component 一節含數字/trend + 誠實標 + 強度說明。
22. `docs/report/phase1.2/chapters/`：beginner-friendly 中文，每 component 獨立章（架構→參數→原理→圖→白話→sim-vs-reference），開頭說明「為何是模擬、gate 較弱、與 1.1 量測級之別」。`tools/report/build_phase1_2_report.py`=**複製** `build_phase1_report.py`（CH/FIG/OUT/ORDER 指 phase1.2）。 → verify：`docs/report/phase1.2/index.html` 產出、圖內嵌、每章有 sim-vs-reference 段。
23. HTML→PDF（headless Chrome → `docs/report/phase1.2/phase1.2-report.pdf`）。 → verify：PDF 非空、頁數>0、圖在。
24. OVERALL.md（Phase 1.2 → 已完成 ✅ + 報告路徑）、LOG.md（1.2 完成 + L1/L2 屬 M1-SPM 釐清 +[選 a] ADR-0002 偏離）。 → verify：OVERALL/LOG 反映完成。
25. secret-scan（`grep -rnI "hf_[A-Za-z0-9]\{20\}"` 乾淨）+ commit（不動 `papers/`）+ `gh pr create` `phase-1.2`→`main`，摘要含 component gate/trend + simulated 標 +[選 a] ADR-0002 偏離。 → verify：grep 無命中；PR 開出；通知 user 最後 review。

Outputs（隨 D2；選 b 時刪 SPM 項）:
- 模型：`m4_cpu.py`(擴)、`m4_npu.py`(實作)、[a]`m1_cim_spm.py`(新) + params `m4_cpu_full.json`、`m4_npu.json`、[a]`m1_cim_spm.json`
- 擬合/建構：`tools/analysis/{fit_m4_cpu_full,build_m4_npu,fit_m2_lpddr5}.py`、[a]`build_m1_spm_whatif.py`、`tools/plotting/phase1_2_figs.py`、`tools/report/build_phase1_2_report.py`
- 驗證：`validation/reports/phase1.2/{m4_cpu_full,m4_npu,m2_lpddr5}.json`、[a]`m1_cim_spm.json`；contracts `m4_cpu.yaml`、`m4_npu.yaml`、`m2.yaml`、[a]`m1.yaml` 更新
- 圖：`docs/figures/phase1.2/{C1,N1,N2}.{png,svg,pdf}`、[a]`{D1,D2}.*`（NPU 標 simulated）
- 報告：`docs/report/phase1.2/index.html` + `phase1.2-report.pdf` + `docs/phase1.2-findings.md`；OVERALL.md、LOG.md；PR `phase-1.2`→`main`
