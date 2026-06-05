# Plan: Phase 1.3 — Wave 2：重型保真模擬引擎（ONNXim / Ramulator2）

範圍：把兩個 **C++ 重型 sim** 插進 Phase 1.2 已就緒的**共用引擎介面**（`UnitEngine.predict(workload,spec)`），當**更高保真的可換引擎**：**ONNXim**（NPU）、**Ramulator2**（記憶體 DRAM）。皆標 `simulated, not silicon-validated`。**前提**：Phase 1.2 已 merge（spec 層 + 共用介面 + analytic 引擎齊備）。分支 `phase-1.3`。可平行（ONNXim ∥ Ramulator2）。

> **定位（grill 收斂）**：重型 sim 的招牌價值（Ramulator2 的多單元競爭；逐 token 整機）要 **Phase 2** 才被用到；本期角色 = (i) **交叉比對 Phase 1.2 的 analytic 單串流趨勢**、(ii) **介面就緒，Phase 2 一接就能用、silicon 一回來就能驗**。即時報酬偏低、真正報酬在 Phase 2——「先建好就緒」的取捨。**單點失敗（C++ build 爆）退 documented fallback、不影響已 merge 的 1.2。**

## 決策（已定）

- **D1 ONNXim（NPU heavy）**：generic systolic-NPU 模擬器配成 RKNPU2-approx（讀 `npu_rknpu2.json`，systolic 維借 Hexagon 32×32 標 borrowed）；當 NPU 的**模擬資料源** → 重 fit / cross-check Phase 1.2 的解析 roofline；HeteroInfer trend 仍交叉驗證。build 失敗 → OVERALL.md risk #7 fallback（解析-only / lookup-override）+ 回報 user。**非對 RKNPU2 silicon 驗證**（同 1.2 honesty）。
- **D2 Ramulator2（記憶體 heavy）**：接其內建 **LPDDR5**（subprocess + **代表性迭代 at 多個 KV 長度**（ADR-0002:128/512/1024，取 per-token-vs-KV 曲線）→ 有效 BW/latency → 解析外推；per-shape 快取）；驗 Phase 1.2 analytic 單串流 BW/latency。**LPDDR4/4X 的 C++ port = 後續**（`assumption`：Ramulator2 出貨 preset 疑只有 LPDDR5、無 LPDDR4——**build 時看 `src/dram/impl/` 確認**，非既定事實）。**SRAM 仍非 Ramulator2**（1.2 CACTI tier）。
- **D3 同介面**：兩者實作 1.2 定的 `Engine(spec, engine=).predict(workload) -> {latency_us,…}`（spec 建構綁、predict 只吃 workload），`engine='onnxim'|'ramulator2'` 與 1.2 `'analytic'` drop-in 互換；heavy 引擎封裝 adapter（shape→ONNX / bytes→trace）+ per-shape 快取。

---

## 0. 前置
1. **前置 gate**：確認 1.2 介面已 merge——`from simulator.models.m2_memory import MemoryModel; MemoryModel(load_spec('mem_lpddr5'), engine='analytic').predict(wl)` 成功（證明 `spec/engine=/predict` API 在、非 1.1 stub）；**API 不在 → STOP + 回報 user，不對 1.1 stub 建**。建 `tools/onnxim/`、`tools/ramulator2/`、`simulated/onnxim/`。 → verify：上述 import+predict 成功、目錄存在。

## A. ONNXim（NPU heavy；可與 B 平行）
2. ONNXim 取得+建置 `tools/onnxim/`（vendored/submodule）+ RKNPU2-approx config（讀 1.2 的 `npu_rknpu2.json`）。 → verify：build 成功 + smoke matmul；**失敗→記 risk#7 fallback + 回報 user**。
3. `tools/analysis/npu_onnxim_trace.py`：由 `measurements/op_inventory/` NPU shapes export ONNX → ONNXim 輸入（ADR-0007：export 次要、fallback=traced graph 直建）。 → verify：對 NPU shapes 產出輸入。
4. 跑 ONNXim → `simulated/onnxim/rknpu2_sim_matmul.json`（標 `simulated (generic-systolic, RKNPU2-approx), NOT silicon`；放 `simulated/` 非 `measurements/`，避免 sim 與 silicon 混）。 → verify：每 shape 有延遲+simulated 標。
5. **編輯 `simulator/models/m4_npu.py` 加 `engine='onnxim'` 分支**（dispatch 到 ONNXim adapter：封閉式對 ONNXim 表 fit + per-shape 快取）；與 1.2 `engine='analytic'` 同建構簽名、drop-in。 → verify：`NpuModel(spec, engine='onnxim').predict(wl)` 同 keys、復現 ONNXim 延遲（fit median/p95）、快取命中。
6. `build_m4_npu_onnxim.py`：**逐 shape ONNXim-vs-1.2-analytic delta** + HeteroInfer trend → `validation/reports/phase1.3/m4_npu_onnxim.json`（simulated；`upgrade=#13(silicon, 獨立、未滿足)`；ONNXim≠#13）。圖 `N3`(analytic vs ONNXim vs HeteroInfer trend)。章節 `chapters/N-npu-onnxim.md`。 → verify：JSON **含每 shape 的 analytic-vs-ONNXim delta 欄**（非只存在）+ 圖 + 章節 + #13 標 silicon-獨立。

## B. Ramulator2（記憶體 heavy；可與 A 平行）
7. Ramulator2 取得+建置 `tools/ramulator2/`（vendored/submodule）+ LPDDR5 config（對齊 1.2 `mem_lpddr5.json`）。 → verify：build 成功 + 對一 LPDDR5 trace 回有效 BW；**失敗→analytic 為主、回報 user**。
8. `tools/analysis/mem_ramulator2.py`：subprocess + 代表性迭代（一 decode/prefill 迭代 trace → 有效 BW/latency）+ per-shape 快取。 → verify：對代表 shape 回有效 BW/latency。
9. **編輯 `simulator/models/m2_memory.py` 加 `engine='ramulator2'` 分支**（dispatch 到 Ramulator2 adapter + per-shape 快取）；與 1.2 `engine='analytic'` 同建構簽名、drop-in。 → verify：`MemoryModel(spec, engine='ramulator2').predict(wl)` 同 keys、config 互換、快取命中。
10. `build_mem_ramulator2.py`：**逐 shape Ramulator2-vs-1.2-analytic（LPDDR5 單串流）delta** → `validation/reports/phase1.3/m2_ramulator2.json`（標：單串流 analytic 已夠準、Ramulator2 招牌競爭在 Phase 2）。圖 `M2-ramulator2`(Ramulator2 vs analytic)。章節 `chapters/M-memory-ramulator2.md`。 → verify：JSON **含 delta 欄** + 圖 + 章節。

## C. 合併
11. 交叉驗證：`engine='analytic'|'onnxim'|'ramulator2'` 三者 drop-in 互換、結果形狀一致、honesty 標一致。 → verify：`check_phase1_3.py` 跑通。
12. 報告 `build_phase1_3_report.py`（章節：ONNXim、Ramulator2、與 1.2 analytic 對照）→ HTML→PDF。`docs/phase1.3-findings.md`。 → verify：HTML/PDF、每章 sim-vs-analytic。
13. OVERALL.md（Phase 1.3 完成）、LOG.md。**reconcile「Ramulator2 階段」**：ADR-0002 加一行修訂（單串流 LPDDR5 cross-check 落 1.3、多單元競爭用 Phase 2）+ 修 OVERALL risk #6 措辭 + 標 `m2_memory.py` docstring 與 `docs/phase1.1-findings.md:19`、`A2-m2-memory.md` 的「Ramulator2→Phase 2」為待更新（follow-up）。secret-scan + commit（不動 papers/）+ `gh pr create` phase-1.3→main + 通知 user。 → verify：grep 無命中、PR 開出、ADR-0002/OVERALL/findings 不再與「Ramulator2 在 1.3」矛盾。

Outputs：`tools/onnxim/`+`tools/ramulator2/`（vendored）；`measurements/onnxim/rknpu2_sim_matmul.json`；heavy 引擎接 `m4_npu.py`/`m2_memory.py`（`engine='onnxim'|'ramulator2'`）；scripts（npu_onnxim_trace/mem_ramulator2/build_*/check_phase1_3）；validation/reports/phase1.3/*.json；圖 N3/M2；報告 index.html+pdf+findings；OVERALL/LOG；PR。
