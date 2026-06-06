# Phase 1.2 — 模組化「引擎 + 可換 spec」元件層

Phase 1.1 把**自家 Metis silicon 量得到**的核心 decode 路徑（CIM 算 + 記憶體搬）校到量測級可信。Phase 1.2 把模擬器補成**完整、可換型號**的元件層：每一個非-micro-benchmark 單元都改寫成「一個**模型引擎** + 一份**可換的 spec 檔**」——**換型號 = 換 spec 檔，引擎程式碼不動**。

## 一條凍結的共用介面

所有單元共用同一個介面（`simulator/models/engine.py`）：

```python
Engine(spec, engine='analytic').predict(workload) -> {latency_us, bound, provenance}
```

- **spec 在建構時綁定**，`predict()` 只吃一個 `Workload`（op + 形狀）。
- 回傳 dict 的 key **凍結**成 `{latency_us, bound, provenance}`（`bound ∈ {compute, memory, floor}`）。
- 有輕/重兩種引擎的單元（記憶體、NPU）預留 `engine=`：Phase 1.3 把 **Ramulator2 / ONNXim** 這些 C++ 重型 sim 用 `engine='ramulator2'|'onnxim'` **drop-in 插進來，不必改 API**。

這條介面先用一個 dummy 引擎寫成 conformance test（`tests/test_engine_iface.py`），在平行開發**之前**就跑綠——它是四個元件各自 fill-in 的合約。

## 這一期交付什麼

| 章 | 單元 | 引擎 | 校準狀態 |
|---|---|---|---|
| **C** | CPU（RK3588 big.LITTLE） | 指令數 roofline `max(compute, memory)+overhead` | **calibrated**（對 fp32 `cpu_ops.json`，per-op 殘差中位數 1.15%） |
| **N** | NPU（RKNPU2） | 解析 systolic-roofline | **simulated**（無 silicon，issue #13；趨勢借 HeteroInfer） |
| **M** | 記憶體 + SRAM | 全-spec analytic（LPDDR4/4x/5）+ CACTI SRAM tier | **mix**：LPDDR4x 24.2 = 量測錨點；LPDDR5 = simulated；峰值 = assumption |
| **G** | GPU（Mali-G610） | 解析 roofline 換型號槽（與 micro-benchmark 並存） | **simulated**（roofline 下界，FP16 校準；INT8 零資料） |
| **CIM-Card** | CIM 重驗 | 同一顆 AIPU 在量產卡上重量 | **calibrated**（Alpha 13 點）+ Card 重驗（見該章） |

## 誠實標註紀律（vs Phase 1.1 一致）

每個數字都帶 provenance 標籤：**`calibrated`**（對我們手上的 silicon 擬合）／**`simulated`**／**`assumption`**／**`borrowed`**。最重要的一條鐵律是 **no fake gate**——**沒有 silicon 就不假裝有數值 gate**。RKNPU2（無板）與 GPU INT8（零資料）因此**沒有** per-op 數值驗收門檻，只有 trend-shape／下界的接受準則，並明寫 issue #13 的 silicon gate 是 *superseded-not-satisfied*（被取代、非達成）。

> **可重現原則（沿用）**：每張圖都是 build artifact（`tools/plotting/` 一圖一 script，只吃 committed 數據重產）；每份 report JSON 都能由 `tools/analysis/` 的 fit/build script 重生。整合 cross-check：`tools/analysis/check_phase1_2.py`（載入十份 spec，含 Phase-1.3 加的 `cim_topo_edge` → 餵各引擎 → 驗凍結 key + 標註一致性 + no-fake-gate + 合約 sanity rules）。

## 與 Phase 1.1 / 1.3 的界線

- **1.1 的校準路徑不動**：CPU/記憶體引擎雖改成 spec-based，但 1.1 的 capstone recompose（8B decode hold-out **9.5%**）重跑仍是 9.5%（gate 只用 op-profile bytes + vendor tok/s + BW fit，與這些引擎無關），1.1 的 fit 腳本重生 byte-identical。
- **1.3（下一期，疊在本期介面上）**：ONNXim（NPU）+ Ramulator2（記憶體 LPDDR5）當更高保真的可換引擎，交叉比對本期 analytic 的單串流趨勢。**招牌價值（多單元競爭、逐 token 整機）在 Phase 2。**
