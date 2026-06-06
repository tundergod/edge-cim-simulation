# Phase 1 總報告 — CIM-centric 異質行動 SoC LLM 推論模擬器：元件建模與驗證

> 本報告把 Phase 1 的三個子階段（1.1 / 1.2 / 1.3）合併成**單一份、自給自足**的回顧，供進入 Phase 2 前做完整 review。讀完這一份即不需再翻舊的分期報告。**核心紀律：不誇大、不軟化；不確定的就標不確定、沒量的就說沒量。** 每個數字都可經文末來源附錄回查其出處檔案。

## 這個模擬器在做什麼，Phase 1 又是哪一段

研究目標是一個**以真實晶片校準的「CIM-enabled 異質行動 SoC 上 LLM 推論」模擬器**。這種晶片並不存在，真實的 Axelera Metis 卡也無法直接當研究對象（Metis Alpha 跑不了完整 LLM、量產 Metis Card 只能透過封閉工具鏈跑），因此策略是：把**研究對象**設為一個*模擬的* CIM-mobile SoC，再用兩塊真實 Metis 板（Aetina Metis Alpha + 量產 Metis Card）當**校準 ground truth**。完整動機、平台假設與貢獻定位見 [`OVERALL.md`](../../../OVERALL.md)，此處不重述。

整個專案的骨幹是 8 個模組 M1–M8 與 6 層驗證 L1–L6。**Phase 1 是「元件層」**：把 Phase 0.3 量到的資料，對每個計算/記憶體單元擬合成**方程式**（而非龐大 lookup table），逐一驗證其準確度並誠實標註其可信邊界。**Phase 1 不做整合**——把校好的元件串成端到端 event-driven 模擬器（M3 事件引擎 + M6 排程器）是 **Phase 2** 的工作。因此本報告的結論不是「模擬器完成了」，而是**「每個元件現在多可信、缺什麼、能不能安全進入 Phase 2」**（見第 9 章 GO/NO-GO）。

## 方法論：為何按「資料來源」拆成三個子階段

Phase 1 刻意**依資料來源、而非依模組**切分，因為一個元件可信到什麼程度，取決於它背後是哪一種證據：

- **Phase 1.1（自家 Metis silicon 校準）** — 凡是能用我們手上的 Metis 量測直接校準、並通過 ADR-0006 gate 的核心 decode 路徑：CIM 的 2D G_eff、Mali GPU attention、CPU 支援算子、量產卡 LPDDR4x、M5 trace、M7 能耗，外加端到端 recompose hold-out。這是「量測級可信」的一層。
- **Phase 1.2（模組化「引擎 + 可換 spec」analytic 層）** — 把每個非-micro-benchmark 單元改寫成「一個模型引擎 + 一份可換 spec 檔」，換型號＝換 spec。補齊 CPU 指令數 roofline、NPU 解析 systolic-roofline、全-spec 記憶體、GPU roofline 槽、CIM 雙拓樸與 Card 重驗。這是「完整、可換型號」的一層。
- **Phase 1.3（重型保真模擬引擎 drop-in）** — 把 ONNXim（NPU）、Ramulator2（記憶體）這類 C++ 重型 sim 透過凍結介面 `engine=` 插進 1.2 的同一個槽，交叉比對 analytic 的單串流趨勢。**這兩者都是 simulated，不是 silicon。** 其招牌價值（多單元競爭、逐 token 整機）留待 Phase 2。

三期共用一條凍結介面 `Engine(spec).predict(workload) -> {latency_us, bound, provenance}`（`simulator/models/engine.py`），所以 Phase 1.1 的校準在 1.2 改成 spec-based 後仍可重現（recompose 8B hold-out 重跑仍是 9.5%）。

## 兩條貫穿全報告的紀律

**1. 方程式優先、ADR-0006 gate。** 元件模型一律是參數化方程式（可外推、可解釋、體積小），擬合誤差以 median / p95 相對誤差記錄；對沒有 silicon 的單元，**遵守 no-fake-gate**——沒有量測就不假裝有數值門檻，只用 trend-shape／下界接受準則，並明寫哪個 silicon gate 是 *superseded-not-satisfied*（被取代、非達成）。

**2. 三段式誠實標籤。** 每個模型的每個來源都標恰好一種：

| 標籤 | 意義 |
|---|---|
| **calibrated** | 對我們手上的真實 Metis silicon 擬合／驗證 |
| **simulated** | 來自模型或外部模擬器，**未**經我們的 silicon 驗證 |
| **assumption** | 規格推算或文獻典型值，repo 內無一手資料來源 |
| **borrowed** | 直接借用他人量測（主要是 HeteroInfer 的 trend） |

這四個標籤不混用。一個元件可以同時含多種來源（例如記憶體 = LPDDR4x calibrated + LPDDR5 simulated + 峰值 assumption），報告會逐項分開標。

## 怎麼讀這份報告

- **按單元編排**（第 2–6 章：CIM / 記憶體 / CPU / GPU / NPU），對應 Phase 2 要組起來的異質 SoC 各計算單元。
- 每個單元一律用**五欄模板**：①模擬什麼（方程式形式）②模型從哪來（provenance + 誠實標籤）③驗證狀態（誤差數字 / gate）④缺口 / 外推區 ⑤進 Phase 2 就緒度。
- **第 7 章**收非單元的元件（M5 trace、M7 能耗）與跨單元結果（decode 端到端 recompose、prefill 整條路徑現況）。
- **第 8 章**是尚未實作的整合層（M3 / M6），明確區分「未實作（設計使然，是 Phase 2 工作）」與「真缺口」。
- **第 1 章**是全局就緒度總表（一眼看完），**第 9 章**是缺口三分類與帶條件的 GO/NO-GO 判定，**文末附錄**是編號來源表。
- 內文數字以上標編號連到附錄來源條目；報告本文不貼檔案路徑與程式碼，以保持可讀性。
