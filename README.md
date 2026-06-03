# edge-cim-simulation

一個以**真實晶片校準**的模擬器，模擬 **CIM-enabled 異質行動 SoC 上的 LLM 推論** — 讓 Compute-in-Memory（CIM）與 NPU + GPU + CPU 在 unified memory 上作為對等計算單元。以真實 Axelera Metis AIPU 晶片校準。

## 這是什麼

「離散 CIM 掛在異質行動 SoC 上跑 LLM」的晶片並不存在，而真實 Metis 卡也無法直接當研究對象（Alpha 跑不了 LLM 計算；量產卡的 LLM 路徑封閉、僅預編譯）。因此我們**模擬**那顆 SoC，並用兩塊真實 Metis 板的量測來**校準**它。研究面是 **CIM-centric 混合精度排程**（哪個 op 在哪個單元、用哪種精度跑），由量測出的各單元特性曲線決定，而非預先設定。

## Repo 導覽

| 路徑 | 內容 |
| --- | --- |
| [overall.md](overall.md) | **專案綱要** — 目標、問題、立場、Phase 0 特性量測計畫、模擬器架構（6 box / M1–M7）、風險、範圍。*初步，可自由修改。*（中文） |
| [voyager-sdk.md](voyager-sdk.md) | **給所有 agent 的 SDK 量測參考** — 如何從 Voyager SDK / Metis 晶片擷取模擬器所需的每一項量測。標注 `[DOC]`/`[FORUM]`/`[MEASURED]`/`[GAP]`。*（英文，agent 用，使用者不參與此實作討論。）* |
| [papers/](papers/) | 文獻筆記 + 真實晶片調查報告（已嚴格篩為 16 篇，並附原始 PDF/HTML）。見 [papers/README.md](papers/README.md)。 |

## 從哪開始

1. 讀 [overall.md](overall.md) 了解目標與計畫。
2. 設計任何量測前，先讀 [voyager-sdk.md](voyager-sdk.md)。
3. 略讀 [papers/metis-silicon/](papers/metis-silicon/) 取得真實晶片 ground truth（校準錨點 L4 + L6）。

## 狀態

Bootstrap 階段。文獻語料、SDK 參考、專案綱要已就位。階段規劃：**Phase 0.1**（生成 trace 與 op inventory，純軟體）→ **Phase 0.2**（真實板量測，除溫度外全部）→ **Phase 1**（每個 component 擬合方程式 + 驗證）→ **Phase 2**（模擬器整合）；**Phase 0.3**（溫度量測）與熱模組 M8 可後續並行加入。詳見 [overall.md](overall.md)。`overall.md` 所述的 `simulator/`、`measurements/`、`characterization/`、`validation/`、`tools/`、`docs/` 目錄為規劃版面，待工作開始時建立。

## 主要外部參考

- Voyager SDK：<https://github.com/axelera-ai-hub/voyager-sdk>
- Axelera 社群論壇（Metis M.2）：<https://community.axelera.ai/metis-m-2-3>
