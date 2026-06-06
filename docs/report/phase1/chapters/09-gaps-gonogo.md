# 第 9 章 — 缺口三分類與進入 Phase 2 的 GO/NO-GO 判定

本章是這份 review 的結論。先把所有缺口攤開分三類，再給帶條件的 GO/NO-GO。

> **這裡的分類是工程判斷，最終由你（專案負責人）拍板。** 每一項都附「為何這樣歸類」，你可以推翻任何一格——特別是 conversion-op 與 multi-tile 兩項的歸類，會直接影響結論強度。

## 9.1　結論摘要（兩段式）

**(1) Phase 1 是否按其誠實標準交付＋驗證？ → 是。**
- decode 主路徑（CIM `G_eff` + 記憶體 + GPU attention + CPU 支援 + 端到端 recompose 8B hold-out 9.5%）達**量測級**可信，且 CIM 已 Card 重驗（4.8%/9.7%，`CARD_REVALIDATED`）。
- 完整、可換型號的 analytic **元件**層齊備（M3/M6 整合層除外，屬 Phase 2 工作）；重型 sim（Ramulator2 / ONNXim）以凍結介面 drop-in 就緒。
- 所有非量測項都誠實標註 simulated / assumption / borrowed，並遵守 no-fake-gate（無 silicon 不假裝有數值門檻）。

**(2) 進入 Phase 2 是否安全？ → 帶條件的 GO。** 沒有任何缺口會擋住「開始 Phase 2（實作 M3+M6 並整合）」；但有兩個條件附在特定宣稱上（見 9.3）。

## 9.2　缺口三分類

### A 類 — 進 Phase 2 前必須先解（blocking）

**（無。）** 沒有任何缺口會阻止 Phase 2 的「開工」。M3/M6 未實作不是缺口，是 Phase 2 的工作內容本身。所有元件介面（凍結 `predict()` 合約）皆已就緒可接。

### B 類 — 可在 Phase 2 內處理（非 blocking，但需在特定里程碑前解）

| 缺口 | 為何歸 B（非 blocking，但要解） | 大致解法方向 |
|---|---|---|
| **conversion-op 成本從未量測（M6/ADR-0004）** | 不擋整合開工，但**擋住混合精度貢獻的宣稱**——這是招牌貢獻，目前其精度邊界 op 無成本基礎。屬高優先。 | Phase 2 早期補量 dequant/requant（便宜 elementwise，CPU/NPU 可量）；在做任何混合精度 latency/energy 宣稱**之前**完成。 |
| **prefill 整條路徑未端到端驗證** | decode 已驗、prefill 只到 BOUNDED-EXTRAPOLATED（反證了線性-M 錯誤模型，但無正面 gate）。Phase 2 本就要建並驗 prefill 路徑。 | Phase 2 整合後對 vendor TTFT 做端到端 prefill 驗證；補 prefill attention S×S softmax 與 host overhead。 |
| **CIM multi-tile（K·N>4.19M）未驗證** | 影響 lm_head 與大 prefill 形狀；唯一 native 點 over-predict +36%。板離線是外部限制。 | 板恢復時重量多-tile 點重校；在此之前該區域標 extrapolated。若板長期離線 → 降為 C 類 limitation。 |
| **KV-cache 係數未隔離量測** | 形式正確、係數未驗；decode 受影響小（已部分吸收進 recompose BW_eff）。 | 板恢復時補 kv_append micro-benchmark 重校。 |

### C 類 — 可接受的 limitation（誠實標註後不擋路）

| limitation | 為何可接受 |
|---|---|
| **NPU 全 simulated/borrowed（#13 無 silicon）** | NPU 是支援單元；模型誠實標 simulated，trend-shape 與 ONNXim 交叉驗證一致；#13 已依 ADR-0006 superseded-not-satisfied。板恢復可升級。**但：Phase 2 整機結果若包含 NPU 路徑，provenance 必須傳播 `simulated (RKNPU2, no silicon)` 標籤（詳見第 6 章 §5）。** |
| **GPU INT8 零資料** | 當前分工 CIM 做 INT8 GEMM、GPU 做 FP16 attention；GPU INT8 GEMM 不在主路徑。若 Phase 2 需要才補。 |
| **LPDDR5 為 simulated（非本 silicon）** | 保守折扣（eff 0.65 < 量測 0.71），且 Ramulator2 device 交叉驗證一致；標籤誠實。 |
| **CPU eta_bw=0.6、A55/多核、fp16** | eta_bw 僅對 qwen vocab（L3）binds，他處 compute/overhead 主導；A55/多核/fp16 明標 simulated。 |
| **M7 能耗全 estimated（無遙測）** | 結論（記憶體主導）對 ±20% 穩健（16 corner、0 翻轉）；絕對 mJ 標 estimated。 |

## 9.3　GO/NO-GO 判定

**判定：條件式 GO。**

可以進入 Phase 2（實作 M3 事件引擎 + M6 排程器，整合成端到端模擬器）。理由：A 類為空——所有元件介面就緒、decode 主路徑量測級可信、缺口皆已誠實標註且不阻止整合開工。

**兩個附帶條件（綁在特定宣稱上，非綁在開工上）：**

1. **混合精度宣稱的前置條件**：在 Phase 2 提出任何混合精度（CIM-INT8 × GPU-FP16）的 latency / energy 結果之前，**必須先補量 conversion-op 成本**（B 類第 1 項）。否則招牌貢獻無成本基礎。
2. **prefill / multi-tile 宣稱的前置條件**：任何 prefill 端到端或大形狀（lm_head、長 prefill）的定量宣稱，需在 Phase 2 取得板存取後以量測支撐；在此之前一律維持 extrapolated 標註。

**這份報告不保證模擬器完美；它保證我們精確知道自己站在哪、缺什麼、那些缺口會不會擋住 Phase 2。** 答案是：decode 地基穩固，整合可以開始，兩個招牌宣稱（混合精度、prefill）在 Phase 2 內補齊量測前不得定稿。

## 9.4　待你拍板的歸類

以下兩項我做了判斷，但你可能有不同看法，請確認或推翻：
- **conversion-op 成本**：我歸 **B（Phase 2 早期補，綁混合精度宣稱）**。若你認為混合精度是 Phase 2 的第一順位、必須先有成本才能設計排程器，可上調為 **A（先解）**。
- **CIM multi-tile**：我歸 **B（待板重校）**。若板恢復無望，應下調為 **C（limitation）** 並在論文明列為範圍限制。
