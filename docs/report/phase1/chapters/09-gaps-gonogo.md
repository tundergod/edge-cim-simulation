# 第 9 章 — 缺口三分類與進入 Phase 2 的 GO/NO-GO 判定

本章是這份 review 的結論。先把所有缺口攤開分三類，再給帶條件的 GO/NO-GO。

> **這裡的分類是工程判斷，最終由你（專案負責人）拍板。** 每一項都附「為何這樣歸類」，你可以推翻任何一格。Phase 1.5 已把原本最影響結論強度的 multi-tile / prefill 缺口補成 Card-native 校準（見下方 B 類註解）。

## 9.1　結論摘要（兩段式）

**(1) Phase 1 是否按其誠實標準交付＋驗證？ → 是。**
- decode 主路徑（CIM `G_eff` + 記憶體 + GPU attention + CPU 支援 + 端到端 recompose 8B hold-out 9.5%）達**量測級**可信，且 CIM 已 Card 重驗（4.8%/9.7%，`CARD_REVALIDATED`）。
- 完整、可換型號的 analytic **元件**層齊備（M3/M6 整合層除外，屬 Phase 2 工作）；重型 sim（Ramulator2 / ONNXim）以凍結介面 drop-in 就緒。
- 所有非量測項都誠實標註 simulated / assumption / borrowed，並遵守 no-fake-gate（無 silicon 不假裝有數值門檻）。

**(2) 進入 Phase 2 是否安全？ → 帶條件的 GO。** 沒有任何缺口會擋住「開始 Phase 2（實作 M3+M6 並整合）」；prefill / 大形狀的定量宣稱有一個量測前置條件附在其上（見 9.3）。

## 9.2　缺口三分類

### A 類 — 進 Phase 2 前必須先解（blocking）

**（無。）** 沒有任何缺口會阻止 Phase 2 的「開工」。M3/M6 未實作不是缺口，是 Phase 2 的工作內容本身。所有元件介面（凍結 `predict()` 合約）皆已就緒可接。

### B 類 — 可在 Phase 2 內處理（非 blocking，但需在特定里程碑前解）

| 缺口 | 為何歸 B（非 blocking，但要解） | 大致解法方向 |
|---|---|---|
| **prefill 整條路徑未端到端驗證** | 元件層 prefill 已於 Phase 1.5 dense 校準（M∈{2..{{cim.prefill_m_max}}}、留出 {{cim.prefill_holdout_pct}}%），但整條 TTFT 路徑（含 attention S×S softmax + host overhead）尚未端到端對 vendor 驗。 | Phase 2 整合後對 vendor TTFT 做端到端 prefill 驗證——元件已備，缺的是組裝。 |

> **Phase 1.5 已解（原 B 類兩項）：**（1）**CIM multi-tile**——不再是缺口：Card-native 直接量到 K·N≤{{cim.native_envelope_m}}M，建立 residency-cliff 模型（median {{cim.multitile_new_median_pct}}%、留出 {{cim.multitile_holdout_pct}}%；取代舊 tile-sum 的 {{cim.multitile_old_median_pct}}%）。（2）**KV-cache 係數**——isolation SPIKE 量到 memory-bound proxy BW {{kv.spike_proxy_bw}} GB/s ≈ M2 {{kv.spike_m2_bw}} GB/s，analytic kv_append 的 BW 假設 board-confirmed（維持 analytic，不需重校）。

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

**附帶條件（綁在特定宣稱上，非綁在開工上）：**

- **prefill 端到端宣稱的前置條件**：元件層 prefill 與 multi-tile 已於 Phase 1.5 Card-native 校準（M∈{2..{{cim.prefill_m_max}}}、multi-tile cliff 模型 median {{cim.multitile_new_median_pct}}%），故大形狀的**元件級**定量已有量測支撐。剩下的是整條 **prefill TTFT 端到端**驗證（attention + host overhead 組裝），屬 Phase 2 整合工作。

> **混合精度成本基礎（非量測前置條件）**：conversion-op（CIM-INT8 × GPU-FP16 邊界的 dequant/requant）於 Phase 2 **解析建模**——memory-bound cast，由既有 M2/M4 per-op 模型 × M6 邊界穿越次數計價（ADR-0004 已於 2026-06-11 修訂）。精度固定於單元，故非量測缺口；Phase 2 把 cast op 插進 op stream 計價即可。

**這份報告不保證模擬器完美；它保證我們精確知道自己站在哪、缺什麼、那些缺口會不會擋住 Phase 2。** 答案是：decode 地基穩固，整合可以開始；Phase 1.5 把 prefill / multi-tile / KV-BW 的元件級量測補齊（並修正了兩個過保守的編譯假設），剩 prefill TTFT 端到端組裝屬 Phase 2；混合精度成本則於 Phase 2 解析建模。

## 9.4　待你拍板的歸類

- **CIM multi-tile**：原列 B（待板重校），**Phase 1.5 已解**——Card-native 量測 + residency-cliff 模型（calibrated）。原「若板長期離線降為 C」的退路已不需要。
- **prefill M>256 / SRAM 牆**：原假設證實**不存在**（編到 M={{cim.prefill_m_max}}），不再是 limitation。

（conversion-op 成本不在此清單：ADR-0004 修訂後它是 Phase 2 解析建模項，非缺口、非量測前置條件。）
