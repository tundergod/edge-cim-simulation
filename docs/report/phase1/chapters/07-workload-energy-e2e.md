# 第 7 章 — 工作負載（M5）、能耗（M7）、端到端與 prefill 現況

本章收三類「不屬於單一計算單元」的成果：M5 工作負載/trace、M7 能耗模型，以及**跨單元**的端到端結果——decode 路徑的 recompose hold-out（已驗證）與 prefill 整條路徑的現況（**analytic、未驗證**，本報告最重要的誠實項之一）。

---

## 7.1 M5 — 工作負載 / trace 生成

**模擬什麼。** M5 把 HuggingFace 模型 + workload 定義，導成每 token 實際執行的 op×shape 流（prefill / decode 分開）。Phase 1 的驗證對象是它的**語意覆蓋**與**零孤兒**：每個 op_profile 用到的 op，都能追溯到 Phase 0.1 從 HF 抽出的 op inventory。

**模型從哪來。** **calibrated**（對已知輸入驗證）——方法是重用 Phase 0.1 的 inventory oracle 做 `expected_ops_check`，加上 0-orphan 檢查；count 直接取自 inventory，**不自行 ×layers 推算**。[^m5a]

**驗證狀態。** 四個目標模型全數通過：llama-3.2-1b / 3b / 8b 各 38 個 distinct op、qwen2.5-7b 39 個，每個模型 4 個任務、孤兒 op 數為 0、語意全覆蓋。[^m5b] `pass_all = true`。[^m5c]

**缺口 / 外推區。** M5 此處驗的是**靜態 op×shape 流的覆蓋正確性**，不是 decode 動態長度展開的逐 token 正確性（後者在 Phase 2 由 M5 按需生成時才完整行使）。embedding gather 在 decode 近似為 0（折進 overhead）、prefill 約 192 MB analytic，未獨立驗證。

**進 Phase 2 就緒度。** 就緒。op DAG 是 M6 排程器的輸入；Phase 2 的工作是把這條 trace 按需展開成逐 token 串流，而非重新驗證覆蓋。

---

## 7.2 M7 — 能耗模型

**模擬什麼。** 逐元件、spec-based 的能耗估算（ADR-0005，無功耗遙測）：CIM 投影、DRAM streaming、CPU 支援的 per-token 能耗加總。

**模型從哪來。** **assumption（全部係數皆規格推算，非量測）**——CIM 15 TOPS/W（廠商 INT8）、LPDDR5 4 pJ/bit、PCIe 5 pJ/bit、A76 0.75 W/core × 4 核。[^m7a] 板子無 on-board 功耗儀表，故能耗一律是**估計、非量測**。[^m7b]

**驗證狀態。** 不做數值 gate（無量測可對），只做 sanity + 敏感度。8B decode 每 token：CIM 投影 {{energy.cim_mj}} mJ、DRAM streaming {{energy.dram_mj}} mJ、CPU 支援 {{energy.cpu_mj}} mJ，合計 {{energy.total_mj}} mJ，**主導項為 DRAM streaming**。[^m7c] 對每個係數做 ±20% 共 {{energy.corners}} 組 corner 的敏感度，**結論翻轉 {{energy.flips}} 次**——「能耗由記憶體搬移主導」這個結論對 ±20% 穩健。[^m7d] 隱含平均功耗 0.692 W，量級合理。[^m7e]

**缺口 / 外推區。** 整個模型是估計值，**沒有任何一點對到真實功耗量測**。RKNPU2 無功耗遙測，NPU 能耗不可定。CPU 支援時間是粗略的 per-token 估計。[^m7b] 結論的價值在「**哪一項主導**」（記憶體），而非絕對 mJ 數字的精確度。

**進 Phase 2 就緒度。** 介面就緒，但**標籤永遠是 estimated**；若 Phase 2 取得功耗遙測（M.2 Max 才有 INA236 可讀），可升級為 measured。

---

## 7.3 decode 端到端 — recompose hold-out（已驗證）

這是 Phase 1 唯一的**端到端**量測級驗證點。decode backbone 建模為 weight-streaming 受限：`tok_s ≈ BW_eff / per_token_weight_bytes`。以 1B + 3B 擬合一個有效頻寬，**外插預測 8B**（hold-out）：

| 模型 | per-token weight bytes (GB) | 量測 tok/s (1 core) |
|---|---|---|
| llama-3.2-1b | 1.237 | 13.07 |
| llama-3.2-3b | 3.214 | 6.38 |
| llama-3.1-8b（hold-out） | 7.507 | 2.7 |

擬合有效頻寬 {{recompose.fit_bw}} GB/s；**8B 預測 {{recompose.pred_8b}} tok/s vs 量測 {{recompose.meas_8b_1}} tok/s，相對誤差 {{recompose.err_8b_pct}}%**，通過 ≤25% gate。[^e2e1] 這個結果是 model-independent 的（只用 op-profile bytes + vendor tok/s + 一個 BW fit），因此 Phase 1.2 把 CPU/記憶體改成 spec-based 引擎後重跑**仍是 9.5%**。

**重要 watch-item（Phase 2）：** 非-streaming 項（CPU 支援 15007 µs、GPU offload attention 下界 260185 µs、KV-cache append 1386.5 µs）在擬合點**已被吸收進 BW_eff**；Phase 2 若把它們再加上去會**重複計算**。[^e2e2] 這是 Phase 2 保真度的已知注意事項，不是 Phase 1 的錯誤。

---

## 7.4 prefill 整條路徑 — 現況：analytic，未端到端驗證

**這是必須講清楚的誠實項。** Phase 1 的校準幾乎全在 **decode** 路徑；**prefill 整條路徑沒有端到端的 silicon 驗證 gate。** 各單元的 prefill 元件成熟度不一：

- **CIM prefill GEMM**：dense M∈{2..508} 在 Card 上量測並擬合（M-amortization，見第 2 章；舊 M_MAX=256 假設低估約 2×，真牆 ~M=510），但典型 LLM prefill 長度（如 8B、M=1024）**仍落在外推區**（M>508 無法編譯量測）。
- **prefill attention 的 S×S softmax scaling**、host overhead：未涵蓋。

目前能說的，是一個**模型比較**而非絕對驗證 gate：以 vendor TTFT 3.794 s 為錨，擬合的 prefill GEMM compute 在 M=1024 約 0.259 s（佔 TTFT 6.8%），記憶體 floor 約 0.409 s。[^e2e3] 這個比較的**鑑別力很弱**（compute ≤ TTFT 對擬合模型幾乎必然成立，界要到 M≈22000 才 bind），其真正價值在**反證**線性-M decode-GEMV 外推——後者單是 compute 就 75.4 s，超過實測 TTFT 約 20×，物理上不可能。[^e2e4]

**結論：** prefill 路徑目前是 **BOUNDED-EXTRAPOLATED**——有界、被反證過明顯錯誤的模型，但**沒有正面的端到端驗證**。這是進 Phase 2 前要正視的缺口（見第 9 章分類）。

---

## 來源（本章腳注）

[^m5a]: 來源 `validation/reports/phase1.1/m5.json` › `method` = "reuse Phase 0.1 inventory oracle (expected_ops_check) + 0-orphan check … counts from inventory, no hand x L"
[^m5b]: 來源 `validation/reports/phase1.1/m5.json` › `per_model`：llama-3.2-1b/3b/8b `n_distinct_ops`=38、qwen2.5-7b=39；各 `n_tasks`=4、`orphan_ops`=[]、`semantic_covered`=true
[^m5c]: 來源 `validation/reports/phase1.1/m5.json` › `pass_all` = true
[^m7a]: 來源 `validation/reports/phase1.1/m7.json` › `params` = {cim_tops_w:15.0, lpddr5_pj_per_bit:4.0, pcie_pj_per_bit:5.0, a76_core_w:0.75, cpu_cores:4}
[^m7b]: 來源 `validation/reports/phase1.1/m7.json` › `limitation` = "energy ESTIMATED not measured (no telemetry, ADR-0005) … CPU support time is a coarse per-token estimate."
[^m7c]: 來源 `validation/reports/phase1.1/m7.json` › `per_token_8b_decode_mJ` = {cim_proj:1.001, dram_stream:240.149, cpu_support:15.0}；`per_token_total_mJ`=256.15；`dominant_term`="dram_stream_mJ"
[^m7d]: 來源 `validation/reports/phase1.1/m7.json` › `sensitivity_pm20pct` = {corners_tested:16, conclusion_flips:0}；`sanity.memory_dominates_robust_to_pm20pct`=true
[^m7e]: 來源 `validation/reports/phase1.1/m7.json` › `sanity.implied_avg_power_W` = 0.692
[^e2e1]: 來源 `validation/reports/phase1.1/recompose.json` › `fit_BW_GBs`=18.33、`pred_8b_tok_s`=2.44、`measured_8b_tok_s`=2.7、`rel_error_8b`=0.095、`GATE_within_25pct`=true；`per_token_weight_bytes`、`measured_tok_s_1c` 如表
[^e2e2]: 來源 `validation/reports/phase1.1/recompose.json` › `standalone_nonstreaming_8b_us` = {decode_stream_backbone:409470.5, cpu_support:15007.0, gpu_offload_attention_lowerbound:260185.0, kv_cache_append:1386.5, _caveat:"already absorbed in BW_eff … ADDING double-counts (Phase-2 fidelity, watch-item)"}
[^e2e3]: 來源 `validation/reports/phase1.1/recompose.json` › `prefill_gemm_compute_BOUNDED` = {status:"BOUNDED-EXTRAPOLATED", M_prefill:1024, M_measured_max:256, vendor_ttft_s:3.794, gemm_compute_fitted_s:0.259, gemm_compute_frac_of_ttft:0.068, memory_floor_s:0.409}
[^e2e4]: 來源 `validation/reports/phase1.1/recompose.json` › `prefill_gemm_compute_BOUNDED.decode_GEMV_linear_s`=75.4、`model_comparison_compute_le_ttft.note`（fitted 鑑別力弱、界於 M≈22000 才 bind；價值在反證 linear-M decode-GEMV ~20×）
