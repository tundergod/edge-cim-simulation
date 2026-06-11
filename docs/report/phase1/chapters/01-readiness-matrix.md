# 第 1 章 — 全局就緒度總表

一眼看完 Phase 1 所有元件的狀態。每列的數字與標籤都在後續對應章節展開、並可經文末附錄回查來源。誠實標籤：**calibrated**（對我們的 Metis silicon 校準）／**simulated**（模型或外部 sim，未經我們 silicon 驗證）／**assumption**（規格／文獻推算）／**borrowed**（借自他人量測）。

## 1.1　元件就緒度矩陣

| 元件 | 模擬什麼 | provenance（誠實標籤） | 驗證狀態 | 進 Phase 2 就緒度 |
|---|---|---|---|---|
| **CIM decode（M1）** | 2D `G_eff(N,K)` 閉式吞吐 | **calibrated**（Alpha 13 點 + Card 重驗） | median **{{cim.decode_median_pct}}%** / p95 **{{cim.decode_p95_pct}}%**；Card 交叉驗證 median **{{cim.card_median_pct}}%** / p95 **{{cim.card_p95_pct}}%**（`CARD_REVALIDATED`） | ✅ 就緒（decode 主路徑量測級可信） |
| **CIM prefill（M1）** | affine M-amortization | **calibrated**（Card，M∈{64,128,256}） | 擬合 median ~{{cim.prefill_median_pct}}%、5 點 full-GEMM max **{{cim.prefill_fullgemm_max_pct}}%** | ⚠ M>256 外推（SRAM tiling 牆）；典型 prefill 長度落在外推區 |
| **CIM multi-tile（M1）** | tile-sum 外推 | **simulated**（解析延伸） | 唯一 native 多-tile 點 over-predict **+{{cim.multitile_overpred_pct}}%**；K·N>4.19M 未驗證 | ⚠ 缺口（板離線無法重量，#2/#11/#17） |
| **記憶體 LPDDR4x（M2）** | eff-BW streaming | **calibrated**（量產卡 decode wall） | {{mem.lpddr4x_eff}} GB/s（峰值 {{mem.lpddr4x_eff_pct}}%），r²=0.997 | ✅ 就緒 |
| **記憶體 LPDDR5（M2）** | eff-BW（前瞻 SoC） | **simulated**（eff {{mem.ram2_system_eff}} 保守） | Ramulator2 device {{mem.ram2_device_eff}} vs system {{mem.ram2_system_eff}} 交叉驗證一致（驗證 ADR-0002） | ✅ 就緒（標籤＝simulated；非本 silicon 量測） |
| **記憶體 PCIe（M2）** | floor + BW | **measured/calibrated**（Alpha 拓樸） | floor {{mem.pcie_floor_us}} µs / p95 1111.7 µs；單調 | ✅ 就緒 |
| **記憶體 SRAM tier（M2）** | CACTI L1/L2 residency | **assumption**（CACTI） | 架構級；8B 權重 ≫32 MiB → 永遠走 DRAM | ✅ 就緒（架構假設明示） |
| **KV-cache（M2）** | analytic pure-BW | **simulated**（係數未驗證） | 形式正確，係數未隔離量測（板離線） | ⚠ 缺口（待板重校） |
| **CPU（M4）** | 指令數 roofline | **calibrated**（fp32 A76 單核） | per-op 殘差 median **{{cpu.resid_median_pct}}%** / p95 **{{cpu.resid_p95_pct}}%** | ✅ 就緒（decode 支援算子） |
| **CPU eta_bw / A55 / fp16** | — | **assumption / simulated** | eta_bw=0.6 假設；A55+多核外推；fp16 解析（非校準） | ⚠ limitation（多數情境 compute/overhead 主導） |
| **GPU attention（M4）** | micro-benchmark `a+b·kv` | **calibrated**（Mali FP16 attn） | median **{{gpu.attn_median_pct}}%** / p95 **{{gpu.attn_p95_pct}}%** | ✅ 就緒（attention offload 主路徑） |
| **GPU roofline 槽（M4）** | 解析 roofline | **simulated**（FP16 下界） | vs 1.1 量測 median **{{gpu.roofline_median_pct}}%** / p95 **{{gpu.roofline_p95_pct}}%**（長尾，非嚴格下界） | ⚠ INT8 零資料；僅形狀趨勢可信 |
| **NPU（M4）** | 解析 systolic-roofline | **simulated / borrowed**（datasheet + HeteroInfer trend） | **無數值 silicon gate**；trend-shape 通過（staircase knee@32、order≤6×、BW 59–66%） | ⚠ 無 silicon（#13 superseded-not-satisfied） |
| **NPU ONNXim（M4）** | 重型 sim 交叉驗證 | **simulated**（cycle-level） | 趨勢一致，但一致高約 4×（median \|delta\| **{{npu.onnxim_median_delta_pct}}%**）；sim-vs-sim | ✅ 介面就緒（兩者皆 simulated） |
| **M5 trace** | op×shape 流 + 覆蓋 oracle | **calibrated**（對 inventory 驗證） | 4 模型全過、0 孤兒、語意全覆蓋 | ✅ 就緒 |
| **M7 能耗** | spec-based 估算 | **assumption**（全規格、無遙測） | 無數值 gate；±20%×16 corner、結論翻轉 0；記憶體主導 | ✅ 就緒（標籤＝estimated） |
| **M3 事件引擎** | discrete-event + 競爭 | — | **未實作**（contract + 3 可調參數） | ⛔ Phase 2 工作（設計使然，非缺失） |
| **M6 排程器** | op→單元 + 精度邊界 | — | **未實作**（contract + 4 可調參數） | ⛔ Phase 2 工作；conversion-op 成本 = Phase-2 解析建模項（非量測） |

## 1.2　怎麼讀這張表

- **✅ 就緒**：元件按其誠實標準交付且通過該層驗證；介面可供 Phase 2 直接接上。注意「就緒」不等於「量測級精確」——例如 LPDDR5、NPU 是 **simulated** 但介面與標籤都誠實、可接，因此就緒。
- **⚠**：有缺口或限制，需在第 9 章歸類（必須先解 / Phase 2 內處理 / 可接受 limitation）。
- **⛔ Phase 2 工作**：M3/M6 未實作是專案設計（Phase 2 才實作整合），不是 Phase 1 缺失。conversion-op 成本不是缺口，是 Phase 2 的解析建模項（memory-bound cast，既有 M2/M4 × 邊界穿越）。

**一句話總結：** decode 主路徑（CIM 算 + 記憶體搬 + GPU attention + CPU 支援 + 端到端 recompose）達量測級可信；prefill、multi-tile、NPU silicon 是已知且誠實標註的缺口；conversion-op 成本是 Phase 2 解析建模項（非量測）；整合層尚待 Phase 2 建。詳細判定見第 9 章。
