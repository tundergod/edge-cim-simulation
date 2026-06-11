# 02 — M1：CIM 計算核（Phase 1 整合章）

> **本章角色：** M1 是異構 SoC 推論架構的主角計算引擎——負責所有 weight-stationary GEMM（Q/K/V/O 投影、FFN gate/up/down、lm_head）。本章整合 Phase 1.1 decode 校準、Phase 1.2 Card 重驗、以及 prefill M-amortization 三個層次的結果，說明模型方程式、資料來源、驗證狀態、已知缺口，並評估 Phase 2 就緒度。

---

## 1　模擬什麼

### 1.1　硬體單元

Metis AIPU 是 **quad-core**，每個 AI-Core 有一塊 **512×512 INT8 數位 in-memory（D-IMC）crossbar**（ISSCC 2024：16 banks × 512 input × 32 output × 4 weight-sets）。[^cim1] 模擬器以「n 個 core」為最小單位，`n_cores` 是可調參數；對 Metis，n_cores=4，有效輸出寬度 W = n_cores × 512 = **2048**。[^cim2] 吞吐量以 INT8 **GOP/s** 計（不是 GFLOP/s）。[^cim3]

### 1.2　decode（M=1）：G_eff 2D 閉合式

M=1 的單步推論（GEMV）使用以下方程式：

```
W = n_cores × 512                              # 有效輸出寬度（= 2048，n=4）
dev_lat(M=1, K, N) = Σ_{tiles} 2·K·n_tile / G_eff(n_tile, K)   [µs]

G_eff(N, K) = Gmax · N/(N + Na) · K/(K + Kb)   [GOP/s]
```

參數（2D 有效吞吐擬合）：[^cim4]

| 參數 | 值 |
|---|---|
| Gmax | 333.67 GOP/s |
| Na | 577.2 |
| Kb | 574.1 |
| n_cores | 4 |
| core_width | 512 |

沿 N 分塊時，每塊**按自己的實際寬度**計費（partial last tile 不會被充整塊的延遲），使 dev_lat 隨 N 持續上升。

### 1.3　prefill（M>1）：M-amortization 仿射擬合

prefill 時，一個 2048×2048 canonical tile 的延遲相對 M 呈**仿射關係**（weight load 被 M 個 activation column 攤薄）：

```
tile_lat(M) = a + b·M   [µs]
full_GEMM_lat(M, K, N) = (K·N / W²) · tile_lat(M)   [fractional area，非 ceil]
```

擬合參數（Card 量測，M ∈ {64, 128, 256}，full 2048×2048 tile）：[^cim5]

| 參數 | 值 |
|---|---|
| a（weight load） | 38.813 µs |
| b（per activation column） | 0.1033 µs/col |
| asymptote | ~81.2 TOPS |

**M=1 decode 的量測錨點**（不在 prefill 擬合之內，不同 regime）：tile 延遲 41.83 µs，GOP/s = 200.5。[^cim6] 線性-M decode 外推（若誤用 G_eff 線性擴展到 M=128）會得到約 65843 µs，比實際量測 724 µs 高出約 **91 倍**（65843 / 724.1）；M-amortization 擬合消除了這個偏差。[^cim7]

---

## 2　模型從哪來

**calibrated（Alpha 13 點，量到的硅）**
13 個原生單-tile G_eff(N,K) 量測點，在 Aetina Metis Alpha 板（pre-production，800 MHz AIPU）以 `characterization/aetina/run_metis_cim.py` + `axrunmodel dev_fps` 取得，K·N ≤ 4.19M（native_max_kn = 2048×2048）。[^cim8] 涵蓋 K ∈ {2048, 3072, 3584, 4096}、N ∈ {64, 128, 256, 480, 512, 544, 1000, 1024, 1536, 2048}。

**calibrated（Card 重驗，同一顆 AIPU）**
量產 Metis Card（800 MHz，16 GiB on-card LPDDR4x）移植同一 1×1-conv matmul proxy，經 `axcompile` 編譯 + `axrunmodel dev_fps` 重量上述 13 個形狀，直接對比 Alpha GOP/s（兩板同 800 MHz，無需 clock 正規化）。[^cim9]

**calibrated（prefill M-amortization，Card 量測）**
Card 上以 `fit_cim_prefill.py` 量 M ∈ {64, 128, 256} 的 full 2048×2048 tile 延遲，仿射擬合。[^cim10]

**assumption**
device 配置 envelope ~14 MB = PCIe-IOMMU window（Alpha 無真正的 on-card DRAM；`zeMemAllocDevice` 映射的是 host LPDDR 的 IOMMU window），論壇記錄預設 ~14 MB。alloc_envelope_param_count 6M 是 SDK weight-alloc limit，與 native_max_kn（4.19M K·N tile 面積）是**不同概念**。[^cim11]

**borrowed**
無需 borrow 外部文獻的數字；所有 fit 參數來自自己的量測。ISSCC 2024 論文提供硬體架構描述（quad-core 512×512），不是數值參數。

---

## 3　驗證狀態

### 3.1　decode G_eff 擬合門檻（ADR-0006，Phase 1.1）

| 指標 | 值 | 門檻 | 結果 |
|---|---|---|---|
| median rel_err（13 點） | **{{cim.decode_median_pct}}%** | ≤ 10% | ✅ PASS |
| p95 rel_err（13 點） | **{{cim.decode_p95_pct}}%** | ≤ 20% | ✅ PASS |
| max rel_err | {{cim.decode_max_pct}}% | — | 呈現不隱藏 |

[^cim12]

### 3.2　Card 重驗（`CARD_REVALIDATED`，Phase 1.2，PR #25）

同一 800 MHz AIPU，13 點交叉驗證 Alpha 擬合 vs Card 量測：

| 指標 | 值 | 容差 | 結果 |
|---|---|---|---|
| median \|rel_diff\|（13 點） | **{{cim.card_median_pct}}%** | ≤ 10% | ✅ PASS |
| p95 \|rel_diff\|（13 點） | **{{cim.card_p95_pct}}%** | ≤ 20% | ✅ PASS |

[^cim13] Alpha 的 `G_eff(N,K)` 擬合因此 Card-confirmed，凍結解除（決策：保留 Alpha 擬合，不重擬）。Card 系統性略低於 Alpha（12/13 點），小-N 差較大（~10%）、大-N 收斂（~1.5%）——SDK 版本跨版小偏移，在容差內。

### 3.3　prefill M-amortization 擬合（Phase 1.2）

Phase 1.5 把擬合基底從 3 點（M∈{64,128,256}）擴成 **{2..320} 共 23 個 dense 點**（直接量 canonical tile）：仿射 `tile_lat = {{cim.prefill_affine_a}} + {{cim.prefill_affine_b}}·M` µs，fit max rel_err **{{cim.prefill_fit_max_pct}}%**，**留出驗證 median {{cim.prefill_holdout_pct}}%**（一半擬合、預測另一半）。代表點：

| M | 量測 µs | 預測 µs | rel_err |
|---|---|---|---|
| 2 | 41.28 | 40.27 | 2.0% |
| 64 | 45.6 | 46.45 | 1.9% |
| 256 | 65.73 | 65.59 | 0.1% |
| 320 | 72.12 | 72.00 | 0.2% |

> **修正先前假設：M_MAX=256「SRAM tiling 牆」高估了約一半。** dense sweep 編到 **M={{cim.prefill_m_max}}** 全部成功、無 error，**M=511 起失敗**（`no_model_json`，至 4096 全失敗）——真正的牆釘在 **~M=510**（M={{cim.prefill_m_max}} 成功、M=511 失敗），約為舊假設 256 的 2×。所以牆是真的（SRAM 相關），只是位置被低估了一倍。prefill 校準範圍延伸到 M≤{{cim.prefill_m_max}}（M> 此外推）。[^cim14]

---

## 4　多-tile residency cliff 與外推區

### 4.1　native multi-tile：residency cliff（Phase 1.5 Card-native，校準）

> **修正先前假設：`axcompile` 不只能編單一 2048×2048 tile。** Phase 1.5 直接在 Card 上 native 編譯 multi-tile GEMM 到 **K·N≈{{cim.native_envelope_m}}M**。M=1 native 吞吐隨 K·N 平滑上升到 ~264 GOP/s 直到 **knee ≈ {{cim.cliff_knee_m}}M params**（權重 resident 於 on-chip SRAM），越過後 **崩落 ~3.5× 到 ~{{cim.cliff_floor_gops}} GOP/s 的 memory-bound floor**（權重溢出到 DRAM，~M2 streaming BW）。knee ≈ on-chip SRAM 權重容量，物理清晰。

舊模型對每個 multi-tile GEMM 用單-tile 吞吐 tile-sum：knee 以下系統性**高估**（編譯器 fuse tile，比 sum 快），knee 以上 **嚴重低估 ~−65%**（完全沒有 cliff），整體 abs median 誤差 **{{cim.multitile_old_median_pct}}%**（max {{cim.multitile_old_max_pct}}%）。兩-regime cliff 模型（resident `lat=a+b·K·N`、spill `lat=2·K·N/floor`）對所有 Card-native 點：**舊 tile-sum median {{cim.multitile_old_median_pct}}% / max {{cim.multitile_old_max_pct}}% → 新 cliff median {{cim.multitile_new_median_pct}}% / max {{cim.multitile_new_max_pct}}%**，留出驗證 median **{{cim.multitile_holdout_pct}}%**（涵蓋完整 K-staircase + N-staircase + mixed 形狀；max {{cim.multitile_new_max_pct}}% 來自 pure K/N-tiling 點，比 f(K·N) 略慢——即沿單一維 tile 比同 K·N 的方形稍慢）。[^cim16] K·N > {{cim.native_envelope_m}}M（native envelope 之外）由 spill floor 外推（memory-bound 線性，低風險，`is_extrapolated=True` 標記）。**Phase 2 影響**：真實 decode FFN/lm_head GEMV（K·N≥16M）正落在 spill regime——舊 tile-sum 低估約 3×，新模型修正後 per-op M1 延遲對 Phase 2 事件引擎才正確。[^cim17]

> **2048 不是 K 也不是 N 的編譯極限（圖 e）。** 圖 P1 的「2048」是**輸出 tile 寬度 W**（4 cores × 512），不是某個維度的上限。兩條對稱的 staircase（M=1）證實：**K-staircase**（固定 N=512、掃 K）與 **N-staircase**（固定 K=512、掃 N）**都一路 native 編譯到至少 16384**、吞吐隨維度上升（K 或 N：2048→~106、12288→~250 GOP/s），且在 **K=N=16384/512（K·N=8.4M）撞上同一個 ~{{cim.cliff_knee_m}}M cliff**（~72 GOP/s）。crossbar 沿 K 用 16 banks × 512-in 內部累加、沿 N 內部分塊，全由 axcompile 處理。**所以 K>2048 與 N>2048 都不需要我們手動 tiling**——它們原生可編、可量；唯一天花板是 K·N residency cliff（K 或 N 任一方向皆同 knee，證實 cliff 是 K·N 總量效應）。我們舊模型沿 N 做 tile-sum（圖 P1 橘虛線）只是早期的近似，已由本章 native N>2048 量測 + cliff 模型取代。

### 4.2　prefill M 橋接 decode：1 ≤ M ≤ {{cim.prefill_m_max}} 連續

dense sweep 顯示仿射律從 M={{cim.prefill_m_max}} 一路平滑到小 M，並 **橋接到 M=1 decode anchor（41.83µs）within 3.5%**——先前「1<M<64 未橋接、disagreement ~2.5×」是稀疏擬合的 artifact。M∈{2..{{cim.prefill_m_max}}} 現皆為量測校準（`prefill_M_min=2`）；M>{{cim.prefill_m_max}} 或 partial-width tile 才標 `prefill_extrapolated=True`。Axis-C 量到 M-axis chunked serving 為加性（total = n×chunk），per-chunk host/DMA overhead 已單獨列出（`m_tiled_chunked`，非 fused large-M 編譯）。

### 4.4　device envelope 是 assumption

~14 MB IOMMU window（allocatable device memory 的實際上限）是**論壇資料 + 推斷**，非板上直接量測。alloc_envelope_param_count 6M（SDK 上限）與 native_max_kn 4.19M 是不同概念。[^cim11]

### 4.5　compute ceiling 未建模

decode 的實測效吞吐約 227 GOP/s，約為峰值 209,600 GOP/s（4 core × 52.4 TOPS/core）的 **0.1%**；decode 完全受記憶體限制，compute ceiling 不需建模（issue #16）。

---

## 5　進 Phase 2 就緒度

| 項目 | 狀態 |
|---|---|
| M=1 decode `dev_lat_us(K,N)` 介面 | ✅ 凍結，calibrated |
| M>1 prefill `dev_lat_us(M,K,N)` 介面（dense M∈{2..{{cim.prefill_m_max}}}） | ✅ 凍結，calibrated（holdout {{cim.prefill_holdout_pct}}%） |
| prefill M > {{cim.prefill_m_max}} | ⚠ 外推，`prefill_extrapolated=True`（真牆 ~M=510；max compiled {{cim.prefill_m_max}}，M=511 fail） |
| multi-tile residency cliff（K·N ≤ {{cim.native_envelope_m}}M） | ✅ calibrated（Card-native，cliff 模型 median {{cim.multitile_new_median_pct}}%；舊 tile-sum 為 {{cim.multitile_old_median_pct}}%） |
| multi-tile K·N > {{cim.native_envelope_m}}M（spill floor 外推） | ⚠ 外推，`is_extrapolated=True`（memory-bound 線性） |
| Card-revalidated，凍結解除 | ✅ CARD_REVALIDATED |
| `CimTileModel` + `params/m1_cim.json` | ✅ 模組化，Phase 2 直接調用 |
| edge topology（`cim_topo_edge.json`） | ✅ 建好；noc_efficiency=0.9 為 assumption，待 edge silicon 校準 |

**Phase 1.5 已補（Card-native）：** multi-tile residency cliff（解鎖 8B FFN/lm_head decode 的 spill-regime 校準）、prefill dense M∈{2..{{cim.prefill_m_max}}}（舊 M_MAX=256 高估約 2×，真牆 ~M=510）、decode↔prefill 橋接、K 維原生可編到 ≥16384（K=2048 不是 K 的極限，見 §4.1）。**KV-append BW SPIKE 為 inconclusive**（proxy 結構上進不了 DRAM；見 §4.3 / 第 3 章）。

**Phase 2 仍需要的：**
- prefill 端到端 TTFT 驗證（用上述 dense prefill + cliff 模型組裝整條路徑）。
- edge topology 的 `noc_efficiency` 校準：當有 edge silicon 可量時進行。
- multi-tile spill 的 M>1 行為（Phase 1.5 cliff 主要在 M=1 量；prefill multi-tile 僅少數點驗證 +13–17%，knee 以上未測）。

無 blocking gap：M1 介面穩定，Phase 1 端到端 recompose 已在 M1 calibrated decode 路徑上通過 8B hold-out（9.5%）。

---

## 圖

**圖 P15 — Phase 1.5 on-Card 量測補量總覽（Metis Card，101 tasks）**

![phase1_5_cim_campaign](../../../figures/phase1.5/phase1_5_cim_campaign.png)

(a) M=1 native multi-tile 的 residency cliff：吞吐隨 K·N 平滑上升到 knee≈{{cim.cliff_knee_m}}M（權重 resident 於 SRAM），越過後崩落 ~3.5× 到 ~{{cim.cliff_floor_gops}} GOP/s 的 DRAM-spill floor；兩-regime 模型線貼合量測。(b) 對 Card-native 點的模型誤差：舊 tile-sum median {{cim.multitile_old_median_pct}}% → 新 cliff 模型 {{cim.multitile_new_median_pct}}%（留出 {{cim.multitile_holdout_pct}}%；含 K-tiling 與 N-tiling）。(c) dense prefill M-sweep：仿射擬合（留出 {{cim.prefill_holdout_pct}}%），舊 M_MAX=256 假設高估約 2×（真牆 ~M=510，max compiled {{cim.prefill_m_max}}），並橋接 M=1 decode anchor。(d) KV-append BW SPIKE：K=1 proxy（黃；實心=掃 M、空心菱形=改掃 N，落在同一條趨勢上）eff_BW 隨傳輸量上升、**結構上無法收斂**——其編譯牆（working set ~{{kv.spike_proxy_max}}M）落在 ~{{kv.spike_knee}}M 的 DRAM knee **之下**，永遠進不了 DRAM regime，量到的是 on-chip throughput。**收斂的 DRAM BW 來自 cliff 的 spill regime**（橘方塊，{{kv.spike_dram_bw}} GB/s，在 K·N 8–17M 平坦）；M2 有效值 {{kv.spike_m2_bw}}。kv-append 維持 analytic，精確係數 = Phase 2。(e) **2048 不是 K/N 極限**：K-staircase（掃 K）與 N-staircase（掃 N）都 native 編到 ≥16384、吞吐隨維度上升，在 K·N=8.4M 撞同一個 cliff；2048 只是輸出 tile 寬度 W，K>2048 與 N>2048 都不需手動 tiling（§4.1）。

---

**圖 P1 — decode 延遲 vs N（K=2048）：原生量測 + 外推邊界**

![P1_cim_staircase](../../../figures/phase1.1/P1_cim_staircase.png)

這裡掃的是**輸出寬度 N**（固定 K=2048）：N ≤ 2048（= W，4 cores × 512）黑實線貼合量測藍點，是 native combined tile；N > 2048 才需沿 N 分塊。**注意這個 2048 是輸出 N 的 tile 寬度，不是 K 的限制**——K（contraction 維）原生可編到 ≥16384（§4.1 K-staircase），不需 tiling。（Phase 1.5 後 N>2048 的 multi-tile 已由 §4.1 的 residency-cliff 模型校準，取代此圖橘虛線的 tile-sum 外推。）

---

**圖 P2 — 2D 有效吞吐 G_eff(N,K)：量測 vs 擬合**

![P2_cim_geff](../../../figures/phase1.1/P2_cim_geff.png)

每個顏色代表一個 K 值；點為 Alpha 原生量測，線為 2D 擬合。同一 N、K 越大（顏色越暖）吞吐越高——K 效應可擬合。高-K 角落（K=4096, N=1024）殘差誠實呈現。

> CIM-Card 13 點交叉驗證（median \|rel_diff\| {{cim.card_median_pct}}%、p95 {{cim.card_p95_pct}}%、`CARD_REVALIDATED`）目前無對應的已 commit 圖檔；數值見上表，原始逐點資料於 `validation/reports/phase1.2/cim_card_revalidate.json`。

---

## 腳注

[^cim1]: 來源 `validation/reports/phase1.1/m1.json` › `architecture` = "quad-core, 512x512 INT8 D-IMC per core; n_cores free (=4 Metis); GOP/s not FLOP/s"
[^cim2]: 來源 `simulator/models/m1_cim_tile.py` › `CimTileModel.width` = `n_cores * core_width`；`simulator/models/params/m1_cim.json` › `n_cores` = 4，`core_width` = 512
[^cim3]: 來源 `validation/reports/phase1.1/m1.json` › `unit_note` = "throughput is INT8 GOP/s (issue #18); the raw JSON field name 'dev_gflops' is legacy."
[^cim4]: 來源 `simulator/models/params/m1_cim.json` › `G_eff_Gmax_gops` = 333.67，`G_eff_Na` = 577.2，`G_eff_Kb` = 574.1
[^cim5]: 來源 `simulator/models/params/m1_cim.json` › `prefill_tile_a_us` = 38.813，`prefill_tile_b_us` = 0.1033；`validation/reports/phase1.2/cim_prefill_fit.json` › `affine_fit_tile_lat_us.asymptote_TOPS` = 81.2
[^cim6]: 來源 `simulator/models/params/m1_cim.json` › `prefill_M_decode_anchor` = {M:1, tile_lat_us:41.83, gops_measured:200.5}；`validation/reports/phase1.2/cim_prefill_fit.json` › `decode_anchor_M1_measured.gops_measured` = 200.5
[^cim7]: 來源 `validation/reports/phase1.2/cim_prefill_fit.json` › `full_gemm_meas_vs_pred` M=128 meas_us=724.1；線性-M 外推值 65843 µs 由 G_eff 公式計算，比值 65843/724.1 ≈ 91×（`simulator/models/m1_cim_tile.py` docstring 概記為 ~80×，本報告採實算比值）
[^cim8]: 來源 `simulator/models/params/m1_cim.json` › `native_max_kn` = 4194304；`_doc` = "G_eff(N,K) 2D throughput (GOP/s, INT8) fit on native single-tile pts (K*N<=4.19M)."
[^cim9]: 來源 `validation/reports/phase1.2/cim_card_revalidate.json` › `honesty` = "CIM = Alpha 13pts calibrated + Card-revalidated (same AIPU, 800MHz, no rescale)."；`compile_path` = "axcompile"
[^cim10]: 來源 `validation/reports/phase1.2/cim_prefill_fit.json` › `honesty` = "…DENSE Phase-1.5 canonical-tile sweep (dense M=2..508 compile; M=511/512 fail with no_model_json). The old M_MAX=256 'SRAM wall' was ~2x too low: the real wall is ~M=510, not 256."
[^cim11]: 來源 `simulator/specs/cim_topo_alpha.json` › `alloc_envelope_MB` = 14，`provenance.alloc_envelope_MB` = "~14 MB vs 1 GiB BAR [assumption]"；`validation/contracts/m1.yaml` › `device_envelope` = "~6M params = PCIe-IOMMU window (default ~14MB, Alpha no on-card DRAM), NOT 32MB L2/SRAM"
[^cim12]: 來源 `validation/reports/phase1.1/m1.json` › `throughput_fit_gate_native.median` = 0.027，`p95` = 0.149，`max` = 0.176，`pass_median_le_0.10` = true，`pass_p95_le_0.20` = true
[^cim13]: 來源 `validation/reports/phase1.2/cim_card_revalidate.json` › `consistency.median_rel_diff` = 0.048，`p95_rel_diff` = 0.097，`n` = 13；`status` = "CARD_REVALIDATED"
[^cim14]: 來源 `validation/reports/phase1.2/cim_prefill_fit.json` › `affine_fit_tile_lat_us` = {a:40.84,b:0.0941,fit_basis:"M in {2..508} (31 dense pts)"}，`fit_quality.max_rel_err` = 0.031，`holdout.median_rel_err` = 0.009；`simulator/models/params/m1_cim.json` › `prefill_M_max` = 508（M=511/512 編譯失敗）
[^cim16]: 來源 `validation/reports/phase1.5/cim_multitile.json` › `old_vs_new` = {old_tilesum_median:0.313,old_tilesum_max:1.00,new_cliff_median:0.028,new_cliff_max:0.125}，`resident_holdout.median_relerr` = 0.033（unique-K*N split，含 k_staircase + n_staircase；gate ≤0.10）
[^cim17]: 來源 `validation/reports/phase1.5/cim_multitile.json` › `model` = {knee_M_params:8.16,spill_floor_gops:69.7,native_envelope_kn:16777216}，`phase2_note` = "real decode FFN/lm_head GEMVs (K*N >= 16M) live in the SPILL regime; the old tile-sum under-predicted them ~3x"；K*N > native_envelope 由 spill floor 外推（`is_extrapolated=True`）
