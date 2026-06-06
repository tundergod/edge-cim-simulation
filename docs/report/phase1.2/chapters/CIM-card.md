# CIM-Card — CIM 計算 kernel 重驗（同一顆 AIPU，未凍結）

## 為什麼 CIM kernel 不該被當成「凍結」

Phase 1.1 的 CIM 計算引擎（M1）校準在 **Metis Alpha 的 13 個 native single-tile 量測點**（`simulator/models/params/m1_cim.json`，2D `G_eff(N,K)` 擬合，2.7%）。Alpha 是 pre-production 板、**跑不了 LLM**（封閉韌體 `-1301` 牆 + 無 on-card DRAM），所以當時無法在 compute-bound regime 多取點。

但**同一顆 quad-core AIPU 在量產 Metis Card 上是活的**（`machines.md`：`metis-0:7:0 16 GiB clock=800MHz`），而且 `axrunmodel` 的 `dev_fps` 是**隔離計算**的指標（dev/system split，不是合成差分）。**兩板都 800 MHz** → Card 的 `dev_lat`/`dev_gflops` 可以**直接對** Alpha 13 點比，**不需 clock 正規化**。

所以 CIM kernel **不是凍結的**：我們可以在 Card 上用同一個 1×1-conv matmul proxy **重新量測/交叉驗證**，並補上 Alpha 量不到的 **prefill / compute-bound** 形狀。這把「Alpha 凍結」的疑慮解除，也補了 decode 量不到的計算上界。

## 方法（已執行 2026-06-06）

移植 Alpha 的 `characterization/aetina/run_metis_cim.py` 到 Card → `characterization/metis_card/run_metis_cim_v16.py`，在 metiscard 上用 axelera-devkit 的 **`axcompile`** 編 1×1-conv matmul proxy → `axrunmodel` 量 `dev_fps`（隔離計算）：

- **編譯路徑（spike 修正）**：v1.6 把低階 `compile` **改名為 `axcompile`**（Artifactory wheel，官方 Beta），**不是移除**——更正先前「v1.6 無 compiler」的結論。raw `MatMul`/`Gemm` 仍編不出（`ONNXGraphCleanerError`）→ 用 1×1-conv proxy（數學等價）。
- **decode（M=1）交叉驗證**：Card 重量 Alpha 13 個 native-tile 形狀，`dev_gflops` 直接對 Alpha（同 800 MHz、無 rescale）。
- **prefill（M>1）—— compiler SRAM-tiling 牆**：v1.6 `axcompile` **編不出大 prefill GEMM**，但**不是** device-DRAM envelope（Card 的 16 GiB 不解此限），而是 **compiler 的 SRAM L1/L2 tiling 牆**：單一 ~2048×2048 tile 只在 **M≤256** 編得出，M=512 或大-N FFN 直接 fail。AIPU 本就把 GEMM 切 native tile，所以照 Alpha 的 `SAFE_KN` 切法量 canonical tile，full GEMM = `(K·N/W²) × tile_lat(M)`（**fractional tile area，非 ceil**：partial-width GEMM 不會被多收一整個 tile）。

## 本期狀態：`CARD_REVALIDATED`

Card 重驗**已執行**（板先前處於 PCIe AER uncorrectable-fault，`axdevice --refresh` 救回）。`validation/reports/phase1.2/cim_card_revalidate.json` 轉 `CARD_REVALIDATED`：

- **decode 交叉驗證（13 點）**：median `|rel_diff|` **4.8%**、p95 **9.6%**（square M=1,K=2048,N=2048：Card 200.5 vs Alpha 203.7，rel_diff=0.015=1.5%）。Card **系統性略低於** Alpha（12/13 點），小-N 差較大（~10%）、大-N 收斂（~1.5%）——同一 kernel 跨 SDK（Alpha v1.3.1 → Card v1.6）的小偏移。**在 decode 擬合容差內（median≤10%）→ Alpha 的 `G_eff(N,K)` 擬合 Card-confirmed、解凍、不重擬**（`m1_cim.json:decode_card_revalidation`）。
- **prefill M-amortization（新，Alpha 量不到）**：`tile_lat(M) = 38.81 + 0.1033·M` µs（asymptote ~81 TOPS），**只擬合在真正的 prefill 點 M∈{64,128,256}**（full 2048×2048 tile）；M=1（41.8µs=200.5 GOP/s）是 decode regime、**分開報告、不混入此線**。擬合 max rel-err **0.6%**。`m1_cim_tile.dev_lat_us` 加 M>1 分支（M=1 decode 路徑**逐字不變**），用 fractional area：`dev_lat_us(128,4096,14336)` = 728µs vs 量測 724µs，**消除原本線性-M 外推的 65843µs（~80× 偏離）**。M>256 或 partial-width tile（K 或 N 非 2048 倍數）標 `prefill_extrapolated`（未校準）。
- **TTFT 交叉驗證（`recompose_e2e.py`，P14）**：8B prefill（M=1024，**> 量測 M_max=256**）擬合 compute = 0.26s = vendor TTFT 3.79s 的 **7%**；舊線性-M decode-GEMV 估計 75s **違界 ~20×（REFUTED）**。狀態 `UNGATED` → **`BOUNDED-EXTRAPOLATED`**（M=1024 為外推）。誠實註記：`compute ≤ TTFT` 對 fitted 模型**鑑別力弱**（任何合理 prefill 長度都遠低於 TTFT，界要到 M≈22000 才 bind）——其價值在**反證線性-M**，而非絕對驗證 fitted。TTFT 餘量 = weight-load 記憶體 + prefill attention + host overhead（Phase-2）。

## Edge-CIM 記憶體牆（前瞻、assumption）

`simulator/specs/cim_topo_edge.json`（topology C）：整合在 mobile SoC 的 edge CIM，**無專屬 on-card DRAM**，weights 從 SoC 共用 **LPDDR5** 經片上 NoC 串流。

- **記憶體牆 = 目標 LPDDR5 eff_BW × noc_efficiency**：`mem_lpddr5.eff_BW_GBs`（33.3，已含 LPDDR5 DRAM-controller 0.65 系統效率）× `noc_efficiency`（0.9，額外 NoC/arbitration overhead，**assumption**）= **30.0 GB/s**。從 eff_BW 起算（**不是** peak×noc，避免丟掉已量到的系統效率）。
- **不是 Card 的 24.2 GB/s**：24.2 是 Card 專屬 on-card **LPDDR4x** 的 decode 牆（topology-specific、較舊記憶體）；edge 用 LPDDR5 → BW 較高。**24.2 不可移植到 edge**。
- **CIM 計算 kernel 共用**：同一顆 800 MHz AIPU、Card-revalidated（topology-agnostic）。**未驗證**：無 edge silicon，`noc_efficiency` 是待校準的 assumption（合理區間 [0.7,1.0] → eff_BW [23.3,33.3]）。
- 註冊進 `validation/contracts/specs.yaml` + `tools/analysis/check_phase1_2.py`（`op="stream"` probe，honesty=assumption，gate exit 0）。

## 一句話

CIM 計算 kernel 在量產 Card 上**重驗通過**（decode 13 點 median 4.8%、解凍 Alpha 擬合；prefill M-amortization 補上、消除 ~80× 偏離），edge 記憶體層以**明標 assumption** 的 target-LPDDR5×NoC 牆建好（非 Card 24.2）。可信度只升不降。
