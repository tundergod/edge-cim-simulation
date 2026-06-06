# G — GPU：Mali-G610 的「解析 roofline 換型號槽」（與 micro-benchmark 並存）

> **這一章你會學到**：為什麼 GPU 在這個 CIM 架構裡是「注意力 offload」的角色、Phase 1.1 量到的那個 micro-benchmark 模型是主力、為什麼這一期又多做一個「解析 roofline」當**可換型號的槽**、它怎麼用 `max(算, 搬)` 一條式子同時抓住 decode（memory-bound）和 prefill（compute-bound）兩個區段、以及一連串**不能不講清楚的誠實邊界**——尤其是 **INT8 在 GPU 上根本零資料、量到的 `20.12` 是 FP16 不是 INT8、峰值 512 是假設**。

---

## G.1 架構考量：GPU 是誰？為什麼這一期要多一個 roofline？

回顧整體設計：CIM 負責「投影矩陣（projection）那種 weight × activation」，而 **GPU 的角色是「注意力 offload」**——把 CIM 不擅長的 activation × activation bmm（QK^T、S·V）丟給它。Phase 1.1 已經量了一個 **micro-benchmark 模型**（`m4_gpu.py` / `MaliGpuModel`）：拿真的 Mali matmul kernel 跑出來的 attn-bmm 斜率 + GEMM 飽和吞吐當下界，這個**仍是 GPU 的主力（PRIMARY）資料源**。

那這一期（Phase 1.2）為什麼還要再做一個東西？因為這個模擬器的設計是「**引擎 + 可換 spec**」（D5）：每個單元都應該能「換一顆型號 = 換一個 spec 檔、引擎不動」。micro-benchmark 模型綁死在「我們手上這顆 Mali-G610 的實測點」，**換一顆 GPU 就沒資料了**。所以這一期補一個 **`GpuRooflineModel`（解析 roofline）當換型號的槽（swap slot）**：它只吃一個 spec（峰值 + 校準效率），就能對任意形狀給一個 latency。

**兩者的關係要講白：micro-benchmark = 主力（實測），roofline = 換型號槽（解析形狀趨勢，非嚴格下界）。** 不是取代，是並存。`m4_gpu.py` 這一期**完全沒動**。

---

## G.2 原理 + 參數：一條 `max(算, 搬)` 同時抓兩個區段

roofline 的式子就是經典的「算 vs 搬，取較慢者」：

```
latency_us = max( compute_us , memory_us )
  compute_us = 2·M·K·N / (eff_compute · fp16_peak)      ← FLOPs ÷ 有效算力天花板
  memory_us  = nbytes   / mem_eff_BW                     ← bytes ÷ 有效頻寬
  nbytes(預設) = (K·N + M·K + M·N) · bytes_per_elem      ← 權重 + 輸入 act + 輸出 act
```

兩個校準參數**都對 `mali_matmul.json` 的 FP16 點校準**（`tools/analysis/fit_gpu_roofline.py`）：

| 參數 | 值 | 怎麼來的 | 標記 |
|---|---|---|---|
| `eff_compute_fp16` | 0.01965 | ksweep 收斂尾端 f16 吞吐 **20.12 GFLOP/s**（最高-M 點，非 M=512 的 20.29 暫態峰；= spec `measured_fp16_gflops`）÷ FP16 峰值 1024 | calibrated（FP16） |
| `mem_eff_BW_GBs` | **1.256 GB/s** | 16 個 decode-GEMV 點，`lat = nbytes/BW` 過原點 `numpy.linalg.lstsq` | calibrated（FP16） |

- **有效算力天花板 = `eff_compute · 1024 ≈ 20.1 GFLOP/s`**：這是 ksweep 大方陣**收斂尾端**（最高 M）的 f16 吞吐——一個**自己寫的、未優化 OpenCL kernel** 的飽和值，所以對「真正調過的 kernel」而言它偏保守（調過會更快）。
- **有效頻寬 ≈ 1.26 GB/s**：decode 是 batch-1 GEMV，純搬權重、memory-bound；把 16 個 decode 點的「latency vs 搬的 bytes」用最小平方擬一條過原點的線，反推出有效頻寬。同樣是這顆未優化 kernel 的下界。

**為什麼一條 `max` 就夠？** 因為 decode（M=1 的 GEMV）會落在 `memory_us` 這一支（搬權重主導）、prefill / ksweep 大方陣會落在 `compute_us` 這一支（算主導）——`max` 自動在兩個區段間切換，`bound` 欄位（`'memory'`/`'compute'`）會告訴你是哪一支贏。

---

## G.3 Simulated vs reference：roofline 對 1.1 量測點的誤差

**圖 G1 — Mali-G610 解析 roofline（FP16）對 micro-benchmark 點**
![G1](../../../figures/phase1.2/G1.png)

- **(a) 左圖**：X = 實測 FP16 latency，Y = roofline 預測，對角虛線是 `y=x`。**橘點 = decode（memory-bound）、藍方 = prefill/ksweep（compute-bound）**。灰帶 = `y=x` 以下（預測 ≤ 實測）。**大多數點落在對角線上或略低，但約 1/3 略高於對角線（最多 +5%）**——所以這是**形狀趨勢擬合、不是嚴格下界**（`frac_pred_le_measured ≈ 0.53`、`frac_within_5pct ≈ 0.67`）。
- **(b) 右圖**：每個形狀的**帶號**相對誤差 `(pred−meas)/meas`。整體 **median |誤差| = 3%、p95 = 36%**。大多數點貼著 0；負的長尾來自兩個離群點：一個 `M=1, K=2048, N=2048` 的 1b decode 點（實測異常慢、−66%），和最小的 `M=64` ksweep（kernel launch 開銷佔比大、−23%）。這些**負偏差 = roofline 在那兩點比實測更樂觀**；但也有約 1/3 的點是正偏差（over-predict，最多 +5%），所以**不是嚴格下界**。

**怎麼讀這張圖（重點）**：median 3% 不是在宣稱「校準很準」——它只是說「**這條 roofline 抓對了 decode/prefill 兩個區段的形狀趨勢，大多坐在實測之下（但非嚴格下界，~1/3 略高 ≤ +5%）**」。誤差數字記在 `validation/reports/phase1.2/m4_gpu_roofline.json` 的 `error_vs_1p1_measured`（30 點逐點 + median/p95/max）。

---

## G.4 誠實標註（非協商）：這一章最重要的一段

這個模型有一連串**不能含糊**的邊界，全部標 `simulated`：

| 項目 | 狀態 | 說明 |
|---|---|---|
| **INT8 GPU GEMM** | ❌ **零資料** | Mali matmul kernel 只有 FP32/FP16；**INT8 完全沒量**。對 int8 workload，`predict()` 仍用 FP16 校準的天花板，並在 provenance 明標「dtype=int8 has ZERO GPU data; FP16 ceilings used」。 |
| **量到的 `20.12`** | ⚠️ **是 FP16** | Phase 1.1 那個 20.12 GFLOP/s 飽和點是 **FP16，不是 INT8**。整個 fit、整個 predict 都是 **FP16 only**。 |
| **FP32 峰值 512** | ⚠️ **assumption** | spec 的 `fp32_peak_gflops=512` 是假設，**可能低估 2–4×**，需驗證。本模型校準對 **FP16**，所以**不依賴**這個 FP32 峰值——但 spec 裡仍誠實標 assumption。 |
| **roofline 本身** | ⚠️ **形狀趨勢（非嚴格下界）** | 未優化 kernel + 只有 **5 個飽和點** → **不是可轉移的校準（NOT transferable）**。大多 `predicted ≤ measured`（~2/3），但 ~1/3 略高（最多 +5%）。**沒有數值驗收 gate**（no INT8 silicon → no fake gate）；驗收 = 趨勢形狀。 |
| `ksweep_saturation_M` | 🪧 dead param | spec 裡這個欄位是**死參數**（保留、不刪，依稽核清單）；本引擎用不到。 |

**為什麼沒有 INT8 sim？** 一句話：**因為沒有任何 Mali INT8 GEMM 的量測資料**。我們不會憑空編一個 INT8 數字來假裝有 gate——那違反「no fake gate」。要 INT8，得先有 silicon 量測（未來工作）。在那之前，這顆 GPU 在模擬器裡只能給 **FP16 的下界趨勢**。

---

## G.5 限制與 gap（誠實清單）

- **可換性已就緒**：`GpuRooflineModel(spec, engine='analytic')` 符合凍結合約 `{latency_us, bound, provenance}`；換一顆 GPU = 換 `roofline_fit` 那塊（重跑 fit 或填新峰值）。
- **micro-benchmark 仍為主**：端到端 recompose / offload 用 `m4_gpu.py`（實測 attn 斜率），roofline 是換型號槽——兩者並存，職責不同。
- **缺口**：(1) INT8 零資料；(2) FP32 峰值 512 待驗證；(3) 5 點飽和 → 不可轉移；(4) 沒有 GPU 記憶體 BW 的獨立 micro-benchmark，`mem_eff_BW` 是從 decode latency 反推的有效值（含 kernel 開銷）。

**一句話總結 G**：GPU 這一期多了一個**解析 roofline 換型號槽**（`max(算,搬)`，兩軸都對 FP16 校準、是**形狀趨勢擬合（非嚴格下界）**），它抓對 decode/prefill 兩個區段的形狀、大多坐在實測之下（median |誤差| 3%，~1/3 略高 ≤ +5%）；但 **micro-benchmark 仍是主力**，而且最重要的是——**INT8 零資料、`20.12` 是 FP16、峰值 512 是假設、roofline 不可轉移**，全部誠實標 `simulated`，**不編任何 gate**。
