# C — M4-CPU：支援算子的「指令數 roofline」（校準到 fp32 silicon）

> **這一章你會學到**：為什麼 CPU 上那些「不是大矩陣乘法」的小算子（RMSNorm、RoPE、softmax、SwiGLU、residual、argmax 取樣）也要算進去、為什麼 `exp()` 是它們裡面真正花時間的那一個、我們怎麼用一條 **roofline**（取 max(算, 搬) + 固定開銷）把它們一次描述完、以及這條 roofline 是怎麼**對手上真 silicon（RK3588 A76 核）校準**出來的——誤差中位數只有 **1.1%**。

---

## C.1 架構考量：M4-CPU 是誰？為什麼這些「小算子」要算？

異質 SoC 上，重活（GEMM/attention）丟給 NPU/GPU/CIM，但 LLM 一層裡還有一堆**非-GEMM 的支援算子**：歸一化（RMSNorm）、位置編碼（RoPE）、注意力的 softmax、FFN 的 SwiGLU 閘、殘差相加（residual）、最後輸出時對 vocab 取 argmax（greedy 取樣）。這些在 profile 裡被指派給 **CPU**（RK3588 的 big.LITTLE：4×A76 + 4×A55）。

它們單看每個都很小（decode 一個 token 約 1–400µs），但**每層每 token 都要做、層數又多**，加總起來在 decode（memory-bound、每個大算子本來就不快）裡並不可忽略。所以 M4-CPU 要能**給每個支援算子一個時間**，而且要能隨「模型大小、kv 長度、精度、核數」變動——這樣端到端模擬器才接得起來。

---

## C.2 spec：cpu_rk3588（每欄位帶 provenance + honesty tag）

引擎吃一份**可換的 spec**（`simulator/specs/cpu_rk3588.json`），所有硬體數字連同 provenance 都在裡面，換型號 = 換這份檔案：

| 欄位 | 值 | provenance / tag |
|---|---|---|
| clusters | A76 4 核 @2.3GHz IPC=2、A55 4 核 @1.8GHz IPC=1 | RK3588 TRM `[measured]` |
| NEON | 128-bit，fp32 4 lanes / fp16 8 lanes | ARMv8.2-A `[measured]` |
| cache | L1d 64KiB/核、L2 512KiB/核、L3 3MiB 共享 | A76 TRM `[measured]` |
| cache_bw_GBs | L1 73.6、L2 36.8、L3 18.4 GB/s（/核） | A76 TRM 推估 **`[assumption]`** |
| calibration_basis | **單 A76 核、單緒、numpy fp32** | Phase 0.3 protocol `[measured]` |

⚠️ **兩個邊界寫進 spec、本章嚴守**：(1) cache BW 是 A76 自己的 SRAM 推估值（`assumption`），**不是** Metis AIPU 的 SRAM tier、**不是**量產卡的 LPDDR4x——WP-CPU 只依這份 spec，不依 WP-MEM。(2) **校準基準是單 A76 核單緒**；多核是把核數加進去**外推**，A55（IPC=1）與多核都標 `simulated`。

---

## C.3 原理：一條 roofline，不是一張查表

前一版（Phase 1.1）這個模組是**查表**——把 cpu_ops.json 的每個 (model, dtype) 量測值存成常數（issue #10：用量測值不用 FLOP）。能用，但**換不了型號**：問「IPC×2 的核會多快」「8 核並行多快」「換更大的 vocab」它都答不出來，因為它只記得量過的那幾個點。

Phase 1.2（D1）改成 **instruction-count roofline**——把每個算子拆成「要做多少次運算」和「要搬多少 bytes」，再除以硬體的吞吐：

```
latency_us = max(compute_us, memory_us) + overhead_op
compute_us = (n_elem * ops_per_elem) / (Σ_assigned[W·IPC·freq]) / η_c
memory_us  = working_set_bytes / (BW_tier(working_set) · η_bw)
```

**逐項白話：**

- **`n_elem`（這個算子處理幾個元素）**：每個算子的「尺寸變數」——softmax = `heads·(kv+1)`、swiglu = `F`（FFN 中間維）、sampling = `V`（vocab）、rmsnorm/residual = `H`（hidden）、rope = `heads·hd`。
- **`ops_per_elem`（每個元素要做幾次運算）= 成本主因在這裡**：**`exp()` 是 transcendental（超越函數），一次 `exp` 在 numpy fp32 上展開成約 30 個融合浮點運算**。softmax 和 swiglu 都含 `exp()`，所以它們的 `ops_per_elem` 給很大（≈30+）；residual/argmax 只是一次加法/比較（≈1）；rmsnorm/rope 介於中間（5–6）。**我們不靠「reduction vs elementwise 二分」——`exp()` 才是真正的軸。**（這是結構性假設 `assumption`，不是 fit 出來的。）
- **`Σ_assigned[W·IPC·freq]`（指派核的算力）**：一個 A76 核的 NEON fp32 吞吐 = 4 lanes × IPC 2 × 2.3GHz = 18.4 G lane-op/s。**多核就把核數乘進去**——這就是「加核會變快」的來源（外推、`simulated`）。
- **`η_c`（實際達成率，校準）**：numpy 的 kernel 不會跑滿 NEON 峰值，實測只到 **≈15.2%**（`η_c=0.152`）。這是**對 fp32 cpu_ops.json 校準**出來的唯一全域算力因子。
- **`BW_tier`（搬資料走哪一層快取）**：依**工作集大小**選 L1/L2/L3——decode 的工作集（幾 KB 到 ~600KB）**全部落在快取裡，永遠不碰 host LPDDR**（「換 LPDDR4→5 重算」只對 prefill）。
- **`η_bw`（快取頻寬達成率）= `assumption`，不是校準**：見 C.5 的誠實說明。
- **`overhead_op`（每個算子的固定開銷，校準）**：rmsnorm/rope/residual/argmax-dispatch 在 decode 尺寸下是**常數主導**的，這個 per-op 固定成本把它吸收掉。

---

## C.4 圖 C1：量測 fp32 vs 校準模型

**圖 C1（M4-CPU roofline）— 6 個子圖、每圖一算子、X=尺寸變數、Y=µs**
![C1](../../../figures/phase1.2/C1.png)

- **怎麼看**：實心圓 = 量測（fp32 中位數，4 個模型各一點）；**×** = 模型在**同一形狀**下的預測。兩者幾乎重疊。
- **紅標題的 softmax / swiglu = `exp()` 主導**：它們的斜率（每多一個元素多花的時間）≈ **12 ns/elem**——**兩個算子斜率一致**，正是「`exp()` 是同一個成本來源」的鐵證。softmax 沿 kv（128→1024）線性上升、swiglu 沿 F 線性上升，模型都抓得很準（中位 0.8% / 1.9%）。
- **sampling_argmax**：掃 vocab 的 reduction，4 個點裡 qwen（V=152K）最大——它的工作集 594KiB **超過 L2、落到 L3**，所以它走的是 **memory（cache）分支**而非 compute（圖上其餘 3 個 128K vocab 走 compute）。這就是 roofline 的 `max()` 真的在切換的地方。
- **residual**：最小、最吵的算子（見 C.5）。

---

## C.5 sim-vs-measured（誠實對照）

**校準強度逐項（對 fp32 cpu_ops.json，單 A76 核）**：

| 算子 | 中位誤差 | 最大誤差 | bound | 註 |
|---|---|---|---|---|
| softmax | 0.8% | 2.4% | compute | `exp()` 主導，斜率 ≈12 ns/elem |
| swiglu | 1.9% | 4.3% | compute | `exp()` 主導，與 softmax 同斜率 |
| sampling_argmax | 1.1% | 1.4% | compute + **memory** | qwen vocab→L3 走 cache 分支 |
| rmsnorm | 1.5% | 2.5% | compute | 常數主導，overhead 16.8µs |
| rope_apply | 1.9% | 4.9% | compute | 常數主導，overhead 22.5µs |
| residual | 5.8% | 13.1% | compute | 見下 |
| **整體** | **1.15%** | **13.1%**（p95 7.3%） | | |

**誠實標註（honesty discipline）：**

- **CPU = `calibrated`**：`η_c` 與 `overhead_op` 對 **fp32** cpu_ops.json 校準。整體中位誤差 1.15%。
- **`exp()` 的 `ops_per_elem`、byte-passes = `assumption`**：指令數的物理估計，不是 fit 出來的（fit 的只有 `η_c` 和 per-op overhead）。
- **`η_bw` = `assumption`，不是校準**：這份 fp32 decode 資料裡**沒有任何「純頻寬解析」的算子**——每個算子的工作集都進得了 L1/L2/L3，而且每個都是 compute-bound 或 overhead-bound；加上 repo 裡**沒有 CPU mem-BW micro-benchmark**（稽核缺口）。所以 `η_bw=0.6` 是文獻常見的快取效率**佔位值**，memory 項只在最大工作集（qwen vocab→L3）才**真的綁住**，而那一點的量測**佐證（不否證）** 這個假設（誤差 1.4%）。它的存在是為了 prefill / 架構研究的更大工作集。
- **residual 誤差 5.8%（最大 13.1%）在它自己的量測噪音內**：residual 是所有算子裡**最吵的**（cpu_ops.json 的 cov 高達 **0.25**，即 25% 量測抖動），值本身只有 1.75–2.04µs（接近 `perf_counter` 解析度、量化成幾個離散階）。它是 **overhead 主導**（固定 floor ~0.8µs），模型誤差小於它自己的量測噪音。
- **fp16 = `simulated` 上界**：fp16 在 A76 上是 **numpy 模擬**（非原生）→ 視為**上界**。引擎的 roofline 校準在 fp32；fp16 路徑回傳同一條 compute roofline（provenance 字串標明「fp16 = emulated UPPER BOUND」）。**swiglu fp16 = 混精度**（silu 的 `exp` 走 fp32）。
- **A55 / 多核 = `simulated`**：校準基準是單 A76 核單緒；多核把核數加進 `Σ`（外推）、A55 用 IPC=1。provenance 字串在非單-A76 路徑會附「A55/multicore EXTRAPOLATED = simulated」。
- **NO FAKE GATE**：沒有發明數值 gate。校準項報的是**對 fp32 silicon 的 per-op 殘差**（真實數字）；非校準項（η_bw、fp16、A55/多核）都明標 assumption/simulated。

---

## C.6 可換性（engine + spec）

- **共用介面**：`CpuModel(spec, engine='analytic')`、`predict(wl)` 回凍結 dict `{latency_us, bound, provenance}`（`bound∈{compute,memory,floor}`）——與其他單元引擎同簽名。換 CPU 型號 = 換 `simulator/specs/*.json`（clusters / cache / BW），引擎碼不動。
- **架構研究就緒**：因為是 roofline 不是查表，可直接問「IPC×2」「8 核並行」「更大 vocab/kv」「更大 L2」這類 what-if——把 `cores`/`cluster` 放進 `Workload.extra`，或改 spec 的 cache 容量/BW 即可。
- **便利路徑**：保留 `op_us(op, model, dtype='fp16', kv=None)`（回 `predict(...).latency_us`），讓 recompose 端到端的接線只需 ~2 行。
- **校準腳本**：`tools/analysis/fit_m4_cpu_instrcount.py` 解 `η_c`/`overhead_op`、報 per-op 殘差 → `validation/reports/phase1.2/m4_cpu.json`；產出的因子寫進 `params/m4_cpu_instrcount.json`，引擎讀它。

**一句話總結 C**：CPU 支援算子用一條 **`max(算, 搬) + 固定開銷`** 的 roofline 描述，**`exp()`（softmax/swiglu）是真正的成本主因**（同一條 ~12 ns/elem 斜率）；`η_c`（≈15% 峰值達成率）與 per-op overhead **對 fp32 真 silicon 校準**（整體誤差中位 1.1%），`η_bw`、fp16、A55/多核都誠實標為 assumption/simulated；換型號只換 spec、引擎不動。
