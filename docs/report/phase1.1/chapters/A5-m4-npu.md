# A5 — M4-NPU：RKNPU2（缺席的第二個 offload 候選）

> **這一章很短，而且故意誠實**：這個元件**本期沒有資料**。它不是被遺忘，而是被一個外部障礙卡住（量測板離線）。我們把「為什麼沒有」「準備好了什麼」「對結論有什麼影響」講清楚——這正是「不藏 gap」的示範。

---

## A5.1 架構考量：NPU 本來該是誰？

§0.4 講過：CIM 不擅長 attention，要 offload。A3 用 **GPU（Mali）** 當 offload 接收方。但其實還有**第二個候選**——**NPU（RKNPU2，RK3588 內建的神經網路加速器）**。

NPU 和 GPU 一樣能做「activation × activation」的運算，所以也能接 attention；而且 NPU 一般預期比 GPU 更省電（這是 NPU 作為專用推論加速器的通則，**本期未實測**）。理想上，Phase 1.1 應該**同時**量 GPU 和 NPU，給排程器（M6）兩個 offload 選項去比較。

---

## A5.2 為什麼是 placeholder（佔位）？

**被一個外部障礙卡住：GitHub issue #13。**

Phase 0.3 量測期間，**aetina 板（跑 RKNPU2 的那塊）離線了**——重開機後仍連不上（連 ping 都沒回應），需要實體到場確認。所以 NPU 的 micro-benchmark（`measurements/aetina/rknpu2_matmul.json`）**沒有收集到**。

這是一個典型的「collect-what-you-can（能量到多少就算多少）」缺口：硬體出狀況時，我們不卡住整個 Phase 1.1，而是**把量得到的（CIM/GPU/CPU/記憶體）先量完、先建模**，把 NPU 標成相依項，等障礙排除再補。

---

## A5.3 已經備妥的部分（所以補起來很快）

雖然沒上板量測，但**前置作業都做完了**，等 aetina 回線就能直接跑：

- **23/23 個 `.rknn` 模型已轉好**（16 個投影 + 7 個 attention bmm），放在 metiscard 上。
- **上板 runner（rknnlite）已就緒**。
- 只差「把 `.rknn` 同步到 aetina → 跑 runner → 產出 `rknpu2_matmul.json`」這幾步。

換句話說，這不是「還沒開始」，而是「**最後一哩卡在硬體上線**」。

---

## A5.4 對 Phase 1.1 結論的影響（其實不大）

**缺 NPU 不會動搖 Phase 1.1 的核心結論**，原因：

1. Phase 1.1 是 **decode-calibrated**，主幹是 **CIM（算）+ 記憶體（搬）**——這兩個都齊了。
2. 「**attention 該 offload**」這個結論，目前**站在 GPU 的數據上就足夠**（A3 的 CIM 31–46ms vs GPU 幾十–幾百µs，差 2 個數量級，對 kernel 品質不敏感）。NPU 只是**第二個佐證**，不是必要條件。

NPU 補上之後會多兩樣東西：

- **第二個 offload 對照點**（NPU 原生 attention vs GPU vs CIM penalty）。
- **「NPU 大投影不需 tiling」的對照**（NPU 不像 CIM 有「輸出超過 4核×512=2048 就要分塊」的限制）。

這些會讓 M6 排程器（A8）在「attention 丟 GPU 還是 NPU」上有更完整的依據，但**不影響 Phase 1.1 已驗證的部分**。

---

## A5.5 給 Phase 1.2 的接口（已寫死）

我們把 NPU 的「未完成」狀態**明確寫進程式與合約**，讓 Phase 1.2 不會誤以為它已完成：

- **`simulator/models/m4_npu.py`**：是一個 stub，被呼叫就 `raise NotImplementedError`，docstring 指向 #13，並說明「等 #13 解掉，就照 A3 的 M4-GPU 方式（`FLOPs/G_eff` + 原生 attention bmm）從 `rknpu2_matmul.json` 擬合」。
- **`validation/contracts/m4_npu.yaml`**：`status: BLOCKED`、`blocked_on: issue #13`，並先列好待調參數（`npu_gemm_gflops`、`npu_attn_bmm_us_per_kv`）。

這就是模組化紀律：**一個沒做完的元件，用「會報錯的 stub + 標清楚的合約」佔位，而不是留一個看起來能用、其實是空的東西**。

---

**一句話總結 A5**：NPU 是 attention 的第二個 offload 候選，本期因 aetina 離線（issue #13）未量測；前置（23 個 `.rknn` + runner）都備妥，等板子回線即可補；缺它不動搖 Phase 1.1 的 decode 校準與「attention 該 offload」結論，只少了第二個佐證。它以「報錯 stub + BLOCKED 合約」誠實佔位。下一章 A6 回到「源頭」——看是誰決定每個 token 到底要跑哪些 op。
