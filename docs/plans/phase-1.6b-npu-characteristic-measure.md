# Plan: Phase 1.6b — 誠實量測 ONNXim / ScaleSim 是否真有 systolic 特性（不預設 knee）

## 0. 錯誤檢討（為什麼需要這個 plan）

先前的 `npu_systolic` 圖是**錯的、且具欺騙性**：

- 我**預設**「有 32 對齊階梯（knee）」，做了一個專門為了「重現 HeteroInfer Fig 3」而設計的 ScaleSim sweep 把階梯畫出來。
- 再把 ONNXim **僅有的 3 個粗點**（N=128/256/512）釘在那條 ScaleSim 階梯上，視覺上**暗示兩個模擬器都有階梯**。
- 事實：ONNXim 在 M=1, K=2048 的既有資料（128→56、256→109、512→217、1024→433 µs）幾乎是 **N 的線性**，**完全沒有證據顯示 32 週期的階**——因為它從來沒被細掃過。

這是循環論證（用「假設有 32-padding」的設定去「找到」32 階）＋跨 sim 的誤導套形狀。**作廢，重做。**

## 1. 原則（這次怎樣才算誠實）

1. **不預設結論。** 虛無假設 H0 =「latency 隨輸出維度平滑單調、沒有 32 週期的階」。只有資料**拒絕 H0** 才能說「有階梯」。
2. **先註冊判準，再看資料**（pre-registered）：在跑之前把「特性存在」的量化判準寫死進程式/報告，避免事後在雜訊裡找圖案。
3. **每個模擬器各自量、各自畫。** 圖上只放每個 sim 自己的原始量測點，**不互相套形狀**。ONNXim 平滑就畫平滑，ScaleSim 有階就有階——**容許兩者結論不同**。
4. **ScaleSim 是 positive control，不是佐證證人。**（review 修正 #4）ScaleSim **本身就是一個 32×32 systolic-array cycle 模擬器**，`ceil(N/32)` tiling 讓 32 量子階梯**幾乎是定義上必然**（既有資料：cycles 每跨 32 邊界恰跳 6080、util 釘在 ~1.05%）。所以「ScaleSim 有 32 階」是**近乎零資訊**的結果——它**不可能不**有。**真正有資訊的是 ONNXim**（它在 systolic 之上再疊 NoC / DRAM scheduling，可能把量子抹平）**有沒有**這個階。ScaleSim 的角色是「確認我們的 probe 抓得到已知存在的階」（positive control）。
5. **兩 sim 的絕對 latency 不可比、不可疊軸。**（review 修正 #5）ScaleSim（1 GHz、/3 cores、util~1%）與 ONNXim（ramulator2-DDR4 25 GB/s + NoC）絕對值差 ~6.7× 且非定值。本研究的問題是**曲線形狀 / 有無階**，不是 magnitude。所以圖**各畫各的 y 軸（或各自對自己 N=128 正規化）**，標「model units, 跨 sim magnitude 不可比（#13）」，**絕不**把兩條絕對曲線疊在同一 y 軸。
6. 全程是 **model，非 RKNPU2 矽**（#13）。HeteroInfer 只是「我們借用同一種 probe 手法」的**動機**，不是答案來源、不是 magnitude 錨點。**新產物不得再用「Fig-3 replica」「HeteroInfer Fig 3」當標籤**（那名字本身夾帶借形狀的假設）。

## 2. 量測設計（pre-registered）

### E1 — 輸出維度細掃（階梯檢定，核心）
- 固定 K=2048，**兩個 M regime 各掃一次**（review 修正 #7）：**M=1（decode GEMV）與 M=128（prefill batch）**。GEMV 與 batched GEMM 的 fill/tiling 行為可能不同，不可只測 M=1 就推廣。
- **N grid（review 修正 #2，真正能解析 32 週期）**：
  - **均勻 step-8 覆蓋 [128, 512]**（49 點，每個 32-block ~4 點：3 個塊內 Δ + 1 個跨界 Δ）。
  - **外加邊界微掃**：在數個 32 倍數附近各取密集叢集，例：N ∈ {158,159,160,161,162, 190,191,192,193,194, 254,255,256,257,258, 382,383,384,385,386}——讓「跨界 Δ」量在**邊界本身**，而非粗塊平均。
  - 全部 N≥128（ONNXim N≤64 SIGFPE）。
- 兩 sim 各跑：ONNXim 記 latency；ScaleSim 記 latency + util%。
- **兩個判準分屬兩個 grid，不可混算**（re-review NEW-1）：step-8 bulk 是**均勻取樣**→只在它上面做 FFT/autocorr；邊界微掃是 **1-spaced 不均勻叢集**→只在它上面做 step_ratio。把兩者串成一條序列做 FFT 會破壞 lag/bin（bulk 的 lag-32＝4 取樣、叢集內 lag-32＝32 取樣，不是同一個 lag）。
- **判準（跑前寫死；以邊界微掃 step_ratio 為主、bulk FFT 為輔）**：對每個 (sim, M) 各自——
  1. **主判準＝邊界微掃 step_ratio（grid 相位無關，最乾淨）**：在每個 32k 邊界叢集（如 158–162）量 `step_ratio = (叢集內最大相鄰 |Δlat|) ÷ (叢集內非跨界相鄰 |Δlat| 中位數)`。叢集偵測的是「**32k 附近有無任何 local 跳階**」——**不假設跳階一定落在 32k+1**（ONNXim 的 tiling/NoC 跳階若存在，未必對齊 ScaleSim 的 32k+1）（re-review NEW-3）。`step_ratio ≥ 3` → 該邊界有階；`< 1.5` → 平滑。
  2. **輔判準＝step-8 bulk 的殘差週期性**：擬合 `lat = a·N^b`（log–log），對 **bulk-only** 殘差做 period-32 autocorrelation / FFT bin。注意：`range(128,512,8)` 給 128,136,144,152,160,…——**每 32-block 4 個取樣點、只有 1 個是 32 倍數**（4 samples/period，超過 Nyquist 2），可解析 period-32；先前 review 誤以為「只落在 32 倍數」是算錯。但 bulk 對相位敏感，故只當交叉確認，不當主證。
  3. **判定**：主判準（≥3 個邊界叢集 step_ratio ≥ 3）→ 階梯；主判準全 < 1.5 **且** bulk 殘差無 period-32（未超洗牌虛無分佈）→ 平滑；其餘 inconclusive。**R² 只當輔助，不單獨當「平滑」證據**（高 R² 仍可藏小漣漪）。門檻常數寫進報告檔頭。
- ScaleSim 可重用既有 sweep 骨架，但 N grid／判準依本 plan 重訂；ONNXim 需把這些 shape 加進 `npu_onnxim_trace.py` 的 `SHAPES`，到 metiscard Docker 重跑。

### E2 — 對齊敏感度
- N=128（32 倍數）vs N=144（非倍數，+16），固定 M,K（M=1 與 M=128 各一）。兩 sim 各量 `ratio = lat(misaligned)/lat(aligned)`。ratio≈1 → 無對齊懲罰，照報。

### E3 — 順序敏感度（**所有對的兩端皆 N≥128**，review 修正 #3）
- 固定 MAC 總量、交換大維：**(M=256, N=128) vs (M=128, N=256)**，K 固定。兩端 N 皆 ≥128（不踩 SIGFPE）。兩 sim 各量 ratio。**規則：E3 任何 pair 的兩個成員都必須 N≥128。**

### E4 — shape / decode 敏感度
- M=1 vs M=32 vs M=128，固定 K,N（N≥128）。兩 sim 各量 latency；**util% 為 ScaleSim-only，照實標「ONNXim 不提供 util」**（review 修正 #7，不為 ONNXim 捏造 util）。

## 3. 板上工作（ONNXim）
- ONNXim 只能在 **metiscard 的 Docker**（x86 Ubuntu）跑；純 CPU 模擬、**不碰 AIPU**，比 thermal 低風險。沿用 `npu_onnxim_trace.py` 的 sweep.sh 逐 shape 機制；**只擴充 `SHAPES`**。
- **tractability（review 修正 #3）**：E1 兩 regime（~49×2 step-8 + 邊界微掃）+ E2/E3/E4 ≈ ~120 個小 shape（M≤256,K≤2048,N≤512，遠比既有 F=14336 巨型便宜）。**不再把「一次 docker run」當硬性限制**；可拆成 2–3 次 docker run，`timeout` 提到 ~3600s，並在 log 記錄實際 wall-clock。
- 守則：N≥128（SIGFPE）；跑前確認 image `onnxim` 還在；giant shape 不混入；跑後清 `~/edge-cim-simulation/onnxim_io` 暫存；不改板上任何全域 config。
- ScaleSim 在本機跑（純 Python，無板）。

## 4. 輸出（actions → verify）
1. 擴充 `tools/analysis/npu_onnxim_trace.py` 的 `SHAPES`（E1 兩 regime step-8 + 邊界微掃 + E2/E3/E4），可拆多次 docker run → verify: 本機 dry-print shape list 含全部、皆 N≥128。
2. 到 metiscard 重跑 ONNXim，更新 `simulated/onnxim/rknpu2_sim_matmul.json` → verify: 所有 E1/E2/E3/E4 點全到、無 missing；log 有 wall-clock。
3. 擴充 `tools/scalesim/run_rknpu2_scalesim.py`：E1 grid（兩 regime）對齊 ONNXim、E2/E3/E4 probe → verify: 本機跑完，輸出含同 grid + util。
4. 新增 `tools/analysis/npu_characteristic_compare.py` → `validation/reports/phase1.6/npu_characteristic_compare.json`：逐 (特性 × sim × M-regime) 的 `{verdict: present|absent|inconclusive, autocorr32, fft32, step_ratio, R2, ratios}` + H0 判定；**判準常數與 ScaleSim=positive-control 註記寫在檔頭** → verify: 重跑冪等、欄位齊、含洗牌虛無分佈門檻。
5. 重畫 NPU 圖（取代 `npu_systolic.png`）：**ONNXim、ScaleSim 各一條自己的量測曲線、各自 y 軸（或各自正規化）**，各自依 compare.json 標 verdict；無階梯就畫平滑線、不畫假階；ScaleSim 明標「positive control（32×32 array，階梯為定義上必然）」 → verify: 圖上每點數值逐一對得回兩個 sim 的 JSON；無共用絕對 y 軸。
6. 改寫 NPU 頁 §1/§3b 為**誠實比較結論**（以實際資料為準；若結論是「ScaleSim 顯示 32 階梯（預期、定義上必然）／ONNXim 在此 grid 為平滑 ∝N^b」就照寫，並明標只在所測 M regime 成立）；更新 `figs.json` sources、`_metrics.py` keys → verify: build `--strict` 綠、figure-staleness 綠、pytest 綠。

## 5. 退場 / 先中和既有欺騙性產物
- **步驟 0（最先做）**：把現有 `npu_systolic.png` 圖與 §1 caption 的「階梯/knee」斷言**降級為 pending（或移除）**，並把 ScaleSim 既有 `stage_staircase_fig3`（含「Fig-3 replica」字樣）一併下架，確保量測完成前**不 ship 任何預設 knee 的圖或標籤**。（目前該圖仍在 working tree、未 commit、未 merge、未對外呈現。）

## 6. 流程（CLAUDE.md per-phase）
本 plan → subagent 審（loop 到乾淨）→ **你核准** → 執行 → after-review subagent → PR。（沿用你對 1.6 要求的 before + after review。）

## 7. 驗證（success criteria）
- 判準（殘差 period-32 autocorr/FFT 門檻、step_ratio 門檻、洗牌虛無分佈）在**跑資料前**寫死於程式與報告。
- `npu_characteristic_compare.json` 對每個 (特性 × sim × M-regime) 都有明確 verdict + 數值，H0 檢定可重跑。
- 圖只含**各 sim 自己的量測點、各自 y 軸**，無套形狀、無預設 knee、無跨 sim 疊絕對值；ScaleSim 明標 positive-control；caption 不出現 validated/agree/parity；HeteroInfer 僅標為 probe 動機；無「Fig-3 replica」標籤。
- 結論若為「ONNXim 平滑」，必須限定在**實測的 M regime**（M=1 與 M=128），不過度推廣。
- build `--strict` / pytest / figure-staleness 全綠。

Outputs: `simulated/onnxim/rknpu2_sim_matmul.json`（細掃，兩 regime）、`simulated/scalesim/rknpu2_sim_matmul.json`（同 grid）、`validation/reports/phase1.6/npu_characteristic_compare.json`、重畫的 NPU 圖（各自 y 軸）、改寫的 `06-npu.src.html` §1/§3b、`figs.json`/`_metrics.py` 同步。
