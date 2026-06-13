# Plan: Phase 0.3 — 真實板特性量測（不含溫度）

> **狀態（2026-06-04）：A1/A1d(CIM)、A5(Mali)、A6(CPU)、B(vendor LLM)、C4、C5 ✅ 已完成並 commit（PR #8）。⏳ A4(RKNPU2) 未完成 —— 卡在 aetina 離線。**
>
> **A4 已備妥（不需 aetina）：** metiscard `~/rknnconv`（uv py3.10）裝好 rknn-toolkit2（依賴 pin：`setuptools<81` 為 pkg_resources、`onnx==1.14.1` 為 onnx.mapping、`numpy<2`）；`characterization/metis_card/convert_rknn.py` 已轉出 **23/23 `.rknn`** 於 `metiscard:~/rknn_out/`（16 投影 + 7 attention bmm，含 2-input 原生 activation×activation）；`characterization/aetina/run_rknpu2.py`（aetina rknnlite runner）就緒。
>
> **A4 待辦（需 aetina 回到網路後）：**
> 1. ☐ **救回 aetina**：重開後仍完全離線（連 ping 都沒）→ 需實體確認開機完成 + 網路起來；先用 `ping 140.112.28.105` 確認上線，再 `ssh aetina`。
> 2. ☐ `rsync metiscard:~/rknn_out/ → aetina:~/rknn_out/`（metiscard 無 GitHub 認證，經 Mac 中轉：先拉回 Mac 再推 aetina）。
> 3. ☐ aetina：`~/edge-cim-simulation/.rknnvenv/bin/python run_rknpu2.py ~/rknn_out` → 產 `measurements/aetina/rknpu2_matmul.json`（rknnlite 上板 latency + GFLOP/s；FP16）。
> 4. ☐ 把 **NPU 原生 attention** 線加進報告 Fig 4（對照 CIM composed 29 ms / Mali 80–500 µs），並把 `docs/phase0.3-findings.md` 的「RKNPU2 not collected」gap 換成數據 + 「NPU 大投影不需 tiling」對照；更新 `docs/report/`、PR #8。
> 5. ☐（選用）HeteroInfer NPU Fig 3（stage 階梯）/ Fig 4（order/shape）連續掃描。

輸入：`measurements/op_inventory/sweep_matrix.json`（580 sigs，9 類：matmul/attention/rope/norm/ffn/residual/softmax/embedding/kv_cache）+ `measurements/op_profile/*`（Phase 0.2：逐-sig 執行次數，定本階段量測優先序）。
驅動（machines.md）：Mac = git 中機；腳本在 `characterization/{aetina,metis_card}/` → `rsync` 到板 → `ssh` 執行 → `rsync` 結果回 `measurements/{aetina,metis_card}/` → Mac commit。板上不留獨有程式碼。
散熱受控（溫度量測 = Phase 0.4）。**每個 session 前先 `axdevice` 確認卡在線**（兩卡都曾掉線；救援見 voyager-sdk.md §11）。
協定：每單元先 Stage-0（CoV）→ Stage-1（依 CoV 定取樣數 n，n 記入該單元 `variance_profile.json`；`validation/contracts/*` 留待 Phase 1）。
**每個 op 量測重複**：warmup 次數丟棄 → N 次 timed iteration × cold-start 重複；**報 median + p95 + CoV（非裸 average，抗 jitter；median 為主值）**，N 由 Stage-0 CoV 決定（範本 `cold_starts=3, iterations_per_run=300`）。
**HeteroInfer 特性圖對應**：A5(Mali)→Fig 1（TFLOPS vs tensor-size）；A4(RKNPU2)→Fig 3（stage 階梯）+ Fig 4（order/shape 敏感度）；A3(contention)→Fig 5（single vs concurrent 總頻寬）。圖在 Phase 0.3 findings 繪出。
**CIM 矩陣原語 = 1×1 conv proxy**（已驗證 M=1 GEMV 與 M>1 prefill 皆可編：8B QKV `[128,4096]×[4096,4096]` 編譯 52.9 s、EXIT=0；raw matmul 編不過）。
**量測優先序（Phase 0.2 profile）+ CIM-deep 主軸（user 定 A+C）**：fine-sweep 預算給 **lm_head / gate-up / down GEMV**（decode+prefill 雙佔 count×FLOPs/bytes 之首）+ **kv_proj 窄-N（GQA exemplar）**；decode GEMV(M=1) 皆 on-grid（measured=true）；唯一要緊的 off-grid 外推 = **LongBench prefill M≈11.8k（grid 上限 1024 的 11×）** → 加 M∈{2048,4096} spot-anchor 驗 roofline 線性，不全展開。kv_cache（decode bytes 22–28%）= 純頻寬 op → A2/A3。**CIM 是論文主軸**：A1d 對 CIM 做 HeteroInfer-風格深掃 + CIM 限制軸（A）；C4 組合式 attention 估計（C）；C5 micro→end-to-end 兩支柱連結。

## A. Machine 1 — aetina

A0. 連線前置 → 容器內 `axdevice` 確認 `metis-0:1:0`；若 `No target device` 則救援（`/sys/.../0000:01:00.0/remove`＋`/sys/bus/pci/rescan`＋`modprobe metis`，再 `docker rm -f axelera-sdk; ~/start-sdk-bg.sh`）。→ verify：`axdevice` 列出 `metis-0:1:0`。

A1. CIM 矩陣（matmul + attention）→ 寫 `characterization/aetina/run_metis_cim.py`：
  - **matmul 類（mm/addmm，105 sigs；linear 投影 q/k/v/o/gate/up/down/lm_head，weight 真 stationary → conv-proxy 忠實）**：從 sig 取 (M,K,N)——**mm = `[[M,K],[K,N]]`；addmm = `[[N],[M,K],[K,N]]`（bias 在前），index 需分支**（P5）→ `Conv2d(in=K, out=N, kernel=1, bias=(op==addmm))` ONNX、input `[1,K,1,M]` → `compile --input X.onnx --input-shape 1,K,1,M --output DIR --overwrite --log-level WARNING`（INT8，自動 100-sample 校準、**無需 imageset**）→ `axrunmodel DIR/compiled_model/model.json --seconds S` 取 dev/host/system FPS。
  - **attention 類（bmm，53 sigs）= conv-proxy 只給「下界」（P1，重要保真度限制）**：QK^T / S·V 的**兩個運算元都是 activation**（K^T、V 是成長中的 KV-cache），而 conv-proxy 第二運算元是編譯期常數 weight。故 conv-proxy 把 K^T/V「烤」成固定 filter，**只量到 compute、漏掉每個 decode step 把成長 K/V 重載入 crossbar 的成本**（CIM 省電正來自 weight 不動，attention 恰違反此）→ **CIM attention 此值明確標為下界**（CIM 本就不擅長 activation×activation attention，符合 CIM-centric 立場）。真實 attention 成本由 **A4/A5（NPU/GPU 原生 activation×activation）**量得；operand-reload / placement 由 Phase 1 決定。仍以單一 head 量、記 B（head 數）metadata。
  - 每形狀對 `dpu_constants_home: global.l2` vs `global.ddr` 各一輪（`extra_kwargs.compilation_config` 覆寫）。
  - 風險（execution 早期先 smoke-test）：lm_head 形狀 N=128256（out-channels 極大）是 conv-proxy 未測極端，可能壓垮/OOM 編譯器；若失敗則沿 out-channel 分塊（tile）量測再分析組合。已驗證上界為 N=4096。
  - 輸出 `measurements/aetina/metis_alpha_matmul.json`；conv-proxy↔matmul 對應 + bias/head 處理寫 `metis_alpha_cnn_proxy.json`。
  → verify：每 matmul 形狀 + 每 attention 單-head 形狀都有 dev latency；l2/ddr 兩組齊；抽一筆 dev FPS>0。

A1d. **CIM 深度特性（A 主軸；conv-proxy 忠實域；焦點 = Phase 0.2 主導 op）** → 擴充 `run_metis_cim.py`，焦點 = lm_head / gate-up / down GEMV（×4 模型，dims H∈{2048,3072,3584,4096}、F∈{8192,8192,14336,18944}、V∈{128256,152064}）+ kv_proj 窄-N（kv_w∈{512,1024}）：
  - **A1d.1 roofline-knee**：FFN/lm_head/q-o 三族在 decode(M=1) + grid prefill M 掃，記 dev FPS→有效 GB/s 與 GOP/s（INT8）。→ verify：有效吞吐 vs (K·N bytes) 曲線顯 memory→compute knee；M=1 點落 memory-bound(intensity≈2) 端。
  - **A1d.2 channel-64 階梯**：M=1、K=H、N 由 64 到 F 每 64 一步（+ 幾個 off-64 如 512±32）**probe 推測的 64-channel granularity 是否 gate conv out-channels**（voyager-sdk §1/§2 為 `[DOC] inferred`、僅證 Pad/Slice，未證 conv 輸出 → 此步即是實證）。→ verify：latency-vs-N 階梯、risers 在 64 倍數；N=512(kv_w) 低利用 vs N=14336。
  - **A1d.3 (M,K,N) aspect 敏感度**：等 MAC 不同長寬（down [F→H] vs gate/up [H→F] vs q/o [H→H]；decode M=1 vs prefill M∈{128,1024}）。→ verify：等 MAC latency 不同 → 量化 aspect 敏感（CIM 版 Fig 4）。
  - **A1d.4 l2 vs ddr 殘留**（每 A1d.1 shape 兩編）：標 L2-spill 門檻（gate/up [4096×14336]≈59 MB > 32 MB L2 → 強制 DDR；1B kv/q 合身）。**Alpha 無 on-card DRAM，"ddr"=host LPDDR over PCIe，l2/ddr gap 高估 production card（其真 on-card LPDDR ~24 GB/s）→ 只取殘留「敏感形狀」，絕對值不外推到 production**。→ verify：每 shape l2/ddr dev-latency 齊；l2 失效 byte 門檻 ≈32 MB L2。
  - **A1d.5 per-call PCIe/DMA fixed-overhead**：用 A2 4-way toggle 跨 3+ 計算量級（kv_proj N=512 → lm_head N≈128k）線性擬合 `latency = fixed + bytes/BW`。→ verify：intercept(固定 floor ms) + slope(有效 GB/s,對照 ~3.9) 抽出；floor 跨 shape 一致。
  - **A1d.6 GQA 窄-N 浪費**：kv_proj [H→512]/[H→1024] vs 寬投影。→ verify：N=512 crossbar 利用率明顯低於 N≥4096。
  - 輸出併入 `metis_alpha_matmul.json` 的 `cim_deep` 區段。**lm_head N≈128k/152k 為 OOM 最高風險（A1 已標）→ 先 smoke-test，失敗則 out-channel 64-對齊分塊量再加總（分塊本身即 A1d.2 資料）。**

A2. PCIe / DMA → 寫 `run_metis_pcie.py`：對一個代表性 compiled model 跑 `axrunmodel`，`--double-buffer/--no-double-buffer` × `--input-dmabuf/--no-input-dmabuf` 四組合，記 Device vs System FPS 差（= 傳輸成本）。輸出 `measurements/aetina/metis_alpha_pcie.json`。→ verify：四組 latency 齊；推得 PCIe 有效 BW（對照 ~3.9 GB/s Gen3×4）。

A3. 記憶體 contention（= **HeteroInfer Fig 5**；ADR-0001：重現 ~60 GB/s 飽和拐點的義務；校準 ADR-0002 的 interconnect-efficiency）→ 寫 `run_contention.py`：各單元跑記憶體串流 loop，**有效 BW = 搬移 bytes ÷ 實測時間**（v1.3.1 無可靠 `axmonitor` DDR-BW readout，故用此法）；單獨 vs 並行（CIM ∥ {RKNPU2 / Mali / CPU}），**掃並行度 1→4（≥3 點，使拐點可解析）**；CIM 經 PCIe 存 host LPDDR、其餘原生存取，分別記錄。輸出併入 `metis_alpha_pcie.json` 的 `contention` 區段。→ verify：≥3 並行度點；單獨 vs 並行總 BW 的飽和拐點可解析。

A4. RKNPU2 matmul **+ attention bmm** → (a) 在 metiscard（x86）裝 `rknn-toolkit2`，**重用 A1 的形狀產生器產出 matmul（linear 105）＋ attention bmm（activation×activation、單-head 53）ONNX**（NPU 原生支援 activation×activation，給真實 attention 成本，P1/P2）→ 轉 `.rknn`（target `rk3588`，INT8 + FP16）；先驗證 import + 轉一個形狀成功再全轉；(b) `rsync` `.rknn` 到 aetina；(c) 寫 `run_rknpu2_matmul.py` 用 `~/edge-cim-simulation/.rknnvenv` 的 `rknnlite` 載入、**warmup + 多 iteration** 計時（RKNPU2 b=1 latency-bound）。(d) **HeteroInfer NPU 特性掃描**：除 sweep_matrix 真實形狀外，加做 **Fig 3（stage）連續 K-size 細掃**（顯示 systolic 對齊 staircase）+ **Fig 4（order/shape）同一 matmul 的維度重排**（`[K,M]×[M,N]` vs `[M,K]×[K,N]` vs 轉置）+ row/col 比例。輸出 `measurements/aetina/rknpu2_matmul.json`（含特性掃描區段）。→ verify：metiscard 成功轉 ≥1 形狀；每形狀 × 精度有 latency；stage 階梯 + order 敏感度資料齊（可繪 Fig 3/4）。

A5. Mali matmul **+ attention bmm** → 寫 `run_mali_matmul/`（自寫 OpenCL GEMM kernel，`gcc … -lOpenCL`），跑 matmul（linear 105）**＋ attention bmm（activation×activation、單-head 53，給真實 attention 成本，P1/P2）**形狀，FP16 主 + FP32 參考；**加做 HeteroInfer Fig 1（GPU TFLOPS vs tensor-size）連續 K-size 掃描**，顯示 memory-bound→compute-bound 轉折。輸出 `mali_matmul.json`（含 TFLOPS 曲線區段）。→ verify：`clinfo` 裝置 = Mali-G610；每形狀有 latency；TFLOPS-vs-size 曲線可繪（Fig 1）。

A6. CPU + 非-GEMM ops → 寫 `run_cpu_ops/`：(i) LLM-support op（sampling、RoPE 控制、KV append/evict、token/quant 邊界）；(ii) **sweep_matrix 非-GEMM 類在 A76 計時**：norm（pow/mean/rsqrt）、rope-elementwise（cos/sin/neg/cat/mul/add）、softmax、residual add、ffn（silu/mul）；`taskset -c 4-7 chrt -f 50`、`clock_gettime` + `perf stat`。輸出 `cpu_ops.json`。→ verify：每 op 有 latency + perf 計數。
  - **覆蓋聲明（sweep_matrix 580 sigs / 9 類全數歸屬）**：
    - **matmul 105**（linear 投影，weight stationary）→ A1(CIM, conv-proxy 忠實) / A4(NPU) / A5(GPU)。
    - **attention 95** = **bmm 53（activation×activation）→ A4/A5 原生量（真實值）＋ A1 conv-proxy 下界（P1）** + 非-bmm elementwise 42（score-scaling mul / mask-bias add，與 rope/residual 同 kernel）→ A6 CPU。
    - **norm 90 / softmax 21 / rope 190（含 10 個 freq-外積小 bmm，非 elementwise，P3）/ residual 20 / ffn 30** → A6 CPU 基準（**精度 = FP16/FP32，與 vendor INT8(B) 實際跑的精度對齊，P4**；亦 NPU/GPU 候選，Phase 1 補）。
    - **kv_cache 9**（KV-append）= 記憶體頻寬 op → A2/A3（bytes÷time，非 compute）。
    - **embedding 20** = host gather，不 micro-bench。

A7. Stage-0 variance → 每單元代表 op，cold-start × iteration 算 CoV → `measurements/aetina/variance_profile.json`（含據此定的 Stage-1 n）。→ verify：四單元各有 CoV + n。

A8. 寫 `characterization/aetina/README.md`（哪支腳本產哪個輸出檔 + 呼叫參數）。→ verify：每支腳本有對應說明。

## B. Machine 2 — metiscard（L4 錨點）

B0. 連線前置 → `cd ~/tundergod/voyager-sdk && source axelera-env/bin/activate`，`axdevice` 確認 `metis-0:7:0`；不回應則 `axdevice --reboot`（等數秒再確認）。→ verify：`axdevice` 顯示 `16GiB`。

B1. 端到端 LLM → 寫 `characterization/metis_card/run_vendor_llm.py`：對已驗證 slug `llama-3-2-1b-1024-static` / `…-4core-static`、`llama-3-2-3b-1024-static` / `…-4core-static`、`llama-3-1-8b-1024-static` / `…-4core-static` 各跑固定 prompt set 的 `axllm <slug> --prompt … --show-stats`，抓 tokenization / prefill / TTFT / gen tok-s / per-token latency / CPU%。**context 固定 1024（llama slug 內建、不可掃）；context sweep 僅 phi3（512 / 1024 / 2048）**。對比 1c vs 4c；模型大小 sweep（1b/3b/8b）。輸出 `measurements/metis_card/vendor_llm_int8.json`。→ verify：3 模型 ×{tok/s, TTFT, prefill} 齊；4c/1c 對比齊；decode 時間 ∝ 權重 bytes 趨勢可見（對照 ~24 GB/s）。

B2. Stage-0 variance → 重複代表 run 算 CoV → `measurements/metis_card/variance_profile.json`（含 Stage-1 n）。→ verify：CoV + n。

B3. 寫 `characterization/metis_card/README.md`（harness 呼叫說明）。→ verify：有說明。

## C. 收尾

C1. 寫 `docs/phase0-aetina-findings.md` + `docs/phase0-metis-card-findings.md`：各單元 latency 摘要、l2 vs ddr、contention 拐點、sweep_matrix 形狀覆蓋率、SDK 意外；**繪 HeteroInfer 風格圖**：Fig 1（Mali TFLOPS-vs-size）、Fig 3/4（RKNPU2 stage / order-shape）、Fig 5（contention 總頻寬）+ 各單元 roofline。→ verify：每節有內容；四張圖的底層資料齊。
C2. 寫 `docs/phase0-L1-L6-mapping.md`：哪個 measurement 檔餵哪個 L1–L6 驗證列（L6 = 重用 Step-1，不重量測）。→ verify：L1–L6 每列都有來源檔。
C4. **組合式 CIM attention 估計（C；analysis，feeds off A1+A2/A3）** → `T_attn_CIM(kv) = T_convproxy_floor(kv) + T_kvreload(kv)`，`T_kvreload = kv_bytes(kv)/BW_eff + n_dma·fixed_overhead`，`kv_bytes = 2·kv·kv_heads·head_dim·1B`（GQA：用 kv_heads 非 heads）；`BW_eff` 取 A2(PCIe)/A3(contended)、`fixed_overhead` 取 A1d.5、`n_dma` = 每 decode step 的 KV-reload DMA 呼叫數（預設 = layers）。在 profile 真實 kv（ShareGPT≈519、LongBench≈11.8k）+ on-grid kv∈{129,513,1025} 求值。**此 composed 值 = Alpha-topology penalty 估計**（服務「CIM 不擅長 attention → offload」論點；非 production 絕對值）。→ verify：每 kv 同報 floor(A1) 與 composed(floor+reload)，reload 份額隨 kv 升；標「CIM attention penalty」並對照 A4/A5 原生 attention 更快（佐證 offload）。輸出 `measurements/aetina/cim_attention_composed.json`。
C5. **兩根支柱連結（micro→end-to-end；ADR-0006 hold-out，非自我驗證）** → 由 Alpha op 成本結構 × Phase 0.2 counts 預測 production card decode tok/s。**ADR-0006 hold-out**：bandwidth/結構**只由 1B+3B fit、預測 8B**（測 size-invariance；ADR-0006 註 4c/1c 加速比 1B→8B 漂移 1.130×→1.081×）。`t_step = Σ_proj count·lat_CIM(proj) + Σ_attn count·T_attn_CIM(kv̄) + Σ_support count·lat_support + overhead`（`kv̄ = P + D/2` 平均 decode kv；`overhead` = 每 step host/sampling floor，取 A6/B1），`tok_s_pred = 1/t_step`。**bandwidth 項用 production card ~24 GB/s（B 量），Alpha 只供相對 op 成本結構；C4 attention 項一併 rescale 到 24 GB/s（否則標 upper-bound）**。**這是結構一致性檢查、非獨立 L4 驗證**。sanity floor（非 verify）：`tok_s_pred ≈ 24.2 GB/s ÷ weight_bytes_per_token` 應近 B 的 ~11–15 tok/s(1B)。→ verify：8B hold-out `|pred−meas|/meas ≤ 0.25`（不再以「隱含 BW=24 GB/s」當 verify，那是循環）。輸出 `measurements/metis_card/twopillar_prediction.json`。
C6. commit → `measurements/aetina/*` + `measurements/metis_card/*` + `characterization/*` + `docs/phase0-*`。→ verify：`git ls-files` 列出，tree 乾淨。

## Outputs（對齊 OVERALL.md §Phase 0.3 完成標準）
- `measurements/aetina/{metis_alpha_matmul（含 cim_deep 區段）, metis_alpha_cnn_proxy, metis_alpha_pcie, rknpu2_matmul, mali_matmul, cpu_ops, cim_attention_composed, variance_profile}.json`
- `measurements/metis_card/{vendor_llm_int8, twopillar_prediction, variance_profile}.json`
- `characterization/aetina/{run_metis_cim.py, run_metis_pcie.py, run_contention.py, run_rknpu2_matmul.py, run_mali_matmul/, run_cpu_ops/, README.md}`、`characterization/metis_card/{run_vendor_llm.py, README.md}`
- `docs/phase0-aetina-findings.md`、`docs/phase0-metis-card-findings.md`、`docs/phase0-L1-L6-mapping.md`
