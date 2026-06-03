# Plan: Phase 0.2 — 真實板特性量測（不含溫度）

輸入：`measurements/op_inventory/sweep_matrix.json`（580 sigs，8 類：matmul/attention/rope/norm/ffn/residual/softmax/embedding）。
驅動（machines.md）：Mac = git 中機；腳本在 `characterization/{aetina,metis_card}/` → `rsync` 到板 → `ssh` 執行 → `rsync` 結果回 `measurements/{aetina,metis_card}/` → Mac commit。板上不留獨有程式碼。
散熱受控（溫度量測 = Phase 0.3）。**每個 session 前先 `axdevice` 確認卡在線**（兩卡都曾掉線；救援見 voyager-sdk.md §11）。
協定：每單元先 Stage-0（CoV）→ Stage-1（依 CoV 定取樣數 n，n 記入該單元 `variance_profile.json`；`validation/contracts/*` 留待 Phase 1）。
**CIM 矩陣原語 = 1×1 conv proxy**（已驗證 M=1 GEMV 與 M>1 prefill 皆可編：8B QKV `[128,4096]×[4096,4096]` 編譯 52.9 s、EXIT=0；raw matmul 編不過）。

## A. Machine 1 — aetina

A0. 連線前置 → 容器內 `axdevice` 確認 `metis-0:1:0`；若 `No target device` 則救援（`/sys/.../0000:01:00.0/remove`＋`/sys/bus/pci/rescan`＋`modprobe metis`，再 `docker rm -f axelera-sdk; ~/start-sdk-bg.sh`）。→ verify：`axdevice` 列出 `metis-0:1:0`。

A1. CIM 矩陣（matmul + attention）→ 寫 `characterization/aetina/run_metis_cim.py`：
  - **matmul 類（mm/addmm，105 sigs）**：每 (M,K,N) → `Conv2d(in=K, out=N, kernel=1, bias=(op==addmm))` ONNX、input `[1,K,1,M]` → `compile --input X.onnx --input-shape 1,K,1,M --output DIR --overwrite --log-level WARNING`（INT8，自動 100-sample 校準、**無需 imageset**）→ `axrunmodel DIR/compiled_model/model.json --seconds S` 取 dev/host/system FPS。
  - **attention 類（bmm，per-head，53 sigs）**：bmm 形如 `[B,S,d]×[B,d,S']`（B = heads）。**取單一 head 的 GEMM 當 conv proxy 量測**，把 B（head 數）記為 metadata（B 顆 core / 順序執行的組合由 Phase 1 處理）。
  - 每形狀對 `dpu_constants_home: global.l2` vs `global.ddr` 各一輪（`extra_kwargs.compilation_config` 覆寫）。
  - 風險（execution 早期先 smoke-test）：lm_head 形狀 N=128256（out-channels 極大）是 conv-proxy 未測極端，可能壓垮/OOM 編譯器；若失敗則沿 out-channel 分塊（tile）量測再分析組合。已驗證上界為 N=4096。
  - 輸出 `measurements/aetina/metis_alpha_matmul.json`；conv-proxy↔matmul 對應 + bias/head 處理寫 `metis_alpha_cnn_proxy.json`。
  → verify：每 matmul 形狀 + 每 attention 單-head 形狀都有 dev latency；l2/ddr 兩組齊；抽一筆 dev FPS>0。

A2. PCIe / DMA → 寫 `run_metis_pcie.py`：對一個代表性 compiled model 跑 `axrunmodel`，`--double-buffer/--no-double-buffer` × `--input-dmabuf/--no-input-dmabuf` 四組合，記 Device vs System FPS 差（= 傳輸成本）。輸出 `measurements/aetina/metis_alpha_pcie.json`。→ verify：四組 latency 齊；推得 PCIe 有效 BW（對照 ~3.9 GB/s Gen3×4）。

A3. 記憶體 contention（ADR-0001：重現 ~60 GB/s 飽和拐點的義務；校準 ADR-0002 的 interconnect-efficiency）→ 寫 `run_contention.py`：各單元跑記憶體串流 loop，**有效 BW = 搬移 bytes ÷ 實測時間**（v1.3.1 無可靠 `axmonitor` DDR-BW readout，故用此法）；單獨 vs 並行（CIM ∥ {RKNPU2 / Mali / CPU}），**掃並行度 1→4（≥3 點，使拐點可解析）**；CIM 經 PCIe 存 host LPDDR、其餘原生存取，分別記錄。輸出併入 `metis_alpha_pcie.json` 的 `contention` 區段。→ verify：≥3 並行度點；單獨 vs 並行總 BW 的飽和拐點可解析。

A4. RKNPU2 matmul → (a) 在 metiscard（x86）裝 `rknn-toolkit2`，**重用 A1 的 matmul ONNX 產生器**產出 ONNX → 轉 `.rknn`（target `rk3588`，INT8 + FP16）；先驗證 import + 轉一個形狀成功再全轉；(b) `rsync` `.rknn` 到 aetina；(c) 寫 `run_rknpu2_matmul.py` 用 `~/edge-cim-simulation/.rknnvenv` 的 `rknnlite` 載入、**warmup + 多 iteration** 計時（RKNPU2 b=1 latency-bound）。輸出 `measurements/aetina/rknpu2_matmul.json`。→ verify：metiscard 成功轉 ≥1 形狀；每形狀 × 精度有 latency。

A5. Mali matmul → 寫 `run_mali_matmul/`（自寫 OpenCL GEMM kernel，`gcc … -lOpenCL`），跑 matmul 形狀，FP16 主 + FP32 參考。輸出 `mali_matmul.json`。→ verify：`clinfo` 裝置 = Mali-G610；每形狀有 latency。

A6. CPU + 非-GEMM ops → 寫 `run_cpu_ops/`：(i) LLM-support op（sampling、RoPE 控制、KV append/evict、token/quant 邊界）；(ii) **sweep_matrix 非-GEMM 類在 A76 計時**：norm（pow/mean/rsqrt）、rope-elementwise（cos/sin/neg/cat/mul/add）、softmax、residual add、ffn（silu/mul）；`taskset -c 4-7 chrt -f 50`、`clock_gettime` + `perf stat`。輸出 `cpu_ops.json`。→ verify：每 op 有 latency + perf 計數。
  - **覆蓋聲明（sweep_matrix 580 sigs 全數歸屬）**：GEMM 類（matmul 105 + attention bmm 53）由 A1/A4/A5 量；非-GEMM 類（norm/softmax/rope-elementwise/residual/ffn）**＋ attention 區段的非-bmm elementwise（score-scaling mul、mask-bias add、KV-cache cat ≈51 sigs，與 rope/residual 同 kernel 類）** 於此量 CPU 基準（亦為 NPU/GPU 候選，Phase 1 視需要補）；`embedding` = host gather，宣告不 micro-bench。

A7. Stage-0 variance → 每單元代表 op，cold-start × iteration 算 CoV → `measurements/aetina/variance_profile.json`（含據此定的 Stage-1 n）。→ verify：四單元各有 CoV + n。

A8. 寫 `characterization/aetina/README.md`（哪支腳本產哪個輸出檔 + 呼叫參數）。→ verify：每支腳本有對應說明。

## B. Machine 2 — metiscard（L4 錨點）

B0. 連線前置 → `cd ~/tundergod/voyager-sdk && source axelera-env/bin/activate`，`axdevice` 確認 `metis-0:7:0`；不回應則 `axdevice --reboot`（等數秒再確認）。→ verify：`axdevice` 顯示 `16GiB`。

B1. 端到端 LLM → 寫 `characterization/metis_card/run_vendor_llm.py`：對已驗證 slug `llama-3-2-1b-1024-static` / `…-4core-static`、`llama-3-2-3b-1024-static` / `…-4core-static`、`llama-3-1-8b-1024-static` / `…-4core-static` 各跑固定 prompt set 的 `axllm <slug> --prompt … --show-stats`，抓 tokenization / prefill / TTFT / gen tok-s / per-token latency / CPU%。**context 固定 1024（llama slug 內建、不可掃）；context sweep 僅 phi3（512 / 1024 / 2048）**。對比 1c vs 4c；模型大小 sweep（1b/3b/8b）。輸出 `measurements/metis_card/vendor_llm_int8.json`。→ verify：3 模型 ×{tok/s, TTFT, prefill} 齊；4c/1c 對比齊；decode 時間 ∝ 權重 bytes 趨勢可見（對照 ~24 GB/s）。

B2. Stage-0 variance → 重複代表 run 算 CoV → `measurements/metis_card/variance_profile.json`（含 Stage-1 n）。→ verify：CoV + n。

B3. 寫 `characterization/metis_card/README.md`（harness 呼叫說明）。→ verify：有說明。

## C. 收尾

C1. 寫 `docs/phase0-aetina-findings.md` + `docs/phase0-metis-card-findings.md`：各單元 latency 摘要、l2 vs ddr、contention 拐點、sweep_matrix 形狀覆蓋率、SDK 意外。→ verify：每節有內容。
C2. 寫 `docs/phase0-L1-L6-mapping.md`：哪個 measurement 檔餵哪個 L1–L6 驗證列（L6 = 重用 Step-1，不重量測）。→ verify：L1–L6 每列都有來源檔。
C3. commit → `measurements/aetina/*` + `measurements/metis_card/*` + `characterization/*` + `docs/phase0-*`。→ verify：`git ls-files` 列出，tree 乾淨。

## Outputs（對齊 overall.md §Phase 0.2 完成標準）
- `measurements/aetina/{metis_alpha_matmul, metis_alpha_cnn_proxy, metis_alpha_pcie, rknpu2_matmul, mali_matmul, cpu_ops, variance_profile}.json`
- `measurements/metis_card/{vendor_llm_int8, variance_profile}.json`
- `characterization/aetina/{run_metis_cim.py, run_metis_pcie.py, run_contention.py, run_rknpu2_matmul.py, run_mali_matmul/, run_cpu_ops/, README.md}`、`characterization/metis_card/{run_vendor_llm.py, README.md}`
- `docs/phase0-aetina-findings.md`、`docs/phase0-metis-card-findings.md`、`docs/phase0-L1-L6-mapping.md`
