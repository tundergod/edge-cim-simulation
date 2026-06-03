# Plan: Phase 0.2 — op 統計 / workload-op profile

輸入（Phase 0.1 產物）：`measurements/op_inventory/{model}.json`（config + 各 phase distinct sigs）、`sweep_matrix.json`（580 sigs / 9 類）、`workload_lengths.json`（per (model,task) 平均 prefill/decode token 數）、`traces/{model}_{task}.json.gz`、`tools/trace_export/*`。
環境：純軟體，這台 Mac `.venv`，**不需上板、不需 GPU**。
粒度：**完整 (op, in_shapes, out_shape) sig**，per (model × workload)，拆 prefill / decode（與 sweep_matrix 1:1、與 Phase 0.3 板 latency 1:1 join）。
計數：**analytic**（per-layer 次數 × layers；decode 的 length-dependent op 逐 kv_len 位置展開），以一條真實 trace 驗證。
精度（算 bytes 用）：GEMM 運算元預設 **INT8（1 byte）**（對齊 Metis 範圍）、非-GEMM 預設 **FP16（2 byte）**（對齊 vendor 實跑，P4）；dtype 為 `op_profile.py` 參數，header 記錄；intensity = FLOPs ÷ bytes 隨之而定。

1. 寫 `tools/trace_export/op_profile.py` — analytic per-(model, prefill_len, decode_len) op 計數器：
   - 從 `op_inventory/{model}.json` 讀 config（layers / hidden / heads / kv_heads / head_dim / ffn / vocab）。
   - 列舉每層 op 的完整 sig，並把次數寫成 (prefill_len, decode 位置 kv_len) 的函數：
     - **length-independent**（shape 不隨 kv_len 變）：matmul q/k/v/o/gate/up/down（decode M=1、prefill M=seq）、norm、rope、residual、ffn(silu/mul) → 每模型每層固定 sig；prefill count = layers、decode count = layers × decode_len；lm_head / embedding 每 token 一次。
     - **length-dependent**：attention QK^T / S·V bmm、softmax、kv_cache append → prefill 在 seq=prefill_len 一個 sig；decode 對 kv_len = prefill_len … prefill_len+decode_len−1 **逐位置展開**，每 (kv_len, op) sig count = layers。
   - 每 sig 記：FLOPs（matmul/bmm = 2·M·K·N，沿 P5 由 mm/addmm 或 bmm 的 in_shapes 取 (M,K,N)；norm/softmax/rope/residual/ffn = element 數 × 每-element 運算數）、bytes（in + weight + out element 數 × dtype_bytes）、intensity = FLOPs ÷ bytes。
   → verify：對 (llama-3.2-1b, prefill=128, decode=1 step)，op_profile 的 distinct (op,shape) 集合 == op_inventory 該點 distinct sigs；且每類 total count == 用 `gen_traces.py` 產生的**非摺疊** trace 之 op 計數（誤差 0）。

2. 把每個 analytic sig 對到 9 類別 — **重用 `sweep_matrix.py` 的 `BY_NAME` / `src_category`**（同一套歸類，含 kv_cache 特例），每 sig 帶 category。→ verify：所有 analytic compute sig 都解析到 9 類之一，無 uncategorized。

3. 產生 Layer-A profile（4 模型 × 4 任務）：用 `workload_lengths.json` 的平均 (prefill, decode) 跑 op_profile → 寫 `measurements/op_profile/{model}_{task}.json`（每 sig row：op / in_shapes / out_shape / category / phase / count / flops / bytes / intensity；header：dtype、(prefill,decode) 長度、每類 total count + total FLOPs + total bytes、prefill/decode 小計）。→ verify：16 檔；每檔每類 count 與 sweep_matrix 出現的類別一致（該 workload 用到的類別 count>0）；手算抽驗一個 matmul sig 的 FLOPs/bytes 與檔內值相符。

4. 產生 Layer-B 合成長度掃描 profile（scaling / roofline 用）：對 Phase 0.1 Layer-B 點（prefill ∈ {128,256,512,1024}、decode kv ∈ {128,512,1024}）× 4 模型跑 op_profile → 寫 `measurements/op_profile/sweep_{model}.json`。→ verify：每模型檔涵蓋所有 Layer-B 點。

5. 寫 `tools/plotting/op_breakdown.py` + `tools/plotting/roofline.py` — **只吃 committed `measurements/op_profile/*.json`**：(a) 每 (model,task) 的 op-類別堆疊長條（count-加權 + FLOPs-加權兩版，prefill/decode 分開）；(b) operational-intensity 散佈（每類一點/一群）疊 roofline 軸。輸出 `docs/figures/phase0.2/*.png`。→ verify：`python tools/plotting/op_breakdown.py` 與 `roofline.py` 從 JSON 重產所有 PNG（無 hardcode 數據）；重跑同檔（byte-stable）。

6. 寫 `docs/phase0.2-op-profile-findings.md`：每任務 prefill/decode op-mix、主導 sig（按 count 與按 FLOPs 各取 top-N）、每 (model,task) 由 intensity 判 compute-bound / memory-bound、嵌入 step 5 圖。→ verify：16 個 (model,task) 各有 mix + 主導 sig + bound 分類；圖被引用。

7. commit → `tools/trace_export/op_profile.py` + `tools/plotting/*` + `measurements/op_profile/*` + `docs/figures/phase0.2/*` + `docs/phase0.2-op-profile-findings.md`。→ verify：`git ls-files` 列出；圖可由 script 重產（不依賴未納版控的中間檔）。

Outputs:
- `tools/trace_export/op_profile.py`、`tools/plotting/{op_breakdown,roofline}.py`
- `measurements/op_profile/{model}_{task}.json`（16）+ `measurements/op_profile/sweep_{model}.json`（4）
- `docs/figures/phase0.2/*.png`（op-breakdown + roofline，皆可從 JSON 重畫）
- `docs/phase0.2-op-profile-findings.md`
