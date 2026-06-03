# Plan: Phase 0.2 — op 統計 / workload-op profile

輸入（Phase 0.1 產物）：`measurements/op_inventory/{model}.json`（含每 (phase, 長度點) 每 sig 的 **`count` 與 `src` 欄位**——q≡o / k≡v / gate≡up 共形已正確聚合，如 1B prefill mm `[[128,2048],[2048,2048]]` count=32=16層×2）、`sweep_matrix.json`（580 sigs / 9 類，僅含**離散追蹤點**：prefill seq∈{128,256,512,1024}、decode kv∈{128,512,1024}）、`workload_lengths.json`（per (model,task) 平均 prefill/decode token 數）、`traces/{model}_{task}.json.gz`（非摺疊 op 流）、`tools/trace_export/*`。
環境：純軟體，這台 Mac `.venv`，**不需上板、不需 GPU**。
粒度：**完整 (op, in_shapes, out_shape) sig**，per (model × workload)，拆 prefill / decode。
計數原則：生成器 = **analytic 每層 op 模板**（提供結構與「哪個軸是長度軸」），oracle/驗證 = **op_inventory**（提供 count，**不自行用「×layers」推算**）。length-independent op 的 per-forward count 跨長度點為常數，length-dependent op 只有 shape 隨長度變、count 亦常數。workload 統計 = prefill-forward（×1）+ decode-forward（×decode_len；length-dependent op 逐 kv 位置展開 shape）。模板於追蹤點生成的 (sig,count) 必須逐筆等於 inventory，否則模板有誤、先修。
精度（算 bytes 用）：GEMM 運算元預設 **INT8（1 byte）**、非-GEMM 預設 **FP16（2 byte）**（對齊 vendor 實跑，P4）；dtype + **weight 計入假設（每 token 串流一次，非常駐）**記於 header。
**join 契約（取代任何「1:1」說法）**：profile 的 sig 集合是 sweep_matrix 的 **superset**（會生成 board 未量的 off-grid M 與 per-position kv）。與 board 成本的接合走 **Phase 1 擬合的 latency 方程式**在每 sig 的 (M,K,N)/kv 上求值（ADR-0006），**非離散 sig 查表**。每 sig 帶 `measured` 旗標（其 (op,shape) 是否落在 sweep_matrix 離散網格上）——此旗標即排序 Phase 0.3 量測的依據。intensity 一律標為 **predicted-side**（量測側 roofline 與 knee 判定屬 Phase 0.3/Phase 1）。

1. 寫 `tools/trace_export/op_profile.py` — **analytic op 模板（生成器）+ op_inventory（計數 oracle + 驗證）**：
   - **模板**：列舉 GQA decoder-only block 的每個語意 op（q/k/v/o-proj、QK^T bmm、softmax、S·V bmm、RMSNorm×2 prims、RoPE prims + freq-外積 bmm、SwiGLU gate/up/down、residual×2、kv_cache append）+ 每模型 lm_head/embedding；每 op 記 `{aten op, src, shape = f(config, 長度), 適用 phase}`。**length-dependence 由 shape 函數是否含長度決定**（明確、不從資料 diff 推斷），**per-phase 各自判定**（同一 op 可能 prefill length-dependent、decode 不變，如 rope-freq bmm）。**decode 的長度軸 inner/seq dim = 位置+1**（新 token attends 過去+自身；錨定 inventory kv=128→inner 129、softmax `[1,H,1,129]`、kv_cache `[1,KVH,129,Dh]`）。
   - **count 一律查 inventory**：模板 op 在某長度實例化出 sig → count 從 `op_inventory` 對應點查得；**絕不自行 ×layers 推算**（共形聚合由 inventory count 反映：Llama q≡o/k≡v/gate≡up 各 2×layers，1B=32、8B=64；但 Qwen q-proj=addmm、o-proj=mm 為相異 sig、不聚合——一律以該模型自身 inventory 為準）。
   - **生成 workload profile**：prefill 於 seq=P 實例化（×1 forward）；decode 的 length-independent op ×decode_len、length-dependent op 對 kv=P+1 … P+D 逐位置實例化（每位置 count = 該 op per-forward count）。lm_head prefill count=1 @ M=P（非逐 token）。每生成 sig 帶 src。
   → verify（純比對 committed JSON、零新 tracer/board）：(a) 在所有追蹤點（prefill {128,256,512,1024}、decode kv {128,512,1024}）模板生成的 **(sig, count) 集合逐筆等於 op_inventory 該點紀錄**——無 orphan inventory compute-sig、無模板 op 缺漏，shape 與 count 全中（此即同時驗證結構、shape 公式含 decode +1 錨、與次數）；(b) 任一 length-dependent decode op，**Σ 位置 count == per-forward count × decode_len**（總呼叫數守恆）。

2. 每 sig 標 category + measured 旗標：(a) category — **重用 `sweep_matrix.py` 的 `BY_NAME`/`src_category`**（含 kv_cache 特例）；analytic sig 已帶 step-1 的 src 故 `categorize` 不變即可用。(b) `measured` = 該 (op, in_shapes, out_shape) 是否存在於 `sweep_matrix.json`（**成員判定僅比對 (op,in,out)、不含 src，與 sweep_matrix 建表方式一致**）。→ verify：所有 compute sig 解析到 9 類之一、無 uncategorized；落在追蹤網格上的 sig `measured=true`、off-grid（如 prefill M=175、decode kv=518）的 `measured=false`。

3. 產生 Layer-A profile（4 模型 × 4 任務）：用 `workload_lengths.json` 平均 (prefill, decode) 跑 op_profile → 寫 `measurements/op_profile/{model}_{task}.json`（每 sig：op/in_shapes/out_shape/category/phase/count/flops/bytes/intensity/measured；header：dtype + weight 假設、(prefill,decode)、每類 total count/FLOPs/bytes、prefill/decode 小計）。FLOPs：matmul/bmm = 2·M·K·N（依 P5 由 in_shapes 取 (M,K,N)：mm `[[M,K],[K,N]]`、addmm `[[N],[M,K],[K,N]]`、bmm `[[B,M,K],[B,K,N]]`）；非-GEMM = element 數 × 每-element 運算數。bytes = (in+weight+out element 數) × dtype_bytes。**自然長度若 >8K（如 LongBench）於 header 標註超出 2K/8K scope**。→ verify：16 檔；手算抽驗一個 mm sig 與一個 bmm sig 的 FLOPs/bytes 與檔內值相符；每類 total count 與 step-1 計數一致。

4. 產生 Layer-B 合成長度掃描 profile（scaling/roofline 用，= sweep_matrix on-grid 的 1:1-joinable 子集）：對 prefill∈{128,256,512,1024} × decode kv∈{128,512,1024} × 4 模型跑 op_profile → 寫 `measurements/op_profile/sweep_{model}.json`。→ verify：每模型檔涵蓋所有 Layer-B 點；其 sigs `measured=true` 全為真。

5. 寫 `tools/plotting/op_breakdown.py` + `tools/plotting/roofline.py` — **只吃 committed `measurements/op_profile/*.json`**：(a) 每 (model,task) op-類別堆疊長條（count-加權 + FLOPs-加權，prefill/decode 分開）；(b) **predicted-side** operational-intensity 散佈（每類一群）。輸出 `docs/figures/phase0.2/*.png`。→ verify：`python tools/plotting/op_breakdown.py` 與 `roofline.py` 從 JSON 重產所有 PNG（無 hardcode）；重跑 byte-stable。

6. 寫 `docs/phase0.2-op-profile-findings.md`：每任務 prefill/decode op-mix、主導 sig（按 count 與按 FLOPs 各取 top-N）、每 (model,task) 的 **predicted** intensity 輪廓（明標「量測側 bound 判定留待 Phase 0.3/1」）、measured-vs-interpolated sig 佔比（給 0.3 量測優先序）、嵌入 step 5 圖。→ verify：16 個 (model,task) 各有 mix + 主導 sig + predicted-intensity + measured 佔比；圖被引用。

7. commit → `tools/trace_export/op_profile.py` + `tools/plotting/*` + `measurements/op_profile/*` + `docs/figures/phase0.2/*` + `docs/phase0.2-op-profile-findings.md`。→ verify：`git ls-files` 列出；圖可由 script 重產（不依賴未納版控的中間檔）。

Outputs:
- `tools/trace_export/op_profile.py`、`tools/plotting/{op_breakdown,roofline}.py`
- `measurements/op_profile/{model}_{task}.json`（16）+ `measurements/op_profile/sweep_{model}.json`（4）
- `docs/figures/phase0.2/*.png`（op-breakdown + roofline，皆可從 JSON 重畫）
- `docs/phase0.2-op-profile-findings.md`
