# Plan: Phase 0.1 — 生成 trace 與 op inventory

範圍：純軟體；Apple M3 Pro / macOS arm64 / CPU。

目標模型：`meta-llama/Llama-3.2-1B`、`meta-llama/Llama-3.2-3B`、`meta-llama/Llama-3.1-8B`、`Qwen/Qwen2.5-7B`（前三為 gated）。
Layer A 任務：ShareGPT(英文)、GSM8K、LongBench-TriviaQA、HumanEval。
Layer B 掃描：prefill ∈ {128,256,512,1024}、decode（= 輸出 token 數，供 step 6 組 trace）∈ {1,64,128,256}。
op inventory 抽 decode op 時的代表點以 `kv_len ∈ {128,512,1024}` 表示（ADR-0002 代表集；2048 無 Llama silicon 錨點故略）。
程式與輸出目錄不存在時先建立（`tools/trace_export/`、`measurements/op_inventory/`、`traces/` 等用 `mkdir(parents=True)`）。
本機已驗證可行的堆疊：torch 2.12.0 / transformers 5.9.0 / Python 3.13 / arm64。

1. 建環境 → 建 `.venv`；`pip install` 並把**實際解析到的版本** pin 進 `requirements.phase0.txt`（本機驗證：`torch==2.12.0`、`transformers==5.9.0`、`datasets`、`accelerate`；注意 transformers 為 **5.x**，attn 實作 API 與 4.x 不同）；確認環境變數 `HF_TOKEN` 已設，且該帳號已在 HF 網站獲准存取三個 meta-llama gated repo。→ verify：對四個模型各跑 `python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('<model>')"` **皆成功**（含三個 gated 的 meta-llama repo）。

2. 寫 `tools/trace_export/op_inventory.py` → 對每個目標模型用 `AutoConfig` 取 config，**強制 eager attention**：`AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")`（或先設 `cfg._attn_implementation="eager"` 再 from_config）——**勿用預設 `sdpa`，它會把 attention 融成單一 op、抹掉隨 kv_len 變化的 QK^T/softmax/S·V**；在 `FakeTensorMode` 下 from_config 實例化（不載實權重）。在 `TorchDispatchMode` 下：(a) 對每個 `prefill ∈ {128,256,512,1024}` 各跑一次 prefill（覆蓋 prefill 形狀軸，供 step 7 sweep matrix）；(b) 對每個 `kv_len ∈ {128,512,1024}`，**先 prefill 該長度建好 `DynamicCache`（`transformers.cache_utils.DynamicCache`），再餵 1-token decode（`past_key_values=cache`）**追蹤一個 decode step。記錄每個 aten op 的 `(op_type, input_shapes, output_shape, prefill|decode, kv_len)`，並標記每個運算元為 **activation 或 weight**（如 lm_head 的 `(2048,128256)` 標 weight），供 step 7 以 op 維度參數化、不被 weight 形狀污染。輸出 → `measurements/op_inventory/{model}.json`。→ verify：4 個 json 產生；op list 非空且每 op 有 shape；decode 的 QK^T / softmax shape 隨 `kv_len` 改變（出現分解後的 `aten._softmax`，而非單一 fused SDPA op）。

3. 寫 `tools/trace_export/expected_ops.py` → 因 traced 是 **aten 基本 op**、語意 op 是其組合，故把每個預期語意 op **展開成 aten 基本 op 集合**並檢查覆蓋：RMSNorm = pow+mean+add+rsqrt+mul；RoPE = cos+sin+neg+cat+mul+add；SwiGLU = silu+mul；QKV/O/FFN/lm_head = mm/addmm；QK^T、S·V = bmm；Softmax = _softmax；embedding = embedding/index。**所有無語意對應的 housekeeping primitive 一律白名單忽略**（view/transpose/_to_copy/expand/slice/arange/clone/where/_unsafe_view/unsqueeze/t/le/alias/lift_fresh/scalar_tensor/prim.* 等——清單為例，遇同類一併納入）；扣除白名單後仍無法歸屬的 traced op 寫進 `unmatched`。→ verify：每個預期語意 op 的基本 op 都在 traced 中找到，且（扣白名單後）`unmatched` 為空。

4. 真實權重交叉檢查（僅 `Llama-3.2-1B`）→ CPU 上載實權重（預設 dtype、不用 device_map/MPS）跑一次 prefill+decode，收集實際 op 集合，與 step 2 的 fake-trace 集合比對。→ verify：1B real-run 與 fake-trace 在**語意/白名單後 op 集合**上相等（raw aten 層的 detach/prim 等差異不計）。

5. 寫 `tools/trace_export/workload_stats.py` → 載入 4 個 dataset，用各模型 tokenizer 算 prefill/decode 長度 mean/median/p95。decode 欄位定義：**ShareGPT** = 取首個 user→assistant 配對（user 串為 prefill、assistant turn 為 decode）；**GSM8K** = `answer`（含 CoT 與 `#### N`）；**LongBench-TriviaQA** = `answers`；**HumanEval** = `canonical_solution`。輸出 `measurements/op_inventory/workload_lengths.json`。→ verify：4 任務 × 4 模型統計齊全；GSM8K ≈ (296,340)、LongBench-TriviaQA ≈ (1787,5) 與 HeteroInfer Table 4 同數量級。

6. 產生代表性 traces → 用 step 2 的 tracer，對每個模型 ×（Layer A 任務的平均長度點 + Layer B 掃描點）產生有序 op×shape 流，輸出 `traces/{model}_{label}.json`（Layer B：label = `{prefill}x{decode}`，對齊 OVERALL.md；Layer A：label = 任務名）。→ verify：每個目標 (model, point) 都有一個非空 trace；隨機抽一條，其 op×shape 與該模型的 op_inventory 一致。

7. 匯出 Phase 0.2 掃描矩陣 → 從所有 op_inventory 抽出去重後的 `(op_type, shape, 候選 precision)` 集合，輸出 `measurements/op_inventory/sweep_matrix.json`。→ verify：sweep_matrix 非空、已去重，且涵蓋 matmul / attention(QK^T,S·V) / norm / rope / elementwise 各類。

8. 寫 `docs/phase0-op-inventory.md` → 各模型 op 集合摘要、shape 參數化（以 hidden/heads/head_dim/seq/kv_len 表示）、dataset 長度 profile、export 過程意外、各 KV 長度的 silicon 錨點狀態（Llama ≤1024）。→ verify：檔案存在且上述每節都有內容。

9. commit。→ verify：`git ls-files tools/trace_export measurements/op_inventory traces docs/phase0-op-inventory.md requirements.phase0.txt` 列出所有新檔（皆已 tracked）。

Outputs:
- `measurements/op_inventory/{llama-3.2-1b,llama-3.2-3b,llama-3.1-8b,qwen2.5-7b}.json`
- `measurements/op_inventory/workload_lengths.json`、`measurements/op_inventory/sweep_matrix.json`
- `traces/{model}_{task}.json.gz`（gzipped；長上下文/完整 trace 由 Phase 2 on-demand）
- `docs/phase0-op-inventory.md`
- `tools/trace_export/{op_inventory.py, expected_ops.py, realweight_check.py, workload_stats.py, gen_traces.py, sweep_matrix.py}`、`requirements.phase0.txt`
