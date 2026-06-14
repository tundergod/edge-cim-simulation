# Plan: Phase 2.2b — Scheduler ABC + value-based conversion-op + CimHetero + group-aware GPU-attention (TDD)

Wave 2.2b of Phase 2.2（接續已 merge 的 2.2a value-flow 基礎；鎖定決策 E1–E8 / 需求 R1–R7、ADR-0003/0004 見 [phase-2.md](phase-2.md)）。**TDD：每步先寫紅燈 test → 實作轉綠。** Action-only。唯一硬 silicon gate = **AllCim 重現 L4 ≤15%（回歸，不得被本 wave 改動）**；CimHetero 異質/混合精度因無並發晶片可量（Aetina 送修，#52）= **simulated**，只證 conversion-op **成本**，不證品質（D3）。

## 範圍
2.2b = 交付專案自身的 **CIM-INT8 matmul × GPU-FP16 attention 混合精度 config + conversion-op 成本**：Scheduler ABC（收斂 2.2a AllCim）、value-based conversion-op(R7 conversion_bytes)、group-aware GPU-attention composite pricing(R2)、CimHetero scheduler、doc sync。**HeteroInfer-style SOTA 重現、tensor-level weight-centric split、topology A/B(2.3)、capacity/SRAM(#58) 不在本 wave**（2.2a 已記「延後排除」）。

> **與 umbrella 的關係**：`phase-2.md` 原 Wave 2.2 = 「M6 排程器 + HeteroInfer SOTA」。實際 decompose 成 2.2a(value-flow 基礎)+ 2.2b(M6 ABC + 專案自身 CimHetero + conversion);**HeteroInfer SOTA 重現順延到後續 wave**(非靜默丟棄,ADR-0003 validation-first 仍成立)。

## Load-bearing 約束（實作須遵守）
- **conversion = int8↔fp16 cast at the GPU boundary**（精確定義,AllCim-safe）：在 value-flow edge (p→c) 插 `convert` **iff**「p/c **恰一端在 GPU**(fp16 attention island) **且** 兩側 sim_precision 不同(非-GPU 端 int8、GPU 端 fp16)」。實際邊界(由 fixture 驗證,非手猜):**`kv_cache(int8,mem)→QK^T/S·V bmm(fp16,gpu)`**(K、V dequant 進 GPU)+ **`S·V(fp16,gpu)→O-proj matmul(int8,cim)`**(requant 出 GPU)。**同精度跨單元 edge(fp16↔fp16,如 GPU bmm↔CPU scale/softmax)= transfer 非 conversion,不在本 wave**(transfer_bytes 延後)。
- **AllCim 插 0 個 convert → L4 不變**：AllCim attention 在 **cim**(無 node 在 gpu)→「恰一端在 GPU」永不成立 → 0 conversion。**precision delta 本身不觸發**(cim int8 matmul→cpu fp16 support 在 AllCim 也有 delta,但無 GPU 端 → 不插)。**crossing count 由 fixture+CimHetero assignment 獨立掃描導出**(red test 用獨立 edge-scan 當 oracle,非手猜固定數)。convert **per crossing-edge**(若一 value fan-out 到多個 GPU 消費者,逐 edge 一個 convert,或 per (value,目標精度) 去重——實作擇一明寫)。
- **R2 = compute-pricing 不變式（#10）**：`m4_gpu.attn_bmm_us` 計 **一個** bmm(非合併 QK^T+S·V);CimHetero 對同一 attention block 的 QK^T+S·V 用 `pricing_group` 使 **compute 只計一次**(group 內 `compute_us` 總和 == 一次 attn_bmm_us)。**各 bmm node 仍各自保有 bytes_streamed**(K-cache、V-cache 是真實獨立 memory stream,引擎 per-node max(compute,mem) 各計,**非雙算**)。∴ 不變式斷言在 **`price()/compute_us`**,**非**引擎 token latency。
- **CimHetero ~2× 慢於 AllCim(誠實預期)**：`attn_bmm_us`(heads 乘入、未優化 lower-bound)使 GPU-attention compute-bound,decode 比 AllCim 慢約 2×——這是既有 m4_gpu 模型性質,**非 bug**;∴ CimHetero 價值是 **conversion 成本 + 忠實結構,非加速**(gotcha #3,batch=1 decode = 依賴鏈),且必標 simulated。
- **conversion 成本 = memory-bound cast**（ADR-0004,無新量測/新 param）：nbytes = n_elem×(read int8 + write fp16),由既有 M2 `stream`/elementwise 模型計價;**convert node 必有 `unit`**(scheduler 設,`_residency` 才能給 mem_domain)。`precision_boundary_placement` = tunable(哪個單元付)。
- **precision 來源 = sim_precision（#8）**,非 trace_dtype。
- **CimHetero 跑並發 = `pipeline=True`**（非僅 concurrency;2.2a 的 serial-vs-overlap 旋鈕是 `pipeline`,`pipeline=True` 自動 flag provenance simulated）。

## Steps (TDD: test 紅 → 實作 綠 → verify)

1. Branch `phase-2.2b` off main → verify: current。

### A. Scheduler ABC（重構,行為不變）
2. **test 紅**：`tests/test_scheduler.py` — `Scheduler(ABC).assign(dag,cfg)->dag` 純（assign 不改傳入 node 的既有欄位語意外的東西;同一 dag 重複 assign 結果一致）;`AllCimScheduler` 對每 node 產生與 2.2a `all_cim_assign` **完全相同** 的 unit + mem_domain;registry `"all_cim"`→AllCimScheduler;acyclic 保持。
3. **實作 綠**：改 `simulator/runtime/scheduler.py` — 加 `Scheduler` ABC,把 2.2a `all_cim_assign`+`_residency` 收斂成 `AllCimScheduler.assign`;`domain_byte_audit` 保留;runner 用 registry（instance/class）。**保留 `all_cim_assign` 薄 wrapper 以免破壞既有 import**(或一次改完所有 caller)。→ verify: test 綠;**AllCim L4 rel_error 逐模型 byte-identical**(非僅 ≤15%:`abs(err − committed) < 1e-4`,committed = `validation/reports/phase2/e2e_l4.json` 的 0.1073/0.0649/0.0311;≤15% 太鬆,抓不到 refactor 漂移);**runner 對 policy=all_cim 仍選 `run_serial`(pipeline=off)**;全 suite 綠。

### B. value-based conversion-op（R7 conversion_bytes）
4. **test 紅**：`tests/test_precision.py` — `insert_conversions(dag)` 在「恰一端 GPU + int8↔fp16」的 edge 插 `convert` OpNode。oracle = **獨立 edge-scan**(test 內直接掃 dag edges 用 precision contract+unit 判定,非呼叫 production code——避免循環),斷言 production `insert_conversions` 的插入點集合 == 獨立掃描集合。**oracle 須逐 (model, phase) 跑**(decode = 3/層:K+V dequant + S·V→O-proj requant;prefill = 2/層:無 kv_cache node——count phase-dependent,勿從 decode 導一次套用 prefill)。關鍵 case:**(a) AllCim DAG 插 0 個**(無 GPU node);**(b) 明確斷言 AllCim 的 cim(int8 matmul)→cpu(fp16 support) edge——有 precision delta 但無 GPU 端——插 0**(防止實作者誤改成「對 int8≠fp16 觸發」而在每條 cim→cpu 插 convert、靜默破壞硬 gate);**(c) CimHetero 的 kv_cache(int8,mem)→bmm(gpu) 與 S·V(gpu)→O-proj(cim) 確有 convert**。convert node `category="convert"`、有 `unit`(→`mem_domain` 經 `_residency`)、`bytes_streamed = n_elem×(read 1 + write 2)`(n_elem 由 producer out_shape 導出);`domain_byte_audit` 互斥含 conversion_bytes(無雙算);acyclic + value 無懸空(convert 重接在 p→c edge 上,`out_value==id`/`in_values==deps` 不變式維持)。
5. **實作 綠**：`simulator/runtime/precision.py` — `insert_conversions(dag, *, placement)`:走 value-flow edge,符合「恰一端 GPU + int8↔fp16」者插 convert node 並重接 deps/in_values/out_value(per crossing-edge,fan-out 處理明寫);convert node 賦 `unit`(per placement)→`_residency` 給 mem_domain。`precision_boundary_placement`(改 `config.py`:加 SimConfig dataclass 欄位 **且** 在 `from_dict` 讀進來——`_KNOWN_SCHED` 已保留此 key 但目前未讀;預設 consumer 單元付;**僅 default+override,不加 sweep/CLI surface**)。`platform.price` 認 `category="convert"`→memory-bound cast,`source_model="convert"`(由 memory 模型計,無新 param);`dag.CATEGORIES` 已含 `convert`。→ verify: test 綠;插入點集合 == 獨立 edge-scan;AllCim 仍 0 conversion → **L4 不變**。

### C. group-aware GPU-attention composite pricing（R2）
6. **test 紅**：`tests/test_pricing.py` — GPU 上的 attention,QK^T 與 S·V 由 `pricing_group` 標為一對;**compute-pricing 不變式:group 內各 node 的 `platform.price(n)["latency_us"]`(=compute) 總和 == 一次 `attn_bmm_us`**(斷言在 **price/compute_us,非引擎 token latency**——各 bmm node 的 `bytes_streamed`(K/V cache memory)由引擎獨立計,是真實 stream 非雙算,不納入此不變式);scale/mask(CimHetero 下固定落 **CPU** support,同 2.2a `wl_from_row`)+ softmax(CPU)**斷言不被當 bmm 計**(`source_model != "m4_gpu"`);**取代 2.2a 對 non-bmm GPU attention 的 fail-loud**(守門改成:有 `pricing_group` 的 bmm → composite,無 → 仍 fail-loud)。
7. **實作 綠**：`workload.py` 為同一 attention block 的 QK^T/S·V 賦相同 `pricing_group`(層內唯一);`platform.price` GPU-attention 分支:同 group 的 **compute** 只計一次(group representative 計 `attn_bmm_us`、同 group 其餘 compute 計 0;**各 node 仍各自帶 bytes_streamed**);scale/mask 固定 CPU support、softmax CPU(per CimHetero placement)。→ verify: test 綠;group `compute_us` 總和 == 一次 attn_bmm_us;scale/mask/softmax 不被當 bmm;2.2a fail-loud 被正確模型取代。

### D. CimHetero scheduler + simulated validation
8. **test 紅**：`tests/test_scheduler.py` — `CimHeteroScheduler`:matmul→cim(int8)、attention(bmm)→gpu(fp16)、softmax/norm/rope/ffn/residual→cpu、kv_cache/embedding→mem;set unit+mem_domain+precision;呼叫 `insert_conversions`;registry `"cim_hetero"`。runner 跑 CimHetero decode 經引擎用 **`pipeline=True`**(多單元 overlap 路徑;`pipeline=True` 自動 flag provenance simulated),metrics 含 conversion crossing count + conversion_bytes + `provenance` 標 simulated。
9. **實作 綠**：`CimHeteroScheduler` 入 scheduler.py;runner registry + CimHetero 走 **`pipeline=True`**(非 AllCim 的 `pipeline=False` 串列);AllCim 預設 `pipeline=False` 不變。`validation/report_mixed_precision.py`(**刻意命名 report 非 validate**——無並發 silicon ground truth,避免 validation-language):報 CimHetero decode + conversion overhead(對比 AllCim,**預期 ~2× 慢、GPU-attention compute-bound**),全標 **SIMULATED**;**不**寫 validated/measured mixed-precision。→ verify: CimHetero 跑(`pipeline=True`→provenance simulated);`reports/phase2/mixed_precision.json` **top-level `"label":"simulated"`** + 列 crossing count/conversion_bytes/decode delta;AllCim L4 路徑(`pipeline=False`)不受影響。

### E. doc sync
10. 更新 `docs/adr/0003`(Scheduler ABC + AllCim/CimHetero plugins 已實作)、`docs/adr/0004`(conversion-op 已 analytic 實作,memory-bound cast,無量測)、`CONTEXT.md` repo index(precision.py、scheduler classes、mixed_precision report)。→ verify: `build.py --strict` 綠;報告/文件無 "validated mixed-precision quality" 字眼(D3 limitation 明列)。

### F. Gate
11. 全 gate:(i) Scheduler ABC 純且單元測試過,AllCimScheduler == 2.2a 行為;(ii) **AllCim L4 rel_error byte-identical**(硬回歸,`abs(err−committed)<1e-4` vs 0.1073/0.0649/0.0311——非僅 ≤15%);(iii) conversion crossing count 可見 + 由既有模型計價(**無新 param**);(iv) R2 composite group 總和 == 一次 attn_bmm_us(無雙算);(v) CimHetero 跑、conversion 成本可見、`mixed_precision.json` top-level **標 simulated**;(vi) mixed-precision **品質未宣稱**(D3 limitation,報告無 validated-mixed-precision 字眼);(vii) `.venv/bin/pytest tests/` + `docs/report/phase1-site/build.py --strict` 綠。

## Outputs
`simulator/runtime/{scheduler,precision,workload,platform,runner,config}.py`;`tests/test_{scheduler,precision,pricing}.py` + 擴充既有 tests;`validation/report_mixed_precision.py` + `reports/phase2/mixed_precision.json`;更新 `docs/adr/0003`、`docs/adr/0004`、`CONTEXT.md`;`docs/plans/phase-2.2b.md`。

## 重用 / 不重造
2.2a value-flow DAG(in_values/out_value/precision/pricing_group/mem_domain)、fixture_io.PRECISION_CONTRACT、`domain_byte_audit`、`Platform.price`、`run_dag/run_serial`(pipeline 旋鈕);m1_cim/m4_gpu(attn_bmm_us)/m4_cpu/m7;ADR-0003(static-first swappable scheduler)/ADR-0004(conversion = explicit op,M2/M4 計價,無量測)。

## 範圍外（2.2b）
HeteroInfer-style SOTA 重現 + validate_sota（延後）、tensor-level weight-centric split、topology A/B(2.3)、sensitivity/holdout/extrapolation(2.3)、capacity/SRAM tier(#58)、多架構 topology(#59)、混合精度**品質**量測(D3,永久 limitation)、可切換 CPU/unit spec config 旋鈕（可選,非阻塞）。

## Workflow
此 action-plan → subagent plan-review（loop 至 clean）→ **使用者批准** → TDD 執行 → subagent code-review → PR `phase-2.2b`→main → 通知 → 使用者 merge。硬 gate = AllCim L4 回歸不變。
