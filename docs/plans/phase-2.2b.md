# Plan: Phase 2.2b — Scheduler ABC + value-based conversion-op + CimHetero + group-aware GPU-attention (TDD)

Wave 2.2b of Phase 2.2（接續已 merge 的 2.2a value-flow 基礎；鎖定決策 E1–E8 / 需求 R1–R7、ADR-0003/0004 見 [phase-2.md](phase-2.md)）。**TDD：每步先寫紅燈 test → 實作轉綠。** Action-only。唯一硬 silicon gate = **AllCim 重現 L4 ≤15%（回歸，不得被本 wave 改動）**；CimHetero 異質/混合精度因無並發晶片可量（Aetina 送修，#52）= **simulated**，只證 conversion-op **成本**，不證品質（D3）。

## 範圍
2.2b = 交付專案自身的 **CIM-INT8 matmul × GPU-FP16 attention 混合精度 config + conversion-op 成本**：Scheduler ABC（收斂 2.2a AllCim）、value-based conversion-op(R7 conversion_bytes)、group-aware GPU-attention composite pricing(R2)、CimHetero scheduler、doc sync。**HeteroInfer-style SOTA 重現、tensor-level weight-centric split、topology A/B(2.3)、capacity/SRAM(#58) 不在本 wave**（2.2a 已記「延後排除」）。

> **與 umbrella 的關係**：`phase-2.md` 原 Wave 2.2 = 「M6 排程器 + HeteroInfer SOTA」。實際 decompose 成 2.2a(value-flow 基礎)+ 2.2b(M6 ABC + 專案自身 CimHetero + conversion);**HeteroInfer SOTA 重現順延到後續 wave**(非靜默丟棄,ADR-0003 validation-first 仍成立)。

## Load-bearing 約束（實作須遵守）
- **conversion 由 scheduler 宣告邊界，非全域規則**：AllCimScheduler 插 **0** 個 convert（all-AIPU 端到端 INT8，vendor tok/s 內無顯式 cast）→ **L4 不變**；CimHeteroScheduler 只在 **CIM↔GPU（int8↔fp16）實體單元邊界** 插 convert。cim↔cpu(AllCim) 是同晶片代理，**非** conversion 邊界。
- **R2 composite（#10）**：`m4_gpu.attn_bmm_us` 是 QK^T+S·V **合併**計價；GPU attention 必須把這對 bmm **計一次**（用 `pricing_group`），**勿雙算**；scale/mask(elementwise)/softmax 不可當 bmm 計（取代 2.2a 的 fail-loud 守門）。
- **conversion 成本 = memory-bound cast**（ADR-0004，無新量測/新 param）：nbytes = n_elem×(read int8 + write fp16)，由既有 M2 `stream`/elementwise 模型計價；`precision_boundary_placement` = tunable（哪個單元付）。
- **precision 來源 = sim_precision（#8）**，非 trace_dtype。
- **batch=1 decode = 依賴鏈（gotcha #3）**：CimHetero 的價值是 conversion 成本 + 忠實結構,**非**加速數字;多單元並發 = 真實但 simulated（無並發 silicon）。

## Steps (TDD: test 紅 → 實作 綠 → verify)

1. Branch `phase-2.2b` off main → verify: current。

### A. Scheduler ABC（重構,行為不變）
2. **test 紅**：`tests/test_scheduler.py` — `Scheduler(ABC).assign(dag,cfg)->dag` 純（assign 不改傳入 node 的既有欄位語意外的東西;同一 dag 重複 assign 結果一致）;`AllCimScheduler` 對每 node 產生與 2.2a `all_cim_assign` **完全相同** 的 unit + mem_domain;registry `"all_cim"`→AllCimScheduler;acyclic 保持。
3. **實作 綠**：改 `simulator/runtime/scheduler.py` — 加 `Scheduler` ABC,把 2.2a `all_cim_assign`+`_residency` 收斂成 `AllCimScheduler.assign`;`domain_byte_audit` 保留;runner 用 registry（instance/class）。**保留 `all_cim_assign` 薄 wrapper 以免破壞既有 import**(或一次改完所有 caller)。→ verify: test 綠;**AllCim L4 rel_error 逐模型 byte-identical**(非僅 ≤15%:`abs(err − committed) < 1e-4`,committed = `validation/reports/phase2/e2e_l4.json` 的 0.1073/0.0649/0.0311;≤15% 太鬆,抓不到 refactor 漂移);**runner 對 policy=all_cim 仍選 `run_serial`(pipeline=off)**;全 suite 綠。

### B. value-based conversion-op（R7 conversion_bytes）
4. **test 紅**：`tests/test_precision.py` — `insert_conversions(dag)` 在 **CIM↔GPU（int8↔fp16）** 邊界插 `convert` OpNode;**規則 key off 實體單元對,非 precision-contract delta**:**明確斷言 AllCim 內 cim(int8 matmul)→cpu(fp16 support) 這條「有 precision delta 但同晶片」的 edge 插 0 個 convert**(這是最易寫錯處——若實作者改成對 int8≠fp16 觸發,會在每條 cim→cpu 邊插 convert、靜默破壞硬 gate);整個 AllCim DAG 也插 0 個;crossing count == 手算(CimHetero 每層:matmul-out(cim)→attention-in(gpu) + attention-out(gpu)→matmul-in(cim));convert node `category="convert"`、`mem_domain` 由 placement 決定、`bytes_streamed = n_elem×(1+2)`;`domain_byte_audit` 互斥含 conversion_bytes(無雙算);acyclic + value 無懸空（convert 接在 producer→consumer edge 上,改 deps/in_values 一致）。
5. **實作 綠**：`simulator/runtime/precision.py` — `insert_conversions(dag, *, placement)`:走 value-flow edge,producer/consumer 落在 scheduler 宣告的 conversion 單元對時插 convert node（重接 deps/in_values/out_value）;`precision_boundary_placement`(改 `config.py` 加此 SimConfig tunable,預設 consumer 單元付;**僅 default+override,不加 sweep/CLI surface**)。`platform.price` 認 `category="convert"`→memory-bound cast,`source_model="convert"`(由 memory 模型計,無新 param);`dag.CATEGORIES` 已含 `convert`。→ verify: test 綠;crossing count == 手算;AllCim 仍 0 conversion → **L4 不變**。

### C. group-aware GPU-attention composite pricing（R2）
6. **test 紅**：`tests/test_pricing.py` — GPU 上的 attention,QK^T 與 S·V 由 `pricing_group` 標為一對;**不變式:對一個 group 的 latency 總和 == 一次 `attn_bmm_us`**(非「第一個非零、其餘零」這種對 node 順序脆弱的寫法);scale/mask(CimHetero 下固定落 **CPU** support,同 2.2a `wl_from_row`)+ softmax(CPU)**斷言不被當 bmm 計**(`source_model != "m4_gpu"` 或不走 attn_bmm_us);**取代 2.2a 對 non-bmm GPU attention 的 fail-loud**(該守門改成:有 `pricing_group` 的 bmm → composite,無 → 仍 fail-loud)。
7. **實作 綠**：`workload.py` 為同一 attention block 的 QK^T/S·V 賦相同 `pricing_group`(層內唯一);`platform.price` GPU-attention 分支:同 group 只計一次 composite(實作擇一明確規則,例:group representative 計 `attn_bmm_us`、同 group 其餘計 0,且**測試斷言 group 總和**而非個別 node);scale/mask 固定 CPU support、softmax CPU(per CimHetero placement)。→ verify: test 綠;group 總和 == 一次 attn_bmm_us;scale/mask/softmax 不被當 bmm;2.2a fail-loud 被正確模型取代。

### D. CimHetero scheduler + simulated validation
8. **test 紅**：`tests/test_scheduler.py` — `CimHeteroScheduler`:matmul→cim(int8)、attention(bmm)→gpu(fp16)、softmax/norm/rope/ffn/residual→cpu、kv_cache/embedding→mem;set unit+mem_domain+precision;呼叫 `insert_conversions`(CIM↔GPU);registry `"cim_hetero"`。runner 跑 CimHetero decode 經引擎(多單元並發 `concurrency=True`,**simulated**),metrics 含 conversion crossing count + conversion_bytes + `provenance` 標 simulated。
9. **實作 綠**：`CimHeteroScheduler` 入 scheduler.py;runner registry + 跑並發路徑(非 AllCim 的 pipeline=off 串列;CimHetero 是真多單元 → concurrency 開,標 simulated/provenance)。`validation/report_mixed_precision.py`(**刻意命名 report 非 validate**——無並發 silicon ground truth,避免 validation-language):報 CimHetero decode + conversion overhead(對比 AllCim),全標 **SIMULATED**;**不**寫 validated/measured mixed-precision。→ verify: CimHetero 跑;`reports/phase2/mixed_precision.json` **top-level `"label":"simulated"`/`provenance:simulated`** + 列 crossing count/conversion_bytes/decode delta;AllCim L4 路徑不受影響。

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
