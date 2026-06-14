# Plan: Phase 2.2a — trace-truth fixture → value-flow DAG → structural oracle → domains → provenance (TDD)

Wave 2.2a of Phase 2.2（umbrella + 鎖定決策 E1–E8 / 需求 R1–R7 見 [phase-2.md](phase-2.md) 與 plan-review）。**TDD：每步先寫紅燈 test → 實作轉綠。** 全程 fidelity-first；唯一硬 silicon gate = AllCim 重現 L4。Action-only。

## 範圍
2.2a = 忠實 value-flow 基礎（fixture + DAG + structural oracle + memory domains + per-op provenance）。**conversion-op / group-aware GPU-attention pricing / CimHetero 屬 2.2b**（2.2a 仍跑 AllCim：attention→CIM 計 0-compute，故 R2 composite/ R7 conversion_bytes 不在本 wave）。

## Steps (TDD: test 紅 → 實作 綠 → verify)

1. Branch `phase-2.2a` off main → verify: current。

### A. trace-truth fixture（R4/R5/R6 — 基礎）
2. **test 紅**：`tests/test_trace_fixture.py` — (R5) 每 `in_value`→ 更早 `out_value` 或明確 external（weight/const/cache）；alias/view 有 `alias_of`；多輸出 op 不丟 value；違反 fail-loud。**(S1-1 adversarial) id-reuse 案例：produce→`del`+`gc`→produce，斷言不得生假 edge**（CPython 會回收 id，必須測）。(R4) 每 op 有 `trace_dtype` 與 `sim_precision` 兩欄。(R6) fixture mirror op_profile anchors+held-outs 多長度。
3. **實作 綠**：擴 `tools/trace_export/`（重用 op_inventory TorchDispatchMode+FakeTensor）記 producer-consumer edges + value-id + trace_dtype。**(S1-1) recorder 對看過的每個 tensor 持 strong-ref（keep-list），首次出現賦單調遞增 value-id**——否則 GC 回收 id 會偽造 edge（已驗：strong-ref → 10/10 unique）。**(S2-3) 先定義 precision-contract 表**（CIM=int8、GPU=fp16、CPU-support=fp16/fp32 per ADR-0004c）作 `sim_precision` 來源，**`sim_precision` 不取 `trace_dtype`**（trace 的 eager-fp16 ≠ 模擬 INT8 placement）。產 4 模型 fixture committed（gz）。→ verify: test 綠（含 id-reuse）;fixture 重建 ordered op-graph;對 op_profile counts 一致。

### B. value-flow DAG（R1/R6）
4. **test 紅**：擴 `tests/test_dag.py`/`test_workload.py` — `OpNode` 有 `out_value/in_values/precision/pricing_group`;`build_token_dag` 由 value-flow template 任意 P/D/KV instantiate。**(S3-1 R6 合成規則，明寫)**：fixture edge 是 per-op-role；count 展開時**把單層內 edge pattern 複製到每個展開 node、層間以 residual value 串接**。attention = **完整鏈，且跨 category/unit**：QK^T(attn)→score_scale·mul(attn)→mask_add·add(attn)→**softmax(獨立 softmax category，AllCim 在 CPU)**→S·V(attn)（**S1-2**：softmax 非 attention category，勿融成單 category）;deps 由 value edges 導出;counts/bytes oracle（含 scale/mask）仍過;acyclic、value 無懸空。
5. **實作 綠**：改 `simulator/runtime/{dag,workload}.py`。→ verify: test 綠;oracle（counts+bytes）對 4 模型過。

### C. structural oracle（R1 — 對 fixture truth，非模板）
6. **test 紅**：`tests/test_dag_structure.py` — DAG 拓樸 == fixture：layer 邊界、Q/K/V fanout、**attention 鏈 QK^T→scale→mask→softmax(CPU,跨 unit)→S·V 消費 Q/K/V**（S1-2 明含 softmax 跨 category+unit boundary）、residual join、gate/up siblings、global node（embedding/final-norm/lm_head/sampling）數+位置。
7. **實作 綠**：structural oracle util（比對 DAG vs **獨立擷取**的 fixture edges，非對 build_token_dag 模板）。→ verify: test 綠;拓樸對 fixture 一致。

### D. memory domains + residency + byte accounting（R7 op_bytes/transfer_bytes — conversion_bytes 留 2.2b）
8. **test 紅**：`tests/test_domains.py` — `OpNode.mem_domain`∈{dram,cpu_cache,none}（禁 local）;Platform `{dram(24.2),cpu_cache(A76)}` 池;events 按 domain fair-share、跨域不互搶;**residency 規則**（intra-unit 小 activation→cpu_cache;cross-unit produced/consumed hidden value transfer→dram）;**byte accounting 互斥**（op_bytes vs transfer_bytes 不雙算）+ oracle;**(S-dc 既有雙算 bug，明確 target)**：CPU-support op 的 memory **已在** `m4_cpu.predict` 內部計（`memory_us=wsb/cache_bw`），引擎又把同 op 的 `bytes_streamed` 灌進 DRAM `SharedBandwidth` → **雙算**。residency 規則須清掉：CPU-support → `mem_domain=cpu_cache` 且**其 bytes 不再灌 DRAM pool**（m4_cpu 內部 cache memory_us 為真值）；red test 斷言「同一 CPU-support op 的 memory 不同時出現在 compute_us(cache) 與 DRAM SharedBandwidth」。**(S2-1) 斷言 residency 規則前後 dram-domain byte 總量差 == CPU-support 份額**（規則作用在競爭資源、效應須量測非假設；實測 1B decode 占比 cim 97.1%/mem 2.6%/cpu 0.4%）。
9. **實作 綠**：改 `resources/events/platform/scheduler.py`(AllCim 設 domain)。→ verify: test 綠;混合域 toy（cache 不拖 dram）;byte oracle 無雙算;**(S2-1) step D 後重跑 L4**（非沿用 2.1 數字）1B/3B/8B ≤15%。

### E. per-op provenance（#55）
10. **test 紅**：擴 `tests/test_runner_e2e.py` — 每 op 有 `compute_provenance`+`source_model`;bound 由引擎決定。
11. **實作 綠**：`Platform.price(node)->{latency_us,compute_provenance,source_model}`（包 m1/m4_gpu;cpu/npu 已有 predict）;events 收 record;runner metrics 摘要。→ verify: test 綠。

### F. Gate
12. 全 gate：(i) fixture 重建==op_profile counts;(ii) structural oracle 對 fixture 全過;(iii) counts/bytes oracle;(iv) **AllCim L4 ≤15%**(硬，**step D 後重跑**);(v) **serial-vs-concurrent AllCim delta = 硬 sub-gate**（非僅報告）：data-deps 下 1B 不得跳到 ~41%（value-flow DAG 移除了「串列鏈使 L4 by-construction 過」的性質，故為真風險）。**Gate-6c：若 value-flow DAG 弄破 L4，回修 model（M1/M3），絕不靠 re-serialize DAG 或整合層調參掩蓋**;(vi) domain/byte/precision/provenance 正確;(vii) `.venv/bin/pytest tests/` + `docs/report/phase1-site/build.py --strict` 綠。

## Outputs
`tools/trace_export/`(fixture 擴充) + committed fixture(gz);`simulator/runtime/{dag,workload,resources,events,platform,scheduler,runner}.py`;`tests/test_{trace_fixture,dag_structure,domains}.py` + 擴充既有 tests;`docs/plans/phase-2.2a.md`。

## 重用 / 不重造
op_inventory TorchDispatchMode+FakeTensor;op_profile `_flops_bytes`/`_key`/`categorize`/`src`;2.1 runtime + oracle_check + validate_e2e_l4;specs cpu_rk3588/mem_lpddr4x;sweep_matrix `src_category`(attention 含 scale/mask)。

## 範圍外（2.2a）
conversion-op、group-aware GPU-attention composite pricing(R2)、conversion_bytes(R7)、CimHetero → 2.2b。tensor-split / topology-spec 池(#59) / "local" domain / HeteroInfer-style → 延後排除。

## Workflow
此 action-plan → subagent plan-review（loop 至 clean）→ **使用者批准** → TDD 執行 → subagent code-review → PR `phase-2.2a`→main → 通知 → 使用者 merge。硬 gate = AllCim L4 回歸。
