# Plan: Phase 2.1 — M5 workload + M3 event engine + 共享 BW 競爭 + L4 分解

Wave 2.1 of Phase 2（context + 鎖定決策見 [phase-2.md](phase-2.md)）。最高風險先做：去循環的 L4 機制驗證。Action-only。

## Steps (action → verify)

1. Branch `phase-2.1` off main → verify: `git branch --show-current` == phase-2.1。

### Step 0 — 入口（先全綠，再寫任何 runtime 程式碼）
2. 重跑 `tools/plotting/site_m8.py` 重產 3 張 stale thermal 圖（`m8_heating`/`m8_load_sweep`/`m8_perf_temp`，只吃 committed `validation/reports/phase0.4/thermal.json`） → verify: `.venv/bin/python docs/report/phase1-site/build.py --strict` 綠（無 stale）。
3. 誠實修正（committed 缺陷，trace 到 JSON）：(a) `docs/adr/0006-*.md` 釐清「≈24 GB/s」(LPDDR4x 串流效率錨) vs committed `fit_BW_GBs:18.33`(decode e2e 擬合 BW) 為不同量；(b) 同檔「4c/1c 1.31×→1.12×」改成實測 `1.130/1.096/1.081`（方向不變[8B 最緊]、量值修正，標 `vendor_llm_int8.json`）；(c) `validation/contracts/m3.yaml` 把 `{type: contention_knee, threshold: 15%}` 改為 `{type: contention_trend, tag: simulated}`（無數值 gate），並移除 acceptance(line 8) + `tunable_params` 註解(line 10) 兩處的「reproduce ~60 GB/s」字眼；(d) `tools/report/_metrics.py` 對 (a)(b) **無需改**（那些數字只在 ADR/m3.yaml prose，不在 metrics）——只確認 `pytest` 仍綠 → verify: `.venv/bin/pytest tests/`（含 `test_report_metrics`）綠；`grep '1.31'` 與 `grep -i 'reproduce.*60 GB/s'` 無命中（錨定字串，**不**裸 grep `24`/`0.65` 以免誤傷正確值）。

### M5 — workload → op DAG
4. `simulator/runtime/dag.py`：`@dataclass OpNode(id, category, wl, deps, unit=None, bytes_streamed=0)` + `Dag(nodes, successors())` → verify: `tests/test_dag.py` —— DAG 無環、每 node `wl` 形狀 sanity（M/K/N≥0、dtype 合法）。
5. `simulator/runtime/workload.py`：`build_dag(model, P, D)` 包既有 `tools/trace_export/op_profile.py::Model`（不重跑 tracer）；`wl_from_row(row, cfg) -> Workload`（aten→Workload，(M,K,N)/flops/bytes 萃取重用 `Model._flops_bytes`）；canonical layer 順序 + chain deps（QKV→attn、gate/up→swiglu 為唯一 fan-out/join）；`count` 取自 profile，**絕不**手乘 layers → verify: per-token DAG op 計數 Σ over (P,D) == `Model.profile(P,D)` 計數（注意 `profile` 已含 ×layers 與 decode count×D 多重性，DAG builder 須重現同樣 multiplicity）；重用 `validate_m5_trace.py` 邏輯：`expected_ops_check.all_semantic_covered`==true + zero-orphans（4 models）。

### M3 — 事件引擎 + 競爭
6. `simulator/runtime/resources.py`：`SramBandwidth`/`SharedBandwidth`（飽和：總需求≤knee → 各 op 全速；>knee → 按 demand 比例縮減使 aggregate 封頂在 knee）+ `ComputeUnit(name, engine, busy_until)` → verify: 單元測試 —— 總需求<knee 全速；>knee aggregate==knee 且按 demand 分配；knee=∞ 時退化成線性疊加。
7. `simulator/runtime/events.py`：自寫 `heapq` 迴圈 `run_dag(dag, platform, bw, *, concurrency=True, contention=True) -> token_latency_us`；`OP_READY`/`OP_DONE`；每 op base latency = `unit.engine.predict(node.wl)['latency_us']`（M1/M2/M4，**絕不自算**）；memory-bound op 經 `SharedBandwidth` 拉伸 → verify: `tests/test_event_engine.py` —— 手算 toy DAG（2 單元 1 依賴）命中已知 finish time；單元內串列不變式；`concurrency=False` == 各 latency 相加；`contention=False` 移除 knee。
8. `simulator/runtime/platform.py`：`Platform` 綁一組 unit engine + memory topology（`load_spec` 餵）= 模擬 SoC；`runner.py`：`run(model, P, D, scheduler, *, ablations) -> metrics dict`（tok/s、ttft(report-only)、energy band、knee 診斷(僅 trend 形狀，非數值 gate)、ablation deltas）串 M5→scheduler→M3→M7 → verify: 跑 llama-3.2-1b / llama-3.1-8b 出非負 tok/s + energy band；ablation flags 改變結果方向正確。

### L4 去循環（D1，本 wave 核心）
9. `AllCimScheduler`（最小映射，2.2 收進 scheduler.py）：matmul→CIM、attention→GPU、norm/rope/swiglu/softmax/residual/sampling→CPU、kv→mem；**decode 路徑不使用 e2e 擬合的 `BW_eff=18.33`**——CIM GEMV 走 `m1_cim_tile`(L1 校準) + 記憶體串流走 `m2_memory` `mem_lpddr4x`（**24.2 GB/s、eff 0.71、measured anchor**；**非** lpddr5 的 0.65 sim 值）+ CPU/GPU/kv 顯式並發 overlap → verify: 每 op provenance 標 silicon-fit 來源；`grep` 確認無 `18.33`/L4-tuned 常數注入 decode 路徑。
10. `validation/validate_contention.py`：合成並發 workload 掃 aggregate demand → 總有效 BW 呈 knee（升至 ~knee 飽和、非線性疊到 peak）；對 Ramulator2 多流飽和 + **Card 4c/1c 實測趨勢**（1.130/1.096/1.081，8B 最緊）做 trend 比對；`contention=False` ablation 移除 knee → verify: `validation/reports/phase2/contention.json` 寫 trend 比對 + 全標 `simulated`、無數值 gate；4c/1c 方向一致。
11. `validation/validate_e2e_l4.py`：分三層、各自誠實標：
    - (a) **smoke**（by-construction，非證據）= AllCim ctx-1024 算術重現 closed-form，明標。
    - (b) **mechanism**（非循環、silicon-anchored、≤15% gate）= 各單元獨立計價之和（CIM-GEMV[L1] + mem_lpddr4x 24.2 + 顯式 CPU/GPU/kv/overlap，**無 e2e refit**）對 **3× 1c** vendor config decode tok/s ≤15%；8B 為 hold-out（對齊 recompose 9.5%）。
    - (c) **simulated 示範**（無 silicon 真值，**不入 ≤15% gate**）= 4c/1c spread（多核競爭趨勢）+ 長文本 KV 流量成長，展示「per-byte streaming 機制隨 核數/context 變動、凍結常數做不到」。注意 LongBench 為 prefill-heavy（prefill 11753/decode 4），長 context 在 prefill，**無 high-context decode 量測**，故此點為 `simulated` extrapolation、僅定性。
    → verify: `validation/reports/phase2/e2e_l4.json` 三層分標（smoke / mechanism 3×1c ≤15% / simulated-demo）；mechanism 未動整合層調參（Gate-6c），PR 描述指出責任元件。

## Outputs
- `simulator/runtime/{dag,workload,resources,events,platform,runner}.py` + `AllCimScheduler`
- `tests/test_{dag,event_engine}.py`
- `validation/validate_{contention,e2e_l4}.py` + `validation/reports/phase2/{contention,e2e_l4}.json`
- 修正後的 `docs/figures/phase0.4/m8_*.png`、`docs/adr/0006-*.md`、`validation/contracts/m3.yaml`、`tools/report/_metrics.py`

## Gate 2.1
入口（`build.py --strict` 綠 + 誠實修正）先過 → M5 過 oracle → M3 toy + ablation 方向正確 → contention knee trend 對 4c/1c 一致（標 `simulated`，無數值 gate）→ L4 mechanism（**3× 1c** decode）≤15%（4c/長文本為 `simulated` 示範、無 gate）→ smoke/mechanism/simulated-demo 明確分標。
**只允許 claim：decode mechanism、smoke vs mechanism、simulated contention trend；不得 claim full prefill/TTFT validation 或「end-to-end validated」**（prefill path 仍 analytic/unvalidated）。
