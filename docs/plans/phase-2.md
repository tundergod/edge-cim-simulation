# Plan: Phase 2 — 整合成端到端 event-driven CIM-LLM 推論模擬器

## Context

Phase 0.1–1.3 已交付：silicon 校準的「engine + 可換 spec」per-unit 元件層（M1 CIM、M2 記憶體、M4 CPU/GPU/NPU、M7 energy），都掛在凍結的 `UnitEngine.predict(wl) -> {latency_us, bound, provenance}` 介面後。**M3（事件引擎）、M5（workload 產生器）、M6（排程器）目前只有 `validation/contracts/*.yaml`，沒有實作。** Phase 2 把這些零件整合成一個模組化、接近真實、可對 silicon 驗證的端到端模擬器，跑完整 prefill+decode，做 L4/L5 + contention knee + hold-out + ablation 驗證。

設計鎖定在 ADR-0001（輕量、**非** cycle-accurate 離散事件；單元並發；共享記憶體頻寬為競爭資源）、ADR-0002（記憶體可換、多單元競爭無 silicon）、ADR-0003（static-first、可換 `Scheduler`、validation-first SOTA）、ADR-0004 rev（精度邊界 = 解析 cast op）、ADR-0006（系統 gate：e2e ≤15%、Gate-6c 不得用整合層調參掩蓋元件失誤）。

**判準是 fidelity，不是發表**：唯一的成功標準是模擬器預測的行為對不對得上真實系統的瓶頸；對不上就如實說對不上。下列頂會模擬器是「如何讓**非** cycle-accurate 引擎仍對得上真機」的**工程先例**（不是投稿賣點）：**ONNXim**（event-driven compute + cycle-level memory 分層，0.23% MAE vs RTL）、**LLMCompass**（解析 per-op、4.1% e2e）、**GenZ**（解析 + 校準效率因子、5.82%）、**ASTRA-sim**（可換 analytical/cycle backend）、**Sniper**（interval、非 cycle-accurate、對真機 25% 仍可信）、**LLMServingSim**（per-iteration 結果重用，不展開整段生成）。有 silicon 真值可對的部分（per-unit + 單串流 + L4）才能宣稱對齊；無 silicon 處（contention、混合精度品質）一律如實標 `simulated`/limitation，不誇稱。

**關鍵重構（plan-review grill 後）—— 避免循環論證**：Phase 1 的 [recompose_e2e.py](../../tools/analysis/recompose_e2e.py) 把 effective decode BW 解到剛好命中 Metis Card 的 tok/s anchor（committed `fit_BW_GBs: 18.33`），所以「all-CIM 在 ctx-1024 重現 L4」是恆等式（by construction）。event engine 若只是包同一個常數再算 `tok_s = BW/bytes`，零證據增益。本計畫據此把 L4 重框為：ctx-1024 = 標記清楚的 smoke-test；真正的非循環驗證 = 各單元**獨立計價**（無 L4-tuned glue）+ event engine 預測凍結常數做不到的點（長文本 decode、實測 4c/1c 趨勢）。

## 已鎖定決策

| # | 決策 | 來源 |
|---|---|---|
| D1 | **L4 重框**：ctx-1024 重現 = by-construction smoke-test（標明）；真驗證 = 各單元獨立計價（移除 decode 路徑的 L4-tuned `BW_eff` glue，改 CIM GEMV[L1] + M2 0.65 效率 + overlap）+ 預測長文本 decode 與 4c/1c 趨勢（1.130/1.096/1.081） | 使用者 |
| D2 | **Contention knee**：現在出貨為「可掃描假設」，由 Ramulator2 多流飽和 + 唯一真實 silicon 訊號（Card 4c/1c 近乎持平縮放）錨定，全程標 `simulated`；m3.yaml 移除「reproduce ~60 GB/s knee」字眼；留 Aetina 修回升級 hook | 使用者 |
| D3 | **混合精度品質 scope out**：v1 只解析計價 conversion-op 的**成本**；品質假設引用文獻、明列 limitation，不做 first-party「validated mixed precision」字眼（現有硬體跑不了：Card 只跑預編譯 INT8、Alpha 不能跑 LLM 且送修） | 使用者 |
| D4 | **Scheduler 範圍**：建可換 `Scheduler` 介面 + `AllCimScheduler` baseline + 一個 SOTA plugin（`HeteroInferScheduler`，ADR-0003 點名的 external-validation）；HPIM 延後 | 使用者 |
| D5 | M3 = 自寫 `heapq` 事件迴圈（不引入 SimPy；~10²–10³ event/token） | 工程 |
| D6 | 新 runtime 套件命名 **`simulator/runtime/`**（不用 `engine/`——與既有 `simulator/engines/` 在 case-insensitive macOS 衝突，已驗證該目錄存在含 onnxim/ramulator2/scalesim） | 工程 |
| D7 | 重構（`models/→units/`、`engine.py→base.py`）為**非阻塞的 Wave 2.0**；2.1 不依賴它完成才能動科學 | 工程 |
| D8 | 三個 sub-wave：2.1 M5+M3+knee+L4-decompose；2.2 M6+HeteroInfer SOTA；2.3 full validation/sensitivity/extrapolation。各自走 CLAUDE.md per-phase workflow（branch→plan→subagent 審→使用者批准→執行→code-review→PR→merge） | 工程 |
| D9 | TTFT/prefill：v1 **報告但不 gate**（TTFT 餘量 = weight-load + prefill-attention + host overhead 仍是開放項）；gate 只看 decode tok/s | 工程 |
| D10 | conversion-op 成本由既有 M2/M4 per-op 模型 × 邊界穿越次數解析計價，**不新增 param、不量測**（ADR-0004 rev） | ADR-0004 |
| D11 | host-MMIO swap 的 per-call floor 用 **Alpha 實測 911µs**，不用 HeteroInfer 的 400µs；Card decode 路徑**不加** per-call sync 項（自家 silicon 說 on-card 無此 floor） | plan-review |

## 架構（6-box，模組化）

```
M5 workload (simulator/runtime/workload.py)  HF op_profile.Model → per-token op DAG (dag.py)
   │ OpNode[]  (category, Workload, deps, bytes_streamed)
M6 scheduler (simulator/runtime/scheduler.py) DAG→DAG 純函式：每 op 標 unit+precision；插 convert op
   │ scheduled DAG
M3 event engine (simulator/runtime/events.py + resources.py)  heapq 迴圈：單元並發 + 共享 BW 競爭(knee)
   │ 每 op latency 由 unit.predict(wl) 取得（M1/M2/M4，絕不自算）
M2/M1/M4 units (simulator/units/*)  既有擬合模型，介面凍結不動
   │ per-op time + bytes
M7 energy (simulator/units/m7_energy.py)  ±20% band，不給點估
   │
Output + inline validation (simulator/runtime/runner.py)  tok/s, ttft, energy band, knee, ablation
```

**M3 事件引擎**（自寫 heapq，全引擎約 80 行）：`OP_READY`/`OP_DONE` 兩種事件；每個 `ComputeUnit`（cim/gpu/npu/cpu）有獨立 `busy_until`（單元內串列、單元間並發）；**唯一競爭資源 = 記憶體頻寬** `SharedBandwidth`：當並發 memory-bound op 的總需求超過 `knee_GBs`，各 in-flight op 的有效 BW 按 demand 比例縮減使總和封頂在 knee（飽和曲線/fair-share，ONNXim/ASTRA-sim/Sniper 形式）。`knee_GBs`、`interconnect_efficiency`、`concurrency_overlap_factor` 為 m3.yaml tunable，全標 `simulated`。Ramulator2 多流**離線**跑來 inform knee，**不**進迴圈（保持 op 粒度、非 cycle-accurate）。

**M6 排程器**：`Scheduler(ABC).assign(dag, cfg) -> dag` 純函式。op→unit 由 characterization 決定：matmul(QKV/O/FFN/lm_head)→CIM INT8；attention(QK^T,S·V)→GPU FP16；norm/rope/swiglu/softmax/residual/embedding/sampling→CPU；kv-append→mem。`precision.py` 在 INT8↔FP16 邊界插 `convert` OpNode（memory-bound cast，由 M2/M4 計價，D10）。

**M5 workload**：包既有 [op_profile.py](../../tools/trace_export/op_profile.py) 的 `Model`（length template + `profile(P,D)` + `(M,K,N)`/flops/bytes 萃取），不在 sim 時重跑 tracer；per-token DAG，decode 跨 token 重用（LLMServingSim 式）。`count` 已含 ×layers，**絕不**手乘。

## Repo 佈局（重構後）

```
simulator/
├── specs/            # 不動
├── engines/          # 不動（heavy-sim cache；複數）
├── units/            # ← models/ 改名：base.py(←engine.py) + m1/m2/m4/m7 + params/
└── runtime/          # ← 新：dag.py workload.py scheduler.py precision.py resources.py events.py platform.py runner.py
validation/
├── contracts/        # m3/m5/m6.yaml 把 phase1_scope: contract_only → validated；m3.yaml 移除 reproduce-knee 字眼
├── reports/phase2/   # ← 新：e2e_l4.json topology_ab.json contention.json sensitivity.json holdout.json sota.json
└── validate_e2e_l4.py validate_topology_ab.py validate_contention.py validate_sensitivity_l5.py validate_holdout.py validate_sota.py
tests/                # ← 新：test_dag.py test_event_engine.py test_scheduler.py test_runner_e2e.py
docs/plans/           # ← 新：phase-2.1.md phase-2.2.md phase-2.3.md（action-only；2.0 可併入 2.1 首步）
```

---

## Wave 2.0 — 重構（非阻塞；可併入 2.1 首 PR）

1. Branch `phase-2.0-refactor` off main → verify: branch current。
2. `git mv simulator/models simulator/units`；`engine.py→base.py`；`params/` 隨之（`_PARAMS` 已相對 `__file__`） → verify: 目錄/檔案就位。
3. 更新 ~30 import（`simulator.models.engine`→`simulator.units.base`、`simulator.models.m*`→`simulator.units.m*`）；用 anchored `grep -rl … | while read -r f; do …` 改，不用 blanket sed（macOS 多行陷阱） → verify: `grep -rn 'simulator.models' --include='*.py' .` 只剩刻意引用（理想 0）。
4. 建空 `simulator/runtime/__init__.py` → verify: 可 import。
Outputs: `simulator/units/`、`simulator/runtime/`（空）。
**Gate 2.0**: `.venv/bin/pytest tests/` 綠；每支 Phase-1 `tools/analysis/fit_*.py` / `validate_*.py` / `recompose_e2e.py` 重跑後 `validation/reports/phase1.*/*.json` 與 committed 逐位元組相同；`docs/report/phase1-site/build.py --strict` 綠。

---

## Wave 2.1 — M5 + M3 + 共享 BW 競爭 + L4 分解（最高風險先做）

Branch `phase-2.1` off main。先寫 `docs/plans/phase-2.1.md`（action-only）→ subagent 審至無 issue → 使用者批准 → 執行。

0. **入口（先過，plan-review 要求提前；這些是 Phase 2 的驗證目標本身，不能留到 2.3）**：
   - (a) 修 `build.py --strict` 的 3 張 stale thermal 圖——重跑 `tools/plotting/site_m8.py`（`m8_heating`/`m8_load_sweep`/`m8_perf_temp`，只吃 committed `validation/reports/phase0.4/thermal.json` 重產；main 上既有的破口）。
   - (b) **誠實修正**：ADR-0006 的「≈24 GB/s」vs committed `fit_BW_GBs:18.33` 釐清（不同量）；「4c/1c 1.31×→1.12×」改成實測 `1.130/1.096/1.081`；`m3.yaml` 移除 `contention_knee` 的「reproduce」字眼、把「≤15%」**降級為「simulated trend 檢查，無數值 gate」**；`tools/report/_metrics.py` expected numbers 同步。
   → verify: `build.py --strict` 綠；`.venv/bin/pytest tests/`（含 `test_report_metrics`）綠；每個改動數字 trace 到 committed JSON。
1. `simulator/runtime/dag.py`：`OpNode`(id, category, wl:Workload, deps, unit, bytes_streamed) + `Dag`(nodes, successors) → verify: `tests/test_dag.py` DAG 無環、每 node 的 wl 形狀 sanity。
2. `simulator/runtime/workload.py`：`build_dag(model, P, D, kv)` 包 `op_profile.Model`；`wl_from_row(row,cfg)` 把 aten op→Workload（(M,K,N) 萃取重用 op_profile `_flops_bytes`，flops/bytes 與 oracle 一致） → verify: per-token DAG op 計數 Σ over generation == `Model.profile(P,D)` 計數（重用 `_sum_by_key` 等式）；m5.yaml `oracle_semantic_covered` + zero-orphans（4 models）。
3. `simulator/runtime/resources.py`：`SharedBandwidth`（飽和 knee + demand 比例 fair-share）+ `ComputeUnit`（busy_until） → verify: 單元測試——總需求 < knee 時各 op 全速；> knee 時總和封頂在 knee。
4. `simulator/runtime/events.py`：自寫 heapq 迴圈 `run_dag(dag, platform, bw) -> token_latency_us`；每 op 經 `unit.predict(node.wl)['latency_us']` 取 base，memory-bound op 經 `SharedBandwidth` 拉伸 → verify: `tests/test_event_engine.py` 手算 toy DAG（2 單元 1 依賴）命中已知 finish time；單元內串列不變式；concurrency-off == 各 latency 相加。
5. `simulator/runtime/platform.py` + `runner.py`：綁一組 unit engine + memory topology = 模擬 SoC；`runner` 串 M5→（stub AllCim 映射）→M3→M7，輸出 metrics dict（含 ablation flags: concurrency_off, contention_off） → verify: 跑 1B/8B 出非負 tok/s/energy band。
6. **`AllCimScheduler`**（2.2 移入 scheduler.py 前的最小映射；亦即 L4 baseline 與 D1 的獨立計價路徑）：decode 路徑**不**用 e2e 擬合的 `BW_eff=18.33`，改 CIM GEMV(L1) + M2 `mem_lpddr4x`（**24.2 GB/s、eff 0.71、measured anchor**；非 lpddr5 的 0.65 sim 值）+ overlap；CPU/GPU/kv 顯式並發 → verify: 各項 provenance 字串標明 silicon-fit 來源、無 `18.33`/L4-tuned 常數。
7. `validation/validate_contention.py`（**knee 驗證移到此處，先於 schedulers**，plan-review 修正）：合成並發 workload 掃 demand → 總有效 BW 呈 knee（升至 ~knee 飽和、不線性疊到 peak）；對 Ramulator2 多流飽和 + **Card 4c/1c 實測趨勢**（1.130/1.096/1.081，8B 最緊）交叉；contention-off 移除 knee → verify: `reports/phase2/contention.json` 寫出趨勢比對 + 全標 `simulated`；4c/1c 方向與量值落在容差。
8. `validation/validate_e2e_l4.py`（**D1 重框**，三層各自誠實標）：(a) **smoke** = AllCim ctx-1024 算術重現 closed-form，by-construction、非證據；(b) **mechanism**（非循環、silicon-anchored、≤15% gate）= 各單元獨立計價之和對 **3× 1c** vendor config decode tok/s ≤15%（無 e2e refit；8B hold-out 對齊 recompose 9.5%）；(c) **simulated 示範**（無 silicon 真值、不入 gate）= 4c/1c spread + 長文本 KV 成長（LongBench 為 prefill-heavy prefill 11753/decode 4，無 high-context decode 量測，故定性） → verify: `reports/phase2/e2e_l4.json` 三層分標；mechanism(3×1c) ≤15% 且未動 integration 調參（Gate-6c）。
Outputs: `simulator/runtime/{dag,workload,resources,events,platform,runner}.py`、`tests/test_{dag,event_engine}.py`、`validation/validate_{contention,e2e_l4}.py`、`reports/phase2/{contention,e2e_l4}.json`、`docs/plans/phase-2.1.md`。
**Gate 2.1**: 入口（build --strict 綠 + 誠實修正）先過；M5 過 oracle；M3 toy + ablation 方向正確；contention knee 趨勢對 4c/1c 一致（標 simulated，無數值 gate）；L4 mechanism（decode）≤15%；smoke/mechanism 明確分標。**2.1 只允許 claim：decode mechanism、smoke vs mechanism、simulated contention trend；不得 claim full prefill/TTFT validation 或「end-to-end prefill+decode validated」**（prefill path 仍 analytic/unvalidated，phase1.1-findings）。

---

## Wave 2.2 — M6 排程器 + HeteroInfer SOTA 重現

Branch `phase-2.2`。先寫 `docs/plans/phase-2.2.md` → subagent 審 → 使用者批准 → 執行。

1. `simulator/runtime/scheduler.py`：`Scheduler(ABC).assign(dag,cfg)->dag` 純函式 + 把 2.1 的 AllCim 收斂成 `AllCimScheduler` → verify: `tests/test_scheduler.py` assign 純（輸入 DAG 不被改）、每 node 有 unit、acyclic 保持。
2. `simulator/runtime/precision.py`：在 INT8↔FP16 邊界插 `convert` OpNode（nbytes = n_elem×(read+write)，由 M2 `stream` 或 CPU elementwise 計價，D10）；`precision_boundary_placement` 為 tunable → verify: 插入點 == 手算 layer 邊界數；crossing count 寫進 metrics（可見、可查）。
3. `HeteroInferScheduler`：依 SOSP'25 notes 編碼（matmul/FFN→NPU、norm/swiglu→GPU、大 matmul tensor-level weight-centric split、**CIM off**）；platform 配成 HeteroInfer-matched（部分 param estimated，標明） → verify: assign 產生對應 unit 分佈。
4. `validation/validate_sota.py`：HeteroInfer config 重現其 speedup **形狀**（非數值 ≤15%——不同 silicon + estimated param） → verify: `reports/phase2/sota.json` 標 `external-reference, shape-match`，不寫 validated。
Outputs: `simulator/runtime/{scheduler,precision}.py`、`tests/test_scheduler.py`、`validation/validate_sota.py`、`reports/phase2/sota.json`、`docs/plans/phase-2.2.md`。
**Gate 2.2**: scheduler 純且單元測試過（不需引擎）；conversion crossing count 可見、由既有模型計價（無新 param）；SOTA 形狀重現且誠實標籤。

---

## Wave 2.3 — full validation + sensitivity + extrapolation + 誠實修正

Branch `phase-2.3`。先寫 `docs/plans/phase-2.3.md` → subagent 審 → 使用者批准 → 執行。

1. `validation/validate_topology_ab.py`（ADR-0006 validate-then-swap）：同 workload 跑 `cim_topo_card`(A, L4-matched) → swap `cim_topo_edge`(B, host-LPDDR5+NoC；per-call floor 用 Alpha 911µs，D11)；輸出 A/B tok/s delta 表，B 標 `simulated/assumption` → verify: `reports/phase2/topology_ab.json` 明列 data-movement delta + honesty 標籤。
2. `validation/validate_sensitivity_l5.py`：±20% 掃 `eff_BW_GBs / knee_GBs / interconnect_efficiency / concurrency_overlap_factor` + M7 係數；定性結論（CIM-centric 主導）須在 band 內穩健；energy 報 band 不報點 → verify: `reports/phase2/sensitivity.json`。
3. `validation/validate_holdout.py`：1B/3B 擬合任何自由耦合 → 預測實測 8B，經完整引擎；須 ≤15% 且**不差於** closed-form 9.5%（若更差 = Gate-6c 信號，回修 M3 不調整合層） → verify: `reports/phase2/holdout.json`。
4. （誠實修正已移至 **Wave 2.1 step 0**——ADR-0006 數字、m3.yaml knee 語言、`_metrics.py` expected numbers 在入口就修，避免後續每個 validator 建在 stale target 上。）
5. 13B/32GB bounded extrapolation + sensitivity（ADR-0006）；mixed-precision **limitation 段落**（D3：品質未在目標 silicon 量測，引用文獻，明列為 limitation；v1 只證 conversion-op 成本） → verify: 報告頁 limitation 段存在、無 validated-mixed-precision 字眼。
6. 把結果注入 `docs/report/phase1-site`（或新 phase2 段）`{{key}}` → verify: `build.py --strict` 綠（未解析 key → fail）。
Outputs: `validation/validate_{topology_ab,sensitivity_l5,holdout}.py`、`reports/phase2/{topology_ab,sensitivity,holdout}.json`、更新的 ADR-0006/m3.yaml、報告頁、`docs/plans/phase-2.3.md`。
**Gate 2.3**: ADR-0006 系統門檻全報 PASS/FAIL + honesty 標籤；數字 trace 到 JSON；smoke vs mechanism、measured vs simulated 全程清楚；`pytest`+`build.py --strict` 綠。

---

## 驗證（端到端怎麼測）

- 每支 validator 讀 committed `measurements/`、寫 `validation/reports/phase2/*.json`（`pass`/`pass_all` bool + `{{key}}`）、印 PASS/FAIL——沿用 repo idiom。
- `tests/`：`test_dag` `test_event_engine`（toy 手算 + ablation 不變式）`test_scheduler`（純函式）`test_runner_e2e`（小模型端到端非負/單調）。
- 綠 gate：`.venv/bin/pytest tests/` + `docs/report/phase1-site/build.py --strict`（未解析 `{{key}}` → fail）+ Phase-1 fit/validate 重跑可重現。
- 圖：`tools/plotting/site_*.py` 每圖一支、只吃 committed 資料、PNG committed、絕不手繪。
- **Gate-6c 程序**：每個 validator 標明「責任元件」；整合失誤回修該元件的 param/模型，PR 描述須說明動了哪個元件——絕不寫「把 knee 調到過 L4」。

## 重用的既有資產（勿重造）

- [simulator/models/engine.py](../../simulator/models/engine.py)（→ `units/base.py`）：`UnitEngine`/`Workload`/`check_return` 凍結契約——M3 dispatch、M5 mapping、M6 都綁它。
- [tools/trace_export/op_profile.py](../../tools/trace_export/op_profile.py)：`Model` length template + `profile(P,D)` + `_flops_bytes`——M5 包它，不重跑 tracer。
- [tools/analysis/recompose_e2e.py](../../tools/analysis/recompose_e2e.py)：靜態 closed-form——M3 取代它；其 op→unit 指派、decode-BW backbone、1B/3B→8B hold-out、prefill/TTFT 開放項是 event engine 必須 match-or-beat 的基準。
- [measurements/metis_card/vendor_llm_int8.json](../../measurements/metis_card/vendor_llm_int8.json)：L4 anchor（6 config）；4c/1c = 唯一真實並發 silicon 訊號。
- [simulator/specs/cim_topo_card.json](../../simulator/specs/cim_topo_card.json) / [cim_topo_edge.json](../../simulator/specs/cim_topo_edge.json)：A/B topology swap。
- 既有 `validate_m5_trace.py`、`tools/report/_metrics.py`、`build_findings.py`：M5 oracle + 報告數字單一真相。

## 範圍外 / limitations（v1）

- **混合精度輸出品質**（D3）：現有硬體不可量；引用文獻、明列 limitation；v1 只證 conversion-op 成本。
- **多單元競爭 silicon 驗證**（D2）：Aetina 送修；knee 為 `simulated` 假設（Ramulator2 + 4c/1c 錨）；留升級 hook。
- **TTFT gate**（D9）：報告不 gate；TTFT 餘量為開放項。
- 既有 OVERALL §範圍外不變（INT4、AIPU Mode 2/3、batch>1、閉環熱、NVIDIA baseline）。

## Per-phase workflow（每 wave 都走）

每個 sub-wave 獨立：branch `phase-2.x` → 寫 `docs/plans/phase-2.x.md`（action-only）→ subagent plan-review（loop 至無 issue）→ **使用者批准** → 執行 → subagent code-review → `gh pr create` → 通知使用者 → 使用者確認後 merge。2.1→2.2→2.3 依序，wave 間硬 gate。
