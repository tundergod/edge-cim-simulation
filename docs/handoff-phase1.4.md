# 🤝 HANDOFF — Phase 1.4（doc-layer + 數字生成 + 整理）收尾與下一步

> 給下一個 session / 我自己。Phase 1.4 的程式碼/文件變更已在分支 **`report/phase1-overall`** 完成且在 sandbox 驗證過；**尚未 rebuild PDF、未 commit、未 merge**。本檔列出收尾步驟 + 之後的科學工作（Phase 2 前的補量測）。先讀 `plans/phase-1.4.md` 的「Execution record」。

## 0. 一句話現狀

Phase 1.1–1.3 元件層完成；Phase 1.4 把(A) conversion-op 改為 Phase-2 解析建模、(B) 報告/findings 數字改由 `validation/reports/*.json` 在 build 時生成（無法手打錯）、(C) repo 整理，全做在 `report/phase1-overall` 分支（pre-merge）。**下一步：在 Mac 上 rebuild + commit + merge，然後進 Phase 2。**

## 1. 立即收尾（Mac，依序）

1. **Rebuild 報告**（需 Chrome + repo 的 `.venv`，sandbox 跑不了）：
   `./.venv/bin/python tools/report/build_phase1_report.py`
   → 重生 `docs/report/phase1/index.html` + `phase1-report.pdf`，把章節的 `{{key}}` 換成 JSON 值、套用 conversion-op 新文字。**rebuild 前那兩個檔是舊的。**
2. **驗證**：`./.venv/bin/pytest tests/test_report_metrics.py`（6 過）；`./.venv/bin/python tools/report/build_findings.py`（應印 `in-sync`）；`./.venv/bin/python tools/analysis/check_phase1_2.py` 與 `check_phase1_3.py`（exit 0，無回歸）。
3. **commit**（`gh` 未裝，git 手動）：審 diff（20 改 / 4 刪 / 4 新增 + 62 個 figure untrack）。確認 `index.html`/PDF 已 rebuild 再一起 commit。push。
4. **merge**：`report/phase1-overall` → `main`（這批含 Phase 1.4 A+B+C）。
5. **cleanup PR**（merge 後另開）：刪 `docs/report/phase1.{1,2,3}/`（保留 `phase1/`、findings、validation、measurements）。見 `plans/phase1-overall-report.md` step 15。

## 2. Phase 1.4 改了什麼（審 diff 時對照）

- **A** conversion-op → Phase-2 解析建模（非量測缺口）：`docs/adr/0004-mixed-precision.md`、`docs/phase1.1-findings.md`、報告 `chapters/01,08,09`、`validation/contracts/m6.yaml`。**注意：第 9 章 GO/NO-GO 條件改了**（剩 prefill/multi-tile 一個量測前置條件）。
- **B** 數字生成：`tools/report/_metrics.py`（46 keys，每個讀 JSON path + 格式化）、`build_phase1_report.py`（build 時換 `{{key}}`、未解析即失敗）、`tools/report/build_findings.py`（findings gate 表 marker 區塊生成）、`tests/test_report_metrics.py`。報告章節 01–07 共 71 個 placeholder。**fact-check subagent 對帳已不需要**（數字無法手打錯）。
- **C** README/OVERALL（1.2/1.3 stale 行）/CONTEXT 更新；`plans/phase-1.3.md` 併入 3 子計畫；PDF+SVG 圖 untrack + gitignore（**PNG 保留**）；刪 orphan `build_phase1_2_report.py`。

## 3. 維護新機制時要記得

- 改報告數字 → 改 `validation/reports/*.json`（事實源），不要動章節裡的 `{{key}}`。新數字 → 在 `_metrics.py` 加 key（附 JSON path + formatter），章節用 `{{key}}`。
- findings gate 表是**生成的**：改它要改 `build_findings.py` 的 template，再跑一次；別手改 `<!-- gen:gate_summary -->` 區塊。
- 圖只 commit PNG；pdf/svg 由 `tools/plotting/*.py` 重生、已 gitignore。

## 4. Phase 2 前的科學工作（Card-only；我們已定案的方向）

Aetina/Alpha 板已永久離線，**只剩 Metis Card**（同顆 800MHz quad-core AIPU，PR #25 已證 `axcompile`+`axrunmodel dev_fps` 可隔離計算）。因此以下**不是永久外推，是待補量測**：

- **CIM compute 補量 sprint**（lift prefill-GEMM / multi-tile K·N>4.19M / compute-bound 到量測級）：用 Card `axrunmodel` 的 dev/system split 隔離純計算，繞過 on-card DRAM 牆；同顆 AIPU、無需 clock 正規化。
- **kv_cache 係數 SPIKE**：試在 Card 構造只做 kv-append 記憶體流量的 proxy 隔離量測；成功則校準係數，失敗則維持 analytic（被 M2 anchor 夾住、非野外推）。
- **conversion-op**：**不需量測**（已定案）。各單元精度固定（CIM INT8 / GPU FP16），邊界 cast 是 memory-bound、由既有 M2/M4 模型 × M6 邊界穿越次數解析計價（ADR-0004 已修訂）。
- **記憶體拓樸分離**：Card on-card DRAM 單獨特性化成 M2 anchor，別污染 M1 compute。

## 5. Phase 2 入口（整合）

- 順序（已定）：**先建 Card 模擬**（可對活 silicon 直接驗證）→ 再用同一 `engine+spec` 換 memory spec 建 **board/host-MMIO 模擬**（前瞻研究目標，靠橋接假設 + 共用 CIM kernel）。`specs/cim_topo_card.json` / `cim_topo_alpha.json` / `cim_topo_edge.json` 已就位。
- 實作 **M3 事件引擎**（頻寬競爭 ~60 GB/s knee）+ **M6 排程器**（op→unit + 精度邊界插 cast op，用既有模型計價）。Ramulator2/ONNXim 的 `engine=` drop-in 在 Phase 2 才發揮多單元競爭價值。
- e2e 驗證：decode 已有 8B recompose hold-out（9.5%）；**prefill 端到端驗證**靠上面的 Card 補量 sprint。
