# Plan: Phase 1 Overall Report — 單一份、自給自足、誠實 review + GO/NO-GO gate

> 形式：`docs/report/phase1/chapters/*.md`（中文）→ build script → 自帶圖 HTML + **PDF**（最終交付物）。
> 流程（輕量版 B）：branch → plan → **subagent plan review loop（fix→review 直到 reviewer 無 issue）** → 使用者批准 → 執行 → fact-check subagent 對帳 → build PDF → 開**報告 PR**（舊報告仍在，便於對照）給使用者最終 review → merge → 另開 **cleanup PR** 刪舊報告 → merge。
> Model 分配：chapter 草稿 + fact-check = **Sonnet**；框架／缺口三分類／GO-NO-GO／最終縫合 = **Opus**。

> **更新（Phase 1.4, 2026-06-11）— 數字由 JSON 生成，取代 fact-check 對帳。** 報告章節的數字格改為 `{{key}}` placeholder，build 時由 `tools/report/_metrics.py`（讀 `validation/reports/*.json`）填入、未解析即 build 失敗；findings 的 gate 數字由 `tests/test_report_metrics.py` 守住不漂移。因此**上面流程裡的兩輪「fact-check subagent 對帳」（步驟 6–7、11）已不需要**——數字無法手打錯，review 只需看散文詮釋。

## 鎖定決策（grill 產出，執行時不得偏離）

- **核心結論兩段式**：(1) Phase 1 是否按其誠實標準交付＋驗證（逐 component 給證據）；(2) 帶條件的 GO/NO-GO，缺口分三類（進 Phase 2 前必須解 / 可在 Phase 2 內處理 / 可接受 limitation）。**不得宣稱「all components ready」**（M3/M6 未實作是設計使然＝Phase 2 工作內容，非 Phase 1 缺失）。
- **完整吸收**：吸收 1.1/1.2/1.3 全部關鍵內容；**舊報告的刪除走獨立的 cleanup PR**（報告 PR merge 後才開），不混進報告 PR 的 diff，讓兩個 PR 的 diff 都被完整 review；保留 `docs/phase1.{1,2,3}-findings.md` + `validation/` + `measurements/` 當資料層。
- **誠實紀律**：reference 方式（內文編號註 → 附錄來源表，不貼行內 path/code）；fact-check subagent 對帳；三段式誠實標籤（calibrated / simulated / assumption / borrowed）不混用。
- **artifacts 優先於 findings**：`validation/reports/*.json`、figures、chapters、caches 為事實來源；`findings.md` 散文可能 stale，遇衝突一律以 artifact 為準。（本次已先修正兩處 stale findings：`phase1.2-findings.md` CIM-card 段 `DEFERRED_FALLBACK`→`CARD_REVALIDATED`（median 4.8%/p95 9.7%）；`phase1.3-findings.md` 移除「Not produced/To finish」殘段。Sonnet 草稿仍須以 JSON 複核任何 findings 敘述。）
- **骨架**：按單元（CIM / 記憶體 / CPU / GPU / NPU）；五欄模板（①模擬什麼 ②模型從哪來 ③驗證狀態 ④缺口/外推區 ⑤Phase 2 就緒度）；非單元組件（M5 trace、M7 能耗）與跨單元 e2e 合在第 7 章；M3/M6 獨立成「尚未實作的整合層」第 8 章。
- **缺口三分類**：Opus 先擬 + 寫理由，使用者最後拍板。

## 章節清單（11 章，每個 Phase 1 deliverable 都有歸屬）

- `00-intro` [Opus] 導論 + 方法論
- `01-readiness-matrix` [Opus] 全局就緒度總表（**最後寫，由已對帳的各章彙整**）
- `02-cim` [Sonnet] M1：含 decode G_eff、prefill M-amortization fit、CIM-card 重驗（**CARD_REVALIDATED**：13 點 cross-val median 4.8%/p95 9.7%，非 deferred）、multi-tile 外推
- `03-memory` [Sonnet] M2：含 LPDDR4x/4x/5、PCIe floor、SRAM CACTI tier、Ramulator2 heavy-sim、KV-cache
- `04-cpu` [Sonnet] M4-CPU：instruction-count roofline、eta_bw 假設、A55/multicore 外推
- `05-gpu` [Sonnet] M4-GPU：Mali attn micro-benchmark、roofline slot、INT8 零資料
- `06-npu` [Sonnet] M4-NPU：systolic-roofline（全 simulated/borrowed、#13 superseded）、ONNXim heavy-sim
- `07-workload-energy-e2e` [Opus] M5 trace（0-orphan、4 模型）、M7 能耗（spec-based、±20% 穩健）、decode recompose hold-out（8B 9.5%）、**prefill 整條路徑現況（analytic/未驗證，不軟化）**
- `08-integration-layer-m3-m6` [Opus] M3/M6 僅 contract；M6 連帶 conversion-op 成本（ADR-0004）從未量測——明確區分「未實作（設計使然）」vs「真缺口」
- `09-gaps-gonogo` [Opus] 缺口三分類 + 帶條件 GO/NO-GO（**對帳後才寫**）
- `99-sources` [Opus] 編號來源表

## 步驟

1. 建立 `docs/report/phase1/chapters/` 上述 11 個 .md 空殼（標題 + 五欄模板註解）→ verify: `ls` 列出 11 檔。
2. **[Opus]** 寫 `00-intro`（Phase 1 做了什麼/怎麼做/為什麼 = goal #1；方法論：按資料來源拆 1.1/1.2/1.3、equation-fit 取代 lookup、ADR-0006 gate、三段式誠實標籤；如何讀本報告）→ verify: 涵蓋 goal #1 三問、無 component 細節。
3. **[Sonnet ×5 並行]** 寫 `02`–`06` 單元章，嚴格套五欄模板。輸入：`docs/phase1.{1,2,3}-findings.md`、`validation/reports/phase1.{1,2,3}/*.json`、`validation/contracts/*.yaml`、`simulator/models/*.py`、`simulator/models/params/*.json`、`simulator/specs/*.json`。每章嵌 1–3 既有圖，**圖連結一律寫 `![NAME](../../../figures/phase1.{1,2,3}/NAME.png)`**（與既有 chapters 同深度，供 build 正則匹配）。欄④缺口文字**須可溯源自 artifact（JSON/figures/chapters/caches）；findings 僅作 narrative input，遇衝突一律以 artifact 為準**（不得自行軟化）；每個數字標記來源供附錄編號。各章 must-cover 見上「章節清單」括號項 → verify: 五欄齊全、欄④與 artifact 一致、所有數字可追到某檔案欄位、圖連結為字面 `../../../figures/phase1.{1,2,3}/NAME.png` 前綴（`grep -L '\.\./\.\./\.\./figures/phase1\.' 各章` 應無漏網）。
4. **[Opus]** 寫 `07`（M5 trace、M7 能耗、decode recompose 9.5%、prefill 未驗證現況）→ verify: M5/M7 各有獨立小節；prefill 明講未驗證、不軟化。
5. **[Opus]** 寫 `08`（M3/M6 僅 contract；conversion-op 真缺口）→ verify: 明確區分「未實作（設計使然）」vs「真缺口（conversion-op）」。
6. **[Sonnet] fact-check 對帳 subagent**（對象：`02`–`08` 所有數字與標籤）。兩類查核：(a) **數值**——報告每個數字 = `validation/reports/phase1.{1,2,3}/*.json` / `contracts/*.yaml` / `simulator/models/params/*.json` 的實際值，對不上或找不到來源即標記；(b) **誠實標籤**——每個 `calibrated` 標籤必須有對應的 `validation/reports/*.json` 驗證條目支撐，否則不得標 calibrated/validated；`simulated`/`assumption`/`borrowed` 不得被寫成 calibrated。產出對帳清單 → verify: 清單含數值 + 標籤兩類，0 筆未處理。
7. **[Opus]** 依對帳清單修正 `02`–`08`（改數字、降級錯標的標籤、或明確標「推斷/估計」）→ verify: 對帳清單每筆已解決。
8. **[Opus]** 寫 `01` 全局就緒度總表（由**已對帳**的 `02`–`08` 彙整）→ verify: 表中每格與來源章一致、無矛盾。
9. **[Opus]** 寫 `09`（缺口三分類表 + 帶條件 GO/NO-GO；「必須先解」每項一句「為何擋路＋大致解法方向」；數字取自已對帳章節）→ verify: 每缺口恰歸一類且附理由；結論不宣稱無條件 ready；引用數字與來源章一致。
10. **[Opus]** 寫 `99` 編號來源表 → verify: 內文每個編號註在附錄有對應條目。
11. **[Sonnet] 第二輪 fact-check（涵蓋 `01`/`09`/`99` + 來源表）**：總表每格、GO/NO-GO 每個引用數字、缺口分類、來源表每條編號，逐一對 `02`–`08` 已對帳內容與底層 JSON 複核；特別查 GO/NO-GO 語氣是否過度（有無把帶條件結論寫成無條件 ready）、缺口分類有無錯置 → verify: 產出第二輪清單，0 筆未處理；**[Opus]** 依清單修正 `01`/`09`/`99`。
12. retarget 既有的 `tools/report/build_phase1_report.py`（**此檔目前是 Phase 1.1 builder，非新建**）指向 `docs/report/phase1/`；`ORDER` = 11 章順序；**`embed_figs` 改雙群組正則** `src="(?:\.\./)+figures/(phase1\.[123])/([^"]+)\.png"`，`repl` 解析 `ROOT/"docs/figures"/m.group(1)/(name+".png")`，可嵌三個 phase 圖源。（`build_phase1_2_report.py` 隨舊報告刪除而變孤兒，於步驟 15 一併處置）→ verify: 跑完無例外、`grep -c 'src="\.\./\.\./\.\./figures/' index.html` = 0（圖全 base64 內嵌，僅針對圖 src、不誤判其他相對路徑）。
13. build PDF → verify: `docs/report/phase1/phase1-report.pdf` 存在、可開、含全部 11 章與圖（**此即 retargeted builder 變更的端到端驗證**）。
14. commit（**只含新增：新報告 + retargeted builder；不刪任何舊檔**）+ 開**報告 PR** `report/phase1-overall` → `main`，PR body 含「關鍵 claim 覆蓋 checklist」（逐條列：舊 1.1/1.2/1.3 每個關鍵結論 → 已吸收進新報告哪一章），通知使用者最終 review。**舊三份報告此刻仍在，便於 review 對照**；使用者批准後 merge → verify: `gh pr view` 顯示純新增的 diff + checklist。
15. **報告 PR merge 後**，另開 branch + **cleanup PR** 刪除 `docs/report/phase1.{1,2,3}/` + 孤兒 builder `build_phase1_2_report.py`（**先 assert** 新報告 `phase1/` 11 章 + PDF 已在 `main`；保留 findings.md / validation / measurements；不動 `docs/report/phase0/`）。**刪除是此 PR 唯一的 diff，獨立被 review，使用者批准後才 merge** → verify: cleanup PR diff 只含刪除；使用者批准；merge 後 `docs/report/` 下 Phase 1 只剩 `phase1/`；舊內容可從 git history 復原。

Outputs:
- `docs/report/phase1/chapters/*.md`（11 章，中文）
- `docs/report/phase1/index.html` + `docs/report/phase1/phase1-report.pdf`（最終交付物）
- `tools/report/build_phase1_report.py`（retargeted 為總報告 build script）
- 修正後的 `docs/phase1.{1,2,3}-findings.md`（兩處 stale 已修）
- **報告 PR**：`report/phase1-overall` → `main`（純新增）
- **cleanup PR**（報告 PR merge 後）：刪除 `docs/report/phase1.{1,2,3}/` + 孤兒 `build_phase1_2_report.py`
