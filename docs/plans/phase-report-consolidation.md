# Plan: Phase report-consolidation — 整合 Phase 0+1 為單一手刻 report 站

## Context

現有兩份 report 高度重疊、圖難懂、內容重複:
- `docs/report/phase0/index.html` — 靜態 HTML(Phase 0.1–0.3 量測描述)
- `docs/report/phase1/` — markdown 章節 + `_metrics.py`(數字唯一來源,從 JSON 注入)→ 單一 `index.html`

目標:把 Phase 0 併入 Phase 1,做成**一份手刻多頁網站**,每個 unit 一頁,呈現該 unit 的 measurement／擬合／模擬資料,**全部用 nature-figure 的圖表示**;每頁分開做、audit subagent 逐項驗、你簽核後才做下一頁。同時補兩個現在做得到的缺口資料(ScaleSim、Metis Card 溫度)。被 Aetina 送修卡死的缺口(NPU 矽 #13、GPU INT8、記憶體多單元競爭矽驗證)誠實標為缺口。

## 已敲定的設計決策(grill-me 討論結果)

1. **產生方式:純手刻 frontend HTML**(用 `/frontend-design`),不走 markdown→HTML build 管線。
2. **數字正確性(已依 review 強化):敘事本文不出裸數字**,只敘事。但 gate/門檻/就緒矩陣等**結構化數字用「從 JSON 注入的生成式 HTML 表」**呈現(保留 `_metrics.py` 式注入 + 缺值 build 失敗的機器保證),量測/擬合趨勢用 nature-figure 的圖(圖內文字由 JSON 當場產生)。audit 是第二道,不是唯一一道。詳見下方「數字保證機制」。
3. **頁面架構:多頁網站**,每 unit 一個獨立 `.html` + 共用導航/CSS。改 A 頁不動 B 頁,audit 範圍乾淨。
4. **不全的 unit 全開獨立頁**,缺資料區塊用 honesty chip 灰標。M4-NPU 缺矽→用 ScaleSim+ONNXim 兩 sim 對照。M3/M6 等 Phase 2 再做,本次只放「Phase 2 預覽」說明頁。
5. **同時收兩個缺口資料**:ScaleSim NPU 引擎(純 Python,無板)+ Metis Card 溫度(板還活著)。
6. **視覺:學術論文為本**(serif、克制配色、留白),「不通俗」靠排版/honesty-chip/導航/圖表裝框,不靠裝飾漸層。
7. **語言:繁中為主**,術語/圖軸保留英文。
8. **audit:每頁一個唯讀 audit subagent**(Agent 工具,opus/high),逐項對 JSON;跨頁迴圈與你的簽核留主對話;**不開 workflow**。
9. **檔案位置:新目錄 `docs/report/phase1-site/` 並存**;舊 phase0/phase1 暫留,新站全做完+你驗收後再用 PR 退場(redirect/刪除)。
10. **流程:當新 sub-wave,off main 開分支**(如 `phase-report-consolidation`),plan 寫 `docs/plans/`,走標準 plan→subagent 審→你核准→執行→code review→PR。
11. **起手:先做 M1 頁鎖模板**,溫度與 ScaleSim 稍後。

## 數字保證機制(取代 _metrics.py fail-loud,B1/S1/S2/B2)

純手刻 HTML 不能丟掉「數字對不上就失敗」的機器保證。三層:
1. **生成式表注入**:gate/門檻/就緒矩陣/擬合參數表 = 用一個 `_metrics.py` 式注入器把 `{{key}}` 解析成值後寫進 HTML 表(可沿用現有 `tools/report/_metrics.py` 的 key→JSON-path 對映)。**任何 `{{key}}` 解不到 → build 失敗**。敘事本文仍零裸數字。
2. **保留並擴充測試**:把 `tests/test_report_metrics.py` **repoint 到新站**(它現在 glob `docs/report/phase1/chapters/*.md`,不可隨章節刪掉,要遷移)。新增 figure-staleness 守門:CI 重跑所有 plotting 腳本後 `git diff --exit-code`,JSON 改了沒重畫圖就失敗。注意:`_style.save()` 只對 **PDF/SVG** 抑制時間戳,**PNG 不抑制**;PNG 的位元級可重現靠 matplotlib 本身的決定性,**CI 要 pin matplotlib/freetype/libpng 版本**才穩。
3. **honesty chip 用明確對映表(B2,逐頁指明來源路徑)**:JSON 的 honesty 欄不統一——只有 3/17 報告有 `status` 列舉(`CARD_REVALIDATED`/`PROXY_INCONCLUSIVE`/nested `BOUNDED-EXTRAPOLATED`),其餘是 freeform `honesty` 字串或 per-key dict,**而鎖模板的 m1.json 兩者皆無**(它的 extrapolation 旗標在 `m1.yaml` contract 裡)。所以 commit 的對映表要**逐頁指明每個 chip 由哪個來源、哪條路徑驅動**(報告 `status` | contract `status`/`honesty` | 新增 committed chip 欄);沒有機器可讀 honesty 的單元(尤其 M1)要新增一個 committed chip 欄。audit 比對「這張表」,不憑感覺。`m1.yaml` 標 extrapolation 的(multi-tile、prefill M>256)不可亮成 calibrated。
4. **pending 區塊不可用 `{{key}}`(B1×B3)**:ScaleSim/thermal 還沒落地前,NPU/溫度頁的待補段用**靜態「pending」標記**,**不可**放 `{{scalesim.*}}`/`{{thermal.*}}` 佔位——那些 key 在 `_metrics` 還不存在,會 hard-fail build、反而把「不阻塞」破壞掉。

## 頁面清單(12 頁;溫度頁 `12-thermal.html` 資料落地後後補,N1)

| 頁 | 檔 | 內容 | 對應舊章 |
|---|---|---|---|
| 導論 | `00-overview.html` | 模擬器是什麼、橋接假設、honesty-tag 圖例 | 00-intro |
| 就緒矩陣 | `01-readiness.html` | 各 unit × {measured/fit/sim} 狀態總表 | 01-readiness |
| M1 CIM | `02-cim.html` | **鎖模板頁** | 02-cim |
| M2 Memory | `03-memory.html` | PCIe floor + LPDDR4x wall(量測)、擬合、Ramulator2(sim) | 03-memory |
| M4 CPU | `04-cpu.html` | A76 instruction-count roofline | 04-cpu |
| M4 GPU | `05-gpu.html` | Mali FP16 量測(INT8 缺口) | 05-gpu |
| M4 NPU | `06-npu.html` | analytic 基線 / ScaleSim / ONNXim 三方並陳(spread,無 ground truth,無矽 #13) | 06-npu |
| M5 Workload | `07-workload.html` | op inventory/profile/roofline(併入 Phase 0.2) | 07 + phase0.2 |
| M7 Energy+E2E | `08-energy-e2e.html` | spec-based 能量 + recompose hold-out | 07 拆分 |
| Phase 2 預覽 | `09-phase2-preview.html` | M3 事件引擎/M6 排程器(未實作,純說明) | 08-integration |
| 缺口/GO-NOGO | `10-gaps.html` | 缺口三分類 + 條件式 GO/NO-GO | 09-gaps |
| 來源 | `11-sources.html` | 編號來源/誠實帳本 | 99-sources |

> M8 溫度頁(`12-thermal.html`)在溫度資料落地後加入(見 Track C)。

## 每頁的 unit 模板(5 段)

```
[unit header: M-code · 名稱 · honesty chips: calibrated|fitted|simulated|assumption|borrowed]
  (N2:用 repo 既有詞彙 calibrated,不用 measured;與 m4_cpu.yaml honesty: calibrated 一致)
§0 本章角色(一段話)
§1 Measurement  — 資料來源/協定/原始資料指標   └ 圖:量測(roofline/staircase/scatter)
§2 Fit          — 方程式、擬合參數表、fit gate   └ 圖:measured-vs-fit(誤差 CDF/疊圖)
§3 Simulation   — heavy-engine 對照或外推         └ 圖:sim-vs-fit
§4 Validation gate — ADR-0006 表(metric|值|門檻|PASS/FAIL,誠實顯示 max)
§5 Phase-2 readiness — 凍結介面✅/外推⚠/缺口
```
honesty chip 由「status→chip 對映表」驅動(見「數字保證機制」),沒矽就不能亮 measured。

> **M1 頁不只鎖視覺,更要鎖「可稽核的表注入機制」**(N4):生成式 gate/參數表怎麼從 JSON 注入、honesty chip 怎麼對映、figure-staleness 怎麼守門——這些在 M1 頁定案,後續頁不再重議數字來源,只填同一個模子。

## 每頁 build+audit 迴圈(主對話驅動)

```
對每一頁 N(M1 起):
  1. 我先出整頁「第一版草稿」:手刻該頁 HTML(frontend-design,學術論文風,
     本文不出裸數字)+ 用 nature-figure 把每一筆數據(measurement/fit/sim 逐筆)
     照我提的呈現方式畫出來。草稿就是討論的載體。
  2. 你直接在具體草稿上改:逐筆數據要換圖型/軸/對照/honesty 標籤都在這裡定。
  3. 我依你的修改重出 → 重複到你滿意這頁每筆數據的呈現。
  4. 派 audit subagent(Agent, 唯讀, opus/high)checklist 驗:
       - 每張圖數字 vs source JSON
       - honesty 標籤 vs JSON status(無矽不可標 measured)
       - gate 表 vs ADR-0006
       - 本文是否誤出裸數字(洩漏)
  5. 依 audit 修 → 重審,直到乾淨
  6. 你最終簽核 → 才做下一頁
```
> 節奏:**draft-first**。我先出整頁草稿(含每筆數據我提的呈現),你在具體草稿上逐筆修,比動工前抽象討論快。每筆數據仍由你拍板,只是改在草稿上。

## 圖表計畫(幾乎全部沿用現有腳本)

每張圖讀 committed JSON,沿用 `tools/plotting/_style.py`(Okabe-Ito、600dpi)。代表性:
- M1:`phase1_figs.py`(staircase/Geff/fit-CDF)+ `phase1_5_cim.py`(多 tile cliff)← 讀 `validation/reports/phase1.{1,5}/`、`params/m1_cim.json`、`measurements/metis_card/`
- M2:`mem_*.py` + `mem_ramulator2_fig.py` ← `phase1.{1,2,3}/m2*.json`
- M4 CPU/GPU/NPU:`cpu_c1.py`/`gpu_g1.py`/`npu_n*.py` ← `phase1.{1,2,3}/m4_*.json`
- M5:`op_breakdown.py`/`roofline.py`(Phase 0.2 圖移入此頁)← `measurements/op_profile/`
- M7/E2E:`phase1_figs.py`(energy/recompose)← `phase1.1/m7.json`、`recompose.json`
- 新圖腳本歸位(S5):新數據(ScaleSim 第三 sim、thermal trace)→ 各一支 `tools/plotting/<name>.py`,import `_style`、用 `_style.save()`,輸出 `docs/figures/phase1-site/`,PNG commit、PDF/SVG gitignore(沿用 CONTEXT 規範)。
- 就緒矩陣 = **生成式 HTML 表**(從 JSON 注入,見「數字保證機制」),不漂移、可 grep/diff。
- NPU 圖框架(B4):三個都是 sim、無矽裁判,且 ONNXim 與 analytic 差 318% median/493% max。**畫成「不確定性帶/spread,無 ground truth」**,不可暗示「三方一致=正確」。audit checklist 明文禁止 page 06 出現 validated/agree 字眼。NPU 第三 sim 待 ScaleSim 落地;未落地前該頁標「third-sim pending」,**不阻塞**收尾。

## 三條工作線

- **Track A(報告頁)**:M1→00/01→M2→CPU→GPU→(NPU 待 ScaleSim)→M5→M7/E2E→Phase2 預覽→缺口→來源。每頁 build+audit+簽核。
- **Track B(ScaleSim)— 獨立 gated 子任務,不阻塞報告(B3)**:這是元件建模、非文件,工程量開放。**切成自己的 sub-wave(`phase-1.6-scalesim`),走自己的 plan→subagent 審→你核准→才執行**(不搭本 plan 的核准)。要有自己的驗收 contract(`validation/contracts/` 的 m4_npu scalesim 條目)+ validation report;落 `measurements/simulated/scalesim/`。NPU 頁(06)**有了就用、沒有就標 third-sim pending 照常收尾**,不被它 gate。
- **Track C(溫度)— 獨立 sub-wave,同樣走自己的核准閘**(也補 `validation/contracts/m8.yaml` 定義協定/驗收)。**排序已定:現在就搶一次最小 `axlogdevice --slog` 原始擷取**(idle→跨跑 LLM→降溫,5 sensor)落 `measurements/metis_card/thermal_*.json`,趁 Metis Card 還活著(偏離 CLAUDE.md thermal-last 是為避免變永久缺口;不影響 M1 鎖模板)。`12-thermal.html` 等資料落地後再做。

## 誠實標為缺口(本次收不到)

- M4-NPU RKNPU2 矽(#13)— Aetina 送修;NPU 頁用 ScaleSim+ONNXim sim 替代,明標無矽。
- M4-GPU Mali INT8 — 只有 FP16;GPU 頁明標 INT8 缺口。
- M2 多單元記憶體競爭矽驗證 — 只能 Ramulator2;M2 頁明標無矽、±20% sensitivity。

## 關鍵檔案

- 新建:`docs/report/phase1-site/*.html` + 共用 `assets/`(CSS/導航)
- 沿用/微調:`tools/plotting/*.py`、`tools/plotting/_style.py`
- 讀取(來源):`measurements/`、`simulator/models/params/*.json`、`validation/reports/phase1.{1,2,3,5}/*.json`、`validation/contracts/m*.yaml`
- 新增資料:`measurements/simulated/scalesim/`(Track B)、`measurements/metis_card/thermal_*.json`(Track C)
- 退場 doc-sync(S4,須列舉每個 referrer):`_metrics.py` 與 `tests/test_report_metrics.py` 是**注入器,要 repoint 到新站、不隨章節退場**(test 現 glob `chapters/*.md` + 讀 `docs/phase1.1-findings.md`,刪章節前先遷移)。退舊 `phase0/`+`phase1/` 還牽動 `build_phase1_report.py`、`LOG.md`/`README.md`/`docs/voyager-sdk.md`/`docs/phase1.3-findings.md`/`docs/handoff-*.md`/`docs/plans/*` 內引用。**退場步驟先 grep 全部 referrer,逐檔決定 redirect/改寫/遷移,並確認 `pytest tests/` 仍綠**。CONTEXT.md repo index、OVERALL.md(Phase 0 併入 Phase 1)同步更新。

## 驗證(success criteria)

- **機器保證(第一道)**:`{{key}}` 全解得到、否則 build 失敗;`pytest tests/test_report_metrics.py` 綠;figure-staleness `git diff --exit-code` 綠。
- **每頁(第二道)**:audit subagent 回報乾淨(圖數字↔JSON、honesty↔對映表、gate↔ADR-0006、本文無裸數字)+ 你簽核。
- 全站:敘事本文零裸數字;結構化數字在生成式表、趨勢在 nature-figure 圖,全部可由 JSON 重現。
- Track B(獨立 sub-wave):ScaleSim 有 contract+validation report;NPU 頁圖框成「不確定性帶、無 ground truth」,無 validated/agree 字眼。
- Track C:thermal_*.json 落地、可重跑;`m8.yaml` 協定就位;溫度頁圖成立(或明確列永久缺口)。
- 退場:新站全做完+你驗收後,grep 全 referrer 逐檔處理,PR 退舊 phase0/phase1,`pytest tests/` 仍綠,CONTEXT.md/OVERALL.md 同步。
- (N3)決定:是否仍需單一可投稿 PDF。多頁 `<img src>` 站不天然產 PDF;若要,補一個 print-CSS 串接或 headless-Chrome 合併步驟。

## 執行前提

- off main 開分支 `phase-report-consolidation`;本 plan 複製進 `docs/plans/`;依 CLAUDE.md 先 subagent 審 plan→你核准→才開工。
