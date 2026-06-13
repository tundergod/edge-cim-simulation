# Plan: Phase 1.6 — ScaleSim NPU 第三引擎（純本地，補 NPU 頁 pending）

> Sub-wave 1.6 ↔ M4-NPU 第三個 sim 引擎。純 Python、**不需板子**。補上 06-npu 頁現在標 `pending` 的 ScaleSim slot。**分支 off `phase-report-consolidation`**（NPU 頁住在那）、PR 回該分支（不是 main）。

## Context

M4-NPU **無矽（issue #13）**，三個都是 sim、無裁判：① analytic systolic-roofline、② ONNXim heavy-sim（Phase 1.3，median |delta| 317.9% vs analytic）、③ **ScaleSim — 之前標 pending、本 sub-wave 建**。ScaleSim 3.0.0 純 Python、pip 裝、本地跑、無板依賴。

**誠實紀律（沿用已 audit-clean 的 NPU 頁框架，不可違反）**：三方全是 sim、**無 silicon ground truth**；ScaleSim 是**第三個不確定性點、不是裁判**。**禁止**任何「validated / agree / cross-validated / brackets / 中間值較可信 / 三取二一致=正確」措辭——第三個 sim 不會讓任何值更可信。NPU primary 仍延到 Phase 2 L4。

**設計哲學（使用者定，本 sub-wave 核心）：模擬器要忠於架構的「原生行為」，不替使用者特地避開。**
- ScaleSim 配成通用 32×32 weight-stationary 陣列（@6 TOPS）後，脈動陣列的原生特性——**對齊-32 padding（staircase knee）、order-sensitivity、shape-sensitivity（大 activation column 退化）**——會**自己長出來**；我們**如實重現、攤開呈現**，**不做**任何「轉置進好 regime」的優化（那是 HeteroInfer 在做的事，不是我們）。
- **這是 Phase 1 的 first-class 發現 + 給未來使用者的設計指引**：Phase 2 / 未來排程器若**利用**此特性（好的 operand 順序/對齊）→ 好 performance；若**忽略**→ 付出代價。模擬器的價值就是把這個代價如實算出來。
- **不以 HeteroInfer 為起點/錨點**：這些行為是「脈動 + weight-stationary」架構的**固有後果（物理）**，不是從 HeteroInfer 匯入。HeteroInfer 只是**獨立佐證**（一顆真實同級 NPU〔Hexagon〕量到同樣原生行為），且**只方向性佐證（同號效應），不比 magnitude**（不同陣列/config，6× 不可直接比）。**不對它校準、不從它配「行為」**。
- **兩個誠實標籤要分清（review B1）**：(a)「**native, not tuned**」= 行為是模型自己長的、沒手動 curate；(b)「**model, NOT silicon**」= 這些 sensitivity 的**數值大小是「一個 32×32-WS 脈動*模型*會產生的」，不是量到的 RKNPU2 事實**。每個 spread 數字 + 設計 note 都要同時帶這兩個標籤。32×32 維度本身**借自 HeteroInfer**（無 RKNPU2 dims）→ 故 HeteroInfer 的佐證「在量測上獨立、在陣列維度上不獨立」,要寫明、不可號稱完全 independent。熱 load-sweep 的呼應是**模型/我方實測內部**,非外部 silicon 驗證。

## 關鍵整合事實（review 確認，務必照做）

- **3-core 聚合（B2，最重要）**：analytic 的 `tops_int8=6` 與 onnxim（config「32×32 ×3」）都是 **3-core 聚合**；單一 ScaleSim 32×32 @1GHz ≈ 2.05 TOPS = **1/3**。若直接 cycles/freq，ScaleSim 會純因核數差 ~3×、被誤呈現成「不確定性 spread」→ **汙染誠實故事**。**必做**：鏡像 onnxim 的 ×3 聚合（layer cycles ÷ 3，等效 3 陣列並行），並在落表前用一個已知形狀 sanity-check：ScaleSim 等效吞吐落在 ~6 TOPS regime（不是 2），才算數。**÷3 本身是一個 load-bearing 假設**（假設理想線性 3-core scaling、無跨核 overhead）——三方對稱套用所以是對的正規化，但要**明標為 assumption**（不是中性的「等效並行」），寫進 provenance + 頁面。
- **ScaleSim 額外輸入是新 assumption surface（S1/S2）**：ScaleSim config 還要 `Ifmap/Filter/OfmapSramSzkB`、`Dataflow`、`Bandwidth(words/cycle)`，spec 都沒有。→ SRAM 三個 buffer 大小（**值的出處要寫清楚**：RKNPU2 datasheet 若有、否則明標「arbitrary/assumed」——它會實質改變 memory-bound layer 的 cycles）+ dataflow=WS（RKNPU2 近似 weight-stationary）+ BW 換算（`bw_GBs.eff ÷ freq ÷ bytes/word`，或用 CALC mode）全部**明列為 assumption-tagged 輸入**，反映進 provenance、chips 的 assumption src、§2/§3 prose。
- **輸出是 CSV、不是 API（S3）**：ScaleSim 寫 `COMPUTE_REPORT.csv`（per-layer `Total Cycles`）到 output dir → runner 解析該欄。
- **欄序（S4）**：ScaleSim GEMM topology CSV 是 `M,N,K`；我們的形狀是 `(M,K,N)` → 明確對映、並對一個已知形狀做 **MAC-count sanity check**（避免 transpose 造假 delta）。
- **形狀集（N1）**：用 onnxim 表的**同一組形狀**（onnxim 已 drop N≤64）；ScaleSim 跑不動的對稱 drop 並註記。**GEMM-only**：attn_bmm 三方都維持 analytic（N4，不要期待 ScaleSim attention 數）。

## 落地點（已確認）

- NPU engine：`NpuModel(spec, engine=)` + `predict(wl)`；onnxim 走 `_onnxim_table()`（讀 `simulated/onnxim/rknpu2_sim_matmul.json` per-shape 表，**先確認其實際 key/value 格式**再鏡像）→ scalesim 加 `_scalesim_table()` + `engine='scalesim'`（同 fallback + frozen predict 契約）。
- config 來源：`npu_rknpu2.json` › systolic_dim[32,32]/cores=3/freq_ghz=1.0 + 上面的 assumption 輸入。

## 步驟（action-only）

1. off **phase-report-consolidation** 開分支 `phase-1.6-scalesim`；plan 進 repo。→ verify: branch + plan
2. `pip install scalesim==3.0.0` 進 `.venv`，**pin 進 requirements**。→ verify: `python -c "import scalesim"` OK
3. 先讀 `simulated/onnxim/rknpu2_sim_matmul.json` 確認 key/value 格式（JSON 不能用 tuple key → 看它怎麼編形狀 + value 是 latency_us 還 cycles）。→ verify: 格式記錄下來
4. 寫 `tools/scalesim/run_rknpu2_scalesim.py`：onnxim 同一組形狀 → ScaleSim topology CSV（M,N,K 欄序）+ config（32×32、WS、assumed SRAM、換算 BW）；跑 → 解析 `COMPUTE_REPORT.csv › Total Cycles` → **÷3 核聚合** → latency = cycles/freq；落 `simulated/scalesim/rknpu2_sim_matmul.json`（鏡像 onnxim 表格式）。先對 1 個已知形狀做 MAC + TOPS-regime sanity。→ verify: JSON 落地、形狀對齊、sanity 通過（~6 TOPS regime、非 2）
5. 接線 `m4_npu.py`：加 `_scalesim_table()` + `engine='scalesim'`（mirror onnxim 分支；空表→analytic fallback；provenance 標 `simulated: scalesim 32x32×3 WS, assumed SRAM/BW, NOT silicon`）。→ verify: `NpuModel(spec, engine='scalesim').predict(wl)` 回值 + provenance 正確
6. `tools/analysis/fit_npu_scalesim.py` → `validation/reports/phase1.6/npu_scalesim.json`：三方（analytic/onnxim/scalesim）同形狀 latency + 兩兩 |delta|；**no_silicon_ground_truth=true、不下 pass/fail、不標任何值較可信**。→ verify: 三方齊、delta 算出
6b. **原生 sensitivity 攤開（設計哲學的 first-class 交付）**：用 ScaleSim 掃三個維度量**原生 best↔worst spread**：① 對齊 vs 非對齊 32、② operand 順序正/反、③ 大 vs 小 activation column(M)。
   - **與 §6 三方 delta 分開（review S-1）**：三方 delta 只用 **canonical orientation**；order-flip 是 6b 專用的 sensitivity 軸。order sweep **必須 MAC-count 不變**（翻轉後算的是同一個 GEMM，才叫 order 效應、不是換了問題）。
   - **報實情（review N-2）**：即使某維度**幾乎無 sensitivity 或與 HeteroInfer 反向**也照實報（不是去 demo HeteroInfer 那三軸）。
   - 落 `npu_scalesim.json › native_sensitivity`，每個 spread 同時帶 **「native, not tuned」+「model, NOT silicon」**兩標籤。→ verify: 三維 spread 算出、雙標籤在、order sweep MAC 不變

7. contract `m4_npu.yaml` 加 scalesim 第三 engine（simulated、無 gate）。→ verify: 欄位齊
8. 更新 06-npu 頁（**精確列出可變 vs 不變**）：
   - 變：§3 ScaleSim `.pending` → 真資料（三方 spread 框架不變）；§5 ScaleSim 那列 pending→built；§6 sources 加 scalesim；chips `simulated` src 改成「ONNXim + ScaleSim 兩個 heavy-sim」。
   - **新增「原生 sensitivity + 設計指引」小節 + 圖 `npu_sensitivity`**：攤開 ScaleSim 原生的 對齊/order/shape spread（best↔worst），明寫「Phase 2 / 未來使用者：利用此特性→快、忽略→付代價」的設計 note；HeteroInfer 一句獨立佐證。
   - **不變（須維持紅/⚠）**：「NPU 絕對延遲未驗證」、「spread 無法定責（兩邊皆無矽）」、「NPU primary L4 延到 Phase 2」、calibrated chip 仍 OFF。
   - `site_npu.py` 的 `npu_spread` 加 scalesim 點 + 新 `npu_sensitivity` 圖；`figs.json` 兩張都登錄 sources（`npu_spread` **加** `phase1.6/npu_scalesim.json` + `m4_npu.py`〔S7〕；`npu_sensitivity` → `phase1.6/npu_scalesim.json`）；在 **`tools/report/_metrics.py`** 加 `npu.scalesim_*` **與 `npu.sens_*`** keys（`_load("phase1.6/npu_scalesim.json")`，比照現有 `npu.onnxim_*`，**同樣 fail-loud**）。→ verify: build --strict 乾淨、零 {{}} 洩漏（含 sens 區）、figs fresh、`pytest tests/test_report_metrics.py` 綠
9. doc-sync：OVERALL.md / NPU 處把 ScaleSim「待建」→「已建（三方 spread）」。→ verify: 不再 stale
10. 派 audit subagent 驗 06-npu，checklist **明文禁止**：validated/agree/cross-validated/「brackets」/「central/likely estimate」/「2-of-3 → correct」；**且禁止把 `native_sensitivity` 數值或設計 note 講成 RKNPU2/measured/silicon 事實——必須讀作「一個 32×32-WS 脈動*模型*的原生預測」（review B1）**；HeteroInfer 只方向佐證、不比 magnitude。確認 3-core 聚合已做（無 ~3× 核數 artifact）、SRAM/BW assumption 已標、calibrated 仍 OFF、L4-deferred 仍在、sensitivity 雙標籤在。→ 修到乾淨 → 你簽核。

## Outputs

`tools/scalesim/run_rknpu2_scalesim.py`、`simulated/scalesim/rknpu2_sim_matmul.json`、`validation/reports/phase1.6/npu_scalesim.json`（含 `native_sensitivity`）、`tools/analysis/fit_npu_scalesim.py`、`m4_npu.py`(+scalesim)、`m4_npu.yaml`(+scalesim)、更新 06-npu 頁（+原生 sensitivity 小節/設計指引）+ `npu_spread` & `npu_sensitivity` 圖 + `npu.scalesim_*`/`npu.sens_*` keys、requirements pin。

## 風險 / 誠實限制

- **ScaleSim ≠ RKNPU2**：通用 systolic 配 RKNPU2-like（32×32 borrowed、SRAM/dataflow/BW 皆 assumed），仍非矽。三方 spread 只表「模型不確定性」、不縮小對真實 RKNPU2 的未知。
- 3-core 聚合 + assumed SRAM/dataflow 是**會影響 cycles 的人為選擇** → 全部明標。
- 三方很可能發散大（onnxim 已 ~4× analytic）→ 誠實呈現為「無矽下的模型發散」，正是 #13 的代價，不是缺陷。
