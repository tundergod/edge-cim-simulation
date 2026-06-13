# Plan: Phase 1.7 — 熱特性化 + 效能-溫度耦合（Metis Card, M8）

> Sub-wave 1.7 ↔ module M8（contract `m8.yaml`、頁 `12-thermal`）。Card-only；Aetina 送修 → 雙平台熱量測延後。

## Context

Track C 獨立 sub-wave。已驗收：量產 Metis Card **能讀核溫**——sysctrl collector 在有 active context 時每秒吐 `core_temps=[5 值]`（板+4 core），`axlogdevice --slog` 抓得到（`axllm --show-temp` 的 tracer 在 SDK 1.6.0 不支援）。實測 1b 輕載 36→38°C。Throttle 門檻：PVT 警告 95 / HW 105 / **freq-downscale 110** / SW 200°C。**無功耗 telemetry**。

**兩個目的：**
1. **熱特性化**：合成負載「難度×時長 → 溫度 a→b」，擬 lumped RC（相對、非線性升溫）。
2. **效能-溫度耦合（新增核心）**：固定負載的**吞吐是否隨溫度改變**（throttle/降頻偵測）→ 判斷 M1–M7 calibrated 延遲熱起來後是否仍成立。**範圍誠實**：throttle 門檻 ≥95°C，而安全上限 60°C → **結構上量不到 throttle**；本目的實際交付的是「正常操作範圍（36–60°C）內吞吐 vs 溫度的漂移、對比噪音底」+「110°C 門檻未達」，不是「觀測到/排除 throttle 極限」。要真的觀測 throttle 需逼近 110°C（共用板過險，不做）。

## 共用板現實（review B2 — 已驗證：執行時實質單人）

**執行前已驗證（2026-06-12）**：`axdevice` 顯示 metis-0:7:0 閒置可用、無他人 context；無任何他人 axrunmodel/axllm/inference 程序；最高 CPU 是本 agent 自己的 session。`uptime` 的「8 users」是閒置登入（chrome、印表機 daemon）。→ **實質單人、device 未爭用**，B2 汙染風險低。下列緩解仍作為**保險**（萬一有人中途加入）保留；但不需要等獨佔時段。

原始風險（Host `wei-tmp-ubuntu` 名義多使用者、同一顆 die）兩個後果:
- **科學汙染**：別人的 inference 同樣加熱這 4 個 core、爭 PCIe/host BW → 我的 T(t) 混入他人散熱（RC 的「T 升因我 P」前提被破壞、熱態不可重現）；吞吐受 host 競爭影響、汙染 perf-vs-temp 斜率。
- **干擾他人**：4-core 持續大 GEMM 數分鐘會壟斷全部 AIPU core、餓死其他人。

**緩解（寫進執行）**：每個 burst 前後記錄 host load + 併發 device context（`axdevice` / slog 掃他人 context）→ **汙染的 burst 丟棄/標記**；只用 **device FPS**（最不受 host 影響）；**短 burst（≤30 s）+ 預設 1-core**、大負載階才短暫 4-core；頁面誠實標「共用 die、熱態不可重現、T(t) 含他人貢獻」。**建議（給你）**：若能跟機器 owner 喬一個**低使用/獨佔時段**，資料品質會好非常多——否則只能交「caveated/flagged」資料。

## 安全機制（review B1 — Blocker，必做）

- **自動 kill-switch**：背景 monitor tail slog 的 `core_temps`，一旦 `max(core_temps) > HARD_CAP`（預設 **60°C**，硬上限 65）**立刻 kill 正在跑的 axrunmodel PID**。每個擷取步驟都在此 guard 下跑。**fail-safe**：若溫度串流中斷/解析失敗/超過 2s 無新讀數 → **預設 kill**（無新讀數 = 不安全，不假設 cool）。
- **短 burst**：`--seconds ≤30`，迴圈間重新檢查溫度，不做單一長時間無人值守 run。
- **pre-flight**（每 session）：`axdevice` 確認裝置在、無他人獨佔；記錄回復路徑（AER fault → `axdevice --refresh`；PCIe rescan 需 sudo，共用板可能要 host admin）。
- **時間/熱預算先算**（review S7）：執行前列出總 core-seconds vs 30 分鐘牆 + 60°C 上限；不夠就砍 N/階數。

## 合成負載機制（非 LLM）

`axrunmodel <gemm.json> --seconds N --aipu-cores K` = 純 matmul 壓力跑 N 秒、回 device 吞吐。難度 = GEMM 大小 × 核數；時長 = `--seconds`。溫度由背景 slog `core_temps` @1Hz 同步抓。

## 步驟（action-only）

1. off main 開 `phase-1.7-thermal`；plan 複製進 repo。→ verify: branch + plan 在
2. `validation/contracts/m8.yaml`（標頭註明 Phase 1.7↔M8）：協定（axrunmodel 合成 GEMM；slog @1Hz；device-FPS；kill-switch 60°C；contention 丟棄規則；RC 相對特性化；perf-vs-temp 固定負載；**誠實限制**：無功耗 telemetry、降溫靠 keepalive、perf 僅 <60°C、共用 die 不可重現）。→ verify: m8.yaml 欄位齊
3. 一次性編 1–2 壓力 GEMM（axcompile，沿用 `characterization/run_metis_cim_v16.py`；中+大 K·N·M）；記錄 build 落點 + **清理 root-owned `quantized/`**（review N2；root-owned → 可能需 sudo/host-admin，清不掉就記錄磁碟佔用、交 owner）。→ verify: gemm.json 產出、5s smoke 跑得動、磁碟有清或已記錄
4. **noise-floor 基線（review S1，先做）**：固定溫度下（或回到 idle 基線後）重複同一 burst ~M 次 → 量 **device-FPS CoV**（repo 無 variance_profile，必須現量）。→ verify: 得 dev-FPS 噪音底（CoV、±band）
5. **擷取 campaign**（kill-switch guard 全程；每 burst 記 host load + 併發 context，汙染者丟棄）:
   a. **熱階梯**：L1 1-core 中、L2 4-core 中、L3 4-core 大；**短 burst 串接**逼近穩態（若 60°C 上限先到 → 標 **T∞ 為外推、非 fitted**）。每階間**回到 idle 基線才換下一階**（review S6；或隨機階序）；記 idle 基線（前/中/後）控漂移。
   b. **效能-溫度**：選最高安全負載階，固定 GEMM+cores，連續等長 burst，每 burst 記 (t, **device FPS**, **max core_temp**, host load) → 吞吐 vs max-core-溫度。
   c. **降溫 τ**：先 smoke-test「能讓 collector 持續吐溫度的最輕 keepalive」並量其穩態溫；重載↔keepalive 交替，**冷卻擬合朝 T∞(keepalive) 非 T_amb，τ_cool 報為 bound**（review S3）。
   → `measurements/metis_card/thermal_steps_*.json` + `thermal_perf_*.json`（含 host-load/contention 旗標）。→ verify: JSON 落地、峰值 <60°C、汙染 burst 已標
6. **分析**（review S2）：per-sensor RC（T_amb / τ；**τ 需 ≫1 s 否則 1Hz 取樣不足、標未解析**；T∞ 達穩態才標 fitted、否則外推）；**不出絕對 R_th**——若給則標 `assumption+warn+外推` 且註明「借 M7 規格估、為 compute-only 非全板功耗」。perf-vs-temp：斜率 + CI **對比 step4 噪音底** → 結論寫「斜率與噪音不可分（|slope|<X）」或「觀測到下降」，**不寫「板子不 throttle」**（110°C 未達 → 標「未達門檻」）。→ verify: 擬合收斂、slope vs noise 判定明確
7. **產committed artifact（review S5）**：`validation/reports/phase1.7/thermal.json`（RC fit quality、per-sensor T∞/τ/T_amb、perf slope+CI、noise floor、throttle-not-reached 旗標）。`simulator/models/params/m8_thermal.json`（RC 參數給 Phase 2）。→ verify: JSON 結構齊
8. `tools/plotting/site_m8.py`（沿用 site_m1 樣式、glyph-clean、讀 committed JSON）：(a) T(t) 升溫+RC 擬合+throttle 門檻線；(b) **device-FPS vs max-core-溫度**（回歸線+噪音帶+「36–60°C 未觀測到 throttle」標註，**非**「不 throttle」）。→ verify: 圖讀 JSON、ASCII-only
9. `docs/report/phase1-site/src/12-thermal.src.html`（5 段）：數字全 `{{key}}` 從 `phase1.7/thermal.json` 注入（加 `_metrics.py` keys via `_load`；`tests/test_report_metrics.py` 保持綠）；honesty chips（calibrated=on 溫度量測、fitted=on RC（達穩態的階）、assumption=on/warn 絕對R_th、其餘按實）；figs/chips 條目、build PAGES `12-thermal`→done。→ verify: build --strict 乾淨、零 {{}} 洩漏、test 綠
10. **doc-sync（review S4）**：清 `OVERALL.md` line 214 的 ⚠（Card 溫度可讀已證實 YES）、記錄 thermal 以 Card-only sub-wave 1.7 跑（Aetina 延後）、更新 CLAUDE.md phase 清單（0.4→併入 1.7 Card 部分）。→ verify: 三處不再 stale
11. 派 audit subagent 驗 12-thermal（數字↔JSON、honesty↔chips、perf-vs-temp 結論↔資料、無 overclaim）→ 修到乾淨 → 你簽核。

## Outputs

`validation/contracts/m8.yaml`、`validation/reports/phase1.7/thermal.json`、`measurements/metis_card/thermal_{steps,perf}_*.json`、`simulator/models/params/m8_thermal.json`、`tools/plotting/site_m8.py` + `docs/figures/phase1-site/m8_*.png`、`docs/report/phase1-site/src/12-thermal.src.html`、doc-sync（OVERALL.md/CLAUDE.md）。M8 = lumped RC `dT/dt=(P·R_th−(T−T_amb))/τ` 給 Phase 2（吃活動/功耗 timeline）。

## 誠實限制（明標於頁）

無功耗 telemetry → 不出絕對 R_th（或標 assumption+外推、M7-compute-only）；降溫 collector idle 即停 → keepalive 偏置、τ_cool 為 bound；perf-vs-temp 僅安全 <60°C、110°C 門檻未達（標「未達」非「不 throttle」）；共用 die → 熱態不可重現、T(t) 含他人貢獻、汙染 burst 已丟棄；RC = 3 階負載相對特性化、非「calibrated 多負載模型」；1Hz 取樣 → τ≫1s 才解析得了。
