# 異質行動 SoC 上的 CIM-Centric LLM 推論 — 模擬器

> 真實晶片校準的「CIM-enabled 異質行動 SoC 上 LLM 推論」模擬器。研究對象是*模擬的* CIM-mobile SoC（CIM 掛 PCIe + MMIO unified host memory，容量為可調參數），以兩塊真實 Axelera Metis 板校準：**Metis Alpha（Aetina RKC-A02）** 提供 CIM/PCIe 計算原語，**量產 Metis Card** 提供 on-card-DRAM 下 INT8 LLM 行為。
>
> **設計紀律 — CIM-centric：** 系統先繞著 CIM 的限制走（擅長 weight-stationary GEMV、不擅長 dynamic attention、tile 對齊 channel×64、weight residency 上限、host-device 來回成本）；GPU/NPU/CPU 為支援層；op→unit 分工由特性量測決定，不預設。
>
> **Workload：** Llama-3 + Qwen-2.5，1B–8B（stretch 13B），INT8，batch=1，prefill+decode。
> **狀態：** Phase 0.1–0.3 + Phase 1.1–1.5 完成；Phase 0.4（熱，Metis Card）待做；**下一步 = Phase 2（整合）**。Aetina 送修中。
> **設計決策見 [docs/adr/](docs/adr/)**（0001 引擎 fidelity · 0002 記憶體 · 0003 scheduler · 0004 混合精度 · 0005 能耗 · 0006 驗證/橋接/外推 · 0007 op inventory）；**文獻見 [docs/papers/](docs/papers/)**（16 篇，README 列篩選理由）；**Metis 量測面見 [docs/voyager-sdk.md](docs/voyager-sdk.md)**。

## 平台假設（橋接）

- **記憶體容量是顯式參數，不是寫死值**：真實板在約 **1–16 GB** 驗證行為*趨勢*（Alpha 1 GB IOMMU window、Card on-card DRAM），確認後模擬器把容量參數**外推**（32 GB+）。BW 效率同樣做成可 sweep 參數。**現行 Metis 無 unified host-device memory — 此為前瞻假設，非當前能力。**
- **橋接假設（寫進 Method）**：CIM-tile 計算 timing 在更換記憶體基底時不變，只有資料搬移 timing 改變。Card 廠商 LLM 量測驗證 CIM + on-card-DRAM 拓樸；模擬器把 on-card DRAM 換成 Alpha 量測參數化的 host-LPDDR + PCIe 模型。

## Workload 範圍

| 軸向 | 選擇 |
| --- | --- |
| **模型家族** | Llama-3 + Qwen-2.5 |
| **模型大小** | 1B / 3B / 7-8B（Llama-3.2-1B、3B、Llama-3.1-8B、Qwen-2.5-7B）。Stretch 13B |
| **階段 / 精度** | Prefill + Decode 端到端；Metis CIM 上 INT8 |
| **混合精度** | 主要研究面：各單元原生精度不同（CIM INT8、NPU INT8/16/FP16、Mali FP16）。**CIM-MLP(INT8) × GPU-attention(FP16)** 的精度邊界管理是結構性新問題 |
| **Context / Batch** | 2K 基準 + 8K stretch；batch=1、AIPU Mode 1（介面保留 `batch` hook） |

**Layer A — 真實任務工作負載**（鋪滿 prefill-heavy ↔ decode-heavy 光譜，batch=1）：

| 任務 | 資料集 | mean prefill | mean decode | 特性 |
| --- | --- | --- | --- | --- |
| 多輪對話 | ShareGPT（英文） | 174.7 | 344.1 | decode 主導 |
| 簡單 QA / 推理 | GSM8K | 296 | 340 | 平衡 |
| 長文本處理 | LongBench-TriviaQA | 1787 | 5 | prefill 主導 |
| 程式碼補全 | HumanEval | 132.3 | 53.8 | on-device 程式助理 |

> GSM8K / LongBench 取自 HeteroInfer Table 4；ShareGPT / HumanEval 由 Phase 0.1 tokenize 得出（表為 llama-3.2 mean，qwen 近似 181.0/348.8、134.1/54.1；完整見 `measurements/op_inventory/workload_lengths.json`）。正確率只在驗證「混合精度對輸出品質」時用到（次要）。

**Layer B — 合成長度掃描**（roofline / scaling）：prefill ∈ {64, 256, 1024} + 動態未對齊 {128…1024} + decode @ prompt 256，再加 2048（8192 stretch）對齊 Metis 靜態 bucket（512/1024/2048）。

## 階段總覽

| 階段 | 內容 | 完成後可做什麼 |
| --- | --- | --- |
| **Phase 0.1** ✅ | 生成 trace 與 op inventory（純軟體，HF 模型 + workload → op×shape 流）。產出 `measurements/op_inventory/`、`traces/`。 | 定出 sweep matrix（去重 580 sig） |
| **Phase 0.2** ✅ | op 統計 / workload-op profile（逐-sig count + FLOPs/bytes/intensity + prefill/decode）。產出 `measurements/op_profile/`。 | 加權合成端到端 + 排序 0.3 量測 |
| **Phase 0.3** ✅ | 真實板量測（除溫度）：各單元 micro-benchmark + 端到端 LLM + variance。產出 `measurements/aetina/`、`measurements/metis_card/`。 | 可開始 Phase 1 / 2 |
| **Phase 0.4** ⏳ | 溫度／熱特性（**Metis Card only**，溫度可讀；Aetina 送修後補）。**最後做**，可與 1/2 並行。 | 熱模組 M8（選用）才加入 |
| **Phase 1**（傘狀） | 每個 component 建模 + 驗證：對量測擬合**方程式**（取代龐大 lookup table），逐一驗證。分子波（依資料來源切）。 | 校好、誠實標註的 component 模型 |
| &nbsp;&nbsp;**Phase 1.1** ✅（CIM 經 1.5 再 review） | silicon 校準並過 ADR-0006 gate：M1-CIM（2.7%）、M4-GPU/CPU、M2-DRAM（Card LPDDR4x）、M5-trace、M7-energy + 端到端 recompose hold-out（8B 9.5%）。**1.5 上 Card 補量**：native multi-tile residency-cliff（31%→2.4%）、dense prefill M=2..320、prefill M-wall=**512**、KV-BW 驗證。報告 `docs/report/phase1/`。 | 量測級可信的核心 decode 路徑 |
| &nbsp;&nbsp;**Phase 1.2** ✅ | 模組化「engine + 可換 spec」校準-analytic 元件層（換型號=換 spec）：CPU（殘差 1.15%）、NPU（解析 systolic-roofline，`simulated`、#13）、記憶體（analytic LPDDR4/4X/5 + SRAM CACTI）、GPU、CIM 雙拓樸。CIM 已 **`CARD_REVALIDATED`**（800MHz 13 點，median 4.8%）。**1.4 再 review**：conversion-op→analytic 重分類、報告數字重生。報告 `docs/report/phase1/`、findings `docs/phase1.2-findings.md`。 | 完整、可換型號的輕引擎模擬器 |
| &nbsp;&nbsp;**Phase 1.3** ✅ | 重型保真引擎插同一 `engine=` 介面：**ONNXim**（NPU）、**Ramulator2**（LPDDR5）皆 LIVE，驗 1.2 單串流趨勢。多單元競爭 / 逐 token 整機在 Phase 2。NPU 第三引擎 **ScaleSim 已建**（Phase 1.6，純 Python；1.6b 進一步實測 ONNXim/ScaleSim 是否有 32-systolic 特性）。 | 高保真引擎就緒 |
| **Phase 2** ⏳ | 整合成端到端 event-driven 模擬器（M3 事件引擎 + M6 排程器），跑完整 prefill+decode，做 L4/L6 端到端 + L5 敏感度 + 混合精度驗證。 | 完整模擬器 |

> **Phase 1.4 / 1.5 是再 review，非新階段**：對 1.1/1.2 的 component 重審 + 重生報告數字，已折進上表。各 phase findings 在 `docs/phase1.{1,2,3}-findings.md`；整併報告為單一 `docs/report/phase1/`。

> **可重現原則**：每次量測存**原始數據**（所有 iteration 原值 + sweep 兩軸 + config + 單位，不只 median/p95）+ 板上 raw log 納版控。所有圖是 build artifact：`tools/plotting/` 每圖一支 script，只吃 committed 數據重產，**絕不手繪**。

## Phase 0 — 量測前置與真實板量測（已完成 0.1–0.3）

- **0.1（純軟體）**：PyTorch runtime tracer（meta/FakeTensor，不載權重、不需 GPU）+ 架構解析交叉檢查 → 每模型 op inventory + sweep matrix（580 sig）+ 代表性 traces。op 集合：QKV/O/FFN MatMul/GEMV、QK^T/S·V、RMSNorm、RoPE、SwiGLU/SiLU、Softmax、residual、embedding、sampling。完整逐-prompt trace 由 M5 在 Phase 2 按需生成。
- **0.2（純軟體）**：由 0.1 trace 算逐-sig `count × (FLOPs/bytes/intensity)`，per (model × workload)、拆 prefill/decode。count 取自 inventory（不自行 ×layers）。是加權合成端到端 `Σ count(sig)×latency(shape)` 的接合點。
- **0.3（真實板）**：兩機並行量測（散熱受控避免 throttling）。Machine 1 = Aetina（CIM Alpha、RKNPU2、Mali、CPU + PCIe）→ `measurements/aetina/{metis_alpha_matmul,cpu_ops,mali_matmul,...}`；Machine 2 = Metis Card（廠商 INT8 LLM tok/s，L4 錨點）→ `measurements/metis_card/vendor_llm_int8.json`。量測協定見 `characterization/*/README.md`。

**交叉驗證矩陣（L1–L6）：**

| L | 驗證對象 | 資料來源 |
| --- | --- | --- |
| L1 | CIM tile 每 op 延遲 | Alpha CIM micro-benchmark（Phase 1 擬合參數方程式為主、lookup fallback） |
| L2 | DRAM / PCIe 來回 | Alpha PCIe + DMA timing |
| L3 | NPU / GPU 每 op | RKNPU2 + Mali matmul micro-benchmark |
| L4 | 端到端 LLM（INT8） | Metis Card 廠商 INT8 tok/s（注意 on-card DRAM ≠ host-MMIO 拓樸，橋接假設明寫） |
| L5 | 敏感度（任一參數 ±20%） | 於 sim 跑時計算，非獨立量測 |
| L6 | 端到端 CNN | Step-1 CNN 特性量測（225 cells，重用免重量測） |

> **Roofline 作為驗證視覺化**（L1/L3/L6）：每單元疊 measured vs predicted roofline，比 knee 位置 + 斜率 + 觀測點分佈。

**Phase 0.4（熱，待做）**：Metis 5 個溫度 sensor（board + 4 core），`axlogdevice --slog`/`core_temp` 可讀且**不像 power 受 M.2-Max 限制**，故 Card 可行（落地先跑 `axlogdevice --slog` 確認）。熱模組 **M8（選用）= 事後附加層**：讀 Phase 2 的 per-op 活動/功耗 timeline → 估溫度，**不**回饋 throttling 進 timing（閉環為 v1 後 stretch）。產出 `measurements/metis_card/thermal_*.json` + `docs/phase0.4-thermal-findings.md`。

## Phase 1 — component 建模與驗證（已完成 1.1–1.5）

- **1a Micro-benchmark → 方程式擬合**：CIM/NPU/GPU/PCIe 以 roofline 參數式起步（`latency ≈ max(compute, mem) + overhead`），記 median/p95/max 誤差；擬合不佳區段保留 lookup fallback（混合策略）。模擬器體積小、可外推、參數可解釋。
- **1b 非 micro-benchmark component 驗證**：M3 事件引擎、M5 trace、M6 排程器、M7 energy 在已知輸入/預期輸出上驗行為正確。

### 1c. 引擎選擇：light（擬合）vs heavy（外部模擬器）— 2026-06-12 決策

每個 unit 可在同一 `engine=` 介面後掛多個引擎；**有 silicon 的 unit 一律以 silicon 校準的擬合元件為 primary**，heavy sim 只在缺 silicon 處當主力或交叉檢查。

- **CIM（M1）**：Metis silicon = primary（最強資產）。NeuroSim / SCALE-Sim-CIM 僅選用交叉檢查或超越-Metis 外推，永不取代 silicon。
- **NPU（M4）**：**無 RKNPU2 silicon**（#13）。掛三引擎 `analytic` + `onnxim` + **`scalesim`（Phase 1.6 已建，純 Python 不需上板）**；皆非 silicon。框架=「擬合到被認可的學術模擬器」。primary 延到 Phase 2 由 **L4 端到端**裁。ONNXim 對 analytic 偏 ~4×（兩個都是模擬器、無真值裁判）。
- **GPU / CPU**：Mali / RK3588 A76 silicon = primary。
- **記憶體單串流（M2）**：silicon 系統效率 **0.65** = 驗證 primary；Ramulator2 device **0.92** = 交叉檢查；BW 效率為可 sweep 參數。
- **記憶體多單元競爭**：**Ramulator2 = primary**（Aetina 送修→並發 silicon 不可得）；單串流先 reconcile 到 0.65。**此層 v1 無 silicon 驗證**（L4 只動 CIM），靠 sim-vs-sim 形狀 + ±20% 敏感度 + HeteroInfer ~60 GB/s 外部參照。詳見 ADR-0001/0002 的 2026-06-12 修訂。
- **排程器（M6）**：Phase 1 只做 **naive 版、不 claim contribution**；貢獻框架延到 Phase 2。
- **定位**：Ramulator2 = 入場券（與純模擬的 PAPI/CENT 打平）；silicon 校準的 per-unit + 單串流 + L4 = 差異化。

## Phase 2 — 模擬器實作（整合，下一步）

照 **CLAUDE.md per-phase workflow** 走（branch → plan → subagent 審 → 批准 → 執行 → code-review → PR → merge），與前四波一致。整合切成 M3 事件引擎、M5 workload、M6 排程器三塊，逐塊對 `validation/contracts/*` + `measurements/` 驗證。

### 模擬器架構（6 box 資料流）

```
① Workload generator (M5)   HF model → PyTorch tracer → per-token op DAG
        │ op stream + tensor metadata
② Scheduler / Mapper (M6) ──貢獻層── 每 op：unit + 精度 + 記憶體放置 + dataflow + pipeline + 精度邊界
        │ (op, target_unit, precision)
③ Per-unit timing (M1+M4)   CIM/NPU/GPU/CPU，event-driven 並行（擬合方程式為主、lookup fallback）
        │ memory access pattern + latency
④ Memory hierarchy (M2+M3)  L1 SPM(4MB×core)─L2 SRAM(32MB)；Host LPDDR5(Ramulator2)；PCIe Gen3×4 DMA
        │ per-op time + access counts
⑤ Energy (M7)   CIM 15 TOPS/W×util；CPU A76 datasheet；memory JEDEC；PCIe 規格/byte
        │ per-inference latency + energy
⑥ Output + Inline Validation   端到端 latency/throughput/energy + per-op timeline/roofline；predicted vs measured
```

實作語言 Python（event loop 自寫；Ramulator2 Python bindings；ONNXim subprocess）。

### Repo 結構

分類軸：**docs（給人讀）/ simulator（程式）/ tools（腳本）/ measurements（資料）/ validation（科學驗證）/ tests（程式測試）。**

```
edge-cim-simulation/
├── CLAUDE.md  CONTEXT.md  OVERALL.md  README.md  LOG.md  requirements.phase0.txt
├── docs/                  # papers/ plans/ adr/ agents/ figures/ report/ + *-findings.md handoff-*.md voyager-sdk.md
├── simulator/
│   ├── specs/             # 可換硬體規格 json（換型號=換 spec）
│   ├── models/            # Phase 1 擬合元件 M1/M2/M4/M7 + engine.py（可換引擎介面）+ params/
│   │                      #   [Phase 2 進入時重構：models/→units/、engine.py→base.py、新增 engine/
│   │                      #    (M3 事件 + M5 workload + M6 排程 + runner)；30+ import 一次改，與 Phase 2 一併做]
│   └── engines/           # 外部 heavy-sim 快取（ONNXim/ScaleSim/Ramulator2 輸出，非 silicon；engine= 後端）
├── tools/                 # analysis/ plotting/ report/ trace_export/ onnxim/ ramulator2/(vendored)
├── characterization/      # 板上量測腳本：aetina/ metis_card/
├── measurements/          # op_inventory/ op_profile/ aetina/ metis_card/（silicon 量測，非 sim）
├── traces/                # 每 token op×shape 串流（per model, workload）
├── validation/            # contracts/(m*.yaml) reports/(phase1.*) validate_m5_trace.py validate_m7_energy.py
└── tests/                 # pytest tests/（vendored upstream/ 不收集）
```

### 模組（Modules）

| M | 模組 | 主要量測來源 | 備註 |
| --- | --- | --- | --- |
| M1 | CIM tile timing | Metis Alpha CNN + matmul | 擬合參數方程式為主、lookup fallback；NeuroSim 選用交叉檢查 |
| M2 | Memory hierarchy | Ramulator2 LPDDR5 + Alpha PCIe DMA | 容量為參數；BW 效率 0.65（silicon）/ 可 sweep；多單元競爭 = Ramulator2 |
| M3 | Event-driven engine | M1 + M2 | Python event loop；op stream 串過各單元 + 記憶體（Phase 2 建） |
| M4 | NPU / GPU / CPU | RKNPU2、Mali OpenCL、A76 | 各單元擬合方程式；NPU 三引擎 analytic/onnxim/scalesim（皆非 silicon，#13） |
| M5 | LLM workload generator | HF → PyTorch tracer → per-token op DAG | 前半即 Phase 0.1（ADR-0007） |
| M6 | Scheduler / Mapper | M3 + M4 + M5 | op→unit + 記憶體 + dataflow + 精度邊界。**Phase 1 naive、貢獻延 Phase 2** |
| M7 | Energy estimation | 廠商規格 + ARM datasheet | 規格 + activity-factor 估算（板無功耗儀表） |
| M8 | Thermal（選用） | Phase 0.4 熱量測 | 事後附加層；v1 不做閉環 throttling |

模組相依：`M3←M1+M2 · M6←M3+M4+M5 · M7←M1..M6`。

### 驗證合約範本（`validation/contracts/m{M}.yaml`）

```yaml
module: m1_cim_tile
measurement_sources: [measurements/aetina/metis_alpha_matmul.json, ...]
acceptance_criteria:
  - {type: median_op_error, threshold: 10%}                    # 指引，非硬承諾
  - {type: roofline_shape_match, metric: knee_position_drift, threshold: 15%}
  - {type: sanity, rules: [no_nan_or_inf, monotonic_with_op_size, latency_positive]}
sample_strategy: {cold_starts: 3, iterations_per_run: 300, budget_seconds: 30}
```

### SSH 回 Aetina 重新擷取（⚠ 阻塞）

⚠ **Aetina 送修中（2026-06-12）** — `measurements/aetina/` 凍結於現有快取，Phase 2 不可遠端重擷。修復回來後流程（也是並發 BW knee 量測入口，可把競爭模型從 sim-anchored 翻成 silicon-anchored）：`ssh aetina 'cd ~/repo/characterization/aetina && ./run_metis_matmul.py' && rsync …`。

## 開放風險

1. ✅ **已解 — NeuroSim 整合成本**：M1 改用 Metis Alpha 量測 + lookup fallback；NeuroSim 降為選用「模型形式」交叉檢查（Metis 是 digital SRAM-CIM ≠ analog RRAM）。
2. 🔶 **保留（Phase 2）— 橋接假設**：L4 錨「CIM + on-card-DRAM」、模擬器用 host-LPDDR + PCIe。需兩拓樸敏感度子實驗（=檢查點 #6）。
3. 🔶 **watch — HPIM 頂會搶先**（[筆記](docs/papers/pim-llm-accelerators/hpim-arxiv2025.md)）。差異化（mobile-SoC、真實晶片校準、混合精度、量測驅動分工）仍成立。
4. ✅ **已解 — agent 自主性**：前四波證明 per-phase workflow 可行；autoresearch 自走迴圈已棄用。
5. ✅ **已解 — ONNX export 品質**（ADR-0007）：op inventory 用 PyTorch runtime tracer，不靠 `torch.onnx.export`。
6. 🔶 **保留（Phase 2）— Ramulator2 多單元競爭**：1.3 已驗單串流；多單元在 Phase 2 為 primary（無 silicon 驗證，ADR-0002 修訂）。LPDDR4/4x 無 Ramulator2 preset（只有 LPDDR5）。
7. ⬇ **降級 — NPU 模擬器契合度**：ONNXim/SCALE-Sim 皆通用 systolic、皆非 silicon（#13），ONNXim 偏 ~4×。緩解：三引擎 + lookup override + L4 裁 + 敏感度（§1c）。
8. ⚠ **Aetina 送修中**：量測凍結、**並發 BW 量測被阻** → 多單元競爭 fallback 到 Ramulator2。Metis Card 仍可用（含 0.4 溫度）。

## 範圍外（v1）

- **Metis INT4**（Voyager 未開放使用者控制）；**AIPU Mode 2/3**（4× weight 或 static batched compile）；**batch > 1**（保留 `batch` hook）。
- **NVIDIA GPU baseline（Accel-Sim）/ Jetson**（保留 `gpu_backend` plugin 槽）。
- **閉環熱 throttling**（耦合 M1/M3；v1 只做 0.4 熱量測 + M8 開環估算）。
- **能量以量測取得**（改規格估算 M7，板缺功耗儀表）；**Intra-frame multi-core CIM 平行**（Voyager v1.3.1 未實作）。

## 開放檢查點

1. 🔶 **Phase 1/2 邊界**（Phase 2 入口）：M3/M6 這類「需先實作才能測」的 component 怎麼驗 — 沿 per-phase workflow 逐塊。
2. ✅ **方程式擬合誤差**（1.x 已答）：roofline 參數式 + lookup fallback，median/p95/max + ADR-0006 gate（CIM 2.4–4.8%、CPU 1.15%）。
3. ✅ **op inventory 完整性**（0.1 已答）：runtime tracer + 架構解析 → 580-sig matrix，decode 逐 kv 展開。
4. 🔶 **記憶體外推有效邊界**（部分答）：容量 + BW 效率已是參數；32GB+ 趨勢佐證待 Phase 2 敏感度。
5. ✅ **熱**（0.4 取代）：Metis Card 量溫度（開環 M8），最後做。
6. 🔶 **L4 橋接驗證強度**（Phase 2 入口）：on-card-DRAM vs host-MMIO 敏感度實驗設計（=風險 #2）。
