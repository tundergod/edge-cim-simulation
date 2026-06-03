# 異質行動 SoC 上的 CIM-Centric LLM 推論 — 模擬器

> **狀態：初步專案綱要。** 以下的計畫、模組拆分與架構皆為*初步*設計，並非定案——可隨特性量測資料與實作現實自由修改。固定不變的只有：**研究目標**（一個以真實晶片校準的「CIM-enabled 異質行動 SoC 上 LLM 推論」模擬器）與我們用來校準的**兩塊真實板子**。

**Workload：** LLM 推論（Llama-3 + Qwen-2.5 系列，1B–8B，stretch 至 13B；prefill + decode）。
**平台：** **模擬的** CIM-enabled 異質行動 SoC（CIM 掛在 PCIe 上，搭配 16 GB MMIO unified host memory），以真實 Metis Alpha（Aetina RKC-A02）＋量產 Metis Card 作交叉驗證。未來平台錨點：**Axelera Europa**（2026 H1 — 128 MB L2 SRAM + LPDDR5 + unified architecture，約 629 TOPS @ 45 W）。
**貢獻：** 待特性量測後確定；預期為 **CIM-centric**——CIM 快又省電但有諸多限制，因此系統設計「先繞著 CIM 的限制走」。

---

## 為什麼要做模擬器

我們想研究 CIM-mobile SoC 上的 LLM，但這種晶片並不存在，而真實 Metis 卡也無法直接當研究對象：Metis Alpha 完全跑不了 LLM 計算（封閉韌體 `-1301` 牆＋無 on-card DRAM，只有 1 GB IOMMU window），量產 Metis Card 則只能透過封閉、預編譯的工具鏈跑 LLM。模擬器的策略是把**研究對象**改成一個*模擬的* CIM-mobile SoC，並以真實 Metis 晶片作為**校準 ground truth**：
- **Metis Alpha（Aetina）：** 在 CNN/matmul workload 上提供 CIM 計算原語＋PCIe 行為。
- **量產 Metis Card：** 提供 on-card-DRAM 拓樸下、INT8 的真實 LLM 行為。

真實晶片細節見 [papers/metis-silicon/](papers/metis-silicon/)；SDK 量測面見 [voyager-sdk.md](voyager-sdk.md)。

## 問題

商用的離散 CIM 加速器（Axelera Metis、Hailo-8/10、Mythic、Untether）一律以 PCIe / M.2 介面卡形式出貨。行動 SoC 則各自把 NPU + GPU + CPU 整合在共享 LPDDR 周邊。這兩者的結合——**CIM 作為異質行動 SoC 上、可存取 unified memory 的對等計算單元**——正是**架構走向**（Axelera Europa 2026 H1 已證實：128 MB L2 SRAM + LPDDR5 + 「unified architecture」+ 內建 decoder），但**文獻上幾乎沒有**在這種組合上跑 LLM 的工作。

相鄰的先前工作只存在於片段中（筆記見 [papers/pim-llm-accelerators/](papers/pim-llm-accelerators/)）：
- HBM-PIM / GDDR-PIM 的 LLM 加速器（NeuPIMs、IANUS、AttAcc、CENT、HPIM、PAPI）— 伺服器級；非 mobile、非 SRAM-CIM
- LPDDR-PIM mobile（LP-Spec）— 用 LPDDR 內部 PIM；非離散 CIM
- Compute-enabled flash（Lincoln、Cambricon-LLM）— flash 基底；非 SRAM-CIM
- HeteroInfer（SOSP'25）— 特性化 GPU+NPU mobile 異質 LLM；無 CIM
- 我方 Metis 量測（[metis-llm-investigation](papers/metis-silicon/metis-llm-investigation-desktop-2026-05-19.md)、[metis-step1-cnn-characterization](papers/metis-silicon/metis-step1-cnn-characterization-2026-05-23.md)）— 真實晶片，但 LLM 計算延伸被廠商封閉

**這個 gap 正好是 (離散-CIM × 異質-mobile-SoC × LLM × 真實晶片校準)** — 在 2024–2026 頂會文獻中近乎空白的一格。

## 立場（Position）

**CIM-centric 設計紀律：** 系統**先繞著 CIM 的限制設計**（擅長 weight-stationary GEMV、不擅長 dynamic attention、tile 需對齊 channel 為 64 的倍數、weight residency 上限、host-device 來回成本）。GPU / NPU / CPU 是**支援層**，負責 CIM 不能或不該做的 op（dynamic K 的 attention、RMSNorm、sampling、KV-cache 管理、混合精度邊界）。

**具體的 op 層級分工交給特性量測決定** — 我們*不*預先假定「CIM 做 MLP、其它做 attention」（這是 HPIM 與多數既有工作的假設）；改由量測出的各單元特性曲線決定分工。這就是把 HeteroInfer 的方法論套用到一個新平台類別。

**與並行工作的差異化：**
- **HPIM（最接近競品 — [筆記](papers/pim-llm-accelerators/hpim-arxiv2025.md)）：** 異質 SRAM-PIM + HBM-PIM，純模擬、僅 FP16、無 energy/area、非 mobile。我們做 SRAM-CIM + GPU + NPU + CPU on mobile-SoC，有真實晶片校準，INT8（Metis），混合精度（CIM INT8 × GPU FP16）。
- **PAPI：** 動態 GPU+PIM 排程、伺服器級、FC-PIM/Attn-PIM 分工。我們是 mobile + CIM-centric（非 GPU-centric）+ 特性量測驅動、非預設分工。
- **LP-Spec：** LPDDR-PIM mobile、NPU+PIM、speculative decode。我們是 CIM（非 LPDDR-PIM）+ 一般 decode（非僅 speculative）。
- **Lincoln：** flash-PIM 50–100B 消費級。不同基底（flash vs SRAM-CIM）、不同規模（極大 vs 1–13B），但同樣是 on-device LLM 的願景。

## 平台假設

**立場：operational + descriptive。** 把「CIM-on-PCIe + MMIO unified host memory」當作模擬器既定的平台假設。

- **描述性證據：** Metis Alpha 已在 1 GB 實作 MMIO 統一（IOMMU window 把 host LPDDR4 映射成 CIM core 可見）；Voyager 已用 `dma-buf` + `cl_khr_external_memory_dma_buf` 做免拷貝 buffer 共享。
- **前瞻錨點：** Axelera Europa（2026 H1）— 128 MB on-chip L2 SRAM、LPDDR5 256-bit @ ~200 GB/s、含 RISC-V vector cores + HEVC265 decoder 的「unified architecture」。
- **保守容量：** 16 GB unified，足以放下 7B INT8 + KV cache + activations；低於 Axelera M.2 Max（已 16 GB on-card）。
- **社群訊號：** Axelera 論壇確認「已知悉更大 unified memory 的需求」（但無承諾 roadmap）。注意：**現行 Metis 並無 unified host-device memory** — 這是前瞻假設，非當前能力。

寫進 Method 段的橋接假設：*「CIM-tile 的計算 timing 在更換記憶體基底時不變，只有資料搬移 timing 改變。量產 Metis Card 的廠商 LLM 量測驗證了 CIM + on-card-DRAM 拓樸；我們的模擬器把 on-card DRAM 換成由 Metis Alpha 量測參數化的 host-LPDDR + PCIe 模型。」*

## Workload 範圍

| 軸向 | 選擇 |
| --- | --- |
| **模型家族** | **Llama-3 + Qwen-2.5** |
| **模型大小** | **1B / 3B / 7-8B**（Llama-3.2-1B、Phi-3-mini 級、Llama-3.1-8B / Qwen-2.5-7B）。Stretch：**13B** |
| **階段** | **Prefill + Decode 端到端** |
| **精度** | **Metis CIM 上 INT8** |
| **🔥 混合精度** | **主要研究面，非 ablation。** 各單元原生精度不同（CIM INT8、NPU INT8/INT16/FP16、Mali FP16、CPU 任意）。**CIM-MLP（INT8）與 GPU-attention（FP16）** 之間的精度邊界管理是結構性的新問題。 |
| **Context 長度** | 2K 基準 + 8K stretch |
| **Batch** | **batch=1，AIPU Mode 1**（單實例）。模擬器介面保留 `batch` hook 以利擴充。 |

## Phase 0 — 真實板特性量測

**Phase 0 先在量產 Metis Card + Aetina 上跑；只有當 Phase 0 量測 commit 進 repo 後，模擬器實作才開始。** 本階段產出模擬器要驗證的所有 ground-truth 資料（下方 L1–L6），因此交叉驗證的可行性是在 *Phase 0 就確定*，而非押注在模擬器完成後。

**做法：拆解為 chip-level invariants + workload-level translations。** Phase 0 分到兩台機器、兩個獨立的 agent 交接。各機器段落自成一體（工具、目標、輸出檔、掃描矩陣）。

### Machine 1 — Aetina RKC-A02（RK3588 + Metis Alpha）

**Agent 角色：** 特性化異質 SoC 的四個計算單元（CIM Alpha、RKNPU2、Mali、CPU）＋ host-device PCIe 邊界。commit 到 `measurements/aetina/`。

**本機需求：** Voyager SDK v1.3.1（Metis Alpha）、RKNN toolkit（RKNPU2）、OpenCL driver（Mali）、`perf`、`eBPF`/`bpftrace`、`taskset`、`chrt`。

| 單元 | 量測內容 | 輸出檔 |
| --- | --- | --- |
| **A. Metis Alpha** | (A1) 以 single-op ONNX 差分法做 CIM tile micro-benchmark — 掃 conv shape：in/out ch、H/W。(A2) `dpu_constants_home: l2` vs `ddr` 的 timing 差 → SRAM vs PCIe 流量比。(A3) Mode 1 每次呼叫的 DMA 階段分解（eBPF + LD_PRELOAD）。(A4) 與 LLM 相關的 matmul micro-benchmark | `metis_alpha_{cnn_proxy,matmul,pcie}.json` |
| **C. RKNPU2** | 跨 LLM 相關 shape 的 matmul micro-benchmark（hidden 2048/4096/8192、seq 1/256/2048），INT8 + INT16 + FP16，batch 掃描 | `rknpu2_matmul.json` |
| **D. Mali** | 自寫 OpenCL matmul kernel（避開 framework 雜訊），FP16 為主 + FP32 參考 | `mali_matmul.json` |
| **E. CPU（RK3588 A76）** | 在板上 micro-benchmark LLM 的 CPU 支援 op（sampling、RoPE 控制、KV-cache append/evict、token/quantization 邊界）。`taskset -c 4-7 chrt -f 50`；`clock_gettime` + `perf stat` | `cpu_ops.json` |

每單元特性掃描（HeteroInfer Fig 2-4 風格）：tile/channel 大小錯配（ch ∈ {64…1024}）、batch（Metis 固定 1；RKNPU2/Mali 1/4/16/32）、weight residency（`l2` vs `ddr`）、op 類型敏感度、精度、sequence 維變化（hidden × seq，seq 1→2048）。

**交付物：** 四個 `measurements/aetina/*.json`；`variance_profile.json`；`characterization/aetina/README.md`（腳本→檔案對應）；最終報告 `docs/phase0-aetina-findings.md`（SDK 意外、op 覆蓋缺口、對模擬器的建議）。

### Machine 2 — Ubuntu + 量產 Metis Card

**Agent 角色：** 擷取真實「CIM 跑 LLM」的錨點。commit 到 `measurements/metis_card/`。

**本機需求：** 含 LLM 預編譯成品（Llama-3 系列）的 Voyager SDK（v1.6）、Python LLM benchmark harness。

| 單元 | 量測內容 | 輸出檔 |
| --- | --- | --- |
| **B. Metis Card** | 廠商預編譯 INT8 LLM：Llama-3.2-1B / 3B / 8B 端到端 tok/s + 每 token 延遲 + 4-core 使用率 + context 長度掃描（2K / 4K / 8K），用 `axllm --show-stats` | `vendor_llm_int8.json` |

**交付物：** `vendor_llm_int8.json`；`variance_profile.json`；`characterization/metis_card/README.md`；最終報告 `docs/phase0-metis-card-findings.md`（各模型 tok/s、隨 context 的 scaling、值得標記的廠商 SDK 行為）。

### 量測協定 — 兩階段取樣

- **Stage 0**（變異特性，半天）：每單元挑代表性 op，cold-start 重複 × 每次多 iteration；算 Coefficient of Variation（CoV）；存 `measurements/{unit}/variance_profile.json`。
- **Stage 1**（正式特性）：用 Stage-0 推導的取樣計畫，套用到完整的 op × shape × precision × batch 掃描；取樣量記入 `validation/contracts/m{M}.yaml`。

### Phase 0 產出的交叉驗證矩陣（L1–L6）

| L | 模擬器驗證對象 | Phase 0 資料來源 |
| --- | --- | --- |
| L1 | CIM tile 每 op 延遲 | Metis Alpha 上 A1 + A4 + Stage 掃描（trace-driven lookup 為主；NeuroSim physics 交叉檢查為選用） |
| L2 | DRAM / PCIe 來回 | A2 + A3（PCIe + DMA 模式 timing） |
| L3 | NPU / GPU 每 op | C + D matmul micro-benchmark |
| L4 | 端到端 LLM（INT8） | B Metis Card 廠商 INT8 LLM tok/s。注意：on-card DRAM ≠ 模擬器的 host-MMIO 拓樸（橋接假設在 Method 明寫） |
| L5 | 敏感度（任一參數 ±20%） | 於 sim 跑時對 L1–L4 計算 — 非獨立量測 |
| **L6** | 端到端 CNN | [metis-step1-cnn-characterization](papers/metis-silicon/metis-step1-cnn-characterization-2026-05-23.md) — 225 cells 已擷取；**重用，免重量測** |

**Roofline 作為驗證視覺化**（L1/L3/L6）：每單元疊出 measured-roofline 與 simulator-predicted-roofline；knee 位置 + 斜率 + 觀測點分佈是否吻合，即同時對 compute + memory 原語做 2D 一致性檢查。資料點於 Phase 0 從同一批 run 抽出。

**混合精度驗證：** 只做最單純的直接案例（如某一模型上 CIM INT8 + GPU FP16 分工）。混合精度本身是方法/貢獻；其驗證刻意簡化。

### Phase 0 完成標準

當**兩台**機器都 commit 後即為完成：所有 `measurements/aetina/*`（4 檔）+ `measurements/metis_card/*`（1 檔）；各機 `variance_profile.json`；各機 `characterization/{aetina,metis_card}/README.md`；各機 `docs/phase0-{aetina,metis_card}-findings.md`；以及一份綜合 `docs/phase0-L1-L6-mapping.md`。兩份報告都 commit 後，**模擬器實作於 Metis Card 機器啟動**。

---

## 模擬器實作

**前置：Phase 0 量測已 commit。** 以下是「在 Phase 0 ground truth 之上建模擬器」的操作交接，設計成初始設定後可（大致）交給自主 agent。

### 做法 — autoresearch 模式、最小基礎設施

**1 個 LLM agent、0 個外部 orchestrator、Python validator 腳本。** 由 agent 自己驅動迴圈。草圖：
```
while not all_modules_passed:
    讀 program.md, HANDOFF.md, log.jsonl, validation/contracts/, measurements/
    依相依圖 + log 狀態挑下一個模組 M
    修改 simulator/modules/m{M}.py
    執行: python simulator/runner.py --module M
    讀 validation_result.json
    if passed:        log; 對 Mi<M 跑 regression; 若 ok 則前進
    elif retryable:   log; 內部診斷; 修改; 重試
    elif stuck:       標記給人; 跳過; 前進
    更新 HANDOFF.md（給下一個 session 的狀態）
```

### 模擬器架構（6 個 box + 資料流）

分層、模組化、event-driven。每個 box 只有一個上游輸入、一個驗證合約、一個量測來源（或上游量測的組合）——所以 agent 可以一次只迭代並驗證一個 box。

```
① Workload generator (M5)
   HuggingFace model → torch.onnx.export → per-token op DAG
   Llama-3 / Qwen-2.5 1B–8B（13B stretch）；prefill+decode；batch=1；對齊 ONNXim trace
            │ op stream + tensor metadata
            ▼
② Scheduler / Mapper (M6)  ── 貢獻層 ──
   每 op 決定：單元（CIM/NPU/GPU/CPU）+ 精度 + 記憶體放置 + dataflow
   + pipeline + 精度邊界插入 + 資源約束檢查
   Plugin 介面：baseline 策略 ↔ 提案策略
            │ (op, target_unit, precision) tuples
            ▼
③ Per-unit timing models (M1 + M4)   event-driven、並行
   CIM (Metis)：自 Metis Alpha 量測做 trace-driven lookup
   NPU (RKNPU2)：ONNXim fork + 對偏離 shape 的 lookup override
   GPU (Mali)：自 Mali OpenCL 量測做 trace-driven lookup
   CPU：ARM A76 instruction-count 模型
   （保留 `gpu_backend` plugin 槽給 Accel-Sim）
            │ memory access pattern + latency
            ▼
④ Memory hierarchy (M2 + M3)
   Metis L1 SPM（4 MB×core）─ L2 SRAM（32 MB 共享）
   Host LPDDR5 — Ramulator2 後端；PCIe Gen3 ×4 DMA 模型（BW + latency + setup）
   TLB-miss penalty（可參數化，預設 0）；on-SoC LPDDR 由 RKNPU2/Mali/CPU 共享
            │ per-op time + memory access counts
            ▼
⑤ Energy estimation (M7)
   Metis CIM：廠商 15 TOPS/W × 使用率；CPU A76：ARM datasheet × activity
   RKNPU2/Mali：INA-delta 或 tech-node 推導；Memory：JEDEC 每存取；PCIe：規格每 byte
            │ per-inference latency + energy
            ▼
⑥ Output + Inline Validation
   端到端 latency / throughput / energy-per-inference；per-op timeline + roofline
   Inline comparator：每 box predicted vs measured → 寫 validation_result.json
```

**實作語言：** Python（event loop 自寫；Ramulator2 透過 Python bindings；ONNXim fork 以 subprocess 整合）。

### Repo 結構（目標）

```
edge-cim-simulation/
├── overall.md                  # 本綱要
├── voyager-sdk.md              # SDK 特性量測參考（給所有 agent；英文）
├── README.md
├── program.md                  # agent 主指令（範本見下）
├── HANDOFF.md                  # 跨 session 狀態
├── log.jsonl                   # 每迭代 log（append-only）
├── papers/                     # 文獻 + 真實晶片筆記（本 commit）
├── simulator/
│   ├── modules/                # M1–M7（m1_cim_tile.py … m7_energy.py）
│   ├── runner.py               # 入口：python runner.py --module M
│   ├── validator.py            # 比對輸出與 measurements
│   └── lib/
├── measurements/               # ground truth，納入版控
│   ├── aetina/                 # metis_alpha_{cnn_proxy,matmul,pcie}, rknpu2_matmul, mali_matmul, cpu_ops
│   └── metis_card/             # vendor_llm_int8.json
├── characterization/           # 重新擷取量測的腳本（aetina/, metis_card/）
├── validation/contracts/       # 每模組驗證規格 YAML（m1.yaml …）
├── tools/                      # analysis/, plotting/（roofline）, trace_export/
├── tests/
└── docs/                       # 給人看（architecture、protocol、phase0 findings）
```

### 兩機分工 — 三次 agent 交接

| 階段 | 機器 | Agent 角色 |
| --- | --- | --- |
| Phase 0 — Aetina | Aetina RKC-A02 | A/C/D/E 量測 → `measurements/aetina/*` + findings |
| Phase 0 — Metis Card | Ubuntu + Metis Card | B 量測 → `measurements/metis_card/*` + findings |
| 模擬器實作 | Ubuntu + Metis Card | autoresearch 迴圈：M1–M7。拉 Phase 0 資料；對 `validation/contracts/*` 迭代。需要時 SSH 回 Aetina 重新擷取 |

兩個 Phase 0 agent 可並行（不同機器/領域、無即時耦合）。模擬器實作只在兩份 Phase 0 報告都 commit 後才開始。以 git push/pull 在共享 repo 同步。

### SSH 存取 — 模擬器 agent → Aetina（僅重新擷取路徑）

模擬器實作期間，若需要 `measurements/aetina/` 沒有的 shape/precision/config，就觸發遠端重新擷取：
```bash
ssh aetina 'cd ~/repo/characterization/aetina && ./run_metis_matmul.py --config tier1'
rsync aetina:~/repo/measurements/aetina/ ./measurements/aetina/
git add measurements/aetina/ && git commit -m "char: new shapes" && git push
```
一次性設定：ed25519 key、`ssh-copy-id`、`~/.ssh/config` 的 Host 條目。所有程式開發在 Metis Card 機器；Aetina 是被遠端驅動的量測機台。

### program.md 範本（草圖）

```markdown
# Project: CIM-Centric LLM Inference Simulator
## Goal
為「CIM-enabled 異質行動 SoC 上的 LLM 推論」實作並驗證一個模擬器。
一次迭代一個模組，直到全部通過對 measurements/ ground truth 的驗證。
## How to work
1. 讀 HANDOFF.md（否則 log.jsonl 尾端）取得當前狀態。
2. 讀目標模組的 validation/contracts/m{M}.yaml。
3. 讀相關 measurements/*.json 當 ground truth。
4. 編輯 simulator/modules/m{M}.py；跑 `python simulator/runner.py --module m{M}`。
5. 讀 validator 輸出。若 passed：regression `--up-to m{M}`；若 ok 則前進。
   若否：把分析 append 到 log.jsonl，提假設，修改，重試（每 session 上限 20 次）。
## Module dependency graph
M1(CIM tile)←metis_alpha_*  ·  M2(memory)←metis_alpha_pcie+Ramulator2  ·  M3(event engine)←M1+M2
M4(NPU/GPU/CPU)←rknpu2+mali+cpu_ops  ·  M5(workload)←HF+torch.onnx.export  ·  M6(scheduler)←M3+M4+M5  ·  M7(energy)←M1..M6
## End-of-session：更新 HANDOFF.md（模組、最後狀態、blockers、下一步）。
```

### 驗證合約範本（每模組）

```yaml
module: m1_cim_tile
measurement_sources:
  - measurements/aetina/metis_alpha_cnn_proxy.json
  - measurements/aetina/metis_alpha_matmul.json
acceptance_criteria:
  - {type: median_op_error, threshold: 10%}        # 指引，非硬承諾
  - {type: roofline_shape_match, metric: knee_position_drift, threshold: 15%}
  - {type: sanity, rules: [no_nan_or_inf, monotonic_with_op_size, latency_positive]}
sample_strategy: {cold_starts: 3, iterations_per_run: 300, budget_seconds: 30}
```

### 模組（Modules）

| M | 模組 | 主要量測來源 | 備註 |
| --- | --- | --- | --- |
| M1 | CIM tile timing | Metis Alpha CNN + matmul micro-benchmark | trace-driven lookup 為主；NeuroSim 選用交叉檢查 |
| M2 | Memory hierarchy | Ramulator2 LPDDR5 + Metis Alpha PCIe DMA | TLB-miss penalty 參數化（預設 0） |
| M3 | Event-driven engine | M1 + M2 | Python event loop；把 op stream 串過各單元 + 記憶體 |
| M4 | NPU / GPU / CPU traces | RKNPU2 matmul、Mali OpenCL matmul、CPU A76 | NPU = ONNXim fork + lookup override；GPU/CPU = 僅 lookup |
| M5 | LLM workload generator | HF Llama-3 / Qwen-2.5 → torch.onnx.export → per-token op DAG | trace 格式對齊 ONNXim 輸入 |
| M6 | Scheduler / Mapper | M3 + M4 + M5 | Plugin：op→unit + 記憶體 + dataflow + pipeline + 精度邊界插入。**貢獻在此。** |
| M7 | Energy estimation | 廠商規格（Metis 15 TOPS/W）、ARM datasheet、INA-delta 或 tech-node | 規格 + activity-factor 估算 |

## 開放風險

1. **NeuroSim 整合成本超出估計** — M1 退路：純 trace-driven lookup Metis Alpha 量測。NeuroSim 由必需降為選用；驗證方法論的引用（NeuroSim <1% 晶片誤差）即使不用其程式碼仍成立。
2. **橋接假設：Metis Card on-card DRAM ≠ 模擬器的 host-MMIO 拓樸** — L4 錨定「CIM + on-card-DRAM」；模擬器以 host-LPDDR + PCIe 替代。需做兩種拓樸下的敏感度子實驗。
3. **HPIM 搶先在頂會發表**（最接近競品，[筆記](papers/pim-llm-accelerators/hpim-arxiv2025.md)）。即使 HPIM 先落地，我們的差異化（mobile-SoC、真實晶片校準、混合精度、特性量測驅動分工）仍成立。
4. **agent 自主性在模擬器規模未經驗證** — 第一次 M1 迭代才是真正的考驗。緩解：若 agent 多個 session 仍不收斂，退回人工開發。
5. **HuggingFace ONNX export 品質** — `torch.onnx.export(Llama-3 / Qwen-2.5)` 出了名地亂（custom op、dynamic shape）。M5 可能需要人工後處理或換工具。依賴前先驗證。
6. **Ramulator2 LPDDR5 + PIM-like 延伸覆蓋度** — 我們的 LPDDR5-PIM-like 用法非 Ramulator2 預設；可能需自寫 plug-in。M2 要預留。
7. **ONNXim 對 RKNPU2 的契合度** — ONNXim 建模通用 systolic NPU；RKNPU2 有 Rockchip 特有行為（op-mix 敏感、depthwise+Swish 弱項 — Step-1 資料）。Plan B：lookup-table override（已在 M4 設計內）。
8. **Aetina 的 SSH 可用性** — 整個 sim 開發期間須可達。若離線，標記 blocker 並用快取量測繼續。

## 範圍外（v1 論文）

- **Metis CIM 上的 INT4** — Voyager 公開文件未開放使用者控制的 INT4。未來工作（若 SDK 開放或出現廠商 INT4 成品）。
- **AIPU Mode 2（4-instance）/ Mode 3（compiler-batched）** — 需 4× weight footprint 或 static-shape batched compile；不符 16 GB unified 上的單 batch dynamic-shape LLM。未來工作（伺服器式 batched 情境）。
- **batch > 1** — mobile 單 batch 是論文範圍。模擬器保留 `batch` hook。
- **NVIDIA GPU baseline（Accel-Sim）/ Jetson Orin / Nano** — 對 M4 的介面化延伸；保留 `gpu_backend` plugin 槽。未來泛化研究。
- **熱建模** — device-dependent、不可泛化；板子缺 on-board 功耗儀表。
- **能量以量測取得** — 改以規格估算（M7），同樣是儀表問題。
- **Intra-frame multi-core CIM 平行** — Voyager v1.3.1 未實作 `cooperative` / `pipeline` 模式。未來 SDK 可能開放。

---

## 參考文獻（本 repo 內）

> 已依「對實作有幫助」嚴格篩選，從 46 篇縮為 16 篇。完整保留/移除清單與理由見 [papers/README.md](papers/README.md)。

- **最接近競品：** [HPIM](papers/pim-llm-accelerators/hpim-arxiv2025.md)
- **直接先前工作（PIM-LLM 加速器）：** [papers/pim-llm-accelerators/](papers/pim-llm-accelerators/) — PAPI、NeuPIMs、IANUS、LP-Spec（其機制可借用到 M6/M2）
- **方法論 / 模擬器：** [papers/methodology-and-simulators/](papers/methodology-and-simulators/) — HeteroInfer（SOSP'25，特性量測模板）、NeuroSim validation（D4 範式）、DNN-NeuroSim v1（CIM tile 物理交叉檢查）、gem5-SALAM（評估後未採用）
- **真實晶片校準來源：** [papers/metis-silicon/](papers/metis-silicon/) — Step-1 CNN 特性量測（L6）、Metis Card LLM 調查（L4）、Aetina 板稽核
- **平台：** [papers/platforms/](papers/platforms/) — Aetina RKC-A02、Axelera Metis Card
- **校準來源構想：** [cnn-dnn-edge-memory-wall-metis-embedded](papers/ideas/cnn-dnn-edge-memory-wall-metis-embedded.md)、來源構想頁 [cim-centric-llm-mobile-soc](papers/ideas/cim-centric-llm-mobile-soc.md)
- **SDK 量測面：** [voyager-sdk.md](voyager-sdk.md)
