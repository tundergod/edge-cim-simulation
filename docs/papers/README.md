# 文獻與真實晶片筆記

本目錄收錄與「CIM-centric LLM-on-mobile-SoC 模擬器」**實作**直接相關的論文筆記與真實晶片量測報告。

> **篩選原則（依使用者要求嚴格執行）**：只保留「對模擬器*實作*有幫助」的文獻——亦即能直接驅動某個模組（M1–M7）、提供建模/設計決策、或作為校準 ground truth 的資料。純粹用於論文「related work / 定位」的文獻已移除。經一個 subagent 逐篇稽核後，**從 46 篇縮減為 16 篇**（移除 30 篇）。
>
> 每篇含：結構化筆記（`.md`，英文，給實作 agent 看的參考）＋ 原始論文（`.pdf`/`.html`，若取得）。筆記內的 `[[wikilink]]` 為 Obsidian 來源語法，請以 `papers/` 內的檔名解析。

**建議起點**：[hpim-arxiv2025](pim-llm-accelerators/hpim-arxiv2025.md)（最接近的競品）、[characterizing-mobile-soc...sosp2025](methodology-and-simulators/characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025.md)（HeteroInfer，我們的方法論模板）、以及 `metis-silicon/` 兩份調查報告（校準 ground truth）。

---

## 保留清單（17 篇）

> 2026-06-05 新增 [configuration-wall-asplos2026](methodology-and-simulators/configuration-wall-asplos2026.md)（DMA-bottleneck triage，見文末）。

圖例：📄 = 已附原始 PDF／HTML；🔗 = 原始檔僅有網址（未附本地檔）。每篇標注**對實作的用途**。

### pim-llm-accelerators/ — PIM/CIM LLM 加速器（競品與可借用機制）
| 筆記 | 原始檔 | 對實作的用途 |
|---|---|---|
| [hpim-arxiv2025](pim-llm-accelerators/hpim-arxiv2025.md) | 📄 pdf | **最接近競品。** SRAM-PIM(attention)+HBM-PIM(FFN) 分工＋intra-token decode pipeline → 直接啟發 **M6**（op→unit 分工、pipeline 設計）；也是論文差異化錨點 |
| [papi-asplos2025](pim-llm-accelerators/papi-asplos2025.md) | 📄 pdf | 以「算術強度門檻」動態決定 FC↔Attn 走哪個單元 → 可借用為 **M6** 的 mapping heuristic |
| [lp-spec-arxiv2025](pim-llm-accelerators/lp-spec-arxiv2025.md) | 📄 pdf | 平台最接近（mobile NPU+PIM）；GEMV→PIM / GEMM→NPU 動態分派建模 → 啟發 **M6**（即使 substrate 不同） |
| [neupims-asplos2024](pim-llm-accelerators/neupims-asplos2024.md) | 📄 pdf | sub-batch interleaving、GEMM/GEMV 並行 → 可重用的 **M6** pipeline 原語；其 ONNXim+DRAMsim3 方法亦對應我們的 **M2/M4** |
| [ianus-asplos2024](pim-llm-accelerators/ianus-asplos2024.md) | 📄 pdf | unified-memory 上的 PIM access scheduling（共享記憶體競爭）→ 啟發 **M2/M6**（我們的 MMIO-unified host memory 競爭模型） |

### methodology-and-simulators/ — 模擬與驗證方法論
| 筆記 | 原始檔 | 對實作的用途 |
|---|---|---|
| [characterizing-mobile-soc...sosp2025](methodology-and-simulators/characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025.md) | 📄 pdf | **HeteroInfer。** 我們整套照搬的方法論（先量測各單元、再決定分工；NPU stage/order/shape 敏感度）→ 驅動 Phase 0 掃描設計與 **M6** |
| [neurosim-validation-frontiers2021](methodology-and-simulators/neurosim-validation-frontiers2021.md) | 🔗 [URL](https://pmc.ncbi.nlm.nih.gov/articles/PMC8219932/) | D4 驗證範式（對真實晶片校準、<1% 誤差）→ **L1** 交叉驗證與 **M1** 可信度所複製的樣板 |
| [dnn-neurosim-v1-iedm2019](methodology-and-simulators/dnn-neurosim-v1-iedm2019.md) | 📄 pdf | CIM tile 的 timing/energy 模型 → **M1** 的選用性物理交叉檢查（NeuroSim cross-check） |
| [gem5-salam-merge-2025](methodology-and-simulators/gem5-salam-merge-2025.md) | 🔗 [URL](https://www.gem5.org/2025/07/30/gem5AccHetSimBlog.html) | 評估後未採用的建構路徑 → 影響 **M3** event-engine 架構決策（含 risk #1 的退路選項） |
| [configuration-wall-asplos2026](methodology-and-simulators/configuration-wall-asplos2026.md) | 📄 pdf | **per-call overhead 的形式化。** Configuration roofline（`I_OC`、`BW_Config`）+ dedup/overlap 編譯優化 → 我方 round-trip-tax thesis 的學理一般化；驅動 **M2/M3**（per-call overhead 時序形式）、**M6**（dedup/overlap 槓桿）、roofline 驗證的第三軸。⚠ 明確不含 DMA 資料搬移（只建模 setup/config 半邊；substrate 為 tightly-coupled，非 PCIe）|

### metis-silicon/ — 我方真實晶片調查（校準 ground truth）
| 筆記 | 原始檔 | 對實作的用途 |
|---|---|---|
| [metis-exp-board-rkc-a02-2026-05-18](metis-silicon/metis-exp-board-rkc-a02-2026-05-18.md) | 📄 html | Aetina 板可改性稽核（L1/L2 編譯期、IOMMU window、DMA tunables）→ **M2** 記憶體模型直接輸入＋Phase 0 可行性 |
| [metis-llm-investigation-desktop-2026-05-19](metis-silicon/metis-llm-investigation-desktop-2026-05-19.md) | 📄 html | **L4** 端到端 LLM 校準錨點（15 tok/s、24.23 GB/s decode wall、prefill TFLOP/s）→ **M2/M4/M7** ground truth |
| [metis-step1-cnn-characterization-2026-05-23](metis-silicon/metis-step1-cnn-characterization-2026-05-23.md) | 📄 pdf | **L6** CNN 校準錨點（225 cells、DMA round-trip floor、三模式資料）→ **M1/M2** ground truth |

### platforms/ — 硬體平台規格頁（vault 原生筆記，無外部 raw）
| 筆記 | 對實作的用途 |
|---|---|
| [system-aetina-rkc-a02](platforms/system-aetina-rkc-a02.md) | 直接寫入模擬器的平台規格（核數、L1/L2 大小、PCIe BW、TOPS、IOMMU）→ **M1/M2/M4/M7** 常數 |
| [system-axelera-metis-card](platforms/system-axelera-metis-card.md) | 量產卡規格＋實測 LLM 內部 IO 分解（98.5% weight-stream、MAC idle）→ **L4** 錨點常數（M2/M7） |

### ideas/ — 專案構想頁（vault 原生筆記，無外部 raw）
| 筆記 | 對實作的用途 |
|---|---|
| [cim-centric-llm-mobile-soc](ideas/cim-centric-llm-mobile-soc.md) | **這就是本專案的規格書**（定義 M1–M7、Phase 0、L1–L6）；`OVERALL.md` 的來源構想頁 |
| [cnn-dnn-edge-memory-wall-metis-embedded](ideas/cnn-dnn-edge-memory-wall-metis-embedded.md) | 校準來源構想：其 Step-1 計畫與 A1/A2/A3 子實驗即 Phase 0 / L6 流程，餵給 M1/M2 |

---

## 移除清單（30 篇，可由 git 還原）

依嚴格標準移除——皆屬「僅 related-work 定位」或「substrate/workload 不符（不影響我們的建模）」：

- **MoE 專屬**（我們 workload 是 dense Llama/Qwen，batch=1）：duplex-moe-pim、context-aware-moe-cxl-ndp、sieve-moe-pim
- **長上下文 / 稀疏注意力 / KV-PIM**（超出 dense batch=1 範圍，且多為 UPMEM/DRAM-PIM 定位）：l3-dimm-pim-longcontext、pimphony-lolpim-longcontext、repa-kvcache-pim、starc-sparse-attention-pim
- **不同 substrate / 伺服器級**（無可轉用機制）：cent-asplos2025（CXL all-PIM）、cxl-pnm-lpddr（datacenter CXL-PNM）、cambricon-llm（flash-PIM 70B）、lincoln-hpca2025（flash-PIM 50–100B）、specpim（speculative decode，明確 out-of-scope）、mi-llm（UPMEM）、pim-llm-pgemmlib（UPMEM GEMM lib）
- **on-device LLM（整個資料夾移除，皆屬 related-work / 稀疏或 flash 機制）**：powerinfer2-smartphone、fast-ondevice-llm-npu、llm-in-a-flash、kvswap-ondevice
- **methodology**：dnn-neurosim-v2（on-chip *training*，我們只做 inference）
- **metis-silicon**：metis-aipu-nn-v2（內部方向/策略報告，非校準資料或建模輸入）
- **concepts（整個資料夾移除，皆為背景概念入門，非實作輸入）**：compute-in-memory、in-memory-computing、sram-imc、memory-centric-computing、processing-in-memory-llm、kv-cache-management、llm-serving、llm-weight-quantization、on-device-llm-inference、speculative-decoding

> **可能想覆寫的邊界判斷**（若您不同意，告知即可還原）：
> - `processing-in-memory-llm`（concept）是唯一含實質工程內容（PIM-LLM substrate 分類表）的概念頁——若想保留一份做論文定位，這是首選。
> - `neupims`/`ianus`（保留）vs `cent`（移除）：三者皆伺服器級 HBM/CXL-PIM；保留前兩者因其 *機制*（sub-batch interleaving、unified-memory access scheduling）可映射到 M6/M2，移除 cent 因無可轉用之 mobile 技術。
> - `dnn-neurosim-v1`（保留）：若最終走純 trace-driven lookup（risk #1 退路），其角色從必需降為選用，但仍建議保留。
> - `gem5-salam`（保留）：屬「評估後未採用」的路徑；若視為純 related-work 可移除。
> - `lp-spec`（保留）：mobile 平台最接近競品，但其具體技術（LPDDR-PIM 上的 speculative decode）超出我們範圍；保留主要取其 mobile NPU+PIM 分派建模。

---

## DMA-bottleneck triage（2026-06-05）

針對 Phase 0.3 觀察到的 per-call DMA round-trip bottleneck，評估 4 篇候選論文是否「直接相關」。**保留 1、排除 3**（同 cent/cxl-pnm 的 substrate-mismatch 標準）：

| 論文 | 判定 | 理由 |
|---|---|---|
| **The Configuration Wall**（ASPLOS'26）| ✅ **ingest** | per-call overhead 的 configuration roofline 一般化我方 round-trip-tax；驅動 M2/M3/M6 + roofline 第三軸。⚠ 明確不含 DMA 資料搬移（只建模 setup/config 半邊；tightly-coupled 非 PCIe）|
| **QuCo**（HPCA'26，Murcia+W&M+NVIDIA）| ❌ excluded | GPU **TMA** tile-transfer 自動配置 HW unit；NVIDIA-GPU substrate 屬 future work（Mali 才是我方 SoC GPU；NVIDIA = 延後的 Accel-Sim plug-in），v1 無模組可驅動。Configuration Wall 的 GPU-DMA 姊妹作 |
| **COMET**（HPCA'26，NUDT+PKU）| ❌ excluded | multi-chiplet-module（MCM）on-package 互連 + 記憶體 co-design；我方拓樸是 host-SoC + 單一 CIM-on-PCIe，非多 chiplet 加速器，comms 模型不對應 |
| **Fastmove**（FAST'23，USTC+SmartX）| ❌ excluded | on-chip DMA（Intel I/OAT 類）做 DRAM↔NVM 儲存搬移 + CPU/DMA load-splitting；主題相鄰但 DMA 領域不同（儲存非加速器 offload），無可轉用之 PCIe round-trip 模型 |

> 若不同意排除判斷（特別是 QuCo——它是「DMA tile-transfer 配置開銷」最貼題的一篇，只差在 GPU substrate），告知即可補 ingest 為 related-work 定位。
