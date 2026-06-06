# M — 記憶體（engine + 可換 spec：三種 host DRAM + Metis SRAM tier）

> **這一章你會學到**：Phase 1.2 把記憶體做成「**一個解析引擎 + 可換 spec**」——同一份 `MemoryModel` 程式碼，換一個 spec 檔（LPDDR4／4x／5、或 CIM 拓樸）就換掉整個記憶體模型；以及為什麼 Metis 晶片上的 SRAM 在 8B 模型下「**永遠放不下權重**」、所以它是 M1-SPM 的一個「架構假設」層、不是 decode 記憶體牆的解方。

---

## M.1 設計：一個引擎、可換 spec（D3 / D5）

Phase 1.1 的 `MemoryModel` 把 PCIe floor、LPDDR、kv_append 寫死在一起。Phase 1.2 改成共用引擎介面（`simulator/models/engine.py` 的 `UnitEngine`）：

```
MemoryModel(spec, engine='analytic').predict(Workload) -> {latency_us, bound, provenance}
```

**spec 建構時綁、`predict` 只吃 workload**。一份 spec 決定套哪種物理：

| spec 種類 | 檔案 | 模型 | bound |
|---|---|---|---|
| **host DRAM** | `mem_lpddr4` / `mem_lpddr4x` / `mem_lpddr5` | `stream/kv = bytes ÷ eff_BW` | `memory` |
| **CIM 拓樸** | `cim_topo_alpha` / `cim_topo_card` / `cim_topo_edge` | `pcie = per_call_floor + bytes ÷ pcie_BW`；edge `stream = bytes ÷ (LPDDR5 eff_BW × noc_eff)` | `floor`／`memory` |

換型號 = 換 spec 檔，引擎碼不動。重型後端（Ramulator2）走**同一個建構簽名**，Phase 1.3 插進來（見 M.5）。

---

## M.2 三種 host DRAM 對「24.2 量測錨點」（誠實標註）

**圖 M1（三 spec 有效頻寬）**
![M1](../../../figures/phase1.2/M1.png)

- **X 軸**：三種可換記憶體 spec。**淺灰柱**：理論峰值（**assumption**——MT/s × 64-bit，in-repo 無 data-rate 出處）。**實心柱**：有效頻寬，依誠實標籤上色。**虛線**：量測錨點 24.2 GB/s。
- **LPDDR4x（藍／calibrated）= 量測錨點**：量產 Metis Card 的 on-card LPDDR4x，batch-1 decode 是純權重串流的記憶體牆（decode 時間 ∝ 權重 bytes，**r²=0.997**，voyager-sdk.md:278）→ 有效頻寬 **24.2 GB/s**，= 34.1 峰值的 **71%**。這是唯一**對我們的 silicon 校準**的數字。
- **LPDDR5（橘／simulated）= 33.3**：51.2 峰值 × **0.65**。**為什麼是 0.65 而不是量測到的 0.71？** LPDDR5 是**另一塊記憶體**、我們**沒有**在它上面量過效率——所以**保守往下折**到 0.65，而不是假設跟 4x 一樣好。這是 simulated，不是 calibrated。
- **LPDDR4（灰／assumption）= 18.2**：25.6 峰值 × 0.71（把 4x 量到的效率「套用」到 4——一塊不同但相關的記憶體）→ eff_BW 是**推導值**，標 assumption。

> **三 spec 都通過 sanity**：stream latency 對 bytes 正且單調；LPDDR4x→24.2、LPDDR5→33.3（見 `validation/reports/phase1.2/m2_memory.json`）。

---

## M.3 CIM 拓樸：那筆 911µs floor 誰付、誰不付

同一個 `MemoryModel`，餵 CIM 拓樸 spec 就變成 PCIe 傳輸模型：

- **`cim_topo_alpha`（量測 Alpha）**：`per_call_floor_us = 911.1`（Phase 0.3 量到的 per-call DMA floor `pcie_floor_A1d5`，**measured**）+ `bytes ÷ 3.9 GB/s`。每次離散 host↔device 搬移都付這 911µs → bound = **floor**。
- **`cim_topo_card`（量產 Card）**：`per_call_floor_us = 0`——on-card 串流權重**不**逐次走 PCIe（**architecture**）。
- **`cim_topo_edge`（前瞻 edge，Phase 1.3 加）**：整合在 SoC、**無專屬 DRAM**，權重從共用 **LPDDR5** 經片上 NoC 串流 → `stream = bytes ÷ (mem_lpddr5.eff_BW 33.3 × noc_efficiency 0.9) = bytes ÷ 30.0 GB/s`，bound = **memory**。**assumption**（無 edge silicon）；**不是** Card 的 24.2（那是 Card 專屬 LPDDR4x，topology-specific）。

**關鍵誠實邊界**：911µs 是 Alpha **沒有 on-card DRAM** 的**拓樸特例**，**不外推**到量產卡（比照 Phase 1.1 A2.2）。floor 只對離散搬移（KV-reload、activation handoff、conversion-op）收費，decode 串流權重的主幹用頻寬項。

---

## M.4 Metis SRAM tier（M1-SPM）：8B 權重「永遠放不下」

SRAM tier 在 **M1-SPM**（`simulator/models/m1_cim_spm.py` 的 `SramTier`，讀 `sram_metis_aipu`），**不在 M2**——M2 管 host DRAM + CIM 拓樸 floor，SRAM 屬 CIM 計算單元的 scratchpad（合約 `m2.yaml` 已註明此歸屬與 ADR-0002 偏離）。

- **容量**：L1 4 MiB/核 + L2 32 MiB 共享 + D-IMC 1 MiB/核 = **52 MiB**（datasheet/ISSCC 2024）。
- **BW/latency = CACTI assumption**：Metis 沒有公開 SRAM 的 BW/latency，所以用 CACTI tier 的代表值（256 GB/s、5 ns）當**假設**。
- **residency = architecture-only**：8B INT8 權重（~8 GB）**≫ 32 MiB L2**，所以 `predict()` 把這種工作集**解析到 DRAM 層**（永不命中 SRAM）。decode 的記憶體牆是 **host LPDDR、不是 SRAM**。

**圖 M3（SRAM tier residency what-if）**
![M3](../../../figures/phase1.2/M3.png)

- 工作集 ≤ 32 MiB L2 牆 → 走 SRAM 層（綠，CACTI assumption 的高頻寬、每 GB 時間低）；超過 → **spill 到 DRAM 層**（藍，calibrated 24.2 錨點，每 GB 時間約 10×）。
- **8B 權重（~8 GB）的點永遠落在 DRAM 那條線上**——這正是「architecture-only」：我們把 SRAM 階層**還原進模擬器**（之後做架構研究時，「如果權重能放進更大的 SRAM，decode 會快多少」是個 load-bearing 變數），但**不宣稱**現在的權重放得下。

---

## M.5 為什麼 Ramulator2 在 Phase 1.3、而且 LPDDR4 不在它的預設裡

**本期記憶體全是 calibrated-analytic**；重型 DRAM 模擬（Ramulator2，cacheline 粒度）**移到 Phase 1.3**，走**同一個建構簽名 + 凍結 contract**（`engine='ramulator2'`，ADR-0002 swappable）。這是相對 ADR-0002 原文「Phase 2」的一個**時程偏離**（已記入 `m2.yaml`）。

> **⚠️ 一個 1.3 要注意的點：LPDDR4(x) 不是 Ramulator2 的一線 DRAM 預設。** Ramulator2 內建的標準預設以 DDR3/4/5、LPDDR5、HBM 為主；**LPDDR4／LPDDR4x 通常沒有現成 preset**。所以 Phase 1.3 接 Ramulator2 時，量產卡那塊 **LPDDR4x（24.2 錨點）必須自行提供/改寫 timing config**（或退而用 LPDDR5 preset 做 cross-check，但那是另一塊記憶體、不能當 4x 的驗證）。本期的 calibrated 24.2 錨點正是 1.3 要對齊的目標值。

---

## M.6 限制與 gap（誠實清單）

| 項目 | 狀態 | 標籤 |
|---|---|---|
| LPDDR4x 24.2 | ✅ 量測錨點 | **calibrated**（量產卡 decode 牆，r²=0.997） |
| LPDDR5 33.3 | 解析 | **simulated**（0.65 折扣 < 量測 0.71，不同記憶體） |
| LPDDR4 18.2 / 各峰值 | 推導 / 數學 | **assumption**（套用 4x 效率／MT/s×64-bit） |
| Alpha 911µs floor | ✅ 量測 | **measured**（Alpha 拓樸特例，不外推 Card） |
| SRAM BW/latency | 假設 | **CACTI assumption**（無公開 Metis SRAM 規格） |
| residency | 架構 | **architecture-only**（8B 權重永不命中 → spill DRAM） |
| kv_append | 未驗證 | **unvalidated**（Phase 0.3 未隔離；純-BW 形式對、待救板校準係數） |
| Ramulator2 | 延 1.3 | swappable（LPDDR4x 無現成 preset，須自備 timing config） |

**一句話總結 M**：記憶體做成「一引擎 + 可換 spec」——`MemoryModel(spec)` 換 spec 就換型號（LPDDR4/4x/5、CIM 拓樸 A/B），唯一**校準**的是量產卡 LPDDR4x 的 **24.2 GB/s** 錨點，LPDDR5(33.3)=simulated、峰值=assumption；SRAM tier 在 M1-SPM、CACTI 假設、residency=architecture-only（8B 權重永遠 spill 到 DRAM）；重型 Ramulator2 在 Phase 1.3 插進同一介面（且 LPDDR4x 需自備 timing config）。
