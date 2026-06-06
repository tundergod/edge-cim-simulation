# M (1.3) — Ramulator2 LPDDR5：解析記憶體的「重型 sim 交叉驗證」

Phase 1.2 的記憶體引擎是**解析的**：LPDDR5 有效頻寬 = 51.2 GB/s 峰值 × **0.65** 效率 = 33.3 GB/s（`mem_lpddr5` spec）。Phase 1.3 把 **Ramulator2**（DRAM 時序模擬的次領域標準）當 `engine='ramulator2'` drop-in 插進**同一介面**，對這個 0.65 假設做交叉驗證。

## 怎麼跑（v2.1，in-process）

Ramulator 2.1 是 **Python-bindings-only**（無 CLI、無 YAML）。我們直接用它**自己的** `latency_throughput` harness（`run_simulation` / `resolve_spec` / `checks.py`），不自寫、不手算 tCK —— v2.1 內部就算好 `total_throughput_MBps`（真實皮秒）。組態 `LPDDR5_8Gb_x16` + `LPDDR5_6400`，飽和串流讀，量兩次：**peak**（關 refresh）+ **achievable**（AllBank refresh）。

> **為什麼非 v2.1 不可**：main 分支的 LPDDR5 在飽和下會 `Failed to send refresh!`（issue #58/#60）；維護者明確指 v2.1 修好（更好的 LPDDR5 + controller 模型、systematic closed-row、WCK2CK sync、prefetch 8→16）。實測 v2.1：streaming 跑得動、無 abort、`channel_width=16`、peak **98.6% 飽和**。build 需補一個 Apple-clang-17 的 `param.h` `template`-keyword patch（`build.sh` 已收）。

## 結果：device 0.92 vs system 0.65 —— 不是矛盾，是兩個層級

![M2-ramulator2](../../../figures/phase1.3/M2-ramulator2.png)

| | 效率 | eff BW | 範圍 |
|---|---|---|---|
| **Ramulator2 v2.1**（DRAM device） | **0.92** | 47.1 GB/s | 只模 DRAM 元件時序（refresh + bank conflict） |
| **解析（system）** | **0.65** | 33.3 GB/s | 系統級，校準到**矽晶 decode 牆**（LPDDR4x 24.2 GB/s） |

Ramulator2 說單串流 LPDDR5 在**元件層**能到峰值的 92%（refresh-limited）—— 比我們的解析 0.65 **高很多**。這**不是打架**：

- Ramulator2 只模 **DRAM 元件**的時序，所以單串流幾乎打滿（元件不是瓶頸）。
- 解析的 0.65 是**系統級**效率，校準到真實量到的 decode 牆（24.2 GB/s），把 controller / NoC / 排隊 / 真實 workload 的開銷都**摺進去了** —— 這些 Ramulator2 的元件模型**沒有**。
- 兩者的差（92% vs 65%）**正是元件之外的系統開銷**。

→ **這直接驗證了 ADR-0002 的設計決定**：「Ramulator 模 DRAM 元件、不模 SoC NoC 仲裁，所以 achievable-vs-peak 的差距是**從 Aetina 並發 micro-benchmark 校準**、不是從 Ramulator import」。Ramulator2 確認了 **DRAM 元件不是單串流瓶頸**；系統級的 0.65（矽晶校準）**仍為主**。

## 誠實標註 + 定位

- `engine='ramulator2'` 回的 47.1 GB/s 是 **device-level ceiling**，標 `simulated (Ramulator2 v2.1)、NOT silicon`。解析 33.3（system-level）**維持為 primary**。
- 本期只驗**單串流**。Ramulator2 的**招牌價值 —— 多單元競爭（CIM+NPU+GPU+CPU 共用 LPDDR）—— 在 Phase 2**（那才是 device-level 時序模型真正派上用場、且系統開銷要顯式建模的地方）。
- **LPDDR4/4x 無 Ramulator2 preset**（實際 build 確認：只有 LPDDR5）→ 只能驗 LPDDR5；LPDDR4x anchor 的交叉驗證待 port（ADR-0002 已記）。

數據可重產：`tools/ramulator2/build.sh`（釘 v2.1 commit `278f1ef`）→ `tools/analysis/mem_ramulator2.py` → `simulated/ramulator2/lpddr5_eff.json` → `build_mem_ramulator2.py` → 本章報告/圖。
