# 第 8 章 — 尚未實作的整合層：M3 事件引擎、M6 排程器

前面各章的單元都是**元件**。把它們串成端到端模擬器需要兩個**整合層**模組：M3（事件引擎，處理共享頻寬競爭與重疊）與 M6（排程器，決定 op→單元映射與精度邊界）。**這兩個在 Phase 1 只有 contract，沒有實作。**

這裡要把兩件事分清楚，因為它們的性質完全不同：

1. **「未實作」不是 Phase 1 的缺失** — M3/M6 的行為驗證需要「先實作才能測」（toy op-stream 時序、競爭 knee、端到端 latency）。依專案規劃，**實作 M3+M6 並整合正是 Phase 2 的定義內容**。Phase 1 的職責只到「定義 contract + 可調參數」。
2. **M6 連帶一個 Phase-2 解析建模項** — conversion-op（精度邊界的 dequant/requant）成本。各單元原生精度固定（CIM INT8、GPU FP16），所以邊界 cast 是 memory-bound op，由既有 M2/M4 per-op 模型 × M6 插入的邊界穿越次數計價（ADR-0004 已於 2026-06-11 修訂）。**這不是測量缺口**，是 Phase 2 的建模工作。

---

## 8.1 M3 — 事件引擎（contract-only）

**規劃要它做什麼。** 一個輕量 discrete-event 引擎（ADR-0001）：把 op stream 串過各單元 + 記憶體，建模 op 級並行、共享頻寬競爭的飽和 knee（~60 GB/s）、CIM ∥ GPU ∥ NPU ∥ CPU 的重疊。

**Phase 1 交付了什麼。** 只有 contract 與可調參數：`bandwidth_contention_knee`、`interconnect_efficiency`、`concurrency_overlap_factor`；驗收門檻（端到端 latency/toks ≤15%、競爭 knee ≤15%）是 **ADR-0006 的 Phase-2 系統級門檻**，**在 Phase 1 不 gate**。[^m3a] per-op 延遲來源（M1、M2）已備妥餵給引擎。[^m3b]

**狀態：** 未實作；行為驗證（toy op-stream 時序、~60 GB/s 競爭 knee 重現）在 Phase 2 實作後進行。

---

## 8.2 M6 — 排程器 / 映射器（contract-only）

**規劃要它做什麼。** 這是**貢獻層**：每個 op 決定單元（CIM/NPU/GPU/CPU）、精度、記憶體放置、dataflow，並在精度邊界插入 conversion op。

**Phase 1 交付了什麼。** 只有 contract 與可調參數：`op_to_unit_mapping_policy`、`precision_boundary_placement`、`memory_placement`、`precision_boundary_conversion_op_cost`（後者於 Phase 2 解析建模，非校準目標）。[^m6a]

**conversion-op 成本 = Phase 2 解析建模，非量測。** 各單元原生精度固定（CIM INT8、GPU FP16，ADR-0004(c)），精度邊界由 M6 的 op→單元映射決定。邊界上的 dequant/requant 是一個 memory-bound cast（讀 int8、寫 fp16、乘 scale），其成本由既有的 M2/M4 per-op 模型 × M6 插入的邊界穿越次數解析求得——**不需要專門的量測 campaign**。[^m6c] 因此「CIM-INT8 × GPU-FP16」混合精度貢獻的成本基礎 = 既有 per-op 模型 + M6 排程；Phase 2 把 cast op 插進 op stream 計價即可。[^m6b]

**狀態：** 未實作；conversion-op 成本是被追蹤的可調參數 + Phase-2 解析建模項（非測量缺口）。

---

## 8.3 小結

| 模組 | Phase 1 狀態 | 性質 | Phase 2 需要做 |
|---|---|---|---|
| **M3 事件引擎** | contract + 3 個可調參數 | 未實作（**設計使然**＝Phase 2 工作） | 實作輕量 event loop；驗證 ~60 GB/s 競爭 knee、端到端時序 |
| **M6 排程器** | contract + 4 個可調參數 | 未實作（**設計使然**） | 實作 op→單元映射、精度邊界插入 |
| **└ conversion-op 成本** | Phase-2 解析建模項 | 非量測（memory-bound cast，既有 M2/M4 × 邊界穿越） | 把 cast op 插進 op stream，用既有 per-op 模型計價 |

M3/M6 的「未實作」本身不擋路（那是 Phase 2 的內容）；conversion-op 成本是 Phase 2 內的解析建模項，非擋路的測量缺口（第 9 章歸類）。

---

## 來源（本章腳注）

[^m3a]: 來源 `validation/contracts/m3.yaml` › `phase1_scope`="contract_only"、`tunable_params`=[bandwidth_contention_knee, interconnect_efficiency, concurrency_overlap_factor]、`acceptance_criteria`=[{e2e_latency_or_toks:15%},{contention_knee:15%}]、`note`（Phase 1 只定 contract，不 gate）
[^m3b]: 來源 `validation/contracts/m3.yaml` › `measurement_sources` = [simulator/models/m1_cim_tile.py, simulator/models/m2_memory.py]
[^m6a]: 來源 `validation/contracts/m6.yaml` › `phase1_scope`="contract_only"、`tunable_params`=[precision_boundary_conversion_op_cost, op_to_unit_mapping_policy, precision_boundary_placement, memory_placement]
[^m6b]: 來源 `validation/contracts/m6.yaml` › `phase2_modeling_item` = {id:conversion_op_cost, adr:ADR-0004, note:"memory-bound cast, analytically costed by existing M2/M4 per-op models × scheduler boundary-crossing count; no measurement target (precisions fixed per unit)."}
[^m6c]: 來源 `docs/adr/0004-mixed-precision.md`（2026-06-11 修訂）› Decision (a)（scheduler 插入 conversion op，其成本於 Phase 2 解析建模）+ Decision (c)（精度固定於單元）+ Consequences（conversion-op 非量測目標）
