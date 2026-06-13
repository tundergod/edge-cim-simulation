# edge-cim-simulation

以真實 Axelera Metis 晶片校準的模擬器，研究 **CIM 作為異質行動 SoC 上的對等計算單元**時的 LLM 推論（CIM + NPU + GPU + CPU、unified memory、INT8、batch=1、prefill + decode）。

## 文件

- **[OVERALL.md](OVERALL.md)** — 專案綱要：目標、Phase 0.1/0.2/0.3/0.4/1/2、6-box 架構（M1–M8）、驗證層（L1–L6）。
- **[docs/adr/](docs/adr/)** — 核心設計決策（ADR-0001…0007）。
- **[docs/voyager-sdk.md](docs/voyager-sdk.md)** — Metis/SDK 量測參考（給 agent，英文）。
- **[docs/papers/](docs/papers/)** — 文獻 + 真實晶片筆記（16 篇 + 原始 PDF）。
- **[CLAUDE.md](CLAUDE.md)** — agent 工作守則（per-phase workflow + karpathy）。

## 狀態

Phase 0.1–0.3（trace / op-profile / 真實板量測）與 Phase 1.1–1.3（元件建模與驗證：CIM、記憶體、CPU、GPU、NPU，加 Ramulator2 / ONNXim 重型 sim drop-in）已完成；合併回顧報告見 [`docs/report/phase1-site/`](docs/report/phase1-site/)（整合 Phase 0+1 的手刻多頁網站）。已知量測缺口（NPU #13、prefill / multi-tile 補量測）誠實標註於各 findings。**下一步 Phase 2**（整合 M3 事件引擎 + M6 排程器，跑端到端 prefill+decode）。

## 外部參考

- Voyager SDK：<https://github.com/axelera-ai-hub/voyager-sdk>
- Axelera 論壇（Metis M.2）：<https://community.axelera.ai/metis-m-2-3>
