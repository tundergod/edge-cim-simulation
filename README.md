# edge-cim-simulation

以真實 Axelera Metis 晶片校準的模擬器，研究 **CIM 作為異質行動 SoC 上的對等計算單元**時的 LLM 推論（CIM + NPU + GPU + CPU、unified memory、INT8、batch=1、prefill + decode）。

## 文件

- **[overall.md](overall.md)** — 專案綱要：目標、Phase 0.1/0.2/0.3/1/2、6-box 架構（M1–M8）、驗證層（L1–L6）。
- **[docs/adr/](docs/adr/)** — 核心設計決策（ADR-0001…0007）。
- **[voyager-sdk.md](voyager-sdk.md)** — Metis/SDK 量測參考（給 agent，英文）。
- **[papers/](papers/)** — 文獻 + 真實晶片筆記（16 篇 + 原始 PDF）。
- **[CLAUDE.md](CLAUDE.md)** — agent 工作守則（per-phase workflow + karpathy）。

## 狀態

Bootstrap：設計已定案（ADR）、文獻與 SDK 參考就位。下一步 **Phase 0.1**（用 PyTorch runtime tracer 抽 op inventory）。

## 外部參考

- Voyager SDK：<https://github.com/axelera-ai-hub/voyager-sdk>
- Axelera 論壇（Metis M.2）：<https://community.axelera.ai/metis-m-2-3>
