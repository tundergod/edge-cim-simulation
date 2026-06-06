# 附錄 — 編號來源表

本報告的每個量化數字都在內文以上標編號（如 `[^cim13]`）標註，**對應的編號來源條目列於各章末尾**（每章一份該章的來源表，緊跟該章內容，方便就地回查、不需翻到全書最後）。編號前綴標示其所屬單元：`cim`=CIM、`mem`=記憶體、`cpu`=CPU、`gpu`=GPU、`npu`=NPU、`m5`=trace、`m7`=能耗、`e2e`=端到端、`m3`/`m6`=整合層。本附錄則給出所有來源所在的 artifact 目錄總覽。

每條來源指向 repo 內可回查的 artifact 檔案與欄位。這些 artifact 是事實基準（findings 散文若與之衝突，以 artifact 為準）。主要來源目錄：

- **驗證報告**：`validation/reports/phase1.{1,2,3}/*.json` — 各模組的驗證數字（誤差、gate 結果、交叉驗證）。
- **驗收合約**：`validation/contracts/m{1..7}.yaml` — 各模組驗收標準、可調參數、測量缺口。
- **擬合參數**：`simulator/models/params/*.json` — 方程式係數（calibrated／assumption 標註於檔內）。
- **可換 spec**：`simulator/specs/*.json` — 各單元／記憶體型號 spec（provenance 標註於檔內）。
- **量測 ground truth**：`measurements/aetina/`、`measurements/metis_card/` — Phase 0.3 真實板量測原始資料。
- **設計決策**：`docs/adr/0001`–`0007` — 鎖定的架構決策（fidelity、記憶體、排程、混合精度、能耗、驗證橋接、op inventory）。

> 逐期的完整 findings 仍保留於 `docs/phase1.{1,2,3}-findings.md`，作為本報告的敘事輸入與歷史記錄；本報告已將其關鍵內容完整吸收。