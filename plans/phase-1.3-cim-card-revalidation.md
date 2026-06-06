# Plan: Phase 1.3 — CIM-Card 重驗 + edge-CIM 記憶體層 + prefill GEMM 擬合

分支：`phase-1.3`（沿用）。執行序：Gate → Spike →（gate）→ Full → Fit → Cross-val → Edge-mem → Honesty/PR。
板存取需使用者授權；Spike 或 Artifactory 不過即 STOP+回報，不進下一段。

## G. 前置 Gate（板存取）

1. 使用者授權 `ssh metiscard`（settings 加一條 Bash allow 規則）。 → verify：`ssh metiscard 'echo ok'` 回 `ok`。
2. 確認板在線＋時脈：`ssh metiscard 'source ~/tundergod/voyager-sdk/axelera-env/bin/activate && axdevice -v'`。 → verify：`metis-0:7:0` 在、`clock=800MHz`（非 800 則 `axdevice --set-clock 800`）。

## S. Spike（可行性，~30–60 min）

3. 查編譯器：`ssh metiscard` 跑 `pip list | grep -i axelera`、`which axcompile`、`axcompile --help`。 → verify：記錄 `axelera-devkit`/`axcompile` 有無；存下 `axcompile --help` 的 flag 名稱。
4. 若 `axcompile` 不在：拋棄式 venv 測 Artifactory `python3 -m venv /tmp/axc && /tmp/axc/bin/pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple 'axelera-devkit[all]'`。 → verify：成功或記下 auth/index 錯誤；被擋即 STOP+回報。
5. 改 `characterization/metis_card/run_metis_cim_v16.py:_try_low_level_compile`：先試 `axcompile --input m.onnx --output out --input-shape 1,K,1,M --log-level WARNING`（flag 名先對步驟 3 的 `axcompile --help` 核對），失敗才退舊 `compile`，兩者皆無才回 ABSENT。 → verify：函式先試 `axcompile`；本機乾跑無語法錯。
6. 跑 `run_metis_cim_v16.py --spike`（conv-proxy：square + 2 staircase + 1 prefill M=256）。 → verify：conv proxy 編出 `out/**/model.json`、`axrunmodel` 回 dev/host/system fps（非 ABSENT）。
7. raw-Gemm 探針：寫一個裸 `Gemm` ONNX，`axcompile` 試編一次。 → verify：記錄「編出」或 `ONNXGraphCleanerError`，據此定 Full 用 raw-Gemm 或 conv-proxy。
8. Spike 結論寫回 `validation/reports/phase1.2/cim_card_revalidate.json`（`status: SPIKE_OK|BLOCKED_<reason>`）；回報使用者。 → verify：報告反映 spike 結果；得使用者 OK 才進 Full。

## F. Full 量測（Spike 過 + 使用者 OK 後）

9. 跑 `run_metis_cim_v16.py` full：`alpha13`（13 形狀）+ `prefill` M∈{128,256,512,1024,2048}（gate_up/q_o/down）。 → verify：alpha13 13 點皆有 `dev_gflops`；prefill M≥512 不再 `no_model_json`。
10. `manifest()` 加一個固定小群組 `matmul_sweep`，從 `sweep_matrix.json` 的 `matrix.matmul` 取一組固定形狀（decode-GEMV M=1, K∈{2048,3072,4096} × N∈{1024,2048,8192} + prefill M∈{128,512}）；無新 flag。 → verify：群組形狀皆量到或各自記 `error`。
11. rsync 結果回庫 → `measurements/metis_card/metis_card_matmul.json`。 → verify：檔在、含 alpha13 + prefill + matmul_sweep 群組。

## P. Prefill GEMM 擬合

12. 新擬合器 `tools/analysis/fit_cim_prefill.py`：在量到的 prefill 點上擬合 `G_eff_prefill(M,K,N)`（含 M amortization）。 → verify：殘差列出；對 M 單調；> 原生 tile 標 extrapolated。
13. prefill 分支接進 `simulator/models/m1_cim_tile.py:dev_lat_us` + `params/m1_cim.json`：M>1 走擬合（M=1 decode 路徑不變）。 → verify：`dev_lat_us(M=128,K=4096,N=14336)` 落在量測 ±擬合容差內（現行線性-M 為 65843µs ≈ 80× 偏離，須消除）。
14. 對 vendor TTFT 交叉驗證：把 prefill-fit 預測**沿 1024-token prompt 組合**後對 vendor `ttft_s=3.794` 比；改 `tools/analysis/recompose_e2e.py` 的 `M_pf=1024` 段改用擬合（取代 implied-TOPS）。 → verify：rel error 列出；prefill 段由 `UNGATED` 升為 gated/有界。

## V. Card 對 Alpha 交叉驗證

15. 跑 `tools/analysis/validate_cim_card.py`：Card 13 點 `dev_gflops` 對 Alpha 13 點（同顆 AIPU、800MHz、無 rescale）算 median/p95 `|rel_diff|`。 → verify：報告轉 `CARD_REVALIDATED`、含 median/p95 + prefill 新點。
16. G_eff 參數決策：`|diff|` 在容差內保留 Alpha 擬合（解凍）；否則對 Card 點重擬。 → verify：`m1_cim.json` provenance 更新（confirmed-on-card 或 re-fit）。

## E. Edge-CIM 記憶體層（從 Alpha + 目標 spec 組，非 Card）

17. 新 spec `simulator/specs/cim_topo_edge.json`：`topology="edge"`（必要，否則 `m2_memory.py` 的 `"topology" in spec` 分支不取）、`on_card_dram=false`、`mem_spec_ref="mem_lpddr5"`、`noc_efficiency`（標 assumption）、`transport`=NoC|PCIe、per-call floor 取 Alpha PCIe 或 0；共用 `m1_cim.json`。 → verify：spec 載入回 dict、含 `topology` + `mem_spec_ref` + 每欄 provenance。
18. 改 `simulator/models/m2_memory.py`：加 `from simulator.specs.loader import load_spec`；topology spec 若 `on_card_dram=false`，`eff_BW_GBs = load_spec(spec["mem_spec_ref"])["peak_GBs"] * spec["noc_efficiency"]`（非 `dram_eff_BW_GBs` 的 None、非 24.2），避免 line 51/77 的 None→AssertionError。 → verify：`MemoryModel(load_spec("cim_topo_edge")).predict(...)` 不報錯、decode BW 隨目標 LPDDR 變、≠ Card 24.2。
19. 註冊新 spec：加進 `validation/contracts/specs.yaml` + `tools/analysis/check_phase1_2.py` 的 `SPECS`，probe 用 `op="stream"`（走新 `eff_BW_GBs` 路徑、非 `op="pcie"`，NoC-default 不會誤 FAIL）。 → verify：`check_phase1_2.py` exit 0 且 `cim_topo_edge` 被 probe。
20. 更新 `docs/report/phase1.2/chapters/CIM-card.md`：加 edge 記憶體牆組成 + provenance（目標 LPDDR + Alpha PCIe + Ramulator，非 Card 24.2）+ CARD_REVALIDATED 結果；移除 DEFERRED_FALLBACK 段。 → verify：章節含 edge provenance + 結果、無 DEFERRED_FALLBACK。

## H. 誠實/回報/PR

21. 更新 `docs/voyager-sdk.md`（§2/§9/§10/§13）：`compile`→`axcompile`、doc 路徑改 `docs/reference/compiler/onnx-support.md`〔路徑待對板上 SDK 核對〕、`axelera-rt`/`axelera-devkit[all]` wheel + **Beta** + Artifactory 權限待證；標 tag。 → verify：相關行更新並標來源。
22. 更新 `validation/reports/phase1.2/cim_card_revalidate.json`：以 axcompile/devkit 發現取代舊「SSH 未授權 / compile 不在」理由，寫最終狀態；保留 `honesty` 欄含 "Alpha 13"+"calibrated"（或 `CARD_REVALIDATED`），勿破壞 `check_phase1_2.py:87-89` gate。 → verify：報告 reason 與現況一致、`check_phase1_2.py` 仍 exit 0。
23. `gh pr create`（`phase-1.3` → `main`）：摘要 + verify 結果。 → verify：PR 開出。

Outputs: `measurements/metis_card/metis_card_matmul.json`；更新後 `m1_cim.json`(+prefill 參數) 與 `m1_cim_tile.py`；`tools/analysis/fit_cim_prefill.py`；`simulator/specs/cim_topo_edge.json` + `m2_memory.py` edge 分支；更新後 `run_metis_cim_v16.py`、`recompose_e2e.py`、`check_phase1_2.py`、`specs.yaml`；`cim_card_revalidate.json`(`CARD_REVALIDATED`)；`validate_cim_card.py` 報告；CIM-card 章節；`docs/voyager-sdk.md` 更新。
