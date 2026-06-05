# Plan: Phase 1 — 每個 component 的建模與驗證

> **狀態（2026-06-05）：subagent grill-review 已通過（3 輪，零剩餘 issue）；user 已批准並加兩項要求（已併入）：output = 報告（HTML→PDF），每張圖明列 x-y 軸。待 user 最終 go 後執行。** branch `phase-1` 已開（off `main`，PR #7/#8 已併）。
>
> **範圍邊界（user 已拍板）：** (1) M3 event engine / M6 scheduler 在 Phase 1 **只定驗證合約 + 標出待調參數**，行為驗證留 Phase 2；M5/M7 在已知輸入上驗證。(2) **不建 event-driven 模擬器**（Phase 2）：擬合/驗證程式碼延伸 `tools/analysis/`，擬合好的方程式+參數落 `simulator/models/`，驗證報告落 `validation/`。(3) 納入端到端重新合成（細節見 §2，已按資料現況重新界定）。(4) **NPU(M4) = 佔位/相依項**（issue #13，aetina 離線）。

## 背景（rationale 集中於此，下方 numbered steps 只放 action + verify）

輸入（皆已 commit 於 `main`）：`measurements/aetina/{metis_alpha_matmul（含 by_group + pcie_floor_A1d5）, metis_alpha_matmul_raw, cim_attention_composed, mali_matmul, cpu_ops}.json`、`measurements/metis_card/{vendor_llm_int8, twopillar_prediction}.json`、`measurements/op_profile/*.json`。先例：`tools/analysis/cim_analysis.py`（已導 PCIe floor、C4 composed attention、C5 兩支柱 hold-out）。**不存在**：`metis_alpha_pcie.json`、`variance_profile.json`（Phase 0.3 collect-what-you-can gap；A2/A3 已折進 A1d.5 per-call floor）。

驗證門檻（ADR-0006，PROVISIONAL）：擬合誤差 **median ≤ 10%、p95 ≤ 20%**（報 median+p95+max）；roofline knee drift ≤ 15%（**僅對有 knee 的曲線**：staircase / prefill / Mali ksweep，**不對 decode GEMV**——見下）；sanity（無 NaN/Inf、latency 為正、**在固定 K 下隨 N 單調**）。

**資料現況（驅動 hold-out 設計；CIM 量測偏 8B）** — `by_group` 各組模型覆蓋：

| group | 覆蓋 | Phase 1 用途 + 驗證紀律 |
|---|---|---|
| `proj_decode` | 1b/3b/8b/qwen（各 4 shape） | **主方程式**（decode GEMV）。**跨模型 hold-out：fit 1b+3b、predict 8b+qwen**（qwen 非-2048 維：M1 以 **padded-tile K·N** 預測 latency → **不需還原**；~1.24× 僅 GOP/s 報告偏差、非 latency 誤差；若仍超門檻則排除、僅描述）。 |
| `lmhead_tile` | 1b/3b/8b/qwen（N=4096 canonical tile, tiles=2） | lm_head **canonical-tile** 值；真實 N≈128k/152k **無量測** → `n_tiles × canonical_tile` 解析合成。跨模型 hold-out 同上。 |
| `proj_prefill` | **8b only**；**僅 M∈{128,256} 有值（皆 tiled_extrapolated）；M≥512 = `canonical_tile_fail`（無延遲）** | 僅 2 個 M 點 → **不設 median/p95 gate**，只做 2-點斜率 sanity；M≥512 + LongBench M≈11.8k = **解析外推、無 ground truth → 明標 unvalidated**。 |
| `staircase64` | 8b only（N 64→3072） | channel-64 階梯模型。within-8B leave-some-out：fit N∈{64…1536}、predict {2048,3072}；verify risers 落 64 倍數。 |
| `staircase_off64` | 8b only（3 pts） | off-64 probe → **sanity 確認 64-量化**（描述性，非擬合）。 |
| `aspect` | 8b only（3 pts） | 等-MAC aspect 敏感（findings: square ~10% penalty）→ 點太少不 hold-out，**記為觀察/參數**。 |
| `attn_floor` | 8b only（6 pts） | CIM attention conv-proxy floor（QK^T+S·V 單-head）= **下界（P1）**，餵 C4 composed；**非擬合方程式**（composed/查值）。 |
| `l2_ddr` | 1b+8b | findings: l2/ddr ratio 1.00–1.01（Alpha 無 on-card DRAM）→ **sanity 確認「無 residency 效應」，不建殘留 gap 模型**。 |

**op-category 覆蓋（對齊 Phase 0.1 的 9 類；補上 kv_cache/embedding 缺口）：**

| op category | Phase 1 timing model | 單元 | 狀態 |
|---|---|---|---|
| matmul（QKV/O/FFN/lm_head） | M1 decode GEMV + staircase + lmhead_tile + proj_prefill；M4 Mali（offload ref） | CIM(+GPU) | ✓ decode 強；prefill M≥512 unvalidated |
| attention bmm（QK^T/S·V） | M4 Mali native bmm（offload）；C4 CIM penalty（佐證） | GPU | ⚠ decode 單-head OK；**prefill scaling 僅 1 點(M=512) → 非-方程式 region** |
| softmax | M4 CPU 3-pt linear-in-kv | CPU | ⚠ decode(1×kv) OK；**prefill S×S 形狀未涵蓋 → 非-方程式 region** |
| norm（RMSNorm） | M4 CPU 常數 | CPU | ⚠ decode；**prefill ×S 解析、unvalidated**（cpu_ops 無 length 軸） |
| ffn（SwiGLU/SiLU） | M4 CPU 常數 | CPU | ⚠ decode；**prefill ×S 解析、unvalidated** |
| rope | M4 CPU 常數 | CPU | ⚠ decode；**prefill ×S 解析、unvalidated** |
| residual | M4 CPU 常數 | CPU | ⚠ decode；**prefill ×S 解析、unvalidated** |
| **kv_cache（KV-append）** | **M2 解析 `kv_bytes/BW_eff`（步驟 2c）** | **M2 記憶體** | **解析、unvalidated**（Phase 0.3 未隔離量測；LongBench decode count-weighted 佔 **12.6–33.5% bytes，8B 22.2%、3B 最高 33.5%**） |
| **embedding（gather）** | **host gather：decode≈0 折進 overhead；prefill ~192MB gather 解析（步驟 9）** | **host/M5** | **顯式處置、unvalidated**（Phase 0.3 不 micro-bench） |
| **conversion（quant/dequant，精度邊界）** | **無**（scheduler 插入、不在 HF trace 9 類）；ADR-0004 指定 Phase 0.2 校準**未做** | CPU/NPU | **❌ headline gap → M6 合約列 tunable + 量測 gap（繫 ADR-0004）** |

端到端 ground truth：`vendor_llm_int8` = **llama 1b/3b/8b ×{1c,4c}，ctx1024**，每筆含 `tok_s_median`（decode）**＋ `ttft_s_median`（prefill anchor，1B/3B/8B = 0.55/1.13/3.79s 隨 size scaling；`prefill_ms_median` 為 degenerate 欄位 ~0.007s 跨模型不變 → 不用）**；**無 Qwen、無 per-task**。故 **decode hold-out = 8B ctx1024 點**（與 C5 同；fit 1b/3b→predict 8b）；**prefill anchor 存在，但 CIM-prefill 輸入無量測 → prefill recompose 為 best-effort、unvalidated、不 gate（步驟 9）**。

> **校準範圍宣告（重要）：Phase 1 = decode-calibrated model；整條 prefill 路徑皆解析、unvalidated。** 具體：proj_prefill M≥512（裝置失敗）、CPU support op 的 prefill ×S scaling（cpu_ops 無 length 軸）、softmax S×S（只量 decode 1×kv）、attention bmm 的 S-scaling（Mali 僅 1 點 M=512）。**唯一 prefill e2e sanity = vendor `ttft_s_median`（ungated；`prefill_ms_median` degenerate、不用）。** LongBench(P≈11.8k) prefill 記憶體 = 65% attention + 26% softmax，正落在此最薄 regime → Phase 2 跑 prefill-heavy workload 須知此為 unvalidated 外推。

方程式↔解析/查值切換：某 region held-out 誤差超門檻 → 退 lookup/解析。**已知非-方程式 region**（不指望封閉式涵蓋，且部分無直接量測）：lm_head N≈128k/152k（解析 tile 合成，無量測）、prefill **M≥512** 含 LongBench M≈11.8k（M≥512 裝置配置失敗，解析外推、unvalidated）、Qwen 非-2048 維（GOP/s 報告偏差 ~1.24×；latency 由 padded-tile 預測、超門檻則排除）、**prefill attention scaling（Mali 僅 1 點 M=512、無 scaling fit）**、**prefill softmax（S×S 形狀、CPU 僅量 decode 式 1×kv）**、**kv_cache append（Phase 0.3 未隔離量測、解析）**。每項在 findings 明寫狀態。

每步皆 **純軟體、這台 Mac、只吃 committed measurements**，不上板。

**報告與圖（工具）：** Phase 1 交付物 = **報告**，**先 HTML 後 PDF**。HTML 版面/UI-UX 用 `frontend-design` skill；**圖一律用 `/nature-figure` 模板**（Python/matplotlib backend，對齊既有 `tools/plotting/`，submission-grade、向量輸出 SVG+PDF）；HTML→PDF 用 headless-Chromium print-to-pdf（或 `aviz85/claude-skills-library@html-to-pdf`）。所有圖 = build artifact，每張一支 script、只吃 committed JSON、可一鍵重產（沿用 OVERALL.md「資料與圖可重現原則」）。圖規格（含 x-y 軸）見 §2 步驟 10。

## 0. Branch + scaffolding

0.1 建目錄：`simulator/models/`（+ `params/`、`__init__.py`）、`validation/contracts/`、`validation/reports/`。→ verify：四目錄存在；`git status` 見新目錄。

## 1a. Micro-benchmark → 方程式擬合

1. **M1 CIM tile** — `tools/analysis/fit_m1_cim.py` → `simulator/models/m1_cim_tile.py` + `params/m1_cim.json`：
   - 主方程式 = **decode GEMV dev latency**：由 `proj_decode` 擬合 `dev_lat_us(M=1,K,N)`，N 對齊 channel-64（ceil 64）；**不用 roofline `max()`**（decode 下 compute、mem 皆 ∝K·N、knee 不可分）——改用 staircase64 擬合的「線性-in-N（每 64 一階）+ tiling」量化模型。
   - tiling：`K·N > 6M → n_tiles = ⌈K/2048⌉·⌈N/2048⌉`、`dev_lat = n_tiles · canonical_tile_lat`（lm_head / large-M 同此式）。
   - hold-out 按上表：proj_decode/lmhead_tile 跨模型（fit 1b+3b → 8b+qwen，qwen 以 padded-tile 預測、不需還原）；staircase 為 within-8B leave-some-out。各 region 記 median/p95/max。
   - → verify：(a) 8b proj_decode median ≤10%、p95 ≤20%（qwen 以 padded-tile 預測，超門檻則排除）；(b) staircase held-out N∈{2048,3072} 預測誤差 ≤門檻且 risers 落 64 倍數，knee drift ≤15%；(c) latency 在固定 K 下隨 N 單調、為正；(d) lm_head（N≈128k/152k）、prefill **M≥512** 含 LongBench、large-M tiling 標為解析合成（無量測、unvalidated）；prefill 僅 M∈{128,256} 2 點做斜率 sanity、**不設 median/p95 gate**；(e) **narrow-N kv-proj 殘差單獨報告**（GQA underfill 與 N 大小相關：1B N=512≈8×64 僅 113 GOP/s vs 寬 204、純 K·N 預測軸偏高 ~1.8×；但 N≥1024≈16×64 已填滿——8B kv-proj K4096×N1024 達 227 GOP/s（≥寬）、underfill 可忽略，故 8B hold-out 較不受此影響）→ 仍列為 CIM-centric 發現（A1d.6 crossbar underfill）、不被平均誤差蓋掉。
2. **M2 記憶體/PCIe**：
   - 2a PCIe/DMA — `simulator/models/m2_memory.py` + `params/m2_pcie.json`：採 `transfer_us = fixed_overhead + bytes/BW`，**`fixed_overhead = 911µs (p95 1112)` 取自 `pcie_floor_A1d5`、`BW = 3.9 GB/s` 取文件化 Gen3×4 值，皆為固定參數**（無 per-shape PCIe sweep → 不重擬 slope）。**911µs floor 的適用邊界（解 step 9 矛盾）：floor 只對「離散 host↔device transfer」收費（KV-reload、activation handoff、conversion-op 流量）；（此為 recompose/production 預測脈絡）decode weight-streaming 主幹用 `BW_eff`、不逐 call 付 floor（注意：Alpha 板上 decode-GEMV 仍逐 call 付 floor，見 step 9 兩平台區分）。此邊界寫進 m2_pcie.json 註記。** → verify：m2_pcie.json 記錄 floor=911µs、BW=3.9 GB/s、floor 適用邊界、及「無 slope refit（Phase 0.3 資料 gap）」註記；`transfer_us` 對任一 bytes 為正且隨 bytes 單調。
   - 2b LPDDR5 後端（**解析模型為 Phase 1 預設交付；Ramulator2 整合延 Phase 2**，ADR-0002 swappable）— `params/m2_lpddr5.json` + wrapper：以 JEDEC LPDDR5 + 量測 decode wall 參數化有效 BW/latency。→ verify：有效 BW ∈ [JEDEC peak×0.4, JEDEC peak]（24.2/51.2≈47%，typical memory-wall 有效率；bound 由 0.5 放寬至 0.4 並記 rationale）且 bracket ~24 GB/s decode wall（來源：ADR-0006 / phase0.3-findings，於檔內 cite）；findings 明記 Ramulator2 deferred + risk #6。
   - 2c kv_cache append（記憶體頻寬 op，Phase 0.3 未隔離量測）— `m2_memory.py` 增 `kv_append_us(kv_bytes) = kv_bytes / BW_eff`（BW_eff 取 2b）；**解析、unvalidated**（無 ground truth）。op_profile 證 LongBench decode kv_cache count-weighted 佔 12.6–33.5% bytes（8B 22.2%）→ 不可省。→ verify：`kv_append_us` 隨 kv_bytes 線性增、為正；findings 標 unvalidated。
3. **M4 GPU（Mali）** — `tools/analysis/fit_m4_gpu.py` → `simulator/models/m4_gpu.py` + `params/m4_gpu.json`：由 `mali_matmul.results` 擬合 GEMM；proj_decode 形狀跨模型驗（fit 1b+3b → 8b+qwen）；native attention bmm latency（單-head ~40–260µs f16；一個 M=512 prefill 案例 ~3.3ms；取自 mali_matmul.json，校正 findings-doc 的 80–500µs 區間）存為 offload 參考。**絕對吞吐 = 下界**（kernel 未優化），故 GPU matmul 只驗 shape-trend，不押絕對值、不設 knee（ksweep M≥128 後 ~20 GFLOP/s 飽和、無清晰 knee）。→ verify：fitted 曲線復現 proj_decode 形狀趨勢（相對誤差記錄）且絕對值標下界；ksweep 飽和點（M≥128）記錄；attention bmm latency 存在。
4. **M4 CPU（A76）** — `tools/analysis/fit_m4_cpu.py` → `simulator/models/m4_cpu.py` + `params/m4_cpu.json`：用 `cpu_ops.ops` 量測 latency（**非解析 FLOPs**）。**按資料**：softmax = 3-pt 線性-in-kv 擬合（kv∈{128,512,1024}）；rmsnorm/rope_apply/residual/swiglu/sampling = per-(model,dtype) **常數**（無 within-op sweep；跨模型「∝hidden」僅記為觀察、非擬合定律）。dtype 採 **fp16（假設為 numpy 模擬 → 視為上界；provenance = Phase-0.3 collection note，非 cpu_ops.json 欄位）**，fp32 留參考。→ verify：每 op 有模型；softmax median ≤10%、p95 ≤20% 對 kv；其餘 op 取值=量測常數、帶 fp16-上界旗標。
5. **M4 NPU（佔位）** — `simulator/models/m4_npu.py`（`raise NotImplementedError`，docstring → #13 + 待擬合 `rknpu2_matmul.json`）+ `validation/contracts/m4_npu.yaml`（標 blocked-on #13）。→ verify：stub + 合約存在且明標相依 #13。

## 1b. 非 micro-benchmark component 驗證

6. **M5 workload/trace 驗證** — `validation/validate_m5_trace.py`：**重用 Phase 0.1 inventory oracle**（不自行 ×layers），對四模型確認 trace 生成器逐-sig (op,in_shapes,out_shape) count **逐筆等於** `op_profile` 的 count。報告落 `validation/reports/m5.json`。→ verify：四模型各 **0 筆 count 不符**；報告列每模型 sig 數 + mismatch=0。
7. **M7 energy 驗證** — `simulator/models/m7_energy.py` + `validation/validate_m7_energy.py`：規格估算（Metis 15 TOPS/W × util；A76 datasheet W × activity；memory JEDEC/access；PCIe/byte，ADR-0005）；板無功耗儀表 → 無 silicon ground truth → 驗 sanity + **±20% 敏感度**。報告落 `validation/reports/m7.json`。→ verify：energy 為正且隨 activity 單調；估得 J/token ∈ [0.5×, 2×] 規格推導界（規格常數於報告列名）；±20% 敏感度界限報出。
8. **M3 / M6 合約 + 待調參數（不做行為驗證）** — 寫 `validation/contracts/m3.yaml`、`m6.yaml`：列 `measurement_sources`、`acceptance_criteria`、`tunable_params`、Phase-2 註記。**M6 合約必含 `tunable_params`：精度邊界 quant/dequant conversion-op 成本，並明標『量測 gap：ADR-0004 指定的 Phase-0.2 校準未執行（measurements 無 quant op）』——否則 headline 混合精度貢獻跑在無成本基礎的 op 上。M3 合約必含 `tunable_params`：bandwidth-contention knee（ADR-0001 義務：重現 ~60 GB/s 飽和拐點）。** → verify：兩合約 acceptance_criteria 引用 ADR-0006 Phase-2 系統門檻（e2e tok/s ≤15%、contention knee ≤15%），非 per-op median/p95；**m6 列 conversion-op cost tunable + ADR-0004 gap；m3 列 contention-knee tunable**；含「行為驗證 = Phase 2」註記。

## 2. 端到端重新合成（capstone；按資料現況重界）

9. **重新合成 L1→L4** — `tools/analysis/recompose_e2e.py`：
   - **decode 主幹 = weight-streaming bandwidth（沿 C5）**：`tok_s = BW_eff / weight_bytes_per_token`，`BW_eff` **fit on 1b+3b、predict 8b**（8b 不參與擬合 → 非循環）。**`weight_bytes` 改由 op_profile 逐-sig bytes 求和**（取代 C5 的封閉式聚合）→ 此為 Phase 1 對 C5 的精化。
   - **Alpha 911µs per-call floor 不進此 production-card decode 預測**（production = on-card DRAM、非 PCIe；拓樸 artifact，findings 明禁外推）。**注意**：Phase-2 模擬的 CIM-on-PCIe 目標平台中，離散 host↔device transfer 仍付此 floor（見 2a 邊界）——兩者是不同平台脈絡，非矛盾。
   - **加性非-streaming 項**：CPU 支援（M4 CPU per-token：rmsnorm/softmax/rope/swiglu/sampling）+ **attention 走 GPU-offload（M4 Mali native bmm）** + **kv_cache append（M2 2c：`kv_bytes/BW_eff`，隨 kv 成長）** + per-step overhead（取 B1/A6）。
   - **embedding gather 處置（明寫，非沉默洞）**：decode ≈0（gather 1 列）→ 折進 per-step overhead；prefill 11.8k 列 gather ~192MB → 記入 prefill 項（解析、unvalidated）。
   - **CIM-attention penalty（C4，31–46 ms/token）單獨報告**為「為何 offload attention」的量化依據，**不入 t_step**。
   - **prefill TTFT recompose（best-effort，不 gate）**：用 vendor `ttft_s_median`（ctx1024；`prefill_ms_median` degenerate、不用）anchor 比對一版 prefill 重新合成；但 CIM-prefill 輸入（proj_prefill M≥512、prefill-attention scaling、prefill-softmax S×S）皆無量測 → **報告為 unvalidated、不設 pass/fail，列 Phase-2 gap**。
   - 輸出 `measurements/metis_card/twopillar_prediction_fitted.json`。
   - → verify：**decode** 8b hold-out `|pred−meas|/meas ≤ 0.25`（對齊 C5 9.6% bar；唯一 gate）；報告標明 decode 主幹用 BW、attention 用 GPU-offload、kv_cache 解析項、embedding 處置、CIM-penalty 僅作佐證；prefill TTFT 數字標 unvalidated/不 gate；註記無 Qwen/per-task e2e。
10. **圖（build artifact，`tools/plotting/phase1_figs.py`，`/nature-figure` 模板，Python）** — 只吃 committed JSON。**規劃 7 張圖（x-y 軸明列）**：

    | 圖 | 內容 | x 軸 | y 軸 | series / 標註 |
    |---|---|---|---|---|
    | P1 | M1 CIM channel-64 階梯（measured vs fitted） | 輸出通道 N（64→3072，線性） | dev latency（µs） | measured staircase64（8B）+ fitted 階梯 + off64 probe；標 64-倍數 risers、held-out N∈{2048,3072} |
    | P2 | M1 CIM decode GEMV 擬合（跨模型） | 權重大小 K·N（params，log） | dev latency（µs，log） | proj_decode 點依模型上色（1b/3b/8b/qwen）+ fitted；標 held-out 8b+qwen（qwen 以 padded-tile 預測） |
    | P3 | M1 擬合誤差分佈 | per-op 相對誤差（%） | 累積比例（CDF） | 垂直線標 median 門檻 10% / p95 門檻 20% |
    | P4 | M4 Mali GEMM 吞吐 vs size（HeteroInfer Fig 1 風） | 方陣維度 M（64→1024） | 吞吐（GFLOP/s） | f16 + f32；標 M≥128 飽和（~20）、「絕對值=下界（kernel 未優化）」 |
    | P5 | M4 CPU 非-GEMM | (a) kv 長度（128/512/1024）；(b) op 名 | latency（µs） | (a) softmax measured + 線性擬合；(b) rmsnorm/rope/residual/swiglu/sampling bar，標 fp16 上界 |
    | P6 | 端到端重新合成 hold-out（C5 refit） | 模型（1B/3B/8B） | decode tok/s（1-core） | measured vs predicted；8B=held-out；±25% band；標各模型 implied BW（GB/s） |
    | P7 | CIM attention penalty vs GPU offload（「為何 offload」） | KV 長度（129/513/1025） | per-token attention latency（µs，log） | CIM composed（C4，含 KV reload 31–46ms）vs Mali GPU-native（40–260µs）；標 ~2 量級（96–370×） |

    → verify：7 張圖各可從 committed JSON 一鍵重產（向量 SVG+PDF）；每圖底層資料齊、x-y 軸與單位如上表。

## 3. 合約 + findings + PR

11. **驗證合約** — `validation/contracts/{m1,m2,m4_gpu,m4_cpu,m5,m7}.yaml`（+ 步驟 5/8 的 m4_npu/m3/m6）：每檔 `module`、`measurement_sources`、`acceptance_criteria`（ADR-0006）、`sample_strategy`（**CPU 取 `cpu_ops` 的 `cov`；其餘標 `TBD — variance_profile 未收集（Phase 0.3 gap）`**）。**m2 合約涵蓋 PCIe(2a)/LPDDR5(2b)/kv_cache(2c)，並顯式記死決定『無 SRAM L1/L2 residency 模型（Alpha 無 on-card DRAM，l2/ddr ratio≈1.0）』供 Phase 2 不誤建。** → verify：每模組一份合約，acceptance_criteria 含對應門檻；sample_strategy 來源真實或明標 gap；m2 合約含 no-SRAM-residency 決定。
12. **findings（內容來源）** — `docs/phase1.1-findings.md`：逐模組方程式形式 + 參數 + 誤差（median/p95/max）+ **op-category 覆蓋矩陣（9 類 + conversion）** + **「Phase 1 = decode-calibrated；prefill 全路徑 unvalidated」校準範圍宣告** + 非-方程式 region 清單（含「無量測/unvalidated」標註：lm_head、prefill M≥512、prefill CPU-support ×S、prefill-attention scaling、prefill-softmax S×S、kv_cache、embedding、conversion-op/ADR-0004 gap、prefill-TTFT anchor 未驗）+ **GQA-narrow kv-proj 殘差（CIM-centric 發現）** + **911µs floor 適用邊界** + 重新合成結果 + NPU/#13 gap + Ramulator2 deferred。→ verify：每模組節列 **方程式形式 + ≥1 參數 + {median,p95,max} 誤差數字**；矩陣各 op 有狀態；decode-calibrated 宣告與非-方程式 region 逐項標狀態。
13. **報告（HTML→PDF）** — 用 `frontend-design` 把步驟 12 的 findings + 步驟 10 的 7 張圖（內嵌向量）組成 `docs/report/phase1.1/index.html`（章節：每模組方程式+誤差、非-方程式 region、端到端 hold-out、gaps/deferred）；再 headless-Chromium print → `docs/report/phase1.1/phase1.1-report.pdf`。→ verify：HTML 開得起來且內嵌 7 圖；PDF 產出且非空、頁數 >0。
14. **secret-scan + commit + PR** — `grep -rnI "hf_[A-Za-z0-9]\{20\}"` 確認乾淨；commit `plans/phase-1.1.md` + `simulator/` + `validation/` + `tools/` + `docs/phase1.1-findings.md` + `docs/report/phase1.1/` + `docs/figures/phase1.1/`（不動無關 `papers/` 變更）；`gh pr create` `phase-1`→`main`。→ verify：grep 無命中；PR 開出。

## Outputs
- `simulator/models/{m1_cim_tile, m2_memory, m4_gpu, m4_cpu, m4_npu(stub), m7_energy}.py` + `params/{m1_cim, m2_pcie, m2_lpddr5, m4_gpu, m4_cpu}.json`
- `tools/analysis/{fit_m1_cim, fit_m4_gpu, fit_m4_cpu, recompose_e2e}.py`、`tools/plotting/phase1_figs.py`
- `validation/contracts/{m1,m2,m3,m4_gpu,m4_cpu,m4_npu,m5,m6,m7}.yaml`、`validation/{validate_m5_trace, validate_m7_energy}.py`、`validation/reports/{m1,m2,m4_gpu,m4_cpu,m5,m7}.json`
- `measurements/metis_card/twopillar_prediction_fitted.json`、`docs/phase1.1-findings.md`
- **報告：** `docs/figures/phase1.1/{P1…P7}.{svg,pdf}`（`/nature-figure`，7 圖 x-y 軸見步驟 10）、`docs/report/phase1.1/index.html`（`frontend-design`）、`docs/report/phase1.1/phase1.1-report.pdf`
