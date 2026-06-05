# CIM-Card — CIM 計算 kernel 重驗（同一顆 AIPU，未凍結）

## 為什麼 CIM kernel 不該被當成「凍結」

Phase 1.1 的 CIM 計算引擎（M1）校準在 **Metis Alpha 的 13 個 native single-tile 量測點**（`simulator/models/params/m1_cim.json`，2D `G_eff(N,K)` 擬合，2.7%）。Alpha 是 pre-production 板、**跑不了 LLM**（封閉韌體 `-1301` 牆 + 無 on-card DRAM），所以當時無法在 compute-bound regime 多取點。

但**同一顆 quad-core AIPU 在量產 Metis Card 上是活的**（`machines.md`：`metis-0:7:0 16 GiB clock=800MHz`），而且 `axrunmodel` 的 `dev_fps` 是**隔離計算**的指標（dev/system split，不是合成差分）。**兩板都 800 MHz** → Card 的 `dev_lat`/`dev_gflops` 可以**直接對** Alpha 13 點比，**不需 clock 正規化**。

所以 CIM kernel **不是凍結的**：我們可以在 Card 上用同一個 1×1-conv matmul proxy **重新量測/交叉驗證**，並補上 Alpha 量不到的 **prefill / compute-bound** 形狀。這把「Alpha 凍結」的疑慮解除，也補了 decode 量不到的計算上界。

## 方法（已就緒、ready-to-run）

移植 Alpha 的 `characterization/aetina/run_metis_cim.py` 到 Card → `characterization/metis_card/run_metis_cim_v16.py`：

- **編譯首選低階 `compile`**（與 Alpha `run_metis_cim.py:65` 同路徑，1:1）；`compile` 在 v1.6 被移除才退 `deploy.py --mode QUANTCOMPILE`。`voyager-sdk.md:339` 記「general MatMul 非 deploy.py 支援 op（YOLO11 whitelist）」→ 所以 deploy.py 只是 best-effort 退路。
- `--spike` 模式（~30 分）先答可行性：低階 `compile` 在不在 + 1×1-conv proxy 編不編得出 model.json。
- 交叉驗證腳本 `tools/analysis/validate_cim_card.py`：Card 的 13 個 alpha-shape `dev_gflops` 對 Alpha 13 點算 `median/p95 |rel_diff|`（同顆 AIPU、800 MHz、無 rescale），另列 prefill/compute-bound 新點。

## 本期狀態：`DEFERRED_FALLBACK`（誠實標註）

本 session **無法上機**：harness 自動把「SSH 進共用 Metis 板」判為未授權動作擋下（量測節點是共用實驗室硬體，需使用者明確授權；當時使用者離線）。**我們不繞過這個 denial。**

這正是 plan 設計好的 fallback 情境（plan step 18 註 + handoff §5：低階 compile 不在／MatMul 不支援／板不可達 → 退「**Alpha 13 點 calibrated（非凍結、待板）** + Card e2e 驗 memory-wall」並**回報 user**、不靜默改路徑）。所以：

- **CIM 計算維持 `calibrated`（Alpha 13 點）**——這是 Phase 1.1 已有的、量測級可信的校準，**不受影響**。
- **Card 重驗延後**，狀態寫進 `validation/reports/phase1.2/cim_card_revalidate.json`（`status: DEFERRED_FALLBACK`），含解鎖步驟。
- 移植腳本 + validator **已寫好並語法/執行驗證**（validator 跑得出 fallback 報告），**只差授權**。

> **解鎖**：給 SSH 權限（settings 加一條 `ssh metiscard` 的 Bash 規則）後：`rsync` 腳本上 Card → 跑 `run_metis_cim_v16.py --spike`（先看可行性）→ full → `rsync` 拉回 → 重跑 `validate_cim_card.py`。報告會自動從 `DEFERRED_FALLBACK` 變成 `CARD_REVALIDATED`（含 13 點一致性 + prefill 新點）。

## 一句話

CIM 計算的可信度**沒有降級**：仍是 Phase 1.1 的 Alpha-13-點量測校準。Card 重驗是**升級路徑**（解凍 + 補 compute-bound），腳本就緒、待一個授權；在那之前，誠實標成 `DEFERRED_FALLBACK`、回報使用者。
