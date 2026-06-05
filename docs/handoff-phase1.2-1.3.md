# 🤝 HANDOFF — Phase 1.2 + 1.3 執行交接

> 給下一個 session。這份文件讓你不必重跑前面的設計討論就能直接執行。**先讀 `CLAUDE.md` → `OVERALL.md` → `CONTEXT.md`（含 `## Repo index`）→ 本文。** 計畫已批准、review 收斂、可開始執行。

---

## 0. 一句話現狀

兩份 plan（`plans/phase-1.2.md` Wave 1、`plans/phase-1.3.md` Wave 2）已經過 **grill-me + 4-subagent 參數稽核 + 2 輪 subagent plan-review（收斂無 blocking）**，使用者**已批准**。現在在 `phase-1.2` 分支，等「開始執行」。使用者要求**全部一起做**（1.2 → 1.3 + CIM-Card 量測），用 **Workflow 平行編排**。

---

## 1. 東西在哪

| | |
|---|---|
| 分支 | **`phase-1.2`**（在這做）。`main` 有 phase 1.1 + repo index。只剩這兩條分支（其餘已 merge 清掉）。 |
| 執行計畫 | **`plans/phase-1.2.md`**（Wave 1，逐步驟 action+verify）、**`plans/phase-1.3.md`**（Wave 2）。**照 plan 執行，不要重新設計。** |
| 導航 | `CONTEXT.md` § Repo index（路徑→內容地圖）、glossary。 |
| 既有量測（校準來源） | `measurements/aetina/*`（CIM/CPU/GPU silicon）、`measurements/metis_card/*`（LLM e2e L4 anchor）。 |
| 既有模型（1.1） | `simulator/models/m*.py` + `params/*.json`。1.2 是擴/換成 spec-based。 |

---

## 2. 怎麼執行（結構）

**Wave 1（phase-1.2，平行 fan-out）：**
1. **序列前置**：`simulator/specs/`（spec 檔 + loader）+ **共用引擎介面** `simulator/models/engine.py`（`UnitEngine` ABC + `Workload` dataclass + 凍結 return keys `{latency_us, bound, provenance}`）+ `tests/test_engine_iface.py` **conformance test（fan-out 前先跑過）**。
2. **平行 fan-out（各一 subagent）**：WP-CPU ∥ WP-NPU(analytic) ∥ WP-MEM(analytic+SRAM) ∥ WP-GPU(roofline) + **CIM-Card 量測**（見 §5）。每個 WP 交付：引擎 + spec + 擬合/驗證 script + report JSON + 圖 + 章節草稿。
3. **序列合併**：cross-check（`check_phase1_2.py`）+ contracts + 報告（HTML→PDF）+ findings + OVERALL/LOG + **PR → 通知 user → 等 user merge**。

**Wave 2（phase-1.3，1.2 merge 後）：** ONNXim(NPU heavy) ∥ Ramulator2(memory heavy)，插進 1.2 就緒的 `engine=` 介面。

> 用 **Workflow 工具**做平行編排（serial 前置 → parallel WP → merge）。每個平行 subagent 的 brief = plan 裡該 WP 的步驟清單。**WP 之間獨立**（只依賴序列前置的 spec+介面）。

---

## 3. 鎖定的設計決定（grill 收斂，不要推翻）

1. **架構 = 引擎 + 可換 spec 模組**。每個非-micro-benchmark 單元 = 模型引擎 + `simulator/specs/*.json`（換型號=換 spec）。
2. **共用介面** = `Engine(spec, engine='analytic'|…).predict(workload) -> {latency_us, bound, provenance}`（**spec 建構時綁、predict 只吃 workload**）。輕/重同簽名。
3. **輕/重二元只存在於記憶體 + NPU**（analytic ↔ Ramulator2 / ONNXim）。CPU/GPU/SRAM/CIM 各一引擎（CACTI 只是填 SRAM 參數、非 runtime 引擎）。**兩個都建、`engine=` 選、不分主次**。
4. **1.2/Phase 2 界線**：1.2 = 單一元件；**整合/競爭/排程/轉換-op/e2e = Phase 2**。
5. **1.2（Wave 1 analytic 核心）/ 1.3（Wave 2 重型 sim）拆分**：1.2 自給自足、可驗證；1.3 把 C++ 重型 sim 插進就緒介面、風險隔離。
6. **CIM 計算 kernel 非凍結**：Metis Card 同一顆 AIPU 活著、可重驗（見 §5）。
7. **誠實標註紀律**：`calibrated`（CPU 對 fp32 cpu_ops.json）/ `simulated`（NPU/GPU-roofline/SRAM）/ `assumption` / `borrowed`。**no fake gate** —— 沒 silicon 就不假裝有數值 gate。

---

## 4. ⚠️ 關鍵注意事項（稽核 + review 抓出來的，務必遵守）

**CPU：**
- **decode 的 CPU op 工作集都駐 cache（L1/L2/L3）、碰不到 LPDDR** → memory 分支用 **cache BW**（不是 LPDDR）；「換 LPDDR4→5 重算」**只對 prefill**。
- 成本主因是 **`exp()` 超越函數**（softmax/swiglu），不是 reduction/elementwise 二分。
- rmsnorm/rope/residual **固定開銷主導**（近常數）；只有 swiglu∝F、sampling∝V 乾淨。需 per-op `overhead` 項。
- size 變數：rope=**heads·hd**（+stack ≥2-pass）、softmax=**heads·(kv+1)**、swiglu=**F**、sampling=**V**。
- **量測基準 = 單 A76 核、單緒 numpy、fp32**（fp16 是 numpy 模擬上界；swiglu fp16 是混精）。多核從 1 核外推、A55 **IPC=1**（單 NEON pipe）、A55+多核**無量測=simulated**。
- cache BW 放進 `cpu_rk3588` spec、標 `assumption`（ARM A76 TRM）、**註「≠ Metis SRAM、WP-CPU 不依 WP-MEM」**。

**GPU：**
- 1.1 的 `20.12 GFLOP/s` 是 **FP16、不是 INT8**；**INT8 GPU GEMM 零資料**（kernel 只有 FP32/FP16）。
- Mali-G610 峰值 ~512 GFLOP/s = **assumption（可能低估 2-4×）**，要查證。
- roofline = **形狀趨勢 + 絕對值下界**（5 點不足以可轉移校準）；Mali 仍以 micro-benchmark 為主。

**NPU：**
- **無 RKNPU2 silicon**（板離線 #13）→ analytic 用 datasheet（6 TOPS）+ **borrowed Hexagon trend**（systolic 維借 32×32 標 borrowed）。
- dtypes 只有 **INT4/8/16/FP16**（無 BF16/TF32）。
- `m4_npu.yaml` BLOCKED→**SIMULATED**：明寫 trend-shape 驗收（staircase 單調+knee 落 32×32、order/shape ≤6×、BW frac 59–66% **分母 68**）、**無 per-op gate**；**#13(silicon) 標 superseded-not-satisfied、≠ ONNXim**。

**記憶體 / SRAM：**
- 記憶體峰值 **34.1/4224 標 `assumption`**（in-repo 無 data-rate 出處），非 jedec/measured。
- SRAM（Metis L1/L2 SPM）= **CACTI tier、不是 Ramulator2**；residency `architecture-only`（8B 權重 ≫ 32MiB → 永不命中）。
- 24.2 GB/s（量產卡 on-card LPDDR4x）和 RK3588 host LPDDR4 是**兩顆不同記憶體**，別混。

**CIM：**
- `alloc_envelope_param_count`（6M 參數量）**≠** `native_max_kn`（4.19M K·N 面積）。
- envelope「~14MB」是**假設**（vs 1GB BAR），非量測。

**Ramulator2（1.3）：** 只有 **LPDDR5**、**無 LPDDR4**（=`assumption`，build 時看 `src/dram/impl/` 確認）；別忘 reconcile `ADR-0002` / `m2_memory.py` docstring / `phase1.1-findings.md`（:19/:74/:130）/ `A2-m2-memory.md`（:90/:124）裡「Ramulator2→Phase 2」的舊字句。

---

## 5. 🔌 CIM-Card 量測任務（特殊，要 SSH 上機）

**目的**：在 Metis Card（同一顆 AIPU）重新量測/驗證 CIM 計算 kernel —— 把 Alpha 凍結的疑慮解除，並補 decode 量不到的 compute-bound regime。

**連線（已驗證可行）：**
```
ssh metiscard          # = tundergod@140.112.28.104，key auth，無密碼
# SDK: ~/tundergod/voyager-sdk ; source axelera-env/bin/activate
# 已確認：AIPU 活（16GiB 卡、4 核 @800MHz）、axrunmodel 可用、onnx/deploy.py 在
```

**方法**：移植 Alpha 的 `characterization/aetina/run_metis_cim.py`（1×1-conv proxy → matmul → axrunmodel → `dev:X fps`）到 Card。

**先跑 SPIKE（~30 分，決定可行性）：**
- 探 v1.6 是否仍有 Alpha 用的**低階 `compile`/`axcompile`**（Alpha `run_metis_cim.py:65` 用 `compile`，**不是** `deploy.py`）。
- **`deploy.py` 撞硬阻**：`docs/voyager-sdk.md:339`「general MatMul 非 deploy.py 支援 op（YOLO11 whitelist）」→ 首選低階 `compile`，deploy.py 才退。
- 確認 1×1-conv proxy 編得出 model.json。

**注意：**
- **兩板都 800MHz** → 直接比 `dev_lat`/`dev_gflops`、**無需 clock 正規化**（Card DVFS 非 800 則先固定）。
- `axrunmodel` 的 `dev_fps` **已是隔離計算**的指標（dev/system split），不是合成差分。
- 交叉驗證對象：Alpha 13 點（square=204、wide/tall=227 GOP/s）；補 prefill/compute-bound shape。
- **Fallback**：低階 compile 不在 且 deploy.py 不支援 MatMul / 量化卡關 → 退「Alpha 13 點 calibrated（待板）+ Card e2e 驗 memory-wall」，**回報 user、不靜默改路徑**。

---

## 6. 執行時要現場確認的（已寫進 plan，非 plan 缺陷）

- [ ] Card v1.6 是否還有低階 `compile`（CIM-Card SPIKE）。
- [ ] Ramulator2 是否真無 LPDDR4（看 `src/dram/impl/`）。
- [ ] Card AIPU 是否需固定 clock=800MHz（DVFS）。
- [ ] ONNXim / Ramulator2 的 C++ build 在環境能不能起來（失敗 → documented fallback + 回報 user）。

---

## 7. 工作流與護欄（必守）

- **Per-phase workflow**：執行完 → subagent code-review → 開 PR（`phase-1.2`→main）→ 通知 user → **只有 user 明確確認才 merge**。1.3 同理（且 1.3 前置 gate：`MemoryModel(spec, engine='analytic').predict(wl)` 能跑才動，否則 STOP —— 不對 1.1 stub 建）。
- **Secrets**：絕不 commit HF token（push 前 `grep -rnI "hf_[A-Za-z0-9]\{20\}"`）、board 密碼。
- **不要 commit 無關的 `papers/` 變更**（working tree 一直有 `papers/README.md` + `configuration-wall-*` 的 pre-existing 改動，**全程排除在每個 commit 外**）。
- **圖 = build artifact**：`tools/plotting/` 每張圖一支 script、只吃 committed 數據重產，絕不手繪。
- **Aetina/Alpha 板壞了**：CIM/CPU/GPU/NPU 的 micro-benchmark **不能重量**，只能用既有 committed 數據；**唯一可上機的是 Metis Card**（§5）。
- **action-only plan**：若要改 plan，維持 action+file+verify 格式，理由放 OVERALL/決策區。

---

## 8. 快速參考（spec 數字，出處在 plan 步驟 3）

- **RK3588 CPU**：4×A76 @2.3GHz（IPC 2）+ 4×A55 @1.8GHz（IPC 1）；NEON 128-bit（fp32 W=4/fp16 W=8）；L1 64K/L2 512K-core/L3 3MB shared。
- **Mali-G610**：4 核 Valhall ~1GHz；FP32 ~512 GFLOP/s（assumption）、FP16 ~1 TFLOP/s、INT8 未量。
- **RKNPU2**：3 核 6 TOPS INT8 ~1GHz；INT4/8/16/FP16；systolic 維 borrowed 32×32。
- **記憶體（64-bit 匯流排）**：LPDDR4 3200→25.6 /（板 4224→33.8）、LPDDR4X 4266→34.1 /（4224→33.8）、LPDDR5 6400→51.2 GB/s；eff 0.71（量測）。
- **Metis SRAM**：L1 4MiB/核×4 + L2 32MiB + D-IMC 1MiB/核×4 = 52 MiB；**無公布 BW** → CACTI/assumption。
- **CIM 1.1 擬合**：Gmax≈333.67 GOP/s、Na≈577、Kb≈574、n_cores=4、512×512/核；g_eff(2048,2048)≈203 GOP/s、dev_lat≈41µs。

---

**接手第一步**：`git checkout phase-1.2` → 讀 `plans/phase-1.2.md` → 跑序列前置（spec+介面+conformance test）→ 開 Workflow 平行 fan-out。**照 plan、守 §4 注意事項、CIM-Card 先 spike。** 祝順利。
