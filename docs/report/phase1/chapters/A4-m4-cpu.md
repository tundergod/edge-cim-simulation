# A4 — M4-CPU：ARM A76 CPU（零碎輔助運算的雜工）

> **這一章你會學到**：除了矩陣乘法和 attention，LLM 還有一堆「零碎的小運算」要做，這些丟給 CPU；它們大多是固定成本（直接存量測值），只有 softmax 隨上下文長度變；以及一個關鍵的誠實標註——我們量的是 numpy 模擬的 FP16，所以這些數字是「上界」。

---

## A4.1 架構考量：CPU 在系統裡是誰？

A1 的 CIM 做大矩陣乘法、A3 的 GPU 做 attention。但一個 LLM token 還有一堆**既不是大 matmul、也不是 attention** 的小運算：

- **RMSNorm**（正規化，讓數值穩定）
- **RoPE**（旋轉位置編碼，告訴模型 token 的位置）
- **SwiGLU**（FFN 的激活函數，§0.3 提的 `ffn` 那一類）
- **residual**（殘差相加）
- **softmax**（把 attention 分數變成機率分佈）
- **sampling**（從機率分佈挑出下一個 token，例如 argmax）

這些**零碎、逐元素或小型 reduction** 的運算，丟給 **ARM A76 CPU**（RK3588 的大核，cores 4–7）。在 §0.7 架構裡 CPU 是「**③ 各單元時間模型**」的一格，是最末端的支援層。

M4-CPU 要回答：給一個輔助 op（和它的尺寸），CPU 算它要多久？

---

## A4.2 原理：這些 op 為什麼「大多是固定成本」？

關鍵觀察：對一個**固定的模型**，這些 op 每個 token 的工作量幾乎是固定的：

- **RMSNorm / RoPE / SwiGLU / residual**：它們的尺寸由 hidden size（模型寬度）決定，而 hidden size 對一個模型是**固定**的。所以每次跑都差不多久 → **一個常數**就描述得了。
- **softmax** 是唯一例外：它要對「目前所有 KV」算機率，**KV 越長、要算越多** → 成本**隨 kv 線性增加**（和 A3 的 attention 同理）。

所以 M4-CPU 的策略很簡單：**softmax 用一條公式，其他用「存下量測值」**。

---

## A4.3 參數設計

**（1）softmax——唯一的真正「公式」：**
```
softmax_us(kv) = a + b · kv          （per 模型、per 精度）
例（8B、FP16）：= 12.34 + 1.553 · kv   →  kv=1024 時約 1603 µs
```
我們在 kv = 128 / 512 / 1024 三個點量測，擬合出每個 (模型, 精度) 的 `a`、`b`。

**（2）其他 5 個 op——「常數查表」，不是擬合：**
這點要講清楚：rmsnorm / rope / residual / swiglu / sampling **沒有做尺寸掃描**（每個只量了一個代表性大小），所以它們**不是擬合出來的公式，而是直接把量到的中位數存起來**（一種 lookup）。例如 8B、FP16：

| op | 量測值（µs） |
|---|---|
| residual | 41.7 |
| rope_apply | 144.4 |
| rmsnorm | 157.2 |
| swiglu | 557.8 |
| sampling_argmax | 985.5 |

> **跨模型的「∝ hidden」只是觀察、不是定律**：我們確實看到大模型這些 op 較慢（hidden 較大），但因為每個 op 沒有 within-op 的尺寸掃描，我們**不**宣稱一條 `成本 ∝ hidden` 的擬合律，只把它當觀察記錄。

**（3）兩個重要的誠實標註：**

1. **用量測值，不是用 FLOPs 算**（對應 issue #10）：這些非-GEMM op 的「理論 FLOPs」是個很粗的下界（1 flop/元素），跟實際延遲差很遠。所以我們**堅持用實測延遲**，不用解析 FLOPs。
2. **FP16 是 numpy 模擬的 → 上界（upper bound）**：我們的量測腳本用 numpy 跑 FP16，但 **A76 沒有原生快速的 FP16 numpy 路徑**，numpy 用軟體模擬 → 慢很多。看數據就知道：

   | op（8B） | FP16 µs | FP32 µs | FP16/FP32 |
   |---|---|---|---|
   | rmsnorm | 157.2 | 23.6 | 6.7× |
   | sampling_argmax | 985.5 | 52.5 | 18.8× |

   FP16 慢 FP32 達 7–19 倍,這不合理(半精度應該更快或差不多)——所以這是**模擬造成的、是上界**。真實硬體的 FP16 會更快。我們把 FP16 當「最壞情況」用,並明確標註 provenance(來源是 Phase 0.3 的收集註記,不是 cpu_ops.json 裡的欄位)。

---

## A4.4 Measurement vs Prediction

**softmax(唯一的擬合)**:每個 (模型, 精度) 用 3 個 kv 點擬合一條直線,共 **8 條擬合(4 模型 × 2 精度),每條 3 個點 → 24 個殘差點(gate 的 n=24)**;這些點對各自直線的相對誤差**中位數 0.3%、p95 1.8%、max 3.4%**,輕鬆過門檻(因為「3 點擬一線」幾乎完美)。

**其他 5 個 op**:它們的「預測」就是「量測值本身」(常數查表),所以談不上擬合誤差——這是誠實的:我們沒有為它們建模型,只是把真值存起來,等 Phase 2 真要用某個沒量過的尺寸時再回頭量。

**圖 A4-1(P5)— CPU 輔助運算**
![P5](../../../figures/phase1/P5_cpu_nongemm.png)
- **左圖(P5a)softmax: 量測 vs 公式**。**X 軸**:KV 長度。**Y 軸**:softmax 延遲(µs,FP16)。**圓點 = 量測,線 = 擬合**,每個模型一色。三點落在線上 → 線性成立、擬合準。
- **右圖(P5b)非-GEMM 常數**。**X 軸**:op 名稱。**Y 軸**:延遲(µs,8B、FP16 上界)。長條 = 各 op 的量測常數。可看出 sampling_argmax 最貴(~986µs)、residual 最便宜(~42µs)。

---

## A4.5 限制與 gap(誠實清單)

| 項目 | 狀態 | 說明 |
|---|---|---|
| softmax 公式 | ✅ 已擬合 | `a + b·kv`,誤差中位 0.3% |
| 其他 5 個 op | ⚠️ 常數查表 | 無尺寸掃描;存量測值,非擬合 |
| FP16 數值 | ⚠️ 上界 | numpy 模擬,慢 7–19×;真硬體會更快;明標 provenance |
| 跨模型 ∝hidden | 📝 僅觀察 | 不宣稱擬合律 |
| prefill 擴展 | ❌ 未驗證 | 這些都是 decode(1 token)成本。prefill:**固定 op**(rmsnorm/rope/residual/swiglu)約 ×S;但 **softmax 因 per-token 本身就隨 kv 成長,對 S 個 token 加總 → 約 S×S(二次)**。皆為解析外推、未驗證 |

**一句話總結 A4**:LLM 的零碎輔助運算交給 CPU,其中只有 softmax 隨上下文長度變(用線性公式,誤差 0.3%),其餘是固定成本(直接存量測值);所有 FP16 數字因 numpy 模擬而偏慢,當上界用。**到這裡,CIM(算 matmul)+ GPU(算 attention)+ CPU(算雜項)+ 記憶體,四個計算路徑的成本模型都齊了**——接下來 A5 補上缺席的 NPU,A6 看「是誰決定每個 token 要跑哪些 op」。
