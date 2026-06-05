"""Phase 1.2 figure N2 — attention offload: CIM/Mali (silicon) vs NPU (simulated) (build artifact).

Per-token attention vs KV length for three offload candidates:
  - CIM composed (Alpha topology) — SOLID, silicon-backed (measurements/aetina/cim_attention_composed.json)
  - Mali GPU-native bmm          — SOLID, silicon-backed (simulator/models/params/m4_gpu.json fit)
  - NPU analytic                 — DASHED, SIMULATED (NpuModel; NO RKNPU2 silicon, issue #13)

The CIM and Mali lines are silicon (the validated Phase-1.1 deliverables); the NPU line is the
Phase-1.2 analytic estimate and is drawn dashed + labeled simulated so the honesty boundary is
visible at a glance. Writes docs/figures/phase1.2/.

Run: ./.venv/bin/python tools/plotting/npu_n2.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plotting"))
import _style as S  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m4_npu import NpuModel  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402

AET = ROOT / "measurements/aetina"
FIG = ROOT / "docs/figures/phase1.2"
HEADS, HD = 8, 128  # 8B-class attention head config for the simulated NPU line


def main():
    # CIM composed (silicon, Alpha topology) — kv = stored kv (129/513/1025 -> 128/512/1024)
    cim = json.loads((AET / "cim_attention_composed.json").read_text())
    cim_kv = [r["kv"] - 1 for r in cim["rows"]]
    cim_ms = [r["composed_us"] / 1e3 for r in cim["rows"]]

    # Mali GPU-native (silicon fit) — single-head a + b*kv, scaled to HEADS heads
    g = json.loads((ROOT / "simulator/models/params/m4_gpu.json").read_text())
    a, b = g["attn_bmm_a_us"], g["attn_bmm_b_us_per_kv"]
    mali_ms = [(a + b * kv) * HEADS / 1e3 for kv in cim_kv]

    # NPU analytic (SIMULATED) — native attn bmm, HEADS heads, padded
    m = NpuModel(load_spec("npu_rknpu2"))
    npu_ms = [m.predict(Workload(op="attn_bmm", kv=kv, heads=HEADS, layers=1,
                                 extra={"hd": HD}))["latency_us"] / 1e3 for kv in cim_kv]

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(cim_kv, cim_ms, "o-", color=S.PALETTE["attention"], ms=4, lw=1.4,
            label="CIM composed (silicon)")
    ax.plot(cim_kv, mali_ms, "s-", color=S.PALETTE["ffn"], ms=4, lw=1.4,
            label=f"Mali GPU-native (silicon, x{HEADS}h)")
    ax.plot(cim_kv, npu_ms, "^--", color=S.PALETTE["matmul"], ms=4.5, lw=1.4,
            label="NPU analytic (SIMULATED)")
    ax.set_yscale("log")
    ax.set_xlabel("KV length")
    ax.set_ylabel("per-token attention (ms, log)")
    ax.set_title("N2  attention offload — CIM/Mali silicon (solid) vs NPU (dashed, sim)",
                 fontsize=7.2)
    ax.legend(loc="center right", fontsize=5.5)
    S.save(fig, str(FIG / "N2_attn_offload"))
    print(f"wrote {FIG/'N2_attn_offload.png'}")


if __name__ == "__main__":
    main()
