"""Phase 1.2 figure N1 — NPU analytic staircase vs HeteroInfer Fig3 SHAPE (build artifact).

SIMULATED (NO RKNPU2 silicon, issue #13). Drives simulator.models.m4_npu.NpuModel over a
compute-bound N-sweep and shows the 32x32-aligned staircase: latency is flat within each
borrowed 32-pad block and steps at every multiple of 32 — the SHAPE of HeteroInfer Fig3 (we
borrow the shape only; no absolute Fig3 values, no silicon). Writes docs/figures/phase1.2/.

Run: ./.venv/bin/python tools/plotting/npu_n1.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plotting"))
import _style as S  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m4_npu import NpuModel  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402

FIG = ROOT / "docs/figures/phase1.2"


def main():
    spec = load_spec("npu_rknpu2")
    m = NpuModel(spec)
    sd = int(spec["systolic_dim"][0])
    Ns = np.arange(1, 257)
    lat = [m.predict(Workload(op="matmul", M=512, K=2048, N=int(n)))["latency_us"] for n in Ns]

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.step(Ns, lat, where="post", color=S.PALETTE["matmul"], lw=1.3,
            label="NPU analytic (simulated)")
    for k in range(sd, 257, sd):  # borrowed 32x32 grid lines = the knees
        ax.axvline(k, ls=":", color="#bbb", lw=0.6, zorder=0)
    ax.set_xlabel(f"output channels N (compute-bound, M=512, K=2048)")
    ax.set_ylabel("dev latency (us)")
    ax.set_title(f"N1  NPU staircase — knee every {sd} (borrowed Fig3 shape; SIMULATED)",
                 fontsize=7.5)
    ax.text(8, max(lat) * 0.88, f"flat within each\n{sd}-pad block, steps\nat multiples of {sd}",
            fontsize=5.2, color="#888")
    ax.legend(loc="lower right", fontsize=5.5)
    S.save(fig, str(FIG / "N1_npu_staircase"))
    print(f"wrote {FIG/'N1_npu_staircase.png'}")


if __name__ == "__main__":
    main()
