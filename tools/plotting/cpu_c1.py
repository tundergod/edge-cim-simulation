"""Phase 1.2 Fig C1 — CPU instruction-count roofline: measured fp32 cpu_ops vs the calibrated model.

One subplot per support op. X = the op's size variable (kv for softmax, F for swiglu, V for sampling,
hidden H for rmsnorm/residual, heads*hd for rope); Y = latency (us). Points = measured fp32 medians
(measurements/aetina/cpu_ops.json); line/markers = m4_cpu.CpuModel.predict() at the same shapes. The
exp()-dominated ops (softmax, swiglu) are annotated as the cost driver.

Reads measurements/aetina/cpu_ops.json + simulator/specs/cpu_rk3588.json (committed). Writes
docs/figures/phase1.2/C1.{png,pdf,svg}.

Run: ./.venv/bin/python tools/plotting/cpu_c1.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plotting"))
from _style import PALETTE, save  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m4_cpu import CpuModel, MODELS, _n_elem  # noqa: E402

AET = ROOT / "measurements/aetina"
FIG = ROOT / "docs/figures/phase1.2"

# subplot order: (op, size-var label, exp-dominated?). softmax & swiglu are the exp() cost drivers.
PANELS = [
    ("softmax", "kv (cache length)", True),
    ("swiglu", "F (intermediate dim)", True),
    ("sampling_argmax", "V (vocab size)", False),
    ("rmsnorm", "H (hidden dim)", False),
    ("rope_apply", "heads*hd", False),
    ("residual", "H (hidden dim)", False),
]
_COLOR = {"softmax": PALETTE["softmax"], "swiglu": PALETTE["ffn"], "sampling_argmax": PALETTE["embedding"],
          "rmsnorm": PALETTE["norm"], "rope_apply": PALETTE["rope"], "residual": PALETTE["residual"]}


def _measured(ops, base):
    """fp32 measured points for a base op: list of (size_var, n_elem, median_us, model)."""
    pts = []
    for v in ops.values():
        if v["dtype"] != "fp32":
            continue
        op = v["op"]
        if (op.startswith("softmax") if base == "softmax" else op == base):
            c = MODELS[v["model"]]
            n = _n_elem(op, c)
            x = int(op[len("softmax_kv"):]) if base == "softmax" and op.startswith("softmax_kv") else _size_var(base, c)
            pts.append((x, n, v["median_us"], v["model"]))
    return sorted(pts)


def _size_var(base, c):
    return {"swiglu": c["F"], "sampling_argmax": c["V"], "rmsnorm": c["H"],
            "rope_apply": c["heads"] * c["hd"], "residual": c["H"]}[base]


def main():
    ops = json.loads((AET / "cpu_ops.json").read_text())["ops"]
    spec = json.loads((ROOT / "simulator/specs/cpu_rk3588.json").read_text())
    model = CpuModel(spec, engine="analytic")

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
    for ax, (base, xlabel, is_exp) in zip(axes.ravel(), PANELS):
        pts = _measured(ops, base)
        color = _COLOR[base]
        xs = [p[0] for p in pts]
        ax.scatter(xs, [p[2] for p in pts], s=22, color=color, edgecolor="white", linewidth=0.3,
                   zorder=3, label="measured (fp32)")
        # model prediction at each measured shape (same n_elem the measurement used).
        preds = []
        for x, n, med, mdl in pts:
            c = MODELS[mdl]
            if base == "softmax":
                wl = Workload(op="softmax", heads=c["heads"], kv=x, dtype="fp32")
            elif base == "rope_apply":
                wl = Workload(op="rope_apply", heads=c["heads"], extra={"hd": c["hd"]}, dtype="fp32")
            elif base in ("rmsnorm", "residual"):
                wl = Workload(op=base, K=n, dtype="fp32")
            else:  # swiglu (F), sampling_argmax (V)
                wl = Workload(op=base, N=n, dtype="fp32")
            preds.append(model.predict(wl)["latency_us"])
        ax.scatter(xs, preds, s=26, marker="x", color=color, linewidth=1.0, zorder=4,
                   label="model")
        ax.set_title(("exp()-dominated: " if is_exp else "") + base, fontsize=7.5,
                     color=("#B00000" if is_exp else "black"))
        ax.set_xlabel(xlabel, fontsize=6.5)
        ax.set_ylabel("latency (us)", fontsize=6.5)
        ax.tick_params(length=2, labelsize=6)
        ax.legend(fontsize=5.8, loc="upper left")
    fig.suptitle("C1 — CPU instruction-count roofline: measured fp32 vs calibrated model "
                 "(single A76 core; exp() = cost driver)", fontsize=8.5, y=1.01)
    fig.tight_layout()
    save(fig, str(FIG / "C1"))
    print(f"wrote {FIG}/C1.png")


if __name__ == "__main__":
    main()
