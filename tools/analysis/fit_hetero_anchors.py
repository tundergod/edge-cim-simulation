"""ADR-0008 — fit the SoC-profile unit models to their anchors (reproducible provenance).

1. GPU  (sd8gen3): 2-regime roofline (m4_gpu_anchor.GpuAnchorRooflineModel) minimax-fitted to the six
   HeteroInfer Table 3 GPU latencies stored in simulator/specs/gpu_adreno750_sd8gen3.json `anchors`.
2. NPU  (sd8gen3): 32-row staircase (m4_npu.NpuStaircaseModel) minimax-fitted to the six Table 3 NPU
   latencies in simulator/specs/npu_hexagon_sd8gen3.json `anchors`; cross-checked against the Fig.3
   staircase levels (not fitted on them).
3. CIM attention: a + b*kv per bmm, least squares on the MEASURED Metis Alpha `attn_floor` points
   (measurements/aetina/metis_alpha_matmul.json) -> simulator/models/params/m1_cim_attention.json.

The objective is the MAX relative error over the anchors (the acceptance band is +/-15% on every
anchor), minimised with a dependency-free Nelder-Mead over log-parameters from several starts. The
models themselves are evaluated (not a re-typed formula), so the committed spec reproduces the
residuals printed here. ALL GPU/NPU numbers are `borrowed` (published, no silicon of ours).

Writes  simulator/specs/{gpu_adreno750_sd8gen3,npu_hexagon_sd8gen3}.json  (`anchor_fit` block)
        simulator/models/params/m1_cim_attention.json
        validation/reports/hetero/anchor_fit.json
Run:    ./.venv/bin/python tools/analysis/fit_hetero_anchors.py     (from the repo root)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m4_gpu_anchor import GpuAnchorRooflineModel  # noqa: E402
from simulator.models.m4_npu import NpuStaircaseModel  # noqa: E402

GPU_SPEC = ROOT / "simulator/specs/gpu_adreno750_sd8gen3.json"
NPU_SPEC = ROOT / "simulator/specs/npu_hexagon_sd8gen3.json"
ALPHA = ROOT / "measurements/aetina/metis_alpha_matmul.json"
CIM_ATTN = ROOT / "simulator/models/params/m1_cim_attention.json"
OUT = ROOT / "validation/reports/hetero/anchor_fit.json"
TOL = 0.15


def nelder_mead(f, x0, iters=3000, step=0.5):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def multistart(f, starts):
    best = None
    for x0 in starts:
        p, v = nelder_mead(f, np.log(x0))
        if best is None or v < best[1]:
            best = (p, v)
    return np.exp(best[0]), best[1]


def _residuals(model, anchors):
    rows = []
    for a in anchors:
        wl = Workload(op="matmul", M=a["M"], K=a["K_in"], N=a["N_out"])
        pred = model.predict(wl)["latency_us"]
        rows.append({**{k: a[k] for k in ("N_out", "K_in", "M", "latency_us")},
                     "pred_us": round(pred, 1), "rel_err": round(pred / a["latency_us"] - 1, 4)})
    return rows


def fit_gpu():
    spec = json.loads(GPU_SPEC.read_text())
    anchors = spec["anchors"]

    def build(p):
        s = dict(spec)
        s["anchor_fit"] = {"gemm_floor_us": p[0], "bw_GBs": p[1], "c_max_tflops": p[2], "m_half": p[3],
                           "weight_bytes_per_param": 0.5}
        return GpuAnchorRooflineModel(s)

    def obj(lp):
        m = build(np.exp(lp))
        return max(abs(r["rel_err"]) for r in _residuals(m, anchors))

    p, v = multistart(obj, [[200, 30, 1.2, 100], [50, 20, 1.0, 50], [300, 40, 1.5, 200]])
    p = [round(float(x), 4) for x in p]
    fit = {"_doc": "2-regime roofline minimax fit of `anchors` (fit_hetero_anchors.py). GEMV: floor + 0.5*K*N/bw; "
                   "GEMM: floor + 2MKN/(C_max*M/(M+M_half)); M>1 = min(M*GEMV, GEMM). [borrowed, fitted]",
           "gemm_floor_us": p[0], "bw_GBs": p[1], "c_max_tflops": p[2], "m_half": p[3],
           "weight_bytes_per_param": 0.5, "objective": "minimax |rel err| over anchors"}
    spec["anchor_fit"] = fit
    res = _residuals(build(p), anchors)
    fit["max_abs_rel_residual"] = round(max(abs(r["rel_err"]) for r in res), 4)
    fit["all_within_15pct"] = bool(fit["max_abs_rel_residual"] <= TOL)
    GPU_SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    return {"params": {k: v for k, v in fit.items() if not k.startswith("_")}, "residuals": res}


def fit_npu():
    spec = json.loads(NPU_SPEC.read_text())
    anchors = spec["anchors"]

    def build(p):
        s = dict(spec)
        s["anchor_fit"] = {"floor_us": p[0], "bw_GBs": p[1], "tops_eff": p[2], "k_penalty_exp": p[3],
                           "k_ref": 4096, "weight_bytes_per_param": 1.0}
        return NpuStaircaseModel(s)

    def obj(lp):
        m = build(np.exp(lp))
        return max(abs(r["rel_err"]) for r in _residuals(m, anchors))

    p, v = multistart(obj, [[50, 38, 7, 1.1], [100, 30, 5, 1.0], [20, 45, 10, 1.3]])
    p = [round(float(x), 4) for x in p]
    fit = {"_doc": "32-row staircase minimax fit of `anchors` (fit_hetero_anchors.py): floor + (K/k_ref)^p * "
                   "(K*N/bw + ceil(M/32)*2*32*K*N/tops_eff). [borrowed, fitted]",
           "floor_us": p[0], "bw_GBs": p[1], "tops_eff": p[2], "k_penalty_exp": p[3], "k_ref": 4096,
           "weight_bytes_per_param": 1.0, "objective": "minimax |rel err| over anchors"}
    spec["anchor_fit"] = fit
    m = build(p)
    res = _residuals(m, anchors)
    fit["max_abs_rel_residual"] = round(max(abs(r["rel_err"]) for r in res), 4)
    fit["all_within_15pct"] = bool(fit["max_abs_rel_residual"] <= TOL)
    # Fig.3 cross-check (NOT fitted): [4096,14336] levels at steps 1..6
    fig3 = []
    for i, lvl in enumerate(spec["fig3_staircase_4096x14336_us"]):
        pred = m.predict(Workload(op="matmul", M=32 * (i + 1), K=14336, N=4096))["latency_us"]
        fig3.append({"steps": i + 1, "fig3_us": lvl, "pred_us": round(pred, 1), "rel_err": round(pred / lvl - 1, 4)})
    NPU_SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    return {"params": {k: v for k, v in fit.items() if not k.startswith("_")}, "residuals": res,
            "fig3_crosscheck_not_fitted": fig3}


def fit_cim_attention():
    rows = json.loads(ALPHA.read_text())["by_group"]["attn_floor"]
    out = {"_doc": "CIM decode-attention per-bmm floor fit a + b*kv (us) on the MEASURED Metis Alpha "
                   "attn_floor points (single head, hd=128, M=1; fit_hetero_anchors.py). qkT: [1x128]x[128xkv]; "
                   "sv: [1xkv]x[kvx128]. Composite per head = (qk_a+sv_a) + (qk_b+sv_b)*kv. [measured basis]",
           "source": "measurements/aetina/metis_alpha_matmul.json by_group.attn_floor dev_lat_us",
           "hd_basis": 128, "points": []}
    kv_fit = sorted({r["N"] if r["family"] == "qkT" else r["K"] for r in rows})
    out["kv_fit"] = kv_fit
    for fam, key in (("qkT", "qk"), ("sv", "sv")):
        pts = [(r["N"] if fam == "qkT" else r["K"], r["dev_lat_us"]) for r in rows if r["family"] == fam]
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        b, a = np.polyfit(x, y, 1)
        out[f"{key}_a_us"], out[f"{key}_b_us_per_kv"] = round(float(a), 4), round(float(b), 7)
        out["points"].extend({"family": fam, "kv": int(xi), "dev_lat_us": round(float(yi), 3),
                              "fit_us": round(float(a + b * xi), 3)} for xi, yi in zip(x, y))
    CIM_ATTN.write_text(json.dumps(out, indent=1) + "\n")
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gpu, npu, cim = fit_gpu(), fit_npu(), fit_cim_attention()
    rep = {"module": "hetero_anchor_fit", "honesty": "GPU/NPU: borrowed (HeteroInfer SOSP'25 published anchors, "
           "no silicon of ours); CIM attention: measured Alpha attn_floor basis", "tolerance": TOL,
           "gpu_sd8gen3": gpu, "npu_sd8gen3": npu, "cim_attention_alpha": cim,
           "pass_all": bool(gpu["params"]["all_within_15pct"] and npu["params"]["all_within_15pct"])}
    OUT.write_text(json.dumps(rep, indent=1) + "\n")
    for name, blk in (("GPU", gpu), ("NPU", npu)):
        print(f"{name} sd8gen3 anchor fit: {blk['params']}")
        for r in blk["residuals"]:
            print(f"  [{r['N_out']},{r['K_in']}] M={r['M']:>3}: anchor {r['latency_us']:>6} pred {r['pred_us']:>8} "
                  f"{100 * r['rel_err']:+.1f}%")
    for r in npu["fig3_crosscheck_not_fitted"]:
        print(f"  Fig.3 steps={r['steps']}: {r['fig3_us']} vs pred {r['pred_us']} {100 * r['rel_err']:+.1f}%")
    print(f"CIM attention floor fit: qk {cim['qk_a_us']} + {cim['qk_b_us_per_kv']}*kv, "
          f"sv {cim['sv_a_us']} + {cim['sv_b_us_per_kv']}*kv (us)")
    print(f"pass_all={rep['pass_all']} -> {OUT}")
    return 0 if rep["pass_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
