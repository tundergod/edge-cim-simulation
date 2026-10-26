"""ADR-0008 round-2 (2026-10-23) — fit the CPU GEMM kernel model END-TO-END to HeteroInfer Fig.12.

The CPU GEMM/GEMV kernel model (m4_cpu.CpuGemmModel) has no silicon. Its two free parameters —
`gemv_efficiency` (fraction of the NEON int8 peak an M=1 matvec kernel reaches) and the per-kernel
`kernel_floor_us` (multi-thread fork/join + dispatch, charged per GEMM/GEMV kernel, twice per
attention composite and once per fused elementwise chain head on the CPU) — are fitted so that the
SIMULATED `cpu_only` decode throughput on the sd8gen3_cim profile equals the SIMULATED `gpu_only`
decode throughput divided by the paper's end-to-end llama.cpp(CPU) / MNN(GPU) decode-time ratio:

    HeteroInfer Fig.12 (cim-mobile-soc/docs/heteroinfer-design-reference.md §4.2):
        InternLM-1.8B  19.38 s / 14.20 s = 1.365     -> proxy llama-3.2-1b
        Llama-7B       63.39 s / 48.76 s = 1.300     -> proxy llama-3.1-8b
        target ratio   1.33 (mean), acceptance band +/-10 % on BOTH models

The previous per-op anchor ("a decode GEMV = 1.33x its W4 pool time", gemv_efficiency 0.217, no
floor) made the end-to-end CPU-only decode BEAT the GPU-only decode (matrix v1 anomaly 1) because the
GPU carries its own kernel floors and the shared pool overlaps the CPU bytes. Fitting the END-TO-END
ratio through the real Platform + scheduler + engine removes that inversion by construction.

Search: the floor on a grid inside the 10-30 us launch band (assumption: a 4-thread NEON kernel's
fork/join + dispatch, same order as the GPU/NPU "tens of us" floors); for each floor the efficiency
that makes the two models' deviations equal and opposite (minimax, bisection: both deviations are
monotone in the efficiency); keep the (floor, efficiency) with the smallest max |deviation|. The
achieved ratios are then re-verified through runner.run on the written profile.

HONESTY: everything here is SIMULATED / ASSUMPTION — two published end-to-end seconds anchor the
whole CPU GEMM model; nothing is a kernel measurement. CPU weight bytes stay 0.5 B/param (llama.cpp
Q4), the CPU single-stream cap 43 GB/s (profile).

Writes  simulator/specs/soc_sd8gen3_cim.json  (units.cpu.gemm.gemv_efficiency / kernel_floor_us +
        provenance), the same values into simulator/specs/soc_rk3588_cim.json (same A76 proxy CPU),
        validation/reports/hetero/cpu_gemv_fit.json
Run:    ./.venv/bin/python tools/analysis/fit_cpu_gemv.py [--dry-run]     (from the repo root)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.events import run_dag  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.scheduler import SCHEDULERS  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
import op_profile  # noqa: E402

PROFILE = "sd8gen3_cim"
MIRROR = "rk3588_cim"            # same cpu_rk3588 A76 proxy: mirrors the fitted values
FIG12 = {"llama-3.2-1b": ("InternLM-1.8B", 19.38, 14.20), "llama-3.1-8b": ("Llama-7B", 63.39, 48.76)}
TARGET = 1.33                    # mean of the two Fig.12 ratios (1.365, 1.300)
BAND = 0.10                      # acceptance: |achieved / TARGET - 1| <= 10 % on both models
FLOORS_US = (10.0, 15.0, 20.0, 25.0, 30.0)
WL = {"context": 1024, "prefill_len": 256, "decode_len": 128}
OUT = ROOT / "validation/reports/hetero/cpu_gemv_fit.json"


def _cfg(model, policy):
    return SimConfig.from_dict({"workload": {"model": model, **WL},
                                "platform": {"soc_profile": PROFILE,
                                             "units": {"gpu": True, "npu": True, "cim": True, "cpu": True}},
                                "scheduler": {"policy": policy}})


def _kv_points(cfg):
    """The runner's decode kv sample points (runner.run: {lo, mid, hi} over [P, min(ctx, P+D)])."""
    P, D = cfg.prefill_len, max(1, cfg.decode_len)
    kv_hi = max(1, min(cfg.context, P + D))
    kv_lo = max(1, min(P, kv_hi))
    return sorted({kv_lo, (kv_lo + kv_hi) // 2, kv_hi})


class _Fitter:
    """cpu_only decode tok/s for (floor, gemv_eff) through the real Platform + scheduler + engine,
    with the two parameters patched on the Platform's CPU GEMM model (the runner's own kv sampling)."""

    def __init__(self, models):
        self.cfg = {m: _cfg(m, "cpu_only") for m in models}
        self.plat = {m: Platform(m, soc_profile=PROFILE) for m in models}
        self.obj = {m: op_profile.Model(m) for m in models}
        self.dags = {m: [build_token_dag(m, "decode", k, _model_obj=self.obj[m]) for k in _kv_points(self.cfg[m])]
                     for m in models}

    def tok_s(self, model, floor_us, gemv_eff):
        plat = self.plat[model]
        plat.cpu_kernel_floor_us = float(floor_us)
        g = plat.cpu_gemm
        g.kernel_floor_us = float(floor_us)
        g.gemv_eff = float(gemv_eff)
        g.eff = max(g.eff, g.gemv_eff)
        plat.cpu_gemv_efficiency = g.gemv_eff
        sched = SCHEDULERS["cpu_only"]
        us = []
        for d in self.dags[model]:
            dag = sched.assign(_fresh(d), self.cfg[model], platform=plat)
            us.append(run_dag(dag, plat, plat.bw))
        return 1e6 / (sum(us) / len(us))


def _fresh(dag):
    import copy
    return copy.deepcopy(dag)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="fit and print, do not write the specs")
    ap.add_argument("--floors", nargs="*", type=float, default=list(FLOORS_US))
    a = ap.parse_args()
    models = list(FIG12)
    gpu = {m: run(_cfg(m, "gpu_only"))["tok_s"] for m in models}
    target_cpu = {m: gpu[m] / TARGET for m in models}
    print("gpu_only decode tok/s:", {m: round(v, 3) for m, v in gpu.items()})
    print("target cpu_only tok/s (gpu / %.2f):" % TARGET, {m: round(v, 3) for m, v in target_cpu.items()})
    fit = _Fitter(models)

    def devs(floor, eff):
        return {m: fit.tok_s(m, floor, eff) / target_cpu[m] - 1.0 for m in models}

    m1, m8 = models
    results = []
    for floor in a.floors:
        # both deviations rise monotonically with eff; the minimax point has dev(1b) == -dev(8b)
        lo, hi = 0.02, 0.5
        for _ in range(40):
            mid = (lo + hi) / 2
            d = devs(floor, mid)
            if d[m1] + d[m8] > 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-5:
                break
        eff = round((lo + hi) / 2, 4)
        d = devs(floor, eff)
        worst = max(abs(v) for v in d.values())
        results.append({"kernel_floor_us": floor, "gemv_efficiency": eff, "deviation": {m: round(v, 4) for m, v in d.items()},
                        "max_abs_deviation": round(worst, 4)})
        print(f"floor {floor:5.1f} us -> gemv_eff {eff:.4f}  dev 1b {d[m1]:+.3%}  8b {d[m8]:+.3%}  max {worst:.3%}")
    best = min(results, key=lambda r: (r["max_abs_deviation"], r["kernel_floor_us"]))
    print("chosen:", best)
    if a.dry_run:
        return 0
    prov_eff = (f"{best['gemv_efficiency']} of NEON int8 peak at M=1, FITTED END-TO-END with kernel_floor_us by "
                f"tools/analysis/fit_cpu_gemv.py: simulated cpu_only decode tok/s = gpu_only / {TARGET} on "
                f"llama-3.2-1b (InternLM-1.8B proxy) and llama-3.1-8b (Llama-7B proxy), ctx 1024 / prefill 256 / "
                f"decode 128 (HeteroInfer Fig.12 llama.cpp/MNN decode 19.38/14.20 s, 63.39/48.76 s) "
                f"[assumption anchored on published end-to-end only, no CPU GEMM silicon]; m_half 32 = one "
                f"register-blocked row tile [assumption]")
    prov_floor = (f"{best['kernel_floor_us']} us per CPU kernel (GEMM/GEMV, x2 per attention composite, once per "
                  f"fused elementwise chain head on the CPU; multi-thread fork/join + dispatch) [assumption: the "
                  f"same 'tens of us' band as the GPU/NPU launch floors; value chosen on a 10-30 us grid by the "
                  f"end-to-end fit above]")
    for name, note in ((PROFILE, ""), (MIRROR, " (mirrored from sd8gen3_cim: same cpu_rk3588 A76 proxy)")):
        path = ROOT / "simulator/specs" / f"soc_{name}.json"
        spec = json.loads(path.read_text())
        g = spec["units"]["cpu"]["gemm"]
        g["gemv_efficiency"] = best["gemv_efficiency"]
        g["kernel_floor_us"] = best["kernel_floor_us"]
        spec["provenance"]["units.cpu.gemm.gemv_efficiency"] = prov_eff + note
        spec["provenance"]["units.cpu.gemm.kernel_floor_us"] = prov_floor + note
        path.write_text(json.dumps(spec, indent=2) + "\n")
        print("wrote", path)
    # authoritative re-check through runner.run on the written profile
    check = {}
    for m in models:
        cpu = run(_cfg(m, "cpu_only"))["tok_s"]
        g_ = run(_cfg(m, "gpu_only"))["tok_s"]
        check[m] = {"gpu_only_tok_s": g_, "cpu_only_tok_s": cpu, "gpu_over_cpu": round(g_ / cpu, 4),
                    "fig12_ratio": round(FIG12[m][1] / FIG12[m][2], 4), "proxy_for": FIG12[m][0],
                    "deviation_from_target": round((g_ / cpu) / TARGET - 1, 4), "within_band": abs((g_ / cpu) / TARGET - 1) <= BAND}
        print(f"{m}: gpu_only {g_:.3f} / cpu_only {cpu:.3f} = {g_ / cpu:.3f} (target {TARGET}, Fig.12 {check[m]['fig12_ratio']})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"_doc": __doc__.split("\n")[0], "target_ratio": TARGET, "band": BAND, "workload": WL,
                               "fig12": {m: {"model": v[0], "llama_cpp_s": v[1], "mnn_s": v[2]} for m, v in FIG12.items()},
                               "grid": results, "chosen": best, "runner_check": check,
                               "honesty": "simulated: CPU GEMM kernel model anchored on two published end-to-end "
                                          "ratios only; no CPU GEMM / kernel-launch silicon"}, indent=1) + "\n")
    print("wrote", OUT)
    return 0 if all(c["within_band"] for c in check.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
