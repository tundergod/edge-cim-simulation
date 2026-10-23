"""ADR-0008 demo — the unit x op COMPUTE-latency table a heterogeneous scheduler sees.

For Llama-3.2-1B / 3.2-3B / 3.1-8B it instantiates the REAL per-token op DAG (build_token_dag) for
decode (M=1, kv=512) and prefill (M=128 / 256), de-duplicates the distinct op shapes, and prices
each on every unit Platform.supports() (cim / gpu / npu / cpu / mem). Printed per shape: the
compute latency per unit (us; '-' = unsupported) plus the op's bytes and its single-stream memory
time at the profile's eff_BW — because the engine takes max(compute, memory), an elementwise op's
tiny launch-floor compute is NOT its cost. Attention rows are the QK^T+S.V COMPOSITE (priced once).

Honesty: the profile's own tags apply (sd8gen3_cim = borrowed HeteroInfer numbers; rk3588_cim =
RK3588-measured units on the simulated edge topology); each unit's provenance string is in the JSON.

Writes validation/reports/hetero/price_table_<profile>.json
Run:   ./.venv/bin/python tools/analysis/price_table.py --profile sd8gen3_cim    (from the repo root)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.runtime.platform import Platform, SOC_PROFILES  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402

MODELS = ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b")
UNITS = ("cim", "gpu", "npu", "cpu", "mem")
PHASES = (("decode", 512), ("prefill", 128), ("prefill", 256))
OUT = ROOT / "validation/reports/hetero"


def _shape_key(n):
    wl = n.wl
    if n.category == "attention" and "hd" in wl.extra:
        return (n.category, "bmm", wl.M, wl.extra["hd"], wl.kv, wl.heads)
    if n.category == "matmul":
        return (n.category, wl.M, wl.K, wl.N)
    return (n.category, wl.extra.get("aten", wl.op), wl.kv, n.bytes_streamed)


def _label(n):
    wl = n.wl
    if n.category == "matmul":
        return f"matmul M={wl.M} K={wl.K} N={wl.N}"
    if n.category == "attention" and "hd" in wl.extra:
        return f"attention(QK^T+SV) heads={wl.heads} hd={wl.extra['hd']} kv={wl.kv} M={wl.M}"
    return f"{n.category} [{wl.extra.get('aten', wl.op)}]" + (f" kv={wl.kv}" if wl.kv else "")


def table(profile, models=MODELS):
    plat = Platform(models[0], soc_profile=profile)
    rows = []
    for model in models:
        plat = Platform(model, soc_profile=profile)
        for phase, L in PHASES:
            seen = set()
            for n in build_token_dag(model, phase, L).nodes:
                k = _shape_key(n)
                if k in seen:
                    continue
                seen.add(k)
                if n.category == "attention" and "hd" in n.wl.extra and n.pricing_group != n.id:
                    continue                                   # the S.V member: composite lives on the rep
                row = {"model": model, "phase": phase, "L": L, "op": _label(n), "category": n.category,
                       "bytes": n.bytes_streamed,
                       "mem_us_single_stream": round(n.bytes_streamed / (plat.bw.eff_BW * 1e3), 2),
                       "units": {}}
                for u in UNITS:
                    if not Platform.supports(u, n.category):
                        continue
                    n.unit = u
                    try:
                        r = plat.price(n)
                    except NotImplementedError as e:   # a subtype exception (Mali non-bmm attention)
                        row["units"][u] = {"latency_us": None, "source_model": "n/a", "provenance": str(e)}
                        continue
                    row["units"][u] = {"latency_us": round(r["latency_us"], 3),
                                       "source_model": r["source_model"], "provenance": r["compute_provenance"]}
                rows.append(row)
    return {"profile": profile, "eff_BW_GBs": plat.bw.eff_BW, "knee_GBs": plat.bw.knee,
            "sync_us": plat.sync_us, "honesty": plat.profile["honesty"], "rows": rows}


def print_table(t):
    print(f"== soc_profile={t['profile']}  eff_BW={t['eff_BW_GBs']} GB/s  knee={t['knee_GBs']}  sync={t['sync_us']}")
    print(f"   {t['honesty']}")
    hdr = f"{'op':56} {'bytes':>11} {'mem_us':>9} | " + " ".join(f"{u:>10}" for u in UNITS)
    cur = None
    for r in t["rows"]:
        if (r["model"], r["phase"], r["L"]) != cur:
            cur = (r["model"], r["phase"], r["L"])
            print(f"\n-- {r['model']} {r['phase']} L={r['L']} (compute us per unit; '-' = unsupported)")
            print(hdr)
        cells = " ".join(
            f"{'-':>10}" if u not in r["units"] else
            (f"{'n/a':>10}" if r["units"][u]["latency_us"] is None else f"{r['units'][u]['latency_us']:>10.1f}")
            for u in UNITS)
        print(f"{r['op'][:56]:56} {r['bytes']:>11} {r['mem_us_single_stream']:>9.1f} | {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=SOC_PROFILES, default="sd8gen3_cim")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    t = table(a.profile, a.models)
    print_table(t)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"price_table_{a.profile}.json"
    p.write_text(json.dumps(t, indent=1) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
