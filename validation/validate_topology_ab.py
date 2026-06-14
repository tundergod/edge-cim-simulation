"""Topology A/B/C swap (Phase 2.3, ADR-0006 validate-then-swap) — SIMULATED except card.

Same decode workload (1B/3B/8B) across the three wired topologies; the decode memory wall is
the ONLY thing that changes (single physics source = the topology spec, resolved via M2
MemoryModel). The point: CIM compute is ~99% idle on decode (voyager-sdk.md:278), so decode
tok/s is set by the memory wall — and the wall depends entirely on where the weights live.

  cim_topo_card  (A, on-card LPDDR4x 24.2 GB/s) — MEASURED L4 anchor. card rows REPRODUCE the
                 committed mechanism predictions (validation/reports/phase2/e2e_l4.json), NOT the
                 measured vendor tok/s (forcing pred==measured would be circular).
  cim_topo_alpha (A-no-DRAM, host PCIe 3.9 GB/s + 911 us per-call floor) — COUNTERFACTUAL: Metis
                 Alpha is LLM-incapable (-1301 closed firmware + no on-card DRAM); the number is
                 "if it could stream weights over PCIe, transport-bound at 3.9 GB/s". PCIe transport
                 is Phase-0.3-measured; the 911 us floor is added to TTFT only (amortized over decode).
  cim_topo_edge  (B, integrated SoC, LPDDR5 33.3 x noc 0.9 = 29.97 GB/s) — SIMULATED (noc_efficiency
                 assumption, no edge silicon).

Residency caveat (#58): topology A's 24.2 GB/s holds ONLY while the model fits the 16 GiB on-card
DRAM; on capacity overflow it degrades toward the alpha host-PCIe wall (spill not modelled in v1 —
fail-loud feasibility gate). 1B/3B/8B all fit 16 GiB, so this table is spill-free.

Reads  validation/reports/phase2/e2e_l4.json (committed card anchor)
Writes validation/reports/phase2/topology_ab.json
Run:   ./.venv/bin/python validation/validate_topology_ab.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402

OUT = ROOT / "validation/reports/phase2"
MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
# (topology, memory_spec, honesty tier, one-line label)
TOPOS = [
    ("cim_topo_card", "mem_lpddr4x", "measured",
     "on-card LPDDR4x 24.2 GB/s — MEASURED L4 anchor"),
    ("cim_topo_alpha", None, "counterfactual",
     "host PCIe 3.9 GB/s + 911us floor — COUNTERFACTUAL (Alpha is LLM-incapable, -1301; "
     "no on-card DRAM; number = 'if it could stream over PCIe')"),
    ("cim_topo_edge", "mem_lpddr5", "simulated",
     "integrated SoC LPDDR5 33.3 x noc 0.9 = 29.97 GB/s — SIMULATED (noc_efficiency assumption, "
     "no edge silicon)"),
]
ANCHOR_TOL = 1e-4


def _cfg(model, topology, memory_spec):
    plat = {"topology": topology}
    if memory_spec is not None:
        plat["memory_spec"] = memory_spec
    return SimConfig.from_dict({
        "workload": {"model": model, "context": 1024},
        "platform": plat,
        "scheduler": {"policy": "all_cim"},
        "tunables": {"pipeline": False},
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    committed = json.loads((OUT / "e2e_l4.json").read_text())["mechanism_independent_pricing"]

    table = {}
    for m in MODELS:
        table[m] = {}
        for topo, mspec, tier, label in TOPOS:
            r = run(_cfg(m, topo, mspec))
            table[m][topo] = {
                "tok_s": r["tok_s"],
                "decode_token_us": r["decode_token_us"],         # = data-movement time (bytes / eff_BW)
                "eff_BW_GBs": round(r["memory_eff_BW_GBs"], 3),
                "ttft_s": r["ttft_s_reported_not_gated"],         # alpha's includes the 911 us per-call floor
                "tier": tier,
                "label": label,
            }

    # (1) card rows must REPRODUCE the committed mechanism predictions (anti-circular: NOT == measured)
    anchor_ok = {}
    for m in MODELS:
        pred = table[m]["cim_topo_card"]["tok_s"]
        committed_pred = committed[m]["pred_tok_s"]
        anchor_ok[m] = abs(pred - committed_pred) < ANCHOR_TOL
    anchor_byte_identical = all(anchor_ok.values())

    # (2) decode tok/s ordering alpha < card < edge — BY CONSTRUCTION (follows from eff_BW 3.9<24.2<29.97),
    #     a sanity check, NOT a discovery. The REPORTED result is the magnitude of the alpha penalty.
    ordering_ok = all(table[m]["cim_topo_alpha"]["tok_s"] < table[m]["cim_topo_card"]["tok_s"]
                      < table[m]["cim_topo_edge"]["tok_s"] for m in MODELS)
    alpha_penalty = {m: round(table[m]["cim_topo_card"]["tok_s"] / table[m]["cim_topo_alpha"]["tok_s"], 2)
                     for m in MODELS}  # how many x SLOWER the host-PCIe (no-on-card-DRAM) path is
    edge_speedup = {m: round(table[m]["cim_topo_edge"]["tok_s"] / table[m]["cim_topo_card"]["tok_s"], 3)
                    for m in MODELS}

    out = {
        "module": "topology_ab", "phase": "2.3",
        "honesty": "card = MEASURED L4 anchor (rows reproduce the COMMITTED mechanism predictions, "
                   "NOT the measured vendor tok/s — pred==measured would be circular). alpha = "
                   "COUNTERFACTUAL (LLM-incapable hardware). edge = SIMULATED (noc assumption). "
                   "The alpha<card<edge ordering is BY CONSTRUCTION (eff_BW constants); the reported "
                   "finding is the alpha host-PCIe penalty magnitude + the edge LPDDR5 speedup.",
        "topologies": {t: {"tier": tier, "label": label} for t, _, tier, label in TOPOS},
        "table": table,
        "card_reproduces_committed_anchor": bool(anchor_byte_identical),
        "card_anchor_committed_pred": {m: committed[m]["pred_tok_s"] for m in MODELS},
        "ordering_alpha_lt_card_lt_edge_by_construction": bool(ordering_ok),
        "alpha_host_pcie_penalty_x_slower": alpha_penalty,
        "edge_lpddr5_speedup_x": edge_speedup,
        "per_call_floor_note": "alpha pays the measured 911 us per-call DMA floor in TTFT ONLY "
                               "(amortized to ~0 over decode tokens, D11); decode tok/s is bandwidth-bound.",
        "residency_caveat": "topology A's 24.2 GB/s holds only within the 16 GiB on-card DRAM; "
                            "overflow degrades toward the alpha host-PCIe wall (spill not modelled; #58). "
                            "1B/3B/8B fit 16 GiB so this table is spill-free.",
        "pass_all": bool(anchor_byte_identical and ordering_ok),
    }
    (OUT / "topology_ab.json").write_text(json.dumps(out, indent=1))

    print(f"topology A/B/C (card=measured, alpha=counterfactual, edge=simulated):")
    for m in MODELS:
        cells = " | ".join(f"{t.split('_')[-1]}={table[m][t]['tok_s']:.2f}t/s@{table[m][t]['eff_BW_GBs']}GB/s"
                           for t, _, _, _ in TOPOS)
        print(f"  {m}: {cells}  [alpha {alpha_penalty[m]}x slower, edge {edge_speedup[m]}x]")
    print(f"  card reproduces committed anchor (<1e-4): {anchor_byte_identical}")
    print(f"  ordering alpha<card<edge (by-construction): {ordering_ok}")
    return 0 if out["pass_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
