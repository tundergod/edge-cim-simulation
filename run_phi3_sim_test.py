from simulator.runtime.config import SimConfig
from simulator.runtime.runner import run

for ctx in [512, 1024, 2048]:
    cfg = SimConfig.from_dict({
        "workload": {
            "model": "phi3-mini",
            "context": ctx,
            "prefill_len": ctx,
            "decode_len": 128,
            "batch": 1,
        },
        "platform": {
            "topology": "cim_topo_card",
        },
        "scheduler": {
            "policy": "all_cim",
        },
        "tunables": {
            "pipeline": False,
        },
    })

    m = run(cfg)
    print(
        ctx,
        "TTFT_s=", m["ttft_s_reported_not_gated"],
        "tok_s=", m["tok_s"],
        "footprint_GB=", m["model_footprint_GB"],
        "BW_GBs=", m["memory_eff_BW_GBs"],
    )
