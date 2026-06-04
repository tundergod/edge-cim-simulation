"""M4 — CPU (RK3588 A76) non-GEMM support-op timing model. params/m4_cpu.json.

Uses MEASURED latencies (not analytic FLOPs, issue #10), from
measurements/aetina/cpu_ops.json. softmax has a kv length axis (kv in {128,512,1024})
-> linear fit per (model,dtype). All other support ops (rmsnorm, rope_apply, residual,
swiglu, sampling_argmax) have no within-op sweep -> per-(model,dtype) CONSTANTS.

dtype: fp16 is numpy-EMULATED on the A76 -> treat as an UPPER BOUND (provenance:
docs/phase0.3-findings.md, not a cpu_ops.json field). fp32 kept as reference.
These are decode (1-token) costs; prefill applies them x S tokens (analytic, UNVALIDATED).
"""
import json
from pathlib import Path

_PARAMS = Path(__file__).parent / "params" / "m4_cpu.json"


class CpuModel:
    def __init__(self, params=None):
        p = params if params is not None else json.loads(_PARAMS.read_text())
        self.const = p["const_us"]           # {op: {model: {dtype: us}}}
        self.softmax = p["softmax_linear"]   # {model: {dtype: {a, b}}}

    def op_us(self, op, model, dtype="fp16", kv=None):
        """Per-token support-op latency (us). fp16 = emulated upper bound."""
        if op == "softmax":
            c = self.softmax[model][dtype]
            return c["a"] + c["b"] * kv
        return self.const[op][model][dtype]
