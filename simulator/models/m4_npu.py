"""M4 — RKNPU2 (RK3588 NPU) timing model — PLACEHOLDER.

BLOCKED ON GitHub issue #13: the RKNPU2 matmul/attention micro-benchmark
(measurements/aetina/rknpu2_matmul.json) was not collected (aetina offline during
Phase 0.3). The .rknn artifacts are staged and the runner is ready; once #13 resolves,
fit this the same way as m4_gpu (FLOPs/G_eff + native attention bmm) from
rknpu2_matmul.json.

Until then NPU is a documented dependency, not a modeled unit. The attention-offload
argument stands on the GPU (Mali) comparison alone (see m4_gpu).
"""


class NpuModel:
    def __init__(self, params=None):
        raise NotImplementedError(
            "M4 NPU not modeled: blocked on issue #13 (rknpu2_matmul.json not collected). "
            "Fit from rknpu2_matmul.json when #13 resolves.")
