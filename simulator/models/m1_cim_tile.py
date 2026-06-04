"""M1 — CIM tile timing model (Phase 1 fit). Loads params from params/m1_cim.json.

Dev (compute) latency of a weight-stationary GEMV/GEMM on the Metis Alpha CIM
(1x1-conv proxy), fit from measurements/aetina/metis_alpha_matmul.json.

Decode (M=1) is the calibrated path. The effective throughput G_eff(N) (GFLOP/s)
rises with output-channel fill N and saturates — fit on the channel-64 staircase
(group `staircase64`). Latency = FLOPs / G_eff(N); FLOPs = 2*M*K*N. The 2048x2048
crossbar-tile count governs the device envelope (K*N > ~6M -> tiled) and feeds the
per-call DMA-floor compounding handled in M2. Prefill (M>1) uses the same form scaled
by M but is UNVALIDATED (no board data at M>=512; see plan §背景).

The per-call host<->device DMA floor (911 us) is M2's, not M1's: this model returns
device compute latency only.
"""
import json
import math
from pathlib import Path

_PARAMS = Path(__file__).parent / "params" / "m1_cim.json"


class CimTileModel:
    def __init__(self, params=None):
        p = params if params is not None else json.loads(_PARAMS.read_text())
        self.Gmax = p["G_eff_Gmax_gflops"]      # saturated effective throughput
        self.Nhalf = p["G_eff_Nhalf"]           # half-saturation output channels
        self.tile = p["crossbar_tile"]          # 2048 x 2048 physical tile
        self.T_tile_us = p["canonical_tile_us"] # full 2048x2048 tile dev latency
        self.envelope = p["device_envelope_params"]  # ~6e6 allocatable K*N
        self.lookup = {int(k): v for k, v in p.get("G_eff_lookup", {}).items()}
        self.use_lookup = p.get("use_lookup", False)

    def g_eff(self, N):
        """Effective throughput (GFLOP/s) vs output-channel fill N (saturating)."""
        if self.use_lookup and self.lookup:
            return self._interp(N)
        return self.Gmax * N / (N + self.Nhalf)

    def _interp(self, N):
        xs = sorted(self.lookup)
        if N <= xs[0]:
            return self.lookup[xs[0]]
        if N >= xs[-1]:
            return self.lookup[xs[-1]]          # saturate above the swept range
        for a, b in zip(xs, xs[1:]):
            if a <= N <= b:
                return self.lookup[a] + (self.lookup[b] - self.lookup[a]) * (N - a) / (b - a)

    def dev_lat_us(self, M, K, N):
        """Device compute latency (us) of (M,K)x(K,N). M=1 = decode (calibrated path).

        Two regimes: narrow output (N < one tile) follows the underfill curve
        FLOPs/G_eff(N); a full/multi-tile output (N >= tile) is n_tiles*T_tile, which
        matches the padded-tile measured latency (incl. Qwen non-2048 dims, no restore).
        The wide-K + narrow-N case (e.g. 8B kv-proj K=4096,N=1024) is a known residual
        the underfill curve over-predicts (reported separately, plan verify e).
        """
        if N < self.tile:                        # narrow output -> underfill curve
            return 2.0 * M * K * N / (self.g_eff(N) * 1e9) * 1e6
        return M * self.n_tiles(K, N) * self.T_tile_us

    def n_tiles(self, K, N):
        return math.ceil(K / self.tile) * math.ceil(N / self.tile)

    def is_tiled(self, K, N):
        return K * N > self.envelope
