"""M1 — CIM tile timing model (Phase 1 fit). Loads params from params/m1_cim.json.

ARCHITECTURE (ISSCC 2024, papers/metis-silicon/metis-aipu-isscc2024.md):
Metis is QUAD-CORE; each AI-Core has a 512x512 INT8 D-IMC crossbar (16 banks of
512-in x 32-out x 4 weight-sets). The simulator's minimum unit is ONE CORE (512 wide);
`n_cores` is a free parameter (= 4 for Metis). The effective output width of the
combined engine = n_cores * 512 (= 2048 at n=4). Throughput is INT8 OP/s (GOP/s), NOT
FLOP/s. Our Phase-0.3 measurements used the default compile (presumed all 4 cores; the
2048 = 4x512 boundary is the evidence) -> the fit is calibrated at n_cores=4.

Device latency (decode/memory-bound regime; M=1 is the calibrated path):
  dev_lat = sum over output-tiles (each <= n_cores*512 wide) of FLOPs_tile / G_eff(n,K)
where G_eff(N,K) (GOP/s) is the effective throughput, rising with BOTH the output width
N and input depth K (2D fill), fit on native single-tile points (K*N <= 4.19M). Tiling
along N keeps the latency RISING (a partial last tile adds its own smaller latency, not a
full tile). Everything above one native tile (K*N > 4.19M) is UNVALIDATED extrapolation.

The 911 us per-call host<->device DMA floor is M2's, not M1's. The compute ceiling
(~52 TOPS/core) is NOT modeled: decode never approaches it (issue #16).
"""
import json
import math
from pathlib import Path

_PARAMS = Path(__file__).parent / "params" / "m1_cim.json"
CORE_WIDTH = 512  # per-core D-IMC crossbar output width (ISSCC 2024)


class CimTileModel:
    def __init__(self, params=None, n_cores=None):
        p = params if params is not None else json.loads(_PARAMS.read_text())
        self.n_cores = n_cores if n_cores is not None else p.get("n_cores", 4)
        self.core_width = p.get("core_width", CORE_WIDTH)
        # 2D effective-throughput closed form: G = Gmax * N/(N+Na) * K/(K+Kb)  (GOP/s)
        self.Gmax = p["G_eff_Gmax_gops"]
        self.Na = p["G_eff_Na"]
        self.Kb = p["G_eff_Kb"]
        self.native_max_kn = p.get("native_max_kn", 4_194_304)  # 2048*2048; above = extrapolated
        self.alloc_envelope = p.get("alloc_envelope_params", 6_000_000)  # SDK weight-alloc limit
        # prefill (M>1): canonical-tile latency is AFFINE in M, tile_lat=a+b*M (Card-measured by
        # fit_cim_prefill.py at M<={1,64,128,256}; M>prefill_M_max extrapolated). None until fit.
        self.prefill_a_us = p.get("prefill_tile_a_us")
        self.prefill_b_us = p.get("prefill_tile_b_us")
        self.prefill_M_max = p.get("prefill_M_max")
        self.prefill_M_min = min(p["prefill_M_fit"]) if p.get("prefill_M_fit") else None  # fit basis lower bound

    @property
    def width(self):
        return self.n_cores * self.core_width   # effective combined output width (2048 at n=4)

    def g_eff(self, N, K):
        """Effective INT8 throughput (GOP/s) vs output width N and input depth K (2D fill)."""
        return self.Gmax * (N / (N + self.Na)) * (K / (K + self.Kb))

    def dev_lat_us(self, M, K, N):
        """Device compute latency (us) of (M,K)x(K,N). Two regimes:

        M<=1 (DECODE, calibrated): tile the output N into passes of width <= n_cores*512; each
        pass costs by its OWN size via the 2D throughput, so latency keeps rising across tiles
        (a partial last tile adds less, not a full tile). K*N > native_max_kn is extrapolation.

        M>1 (PREFILL, Card-measured): the 2048x2048 weight tile's load is amortized over M
        activation columns -> tile latency is AFFINE in M (a + b*M, fit on the FULL-tile prefill
        points M in {64,128,256}; the M=1 decode point is NOT in this fit). A GEMM costs
        (K*N / W^2) * (a + b*M): FRACTIONAL tile area (NOT ceil), so compute and weight-load scale
        with the actual K*N -- a partial-width GEMM is not over-charged a full tile, and the value
        equals the integer tile count for W-multiple shapes. Linear-in-M extrapolation of the DECODE
        law would over-predict ~80x. UNVALIDATED where prefill_extrapolated() is True: M>prefill_M_max
        (256), M<prefill_M_min (the 1<M<64 below-fit band), or partial-width tiles (K or N not a
        multiple of W) -- the fit used only full tiles at M in {64,128,256}.

        NO monotonicity clamp across the decode<->prefill boundary (issues #35/#39): decode (M=1) and
        prefill (M>=64) are two separately-calibrated models with NO data between, disagreeing up to
        ~2.5x at the narrow-K corner. Any clamp either over-charges partial-width (#39) or mixes cost
        bases (#35). So the un-bridged region -- where the prefill value can dip BELOW the M=1 decode
        value -- is surfaced HONESTLY by prefill_extrapolated()==True rather than fake-monotonized. In
        the CALIBRATED region (M>=64, full-width) prefill is monotone in M and sits above the M=1 decode.
        """
        if M <= 1:
            return self._decode_lat_us(K, N) * M   # M=1 = the calibrated decode floor (linear in M)
        if self.prefill_a_us is None:
            raise ValueError("prefill (M>1) latency needs the M-amortization fit; "
                             "run tools/analysis/fit_cim_prefill.py")
        W = self.width   # canonical tile edge (= n_cores*512 = 2048); native_max_kn = W*W
        area = (K * N) / (W * W)   # fractional tile area (not ceil): no padded-tile over-charge
        return area * (self.prefill_a_us + self.prefill_b_us * M)

    def _decode_lat_us(self, K, N):
        """M=1 decode latency (us): output N tiled into passes of width <= W, each costed by its own
        size via the 2D throughput (partial-fill-aware). The calibrated path."""
        W = self.width
        lat, rem = 0.0, N
        while rem > 0:
            n = min(W, rem)
            lat += 2.0 * K * n / (self.g_eff(n, K) * 1e9) * 1e6
            rem -= n
        return lat

    def is_extrapolated(self, K, N):
        """True if (K,N) is beyond the largest natively measured single tile (decode regime)."""
        return K * N > self.native_max_kn

    def prefill_extrapolated(self, M, K, N):
        """True if an M>1 prefill prediction is outside the calibrated range: M>prefill_M_max (only
        M<=256 compiles/was measured), M<prefill_M_min (below the M>=64 fit basis -- 1<M<64 is
        extrapolation, NOT calibrated), or a partial-width tile (K or N not a multiple of W -- the
        affine fit used only full 2048x2048 tiles, so sub-tile-width prefill is uncalibrated)."""
        W = self.width
        return (M > (self.prefill_M_max or M) or M < (self.prefill_M_min or 0)
                or (K % W != 0) or (N % W != 0))

    def n_tiles(self, N):
        return math.ceil(N / self.width)
