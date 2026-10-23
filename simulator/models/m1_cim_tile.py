"""M1 — CIM tile timing model (Phase 1 fit). Loads params from params/m1_cim.json.

ARCHITECTURE (ISSCC 2024, docs/papers/metis-silicon/metis-aipu-isscc2024.md):
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
full tile).

Multi-tile M=1 (some dim > W), Card-measured Phase 1.5 — RESIDENCY CLIFF (_decode_multitile_lat_us):
resident (K*N <= knee ~8M) the weights fit on-chip SRAM and latency is affine in K*N; above the
knee they spill to DRAM and throughput collapses ~3.5x to a memory-bound floor (~70 GOP/s). This
replaced the old tile-sum extrapolation (over below the knee, ~-65% under above it; +31% abs median
overall). Native envelope measured to K*N ~16.8M; beyond that the floor extrapolates (memory-bound
linear). NB a ~5% latency seam at the single-tile<->resident boundary (K*N=4.19M): two independently
calibrated models, no data exactly at the seam (cf. the decode<->prefill seam) — left un-clamped, not faked.

The 911 us per-call host<->device DMA floor is M2's, not M1's. The compute ceiling
(~52 TOPS/core) is NOT modeled: decode never approaches it (issue #16).

n_cores SCALING (ADR-0008 revision F12). `scale_mode="legacy"` (default, the L4 path) keeps the
Phase-1.5 numerics exactly: n_cores only widens the pass W = n_cores*512 (prefill area K*N/W^2
-> 1/n^2, the multi-tile cliff independent of n). `scale_mode="physical"` (SoC profiles) scales
the calibrated 4-core fit by s = n_cores / n_cal with explicit physics: compute throughput
proportional to cores (single-tile: s x G_eff(N/s, K) — the output columns are spread over the
cores, each at the per-core fill law; prefill tile throughput x s -> latency 1/n; resident slope
b/s), SRAM residency proportional to cores (multi-tile knee x s), and the DRAM-spill floor kept
as the array's measured ingest rate (n-independent kernel time; the shared pool meters the DRAM
bytes themselves, F2). At s = 1 physical == legacy by construction. No silicon at s != 1.
"""
import json
import math
from pathlib import Path

_PARAMS = Path(__file__).parent / "params" / "m1_cim.json"
CORE_WIDTH = 512  # per-core D-IMC crossbar output width (ISSCC 2024)


_SCALE_MODES = ("legacy", "physical")


class CimTileModel:
    def __init__(self, params=None, n_cores=None, scale_mode="legacy"):
        p = params if params is not None else json.loads(_PARAMS.read_text())
        self.n_cores = n_cores if n_cores is not None else p.get("n_cores", 4)
        self.n_cal = p.get("n_cores", 4)             # the core count the fit is calibrated at (Metis 4)
        if scale_mode not in _SCALE_MODES:
            raise ValueError(f"CimTileModel scale_mode must be one of {_SCALE_MODES}, got {scale_mode!r}")
        self.scale_mode = scale_mode
        self.core_width = p.get("core_width", CORE_WIDTH)
        # 2D effective-throughput closed form: G = Gmax * N/(N+Na) * K/(K+Kb)  (GOP/s)
        self.Gmax = p["G_eff_Gmax_gops"]
        self.Na = p["G_eff_Na"]
        self.Kb = p["G_eff_Kb"]
        self.native_max_kn = p.get("native_max_kn", 4_194_304)  # 2048*2048 = one combined tile
        self.alloc_envelope = p.get("alloc_envelope_params", 6_000_000)  # SDK weight-alloc limit
        # native multi-tile (M=1, some dim > W): RESIDENCY CLIFF (Card-measured, Phase 1.5). Resident
        # (K*N <= knee) the weights fit on-chip SRAM -> latency affine in K*N; above the knee they
        # spill to DRAM -> memory-bound floor throughput (~3.5x slower). Replaces the old tile-sum
        # (over below the knee, ~-65% under above it; +31% abs median overall). None until fit_cim_multitile.py.
        self.mt_knee_kn = p.get("multitile_knee_kn")
        self.mt_resident_a_us = p.get("multitile_resident_a_us")
        self.mt_resident_b_us = p.get("multitile_resident_b_us")   # us per K*N param (resident slope)
        self.mt_floor_gops = p.get("multitile_floor_gops")         # DRAM-spill memory-bound floor
        self.native_envelope_kn = p.get("native_envelope_kn", self.native_max_kn)  # max natively measured K*N
        # prefill (M>1): canonical-tile latency is AFFINE in M, tile_lat=a+b*M (Card-measured by
        # fit_cim_prefill.py on the dense sweep M=2..prefill_M_max (508); M>prefill_M_max extrapolated). None until fit.
        self.prefill_a_us = p.get("prefill_tile_a_us")
        self.prefill_b_us = p.get("prefill_tile_b_us")
        self.prefill_M_max = p.get("prefill_M_max")
        self.prefill_M_min = min(p["prefill_M_fit"]) if p.get("prefill_M_fit") else None  # fit basis lower bound

    @property
    def width(self):
        return self.n_cores * self.core_width   # effective combined output width (2048 at n=4)

    @property
    def core_scale(self):
        """s = n_cores / calibrated cores (1.0 in legacy mode, where n_cores only widens the pass)."""
        return (self.n_cores / self.n_cal) if self.scale_mode == "physical" else 1.0

    @property
    def mt_knee(self):
        """Resident-weight capacity (K*N elements) at this core count: the calibrated knee x s."""
        return None if self.mt_knee_kn is None else self.mt_knee_kn * self.core_scale

    def g_eff(self, N, K):
        """Effective INT8 throughput (GOP/s) vs output width N and input depth K (2D fill)."""
        return self.Gmax * (N / (N + self.Na)) * (K / (K + self.Kb))

    def dev_lat_us(self, M, K, N):
        """Device compute latency (us) of (M,K)x(K,N). Two regimes:

        M<=1 (DECODE, calibrated): tile the output N into passes of width <= n_cores*512; each
        pass costs by its OWN size via the 2D throughput, so latency keeps rising across tiles
        (a partial last tile adds less, not a full tile). K*N > native_max_kn is extrapolation.

        M>1 (PREFILL, Card-measured): the 2048x2048 weight tile's load is amortized over M
        activation columns -> tile latency is AFFINE in M (a + b*M, fit on the dense full-tile
        prefill sweep M=2..prefill_M_max (508); the M=1 decode point is NOT in this fit). A GEMM costs
        (K*N / W^2) * (a + b*M): FRACTIONAL tile area (NOT ceil), so compute and weight-load scale
        with the actual K*N -- a partial-width GEMM is not over-charged a full tile, and the value
        equals the integer tile count for W-multiple shapes. Linear-in-M extrapolation of the DECODE
        law would over-predict ~80x. UNVALIDATED where prefill_extrapolated() is True: M>prefill_M_max
        (508; M=511/512 fail to compile), or partial-width tiles (K or N not a multiple of W) --
        the fit used only full 2048x2048 tiles.

        NO monotonicity clamp across the decode<->prefill boundary (issues #35/#39): decode (M=1) and
        prefill (M=2..508) are two separately-calibrated models with NO data between, disagreeing up to
        ~2.5x at the narrow-K corner. Any clamp either over-charges partial-width (#39) or mixes cost
        bases (#35). So the un-bridged region -- where the prefill value can dip BELOW the M=1 decode
        value -- is surfaced HONESTLY by prefill_extrapolated()==True rather than fake-monotonized.
        """
        if M <= 0 or K <= 0 or N <= 0:
            raise ValueError(f"dev_lat_us requires M,K,N > 0; got M={M}, K={K}, N={N}")
        if M <= 1:
            return self._decode_lat_us_full(K, N) * M   # M=1 = the calibrated decode floor (linear in M)
        if self.prefill_a_us is None:
            raise ValueError("prefill (M>1) latency needs the M-amortization fit; "
                             "run tools/analysis/fit_cim_prefill.py")
        if self.scale_mode == "physical":
            # F12: tile throughput proportional to cores -> the calibrated-tile area / s (1/n scaling)
            W = self.n_cal * self.core_width
            return (K * N) / (W * W) / self.core_scale * (self.prefill_a_us + self.prefill_b_us * M)
        W = self.width   # canonical tile edge (= n_cores*512 = 2048); native_max_kn = W*W
        area = (K * N) / (W * W)   # fractional tile area (not ceil): no padded-tile over-charge
        return area * (self.prefill_a_us + self.prefill_b_us * M)

    def _decode_lat_us(self, K, N):
        """M=1 decode latency (us): output N tiled into passes of width <= W, each costed by its own
        size via the 2D throughput (partial-fill-aware). The calibrated single-tile path; for genuine
        multi-tile (some dim > W) this is the TILE-SUM — replaced by the cliff model, kept as the
        fallback + the old-vs-new baseline for fit_cim_multitile."""
        W = self.width
        s = self.core_scale
        lat, rem = 0.0, N
        while rem > 0:
            n = min(W, rem)
            # physical mode: the n output columns spread over s x the calibrated cores, each at the
            # per-core fill law -> throughput s * G_eff(n/s, K) (= G_eff(n, K) at s = 1)
            lat += 2.0 * K * n / (s * self.g_eff(n / s, K) * 1e9) * 1e6
            rem -= n
        return lat

    def _decode_lat_us_full(self, K, N):
        """M=1 latency dispatcher: single-tile (both dims <= W) uses the G_eff tiling fit; genuine
        multi-tile (some dim > W) uses the native residency-cliff model when fit, else the tile-sum."""
        if (K <= self.width and N <= self.width) or self.mt_knee_kn is None:
            return self._decode_lat_us(K, N)
        return self._decode_multitile_lat_us(K, N)

    def spills(self, K, N):
        """True iff an M=1 multi-tile weight set (K*N) exceeds the resident capacity at this core
        count (physical mode: knee x s). Single-tile shapes never spill."""
        if (K <= self.width and N <= self.width) or self.mt_knee_kn is None:
            return False
        return K * N > self.mt_knee

    def _decode_multitile_lat_us(self, K, N):
        """Native multi-tile M=1 latency (us), Card-calibrated (Phase 1.5). Resident (K*N <= knee):
        weights fit SRAM -> latency affine in K*N. Spill (K*N > knee): weights stream from DRAM ->
        memory-bound floor throughput. K*N > native_envelope_kn is floor-extrapolated (memory-bound
        linear, low risk; is_extrapolated flags it)."""
        kn = K * N
        if kn <= self.mt_knee:
            return self.mt_resident_a_us + self.mt_resident_b_us / self.core_scale * kn
        return 2.0 * kn / (self.mt_floor_gops * 1e9) * 1e6     # spill: array ingest rate (n-independent)

    def is_extrapolated(self, K, N):
        """True if (K,N) is beyond the natively measured M=1/K*N envelope ONLY. Single-tile (both dims
        <= W) and multi-tile up to native_envelope_kn are Card-measured; beyond that the spill floor is
        memory-bound-linear extrapolated. Prefill (M>1) extrapolation is NOT covered here -- use
        prefill_extrapolated() for the M-axis (M>prefill_M_max / partial-width) calibration limits."""
        return K * N > self.native_envelope_kn

    def prefill_extrapolated(self, M, K, N):
        """True if an M>1 prefill prediction is outside the calibrated range: M>prefill_M_max (508;
        M=511/512 fail to compile), M<prefill_M_min (below the dense fit basis -- prefill_M_min =
        min(prefill_M_fit) = 2, so M=2..508 IS calibrated), or a partial-width tile (K or N not a
        multiple of W -- the affine fit used only full 2048x2048 tiles, so sub-tile-width prefill is
        uncalibrated)."""
        W = self.width
        return (M > (self.prefill_M_max or M) or M < (self.prefill_M_min or 0)
                or (K % W != 0) or (N % W != 0))

    def n_tiles(self, N):
        return math.ceil(N / self.width)


_ATTN_PARAMS = Path(__file__).parent / "params" / "m1_cim_attention.json"


class CimAttentionModel:
    """CIM decode-attention latency from the MEASURED Metis Alpha `attn_floor` points
    (measurements/aetina/metis_alpha_matmul.json by_group.attn_floor: single-head QK^T [1x128]x[128xkv]
    and S.V [1xkv]x[kvx128] dev_lat_us at kv = 129/513/1025, hd = 128, Llama-3.1-8B), fit as
    a + b*kv per bmm (tools/analysis/fit_hetero_anchors.py -> params/m1_cim_attention.json).

    The measured basis is a tiny per-call compute FLOOR (~9 us per bmm: the AIPU's standalone-model
    call cost, far above the G_eff single-tile law for such shapes) that grows weakly with kv.
    Composite per head = (qk_a + sv_a) + (qk_b + sv_b) * kv; heads are SERIAL calls (ASSUMPTION — Alpha
    measured one head per model). EXTRAPOLATED when: hd != 128, kv > 1025, or M > 1 (prefill rows:
    per bmm = max(floor fit, the M1 prefill-tile law dev_lat_us(M, K, N)) — no prefill attention
    silicon). NB the L4 Card anchor passes with CIM attention priced at ZERO compute (the compiled
    axllm graph shows no per-call floor), so this floor is the Alpha standalone-call basis, and it is
    enabled only through a SoC profile (`attention_model: alpha_floor_fit`), never on the legacy path."""

    HEADS_MODES = ("serial", "batched")

    def __init__(self, tile: "CimTileModel", params=None, heads_mode="serial"):
        p = params if params is not None else json.loads(_ATTN_PARAMS.read_text())
        if heads_mode not in self.HEADS_MODES:
            raise ValueError(f"CimAttentionModel heads_mode must be one of {self.HEADS_MODES}, got {heads_mode!r}")
        self.tile = tile
        self.heads_mode = heads_mode
        self.qk_a, self.qk_b = float(p["qk_a_us"]), float(p["qk_b_us_per_kv"])
        self.sv_a, self.sv_b = float(p["sv_a_us"]), float(p["sv_b_us_per_kv"])
        self.hd_basis = int(p.get("hd_basis", 128))
        self.kv_max = int(max(p.get("kv_fit", [1025])))

    def composite_us(self, kv, heads=1, hd=128, M=1):
        """QK^T + S.V latency (us) over `heads` heads. Returns (latency_us, extrapolated).
        heads_mode (ADR-0008 revision F6, a profile sensitivity axis `cim.attention_heads_mode`):
          'serial'  = one standalone call per head per bmm: (floor_a + b*kv) x heads — the measured
                      basis (Alpha ran one head per model), the pessimistic default;
          'batched' = one call per bmm with the heads batched inside the tile: floor_a paid ONCE per
                      bmm, the kv-proportional work (b*kv, or the M1 tile law for M>1) x heads —
                      what a compiled multi-head graph would do (no silicon for it)."""
        kv, heads, M = max(1, int(kv)), max(1, int(heads)), max(1, int(M))
        qk_w, sv_w = self.qk_b * kv, self.sv_b * kv            # per-head kv-proportional work
        extrap = (hd != self.hd_basis) or (kv > self.kv_max) or (M > 1)
        if M > 1:   # prefill rows: the per-call floor still applies; the tile law may exceed it
            qk_w = max(qk_w, self.tile.dev_lat_us(M, hd, kv) - self.qk_a)
            sv_w = max(sv_w, self.tile.dev_lat_us(M, kv, hd) - self.sv_a)
        if self.heads_mode == "serial":
            return (self.qk_a + qk_w + self.sv_a + sv_w) * heads, extrap
        return self.qk_a + self.sv_a + (qk_w + sv_w) * heads, extrap
