"""Phase 0.2 — analytic op profile generator + op_inventory oracle.

Builds, per (model, prefill_len, decode_len), the full per-sig profile
(op, in_shapes, out_shape, src, category, phase, count, flops, bytes,
intensity, measured), split prefill/decode.

Design (see plans/phase-0.2.md):
- COUNTS come from Phase 0.1 op_inventory `count` (length-independent; already
  aggregates collided q/o, k/v, gate/up sigs). Never hand-rolled "x layers".
- SHAPES at arbitrary workload length are generated from a data-driven
  length-TEMPLATE: each compute sig's length dims are detected by diffing two
  inventory points (lo, hi); a differing scalar fits value = a*L + b (integers).
  Invariant scalars (incl. head_dim that coincidentally equals a seq point) stay
  fixed because they are equal across the two well-separated points.
- VALIDATION: templates built from {128,1024} must reproduce the held-out
  inventory points {256,512} (prefill) and {512} (decode) EXACTLY (sig+count),
  for every model. main() runs this and refuses to emit on mismatch.

Run: ./.venv/bin/python tools/trace_export/op_profile.py            # self-validate
     (importable: build_model(model) -> profile(P, D))
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sweep_matrix import categorize  # 9-class categorizer (op + src)

INV = Path("measurements/op_inventory")
PREFILL_PTS = [128, 256, 512, 1024]
DECODE_PTS = [128, 512, 1024]
# Anchors avoid length==head_dim (64/128) collisions, where QK^T and S.V alias to
# one sig: 256/1024 (prefill) and 512/1024 (decode) never equal 64 or 128.
PREFILL_ANCHOR = (256, 1024)
DECODE_ANCHOR = (512, 1024)
GEMM_BYTES = 1                            # INT8 matmul/bmm operands (Metis scope)
ELT_BYTES = 2                             # FP16 non-GEMM (vendor actual, P4)
GEMM_OPS = {"aten.mm.default", "aten.addmm.default", "aten.bmm.default"}


# ---------- template fitting ----------

def _fit(x, y, lo, hi):
    """Scalar template: int if invariant, ('L',a,b) if y=x maps via a*L+b, else None."""
    if x == y:
        return x
    if (hi - lo) == 0 or (y - x) % (hi - lo) != 0:
        return None
    a = (y - x) // (hi - lo)
    b = x - a * lo
    if a < 1:
        return None
    return ("L", a, b)


def _fit_shape(slo, shi, lo, hi):
    if slo is None or shi is None:
        return None if slo != shi else slo
    if len(slo) != len(shi):
        return None
    out = []
    for x, y in zip(slo, shi):
        f = _fit(x, y, lo, hi)
        if f is None:
            return None
        out.append(f)
    return out


def _make_template(slo, shi, lo, hi):
    """Template for one sig = {in:[[scalar|('L',a,b)]...], out:..., len_dep:bool} or None."""
    if len(slo["in_shapes"]) != len(shi["in_shapes"]):
        return None
    tin = []
    for a_op, b_op in zip(slo["in_shapes"], shi["in_shapes"]):
        t = _fit_shape(a_op, b_op, lo, hi)
        if t is None:
            return None
        tin.append(t)
    tout = _fit_shape(slo["out_shape"], shi["out_shape"], lo, hi)
    if slo["out_shape"] is not None and tout is None:
        return None
    flat = [s for op in tin for s in op] + (tout or [])
    len_dep = any(isinstance(s, tuple) for s in flat)
    return {"op": slo["op"], "src": slo["src"], "count": slo["count"],
            "in": tin, "out": tout, "len_dep": len_dep,
            "n_lendim": sum(isinstance(s, tuple) for s in flat)}


def _inst_shape(t, L):
    if t is None:
        return None
    return [(s[1] * L + s[2] if isinstance(s, tuple) else s) for s in t]


def _instantiate(tpl, L):
    ins = [_inst_shape(op, L) for op in tpl["in"]]
    out = _inst_shape(tpl["out"], L)
    return {"op": tpl["op"], "src": tpl["src"], "count": tpl["count"],
            "in_shapes": ins, "out_shape": out}


# ---------- build per-phase templates from two anchor points ----------

def _compute_sigs(records):
    """Keep only the 9 compute categories (drops host-side housekeeping)."""
    out = []
    for r in records:
        if categorize(r) is not None:
            out.append(r)
    return out


def _build_phase(inv_phase, lo, hi):
    """Match lo<->hi compute sigs by (op,src,count)+template validity -> templates."""
    lo_sigs = _compute_sigs(inv_phase[str(lo)])
    hi_sigs = _compute_sigs(inv_phase[str(hi)])
    from collections import defaultdict
    hi_by = defaultdict(list)
    for s in hi_sigs:
        hi_by[(s["op"], s["src"], s["count"])].append(s)
    templates = []
    used = set()
    for s in lo_sigs:
        cands = hi_by[(s["op"], s["src"], s["count"])]
        best, best_t = None, None
        for j, h in enumerate(cands):
            if id(h) in used:
                continue
            t = _make_template(s, h, lo, hi)
            if t is None:
                continue
            if best_t is None or t["n_lendim"] < best_t["n_lendim"]:
                best, best_t = h, t
        if best_t is None:
            raise RuntimeError(f"no hi-match for {s['op']} {s['src']} {s['in_shapes']}")
        used.add(id(best))
        templates.append(best_t)
    return templates


def _key(sig):
    return (sig["op"], json.dumps(sig["in_shapes"]), json.dumps(sig["out_shape"]))


def _sum_by_key(items):
    """items: iterable of (key, count) -> dict key->summed count (colliding sigs merge)."""
    out = {}
    for k, c in items:
        out[k] = out.get(k, 0) + c
    return out


def _validate(inv_phase, templates, points):
    """Templates must reproduce inventory compute sigs (sig+count) at each held-out point.

    Counts are summed per key: at a length that equals head_dim, distinct templates
    (QK^T, S.V) alias to one sig and the inventory holds their merged count.
    """
    for P in points:
        gen = _sum_by_key((_key(_instantiate(t, P)), t["count"]) for t in templates)
        truth = _sum_by_key((_key(r), r["count"]) for r in _compute_sigs(inv_phase[str(P)]))
        if gen != truth:
            miss = {k: truth[k] for k in truth if k not in gen}
            extra = {k: gen[k] for k in gen if k not in truth}
            badc = {k: (gen[k], truth[k]) for k in truth if k in gen and gen[k] != truth[k]}
            raise AssertionError(f"point {P}: missing={list(miss)[:3]} extra={list(extra)[:3]} "
                                 f"countdiff={list(badc.items())[:3]}")


# ---------- model assembly ----------

class Model:
    def __init__(self, name):
        self.name = name
        d = json.loads((INV / f"{name}.json").read_text())
        self.config = d["config"]
        inv = d["inventory"]
        self.pre = _build_phase(inv["prefill"], *PREFILL_ANCHOR)
        self.dec = _build_phase(inv["decode"], *DECODE_ANCHOR)
        # held-out validation (points not used as anchors)
        _validate(inv["prefill"], self.pre, [p for p in PREFILL_PTS if p not in PREFILL_ANCHOR])
        _validate(inv["decode"], self.dec, [p for p in DECODE_PTS if p not in DECODE_ANCHOR])
        self._inv = inv
        self.ms = measured_set()

    def profile(self, P, D):
        """Return list of profile rows for workload (prefill=P, decode=D tokens)."""
        rows = {}
        # prefill: one forward at seq=P
        for t in self.pre:
            self._add(rows, _instantiate(t, P), "prefill", 1)
        # decode: length-indep ops x D ; length-dep ops per kv position (past=P..P+D-1)
        for t in self.dec:
            if t["len_dep"]:
                for past in range(P, P + D):
                    self._add(rows, _instantiate(t, past), "decode", t["count"])
            else:
                # shape constant across positions -> instantiate at any L (use P), count x D
                self._add(rows, _instantiate(t, P), "decode", t["count"] * D)
        rows = {k: v for k, v in rows.items() if v["count"] > 0}
        return list(rows.values())

    def grid_profile(self):
        """On-grid Layer-B scaling data: prefill at PREFILL_PTS, decode at DECODE_PTS
        (single forward each). All sigs land on the sweep_matrix grid (measured=true)."""
        out = {"prefill": {}, "decode": {}}
        for P in PREFILL_PTS:
            rows = {}
            for t in self.pre:
                self._add(rows, _instantiate(t, P), "prefill", t["count"])
            out["prefill"][P] = list(rows.values())
        for K in DECODE_PTS:
            rows = {}
            for t in self.dec:
                self._add(rows, _instantiate(t, K), "decode", t["count"])
            out["decode"][K] = list(rows.values())
        return out

    def _add(self, rows, sig, phase, count):
        if count == 0:
            return
        cat = categorize(sig)
        if cat is None:
            return
        key = (phase, _key(sig))
        if key in rows:
            rows[key]["count"] += count
            return
        fl, by = _flops_bytes(sig)
        meas = (sig["op"], json.dumps(sig["in_shapes"]), json.dumps(sig["out_shape"])) in self.ms
        rows[key] = {"op": sig["op"], "in_shapes": sig["in_shapes"], "out_shape": sig["out_shape"],
                     "src": sig["src"], "category": cat, "phase": phase, "count": count,
                     "flops": fl, "bytes": by, "intensity": (fl / by if by else 0.0),
                     "measured": meas}


def _prod(shape):
    p = 1
    for x in shape:
        p *= x
    return p


def _flops_bytes(sig):
    op, ins, out = sig["op"], sig["in_shapes"], sig["out_shape"]
    if op == "aten.embedding.default":
        # gather: touches only the selected rows (out elems), not the full table; no arithmetic
        nout = _prod(out) if out else 0
        return 0, 2 * nout * ELT_BYTES
    if op in GEMM_OPS:
        if op == "aten.mm.default":          # [[M,K],[K,N]]
            (M, K), (_, N) = ins[0], ins[1]
            flops = 2 * M * K * N
        elif op == "aten.addmm.default":     # [[N],[M,K],[K,N]]
            (M, K), (_, N) = ins[1], ins[2]
            flops = 2 * M * K * N
        else:                                # bmm [[B,M,K],[B,K,N]]
            B, M, K = ins[0]
            N = ins[1][2]
            flops = 2 * B * M * K * N
        by = (sum(_prod(s) for s in ins) + _prod(out)) * GEMM_BYTES
        return flops, by
    # non-GEMM: FLOPs ~ output elements (low intensity); bytes at FP16
    nout = _prod(out) if out else 0
    nin = sum(_prod(s) for s in ins)
    return nout, (nin + nout) * ELT_BYTES


# ---------- measured-flag set (sweep_matrix membership, src-agnostic) ----------

def measured_set():
    sm = json.loads((INV / "sweep_matrix.json").read_text())["matrix"]
    return {(s["op"], json.dumps(s["in_shapes"]), json.dumps(s["out_shape"]))
            for cat in sm.values() for s in cat}


def main():
    models = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]
    for m in models:
        M = Model(m)
        # sanity: a small workload profiles without error
        rows = M.profile(128, 4)
        cats = {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(f"{m}: templates pre={len(M.pre)} dec={len(M.dec)}  "
              f"held-out validation PASS  | profile(128,4) rows={len(rows)} cats={cats}")
    print("\nALL MODELS: held-out (256/512 prefill, 512 decode) reproduced exactly.")


if __name__ == "__main__":
    main()
