"""Phase 1.6b — MEASURE whether ONNXim exhibits the systolic characteristics (no presupposed knee).

Independent of the committed analytic-vs-ONNXim SPREAD: this writes its OWN output
(simulator/engines/onnxim/rknpu2_characteristic.json) and does NOT touch npu_onnxim_trace.py's canonical
SHAPES (which back the 317.9% spread / delta report). We sweep ONNXim on a grid fine enough to
resolve a 32-period staircase and let the data REJECT or confirm H0 = "smooth, no 32-step".

H0 = smooth ∝N^b, no 32-quantum step. Burden of proof is on rejecting it. ScaleSim (a literal
32x32 array sim) is a POSITIVE CONTROL run separately; ONNXim (adds NoC/DRAM scheduling) is the
informative one. simulated, NOT silicon (#13).

Runs ONE-or-more docker runs on metiscard (reusing npu_onnxim_trace's sweep.sh pattern). N>=128
(ONNXim SIGFPE on N<=64). Usage:
  ./.venv/bin/python tools/analysis/npu_characteristic_trace.py pilot   # small boundary subset first
  ./.venv/bin/python tools/analysis/npu_characteristic_trace.py         # full E1-E4 grid
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tools/onnxim/rknpu2_approx.json"
OUT = ROOT / "simulator/engines/onnxim/rknpu2_characteristic.json"
HOST = "metiscard"
IO = "~/edge-cim-simulation/onnxim_io"
ONNXIM_COMMIT = "a1e86296"
K = 2048   # fixed contraction for the output-dim sweep (decode-projection regime)


def e1_n_grid():
    """N grid that can resolve a 32-period: uniform step-8 bulk + 1-spaced boundary microsweeps
    straddling several 32-multiples. step-8 gives 4 samples/32-block (Nyquist OK); microsweeps put
    the across-boundary Δ AT the boundary."""
    bulk = list(range(128, 513, 8))
    micro = [c + d for c in (160, 192, 256, 384) for d in (-2, -1, 0, 1, 2)]
    return sorted(set(bulk + micro))


def shapes(pilot=False):
    if pilot:
        # cheap, high-information: 1-spaced clusters straddling 160 & 256 + a few step-8 anchors, M=1.
        ns = sorted(set([144, 152, 160, 176, 192, 224, 288]
                        + [160 + d for d in (-2, -1, 0, 1, 2)]
                        + [256 + d for d in (-2, -1, 0, 1, 2)]))
        return [(1, K, n) for n in ns]
    e1 = [(M, K, n) for M in (1, 128) for n in e1_n_grid()]   # staircase test, two M regimes
    e3 = [(256, K, 128), (128, K, 256)]                       # order (both N>=128)
    e4 = [(M, K, 256) for M in (1, 32, 128)]                  # shape/decode (E2 align 128/144 ⊂ e1)
    return sorted(set(e1 + e3 + e4))


def sweep_sh(shps):
    shape_args = " ".join(f'"{M} {Kk} {N}"' for (M, Kk, N) in shps)
    return f"""set -e
cp /io/rknpu2_approx.json $ONNXIM_HOME/configs/rknpu2_approx.json
cd $ONNXIM_HOME
for s in {shape_args}; do
  set -- $s; M=$1; K=$2; N=$3
  sed -i "s/^size_list = .*/size_list = [[$M, $K, $N]]/" scripts/generate_matmul_onnx.py
  python3 scripts/generate_matmul_onnx.py >/dev/null 2>&1
  out=$(./build/bin/Simulator --config configs/rknpu2_approx.json \\
        --models_list model_lists/matmul_${{M}}_${{K}}_${{N}}.json 2>/dev/null \\
        | grep -oE "Finished at [0-9]+ cycle [0-9.]+ us" | head -1)
  cyc=$(echo "$out" | grep -oE "at [0-9]+ cycle" | grep -oE "[0-9]+")
  us=$(echo "$out"  | grep -oE "cycle [0-9.]+ us" | grep -oE "[0-9.]+")
  echo "RESULT $M $K $N $cyc $us"
done
"""


def scp(text, remote):
    subprocess.run(["ssh", HOST, f"cat > {remote}"], input=text, text=True, check=True)


def run(shps, label):
    scp(CONFIG.read_text(), f"{IO}/rknpu2_approx.json")
    scp(sweep_sh(shps), f"{IO}/char_sweep.sh")
    print(f"[{label}] running {len(shps)} shapes on {HOST}...")
    t0 = time.time()
    p = subprocess.run(
        ["ssh", HOST, f"docker run --rm -v {IO}:/io onnxim bash -lc 'bash /io/char_sweep.sh'"],
        capture_output=True, text=True, timeout=5400)
    wall = round(time.time() - t0, 1)
    rows = []
    for line in p.stdout.splitlines():
        m = re.match(r"RESULT (\d+) (\d+) (\d+) (\d+) ([\d.]+)\s*$", line.strip())
        if m:
            rows.append({"shape": [int(m[1]), int(m[2]), int(m[3])],
                         "cycles": int(m[4]), "latency_us": round(float(m[5]), 3)})
    got = {tuple(r["shape"]) for r in rows}
    missing = [list(s) for s in shps if s not in got]
    print(f"[{label}] {len(rows)}/{len(shps)} in {wall}s; missing {missing[:8]}")
    if not rows:
        sys.exit(f"FAIL: no rows.\nstderr:\n{p.stderr[-1200:]}")
    return rows, missing, wall


def main():
    pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"
    shps = shapes(pilot=pilot)
    rows, missing, wall = run(shps, "pilot" if pilot else "full")
    if pilot:
        # just print the boundary structure for inspection; do NOT write the canonical file.
        for r in sorted(rows, key=lambda r: r["shape"][2]):
            print(f"  N={r['shape'][2]:>4}  {r['cycles']:>10d} cyc  {r['latency_us']:>8.3f} us")
        return
    out = {
        "_doc": "ONNXim v%s RKNPU2-approx characteristic sweep (Phase 1.6b): does ONNXim exhibit a "
                "32-period staircase / alignment-order-shape sensitivity? simulated, NOT silicon "
                "(#13). H0 = smooth, no 32-step; burden on rejecting it. Independent of the committed "
                "spread (npu_onnxim_trace.py)." % ONNXIM_COMMIT[:10],
        "config": "rknpu2_approx (32x32 x3, INT8, dram=ramulator2)", "onnxim_commit": ONNXIM_COMMIT,
        "fixed_contraction_K": K, "wall_seconds": wall,
        "honesty": "simulated (ONNXim generic-systolic, RKNPU2-approx), NOT silicon",
        "missing_shapes": missing,
        "rows": sorted(rows, key=lambda r: r["shape"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(rows)} shapes -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
