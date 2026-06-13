"""Phase 1.3 — run ONNXim RKNPU2-approx over the NPU matmul shapes (on metiscard via Docker).

Drives ONE `docker run` on metiscard over a canonical (M,K,N) list, parses ONNXim's
`Simulation Finished at {cycle} cycle {us} us`, writes simulator/engines/onnxim/rknpu2_sim_matmul.json
LOCALLY. ONNXim v a1e86296 is built as image `onnxim`. The committed RKNPU2-approx config and a
generated sweep script are sent to metiscard:~/edge-cim-simulation/onnxim_io (mounted at /io;
copied into $ONNXIM_HOME/configs inside the container — NOT bind-mounted over /workspace, which
would shadow the in-image build). The generator's `size_list` is hardcoded (no args), so the sweep
`sed`s the active line per shape (anchored `^size_list` to skip the commented lines).

The (M,K,N) list here is the single source of truth (N4): build_m4_npu_onnxim.py reads back exactly
these shapes for the analytic-vs-ONNXim delta. simulated, NOT silicon; ONNXim != issue #13.

Run (from the Mac): ./.venv/bin/python tools/analysis/npu_onnxim_trace.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tools/onnxim/rknpu2_approx.json"
OUT = ROOT / "simulator/engines/onnxim/rknpu2_sim_matmul.json"
HOST = "metiscard"
IO = "~/edge-cim-simulation/onnxim_io"
ONNXIM_COMMIT = "a1e86296"

# Canonical NPU LLM GEMM shapes (llama-3.1-8b H=4096, F=14336): projections at decode M=1 + prefill
# M=256; + the K=2048 channel staircase (M=1) for the HeteroInfer trend.
# N>=128 only: ONNXim SIGFPE-crashes on N<=64 GEMMs (degenerate 32x32 tiling) — a documented
# ONNXim limit, so the staircase starts at 128 (still shows the rising channel trend).
H, F = 4096, 14336
_SHAPES = []
for M in (1, 256):
    _SHAPES += [(M, H, H), (M, H, 1024), (M, H, F), (M, F, H)]      # q/o, kv, gate/up, down
_SHAPES += [(1, 2048, N) for N in (128, 256, 512, 1024, 1536, 2048, 3072)]  # staircase (N>=128)
SHAPES = sorted(set(_SHAPES))


def _sweep_sh():
    shape_args = " ".join(f'"{M} {K} {N}"' for (M, K, N) in SHAPES)
    return f"""set -e
cp /io/rknpu2_approx.json $ONNXIM_HOME/configs/rknpu2_approx.json
cd $ONNXIM_HOME
for s in {shape_args}; do
  set -- $s; M=$1; K=$2; N=$3
  sed -i "s/^size_list = .*/size_list = [[$M, $K, $N]]/" scripts/generate_matmul_onnx.py
  python3 scripts/generate_matmul_onnx.py >/dev/null 2>&1
  us=$(./build/bin/Simulator --config configs/rknpu2_approx.json \\
        --models_list model_lists/matmul_${{M}}_${{K}}_${{N}}.json 2>/dev/null \\
        | grep -oE "cycle [0-9.]+ us" | grep -oE "[0-9.]+" | head -1)
  echo "RESULT $M $K $N $us"
done
"""


def _scp(text, remote):
    subprocess.run(["ssh", HOST, f"cat > {remote}"], input=text, text=True, check=True)


def main():
    _scp(CONFIG.read_text(), f"{IO}/rknpu2_approx.json")
    _scp(_sweep_sh(), f"{IO}/sweep.sh")
    print(f"running {len(SHAPES)} shapes on {HOST} (one docker run)...")
    p = subprocess.run(
        ["ssh", HOST, f"docker run --rm -v {IO}:/io onnxim bash -lc 'bash /io/sweep.sh'"],
        capture_output=True, text=True, timeout=2400)

    rows = []
    for line in p.stdout.splitlines():
        m = re.match(r"RESULT (\d+) (\d+) (\d+) ([\d.]+)\s*$", line.strip())
        if m:
            rows.append({"shape": [int(m[1]), int(m[2]), int(m[3])], "latency_us": round(float(m[4]), 3)})
    got = {tuple(r["shape"]) for r in rows}
    missing = [s for s in SHAPES if s not in got]
    if missing or not rows:
        sys.exit(f"FAIL: {len(rows)}/{len(SHAPES)} shapes; missing {missing}\nstderr:\n{p.stderr[-800:]}")

    out = {
        "_doc": "ONNXim v%s RKNPU2-approx (32x32 x3 systolic, INT8, ramulator2-DDR4) per-shape GEMM "
                "latency. simulated, NOT silicon; ONNXim != issue #13." % ONNXIM_COMMIT[:10],
        "config": "rknpu2_approx (32x32 x3, INT8, dram=ramulator2)",
        "onnxim_commit": ONNXIM_COMMIT,
        "honesty": "simulated (ONNXim generic-systolic, RKNPU2-approx), NOT silicon",
        "rows": sorted(rows, key=lambda r: r["shape"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(rows)} shapes -> {OUT}")


if __name__ == "__main__":
    main()
