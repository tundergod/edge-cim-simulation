"""Smoke test: one 1x1-conv-proxy shape end to end (build ONNX -> compile -> axrunmodel).
Run inside the aetina SDK container. Dumps raw output so we can nail the parser."""
import subprocess, time, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

WORK = Path("/tmp/cim_work/smoke"); WORK.mkdir(parents=True, exist_ok=True)
SDK = "/home/ubuntu/voyager-sdk"
M, K, N = 1, 2048, 2048  # 1B q-proj decode GEMV

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, K, 1, M])
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, N, 1, M])
W = numpy_helper.from_array(np.random.randn(N, K, 1, 1).astype(np.float32), "W")
node = helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
g = helper.make_graph([node], "m", [X], [Y], [W])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), str(WORK / "m.onnx"))
print("onnx written")

t0 = time.time()
r = subprocess.run(["compile", "--input", str(WORK / "m.onnx"), "--input-shape", f"1,{K},1,{M}",
                    "--output", str(WORK / "out"), "--overwrite", "--log-level", "WARNING",
                    "--dataset-len", "20"], cwd=SDK, capture_output=True, text=True, timeout=900)
print(f"=== COMPILE rc={r.returncode} ({time.time()-t0:.1f}s) ===")
print("STDOUT tail:", r.stdout[-800:])
print("STDERR tail:", r.stderr[-800:])
mj = list((WORK / "out").rglob("model.json"))
print("model.json:", mj)
if r.returncode == 0 and mj:
    r2 = subprocess.run(["axrunmodel", str(mj[0]), "--seconds", "5"], cwd=SDK,
                        capture_output=True, text=True, timeout=120)
    print(f"=== AXRUNMODEL rc={r2.returncode} ===")
    print("STDOUT:", r2.stdout)
    print("STDERR tail:", r2.stderr[-500:])
