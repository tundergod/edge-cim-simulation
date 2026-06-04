"""Phase 0.3 A4 (converter, runs on metiscard x86, py3.10 venv ~/rknnconv).

Builds MatMul ONNX for each target shape and converts to .rknn (target rk3588, FP16):
  - projection  (weight-stationary): 1 runtime input A[M,K], const weight W[K,N]  -> [M,N]
  - attention   (activation x activation): 2 runtime inputs A[M,K], B[K,N]         -> [M,N]
    (this is the native attention the NPU CAN do but the CIM conv-proxy cannot).
Writes .rknn files + manifest.json to ~/rknn_out/. rsync to aetina, run with rknnlite.

Run: ~/rknnconv/bin/python convert_rknn.py
"""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from rknn.api import RKNN

OUT = Path.home() / "rknn_out"; OUT.mkdir(exist_ok=True)
TMP = Path.home() / "rknn_tmp"; TMP.mkdir(exist_ok=True)

MODELS = {  # decode-GEMV projection families per model (M=1)
    "1b": dict(H=2048, F=8192, kv=512),  "3b": dict(H=3072, F=8192, kv=1024),
    "8b": dict(H=4096, F=14336, kv=1024), "qwen": dict(H=3584, F=18944, kv=512),
}


def onnx_matmul(M, K, N, two_input, path):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    if two_input:  # activation x activation (attention bmm, single head)
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [K, N])
        node = helper.make_node("MatMul", ["A", "B"], ["Y"])
        g = helper.make_graph([node], "mm", [A, B], [Y])
    else:          # weight-stationary projection (B is a constant)
        W = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), "B")
        node = helper.make_node("MatMul", ["A", "B"], ["Y"])
        g = helper.make_graph([node], "mm", [A], [Y], [W])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def convert(tag, M, K, N, two_input):
    op = TMP / f"{tag}.onnx"
    onnx_matmul(M, K, N, two_input, op)
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588")
    if r.load_onnx(model=str(op)) != 0:
        r.release(); return {"error": "load_onnx"}
    if r.build(do_quantization=False) != 0:   # FP16 (matches Mali FP16 comparison)
        r.release(); return {"error": "build"}
    out = OUT / f"{tag}.rknn"
    rc = r.export_rknn(str(out))
    r.release()
    return {"ok": rc == 0, "rknn": out.name if rc == 0 else None}


def manifest():
    tasks = []
    for m, c in MODELS.items():
        H, F, kv = c["H"], c["F"], c["kv"]
        for fam, (K, N) in {"q_o": (H, H), "kv": (H, kv), "gate_up": (H, F), "down": (F, H)}.items():
            tasks.append((f"proj_{m}_{fam}", 1, K, N, False))
    # attention bmm (single head, hd=128, 8B) — the native activation x activation case
    for kv in [129, 513, 1025]:
        tasks.append((f"attn_qkT_kv{kv}", 1, 128, kv, True))   # QK^T: [1,hd]x[hd,kv]
        tasks.append((f"attn_sv_kv{kv}", 1, kv, 128, True))    # S.V:  [1,kv]x[kv,hd]
    # prefill QK^T sample (seq=512)
    tasks.append(("attn_qkT_prefill512", 512, 128, 512, True))
    return tasks


def main():
    results = {}
    for tag, M, K, N, two in manifest():
        r = convert(tag, M, K, N, two)
        results[tag] = {"M": M, "K": K, "N": N, "two_input": two, **r}
        print(f"{tag:24s} M{M}K{K}N{N} two={two} -> {r.get('error', 'OK' if r.get('ok') else 'FAIL')}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(results, indent=1))
    n_ok = sum(1 for v in results.values() if v.get("ok"))
    print(f"\nconverted {n_ok}/{len(results)} -> {OUT}/")


if __name__ == "__main__":
    main()
