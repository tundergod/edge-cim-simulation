# ADR-0007 — Op inventory / trace extraction

Status: Accepted (2026-06-03)

## Context
OVERALL.md originally proposed `torch.onnx.export` for the op inventory, but HF→ONNX export of Llama/Qwen is fragile (risk #5: custom ops, fusion, dynamic shapes). A concern was raised that running an LLM is only possible on the Metis Card — but that conflates *measuring performance* (device-dependent, the L4 anchor) with *extracting the op graph* (a property of the model, device-independent).

## Decision
- **op inventory = PyTorch runtime tracer on the dev machine.** Run the HF model's forward in plain PyTorch (eager) with dispatch/FX hooks over **meta / FakeTensor** inputs — shapes propagate through every aten op **without real weights, real compute, a GPU, or the Metis Card**. So 1B–13B op inventory is seconds on the dev Mac. It uses PyTorch's dispatch/FX tracing over meta tensors — a more robust path than ONNX export for dynamic-shape decode, and (unlike ONNX export) not subject to its custom-op/fusion fragility.
- **Cross-check completeness** against an analytical per-architecture op enumeration (QKV/O/FFN matmuls, RMSNorm×2, RoPE, attention QK^T/SV, SwiGLU, residual, sampling). traced ≡ expected ⇒ complete.
- **Dynamic shapes:** trace decode at several KV lengths (shared with ADR-0002's representative iterations).
- **ONNX export is secondary**, only for the ONNXim NPU path, with fallbacks (build ONNXim input from the traced graph, or M4 lookup-override per risk #7). The op inventory does **not** depend on fragile ONNX export.

## Consequences
Makes explicit the three-layer separation: **WHAT** (op×shape, HF trace, device-independent) vs **HOW-LONG** (per-unit micro-benchmarks on real silicon) vs **the simulated unified-memory CIM device** (composition of the two + our memory topology). De-risks risk #5.
