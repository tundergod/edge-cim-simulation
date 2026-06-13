"""Phase 2 runtime: end-to-end event-driven CIM-LLM inference simulator.

M5 (workload.py) builds a per-token op DAG (dag.py) from the Phase-0.2
op_profile templates; M6 (scheduler.py) annotates each op with a unit +
precision; M3 (events.py + resources.py) walks the DAG with units running
concurrently and shared memory bandwidth as the one contended resource.
SimConfig (config.py) is the declarative user-input contract; runner.py wires
it all together. Per-op latency always comes from the frozen Phase-1 unit
models via UnitEngine.predict() — the runtime never computes a latency itself.
"""
