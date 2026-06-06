#!/usr/bin/env bash
# Reproducible Ramulator 2.1 build for the engine='ramulator2' backend (Phase 1.3).
# v2.1 is Python-bindings-only (no CLI/YAML): builds libramulator.so + the `ramulator` Python
# extension under python/ramulator/ + runs codegen. The main-branch LPDDR5 refresh bug is fixed
# in v2.1 (issues #58/#60/#89). VERIFIED 2026-06-06 on macOS (Apple clang 17, cmake 4.3, Python 3.13):
# LPDDR5_6400 streaming runs clean, saturates at 98.6% of peak. upstream/ is gitignored.
#
# After building, drive it via tools/analysis/mem_ramulator2.py (reuses v2.1's latency_throughput
# harness in-process; no CLI). No YAML config — the main-branch lpddr5.yaml/ddr4_smoke.yaml are retired.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
VENV="$REPO_ROOT/.venv/bin/python"
SHA=278f1effc3838099a6ffe0ad5f9f572fea80c948   # v2.1 (pinned)

[ -d upstream ] || git clone https://github.com/CMU-SAFARI/ramulator2 upstream
cd upstream
git fetch --depth 1 origin "$SHA" 2>/dev/null || git fetch origin v2.1
git checkout "$SHA"

# Patch — Apple clang 17 requires the `template` keyword for a dependent template name (base/param.h).
# Idempotent + grep-guarded (skip if v2.1 ever ships it fixed).
grep -q 'config\[name\]\.as<T>()' src/ramulator/base/param.h && \
  perl -0pi -e 's/(\bconfig\[name\])\.as<T>\(\)/$1.template as<T>()/g' src/ramulator/base/param.h || true

mkdir -p build && cd build
# -DPython_EXECUTABLE is REQUIRED so the extension + codegen target the venv, not system Python.
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 for cmake 4.x vs bundled yaml-cpp.
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$VENV" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j4

echo "Built. Verify: $VENV -c \"import sys; R='$REPO_ROOT/tools/ramulator2/upstream'; sys.path[:0]=[R,R+'/python']; import ramulator; print('ok')\""
echo "Drive:  $VENV tools/analysis/mem_ramulator2.py"
