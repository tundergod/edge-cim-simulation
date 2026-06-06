#!/usr/bin/env bash
# Reproducible Ramulator2 build for the engine='ramulator2' backend (Phase 1.3).
# VERIFIED 2026-06-06 on macOS (Apple clang 17, cmake 4.3, C++20): builds + runs DDR4 e2e.
# Two patches were needed for this toolchain (captured below). upstream/ is gitignored.
set -euo pipefail
cd "$(dirname "$0")"

[ -d upstream ] || git clone --depth 1 https://github.com/CMU-SAFARI/ramulator2 upstream
cd upstream

# Patch 1 — Apple clang 17 requires the `template` keyword for a dependent template name (param.h:91).
perl -0pi -e 's/_config\[_name\]\.as<T>\(\)/_config[_name].template as<T>()/g' src/base/param.h

mkdir -p build && cd build
# Patch 2 — cmake 4.x removed <3.5 policy compat (bundled ext/yaml-cpp); set the minimum policy version.
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j4

echo "Built: $(pwd)/ramulator2"
echo "Smoke (DDR4, known-good):  ./ramulator2 -f ../../ddr4_smoke.yaml   # needs an LD/ST trace"
echo "NOTE: the LPDDR5 config (../../lpddr5.yaml) currently throws 'Failed to send refresh' under"
echo "      saturated streaming — a ramulator2/LPDDR5 refresh-config tuning item (see README.md)."
