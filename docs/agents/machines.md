# Machines & unified management

Three machines, one private repo (`tundergod/edge-cim-simulation`) as the single source of truth.

## Topology

| alias | host | role | key specs |
|---|---|---|---|
| *(local)* Mac | this machine (Apple M3 Pro, 18 GB, arm64) | **control / dev / git hub** — writes code, orchestrates boards over SSH, owns GitHub auth | macOS, Python venv at `.venv` |
| `aetina` | `aetina@140.112.28.105` | **measurement node 1** — Metis Alpha CIM + RKNPU2 + Mali + CPU micro-benchmarks | Ubuntu 22.04.4 / k5.10.110, RK3588 8-core / 15 Gi, Voyager SDK **v1.3.1** (docker `axelera-sdk-ubuntu-2204-arm64:1.3.1`), RKNPU2 `librknnrt` + `rknn_server`, Mali-G610 (OpenCL) |
| `metiscard` | `tundergod@140.112.28.104` (`wei-tmp-ubuntu`) | **measurement node 2** — production Metis card end-to-end LLM (L4 anchor) | Ubuntu 24.04.4 / k6.8, i9-12900K 24-core / 125 Gi, Metis prod card `/dev/metis-0:7:0`, RTX 3090 (24 GB, drv 535), Voyager SDK at `~/tundergod/voyager-sdk` |

SSH is key-based (`~/.ssh/edgecim_ed25519`, aliases in `~/.ssh/config`): `ssh aetina`, `ssh metiscard` — no password.

## Management model (chosen: Mac is the sole git hub)

GitHub credentials live **only on the Mac**. The boards are pure measurement workhorses with a plain working dir `~/edge-cim-simulation/{characterization,measurements}` (not a git clone). Sync flow per measurement task:

```
# Mac -> board: push the characterization scripts for a unit
rsync -a characterization/<board>/ <alias>:~/edge-cim-simulation/characterization/

# board: run (driven from the Mac over SSH)
ssh <alias> 'cd ~/edge-cim-simulation/characterization && <run cmd>'

# board -> Mac: pull the produced measurements back into the repo
rsync -a <alias>:~/edge-cim-simulation/measurements/ measurements/<board>/

# Mac: commit + push on the phase branch (per the per-phase workflow in CLAUDE.md)
```

The Mac stays in the loop for every sync — no GitHub keys on the shared lab machines. (If fully-autonomous overnight board runs are ever needed, upgrade to per-board write deploy keys.)

## Per-phase coordination

Same per-phase workflow as everything else (CLAUDE.md): a phase = branch off `main` → plan → subagent review → user approval → execute (drive boards via the rsync flow above) → subagent code review → PR → user review → merge.

## Board readiness (verified 2026-06-03)

Both Metis cards were found in **bad states and recovered** — always run `axdevice` (in the SDK env) to confirm a card responds **before** measuring:
- **aetina (Alpha)** had dropped off the PCIe bus (slot showed garbage `16c3:abcd`, no `/dev/metis`, `metis.ko` unloaded). Recovered via PCIe `remove`+`rescan`+`modprobe metis`, then recreate the SDK container. Now: `metis-0:1:0 1GiB m2 flver=1.3.0 clock=800MHz`. (Procedure in `docs/voyager-sdk.md` §11.)
- **metiscard (production)** card was present but **not responding** (comm timeout, `board_type=unknown`). Recovered via `axdevice --reboot`. Now: `metis-0:7:0 **16 GiB** pcie flver=1.4.0 clock=800MHz`. (16 GB → holds 8B + longer ctx; check whether power telemetry is readable — possible M7 bonus.)

Toolchain availability:
- **aetina**: SDK v1.3.1 docker ✓; Mali OpenCL + gcc + CL headers ✓ (custom matmul kernel buildable); CPU gcc ✓. **RKNPU2:** `librknnrt.so` + `rknn_server` ✓; `rknn-toolkit-lite2` (on-board inference) **installed** at `~/edge-cim-simulation/.rknnvenv` ✓. `rknn-toolkit2` (ONNX→`.rknn` converter) fails to build on aarch64 → convert on an x86 host (metiscard) then run on-board via rknnlite.
- **metiscard**: SDK venv `~/tundergod/voyager-sdk/axelera-env`, `axllm` ✓ (Gradio absent — UI only, irrelevant); **all target precompiled LLMs present** — llama-3.2-1b/3b, llama-3.1-8b, phi3-mini 512/1024/2048, velvet-2b (each 1c+4c). RTX 3090 ✓.

## Caveats

- aetina `/userdata`: cleaned 2026-06-03 (94% → 49%, 7.2 GB free); Voyager builds still target `/userdata/voyager-sdk/build`.
- aetina PCIe recovery, build cleanup, and some introspection (`lspci`/`lsmod`/rknpu debug) need `sudo` (password auth available).
