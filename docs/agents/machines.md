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

## Caveats (verified 2026-06-03)

- **aetina `/userdata` is 94% full (~960 MB free)** — Voyager builds default to `/userdata/voyager-sdk/build`. Heavy Phase 0.2 compiles may need cleanup first. The board working dir `~/edge-cim-simulation` is on `/` (8.1 GB free) — fine for scripts + result JSONs, but watch build outputs.
- **metiscard precompiled LLMs = llama-3.1-8b (1c+4c) + phi3-mini (512/1024/2048)** — not the llama-3.2-1b/3b the earlier investigation used. Re-fetch those if the end-to-end sweep needs them.
- Several board introspection commands (`lspci`, `lsmod`, rknpu debug) need `sudo` (password auth available); SDK/device presence already confirmed without it.
