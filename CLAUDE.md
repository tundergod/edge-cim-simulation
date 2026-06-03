# CLAUDE.md

Guidance for any agent working in this repo. Read this first, then [overall.md](overall.md) (project brief + phases) and [voyager-sdk.md](voyager-sdk.md) (Metis/SDK measurement reference).

## Orientation

- **What this is:** a real-silicon-calibrated simulator of LLM inference on a CIM-enabled heterogeneous mobile SoC, calibrated against two Axelera Metis boards. Goal, phases, modules (M1–M8), validation layers (L1–L6): [overall.md](overall.md).
- **Authoritative docs (don't re-derive):** [overall.md](overall.md) = plan & phases (preliminary — revise freely). [voyager-sdk.md](voyager-sdk.md) = how to measure Metis (tagged `[DOC]`/`[FORUM]`/`[MEASURED]`/`[GAP]`). [papers/](papers/) = literature + real-silicon notes (16 curated). [CONTEXT.md](CONTEXT.md) = domain glossary.
- **Hard scope:** dense Llama-3 / Qwen-2.5, 1B–8B, INT8, batch=1, prefill+decode. See `overall.md` § 範圍外 for what's excluded.
- **Secrets:** never commit tokens/keys. The HF token lives in the user's environment (`HF_TOKEN`), not in the repo.

## How to work (Karpathy guidelines)

Full text: `/karpathy-guidelines`. The four rules, applied here:

1. **Think before coding.** State assumptions explicitly; if multiple interpretations exist, surface them — don't pick silently. If a simpler approach exists, say so.
2. **Simplicity first.** Minimum code that solves the task. No speculative features, abstractions, config, or error handling for impossible cases.
3. **Surgical changes.** Touch only what the task requires. Match existing style. Don't refactor or "improve" adjacent code; mention unrelated dead code, don't delete it.
4. **Goal-driven execution.** Turn each task into a verifiable check and loop until it passes. For module work, that means: validate against `validation/contracts/*` and the `measurements/` ground truth.

### Process precedence (multiple skill packs are installed)

This repo has several process-skill sources: **this file's per-phase workflow** (authoritative), `karpathy-guidelines`, matt-pocock skills (`tdd`/`diagnose`/`to-issues`), and the `superpowers` plugin (`writing-plans`, `test-driven-development`, `systematic-debugging`, `verification-before-completion`, `requesting-code-review`, `subagent-driven-development`, …). They don't change the simulator design (that's the ADRs) — only how you work.

- **This file's per-phase workflow is authoritative.** Use the skill packs as a *toolbox* where they fit: TDD when implementing M-modules, `systematic-debugging` when a validation fails, `verification-before-completion` before claiming a phase done, `subagent-driven-development` for parallel module work.
- **On conflict, CLAUDE.md + karpathy simplicity win.** Don't pile on ceremony or invoke many overlapping skills for one small task. Plans always use the action-only format below, regardless of what a plan-writing skill suggests.

## Per-phase workflow (required)

Phases are defined in [overall.md](overall.md) § 階段總覽 (Phase 0.1, 0.2, 0.3, 1, 2). **Every phase follows this loop. Do not skip the gates.**

1. **Write the plan** → `plans/phase-<id>.md` (e.g. `plans/phase-0.1.md`). **Action-only**: list the steps to take, files to create/edit, commands to run, outputs to produce, and a one-line verification check per step. **No purpose, motivation, background, or rationale** — those live in `overall.md`.
2. **Plan review by subagent.** Spawn a subagent to review the plan. Apply its findings, re-review, and **loop (fix → review) until the reviewer reports no issues.**
3. **User approval gate.** Present the clean plan to the user. **Wait for explicit approval. Do not start executing before the user approves.**
4. **Execute** the phase exactly per the approved plan.
5. **Code review by subagent.** After execution, spawn a subagent to code-review the result. Address its findings, then report.

Plan file shape (action-only):

```markdown
# Plan: Phase <id> — <short title>

1. <action> → verify: <check>
2. <action> → verify: <check>
...
Outputs: <files / artifacts produced>
```

## Agent skills

### Issue tracker

Issues live as **GitHub issues** (use the `gh` CLI). No GitHub remote is configured yet — create one (`gh repo create` / `git remote add origin …`) before running `gh issue` commands. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
