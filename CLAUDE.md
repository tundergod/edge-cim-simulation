# CLAUDE.md

Guidance for any agent working in this repo. Read this first, then [OVERALL.md](OVERALL.md) (project brief + phases) and [docs/voyager-sdk.md](docs/voyager-sdk.md) (Metis/SDK measurement reference).

## Orientation

- **What this is:** a real-silicon-calibrated simulator of LLM inference on a CIM-enabled heterogeneous mobile SoC, calibrated against two Axelera Metis boards. Goal, phases, modules (M1–M8), validation layers (L1–L6): [OVERALL.md](OVERALL.md).
- **Authoritative docs (don't re-derive):** [OVERALL.md](OVERALL.md) = plan & phases (preliminary — revise freely). [docs/voyager-sdk.md](docs/voyager-sdk.md) = how to measure Metis (tagged `[DOC]`/`[FORUM]`/`[MEASURED]`/`[GAP]`). [docs/papers/](docs/papers/) = literature + real-silicon notes (16 curated). [CONTEXT.md](CONTEXT.md) = domain glossary **+ repo index** (`## Repo index` — a directory-level map of where everything lives; consult it to locate code/docs/data fast, before grepping blindly).
- **Hard scope:** dense Llama-3 / Qwen-2.5, 1B–8B, INT8, batch=1, prefill+decode. See `OVERALL.md` § 範圍外 for what's excluded.
- **Secrets:** never commit tokens/keys. The HF token lives in the user's environment (`HF_TOKEN`), not in the repo.

## How to work (Karpathy guidelines)

Behavioral guidelines to reduce common LLM coding mistakes (from [Karpathy's observations](https://x.com/karpathy/status/2015883857489522876)). They bias toward caution over speed; for trivial tasks, use judgment. The four rules in full, each with how it applies here:

**1. Think before coding — don't assume, don't hide confusion, surface tradeoffs.**
Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**2. Simplicity first — minimum code that solves the problem, nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility"/configurability that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**3. Surgical changes — touch only what you must, clean up only your own mess.**
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove only the imports/variables/functions that YOUR changes made unused; don't remove pre-existing dead code unless asked.
- The test: every changed line should trace directly to the user's request.

**4. Goal-driven execution — define success criteria, loop until verified.**
- Transform tasks into verifiable goals: "Add validation" → "write tests for invalid inputs, then make them pass"; "Fix the bug" → "write a test that reproduces it, then make it pass".
- For multi-step tasks, state a brief plan (`1. step → verify: check`).
- Strong success criteria let you loop independently; weak criteria ("make it work") require constant clarification.
- **Here:** turn module work into a verifiable check and loop until it passes — validate against `validation/contracts/*` and the `measurements/` ground truth.

### Process precedence (multiple skill packs are installed)

This repo has several process-skill sources: **this file's per-phase workflow** (authoritative), `karpathy-guidelines`, matt-pocock skills (`tdd`/`diagnose`/`to-issues`), and the `superpowers` plugin (`writing-plans`, `test-driven-development`, `systematic-debugging`, `verification-before-completion`, `requesting-code-review`, `subagent-driven-development`, …). They don't change the simulator design (that's the ADRs) — only how you work.

- **This file's per-phase workflow is authoritative.** Use the skill packs as a *toolbox* where they fit: TDD when implementing M-modules, `systematic-debugging` when a validation fails, `verification-before-completion` before claiming a phase done, `subagent-driven-development` for parallel module work.
- **On conflict, CLAUDE.md + karpathy simplicity win.** Don't pile on ceremony or invoke many overlapping skills for one small task. Plans always use the action-only format below, regardless of what a plan-writing skill suggests.

## Honesty discipline (results must not confirm their own assumptions)

**Be honest.** Making an assumption or hypothesis *before* an experiment or validation is fine. The sin is the opposite move — rigging the method, the data, or the figure so the result confirms the prior. Let the data — including a null or a disagreement — stand.

1. **No circular reasoning.** A method that *bakes in* X cannot be evidence *for* X — test X with something that doesn't assume it. Pre-register the criterion + null before seeing data; a correction may only make a positive claim harder, never easier.
2. **No manufactured agreement.** Measure each source independently; never pin one source's points onto another's curve, and label by-construction results as such (not discoveries). A spread is uncertainty, not agreement.
3. **No validation language without ground truth.** Don't write validated/agree/parity/consistent/calibrated/measured (or their Chinese equivalents) unless a real measurement backs it — every such number traces to committed JSON (`{{key}}`, fail-loud), honesty tags ([MEASURED]/[GAP]/…) match true provenance.

## Per-phase workflow (required)

Phases are defined in [OVERALL.md](OVERALL.md) § 階段總覽 (Phase 0.1, 0.2, 0.3, 0.4, 1, 2). **Every phase follows this loop. Do not skip the gates.**

0. **Branch.** Before starting a phase, create branch `phase-<id>` off `main`. The phase's plan + code all live on it.
1. **Write the plan** → `docs/plans/phase-<id>.md` (e.g. `docs/plans/phase-0.1.md`). **Action-only**: list the steps to take, files to create/edit, commands to run, outputs to produce, and a one-line verification check per step. **No purpose, motivation, background, or rationale** — those live in `OVERALL.md`.
2. **Plan review by subagent.** Spawn a subagent to review the plan. Apply its findings, re-review, and **loop (fix → review) until the reviewer reports no issues.**
3. **User approval gate.** Present the clean plan to the user. **Wait for explicit approval. Do not start executing before the user approves.**
4. **Execute** the phase exactly per the approved plan (on the phase branch).
5. **Code review by subagent.** After execution, spawn a subagent to code-review the result; address its findings.
6. **Open a PR.** `gh pr create` from `phase-<id>` → `main`, summarizing what was done + the verify results.
7. **Notify the user.** Tell the user the PR is up; **the user does an additional code review.**
8. **Merge.** Only after the user's explicit confirmation, merge the PR into `main` (the repo's default branch; the user's "master").

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
