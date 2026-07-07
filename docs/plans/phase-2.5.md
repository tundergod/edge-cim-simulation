# Plan: Phase 2.5 — spec-driven CIM compute engine (minimal honest generality)

Branch `phase-2.5` off `phase-2.4` tip (stacked; PR base = phase-2.4). Invariant: all three wired
topologies keep `cim_compute_params = "simulator/models/params/m1_cim.json"`, so the resolved params
equal today's default → AllCim L4 stays 0.1073/0.0649/0.0311 byte-identical.

## Steps

1. `simulator/runtime/platform.py` (the `self.cim = CimTileModel()` site): when a topology is given, read
   `topo_spec["cim_compute_params"]` (a repo-root-relative path), resolve it against the repo root
   explicitly — `repo_root = Path(__file__).resolve().parents[2]`. The path MUST stay inside the repo:
   reject `Path(_ccp).is_absolute()` and, after `(repo_root / _ccp).resolve()`, reject
   `not .is_relative_to(repo_root)` (fail-loud) — a spec may only reference in-repo params so the same
   commit is reproducible on any machine (review P2). Then `params = json.loads(resolved.read_text())` and
   pass `CimTileModel(params=params)`. If the field is absent OR no topology, keep `CimTileModel()` (Metis
   default). Reuse the topo spec already loaded for the BW resolver (no second load). Geometry
   (n_cores/core_width) follows from the params dict via `CimTileModel`'s `p.get(...)` — no separate
   plumbing → verify: `Platform(model, topology="cim_topo_card").cim` has params equal to `m1_cim.json`
   (dev_lat_us identical to today); an absolute or `..`-escaping path raises ValueError.
2. `simulator/runtime/config.py`: add `_CAL_CIM_PARAMS = "simulator/models/params/m1_cim.json"`. In
   `_flag_provenance()`, `load_spec(self.topology).get("cim_compute_params")`; if it is present and
   `!= _CAL_CIM_PARAMS`, append `simulated: CIM compute params '<x>' (non-Metis geometry, no silicon
   ground truth; not the Metis-calibrated m1_cim.json)`. (Exact-string compare by design — a differently
   formatted but identical path reads as non-Metis, the conservative/honesty-safe direction.) Do NOT touch
   `is_calibrated_anchor()` — it is `not self.provenance`, so the appended tag already drives it to False
   → verify: card/alpha/edge (all m1_cim.json) keep their current provenance/anchor exactly; a synthetic
   topology with a different `cim_compute_params` gets the `simulated:` tag and `is_calibrated_anchor()`
   returns False.
3. `simulator/specs/cim_topo_{card,alpha,edge}.json`: edit the `_doc`/`honesty` strings so they state the
   CIM compute engine is RESOLVED FROM `cim_compute_params` (still the shared Metis `m1_cim.json` for all
   three) instead of "topology-agnostic and SHARED (hardcoded)". No numeric field changes → verify: `git
   diff simulator/specs/` shows only `_doc`/`honesty` prose; `grep cim_compute_params simulator/specs/
   *.json` all still = m1_cim.json.
4. `CONTEXT.md`: add one repo-index/glossary line — `cim_compute_params` is the compute-engine swap
   surface (parallels the memory/topology swap); a non-Metis value is tagged `simulated` (no non-Metis
   silicon), consistent with the existing `simulated` tier definition → verify: text present; the glossary
   `simulated`/`extrapolated` tier definitions are unchanged (the tag matches `simulated`).
5. NEW `tests/test_cim_params_swap.py` — mechanism proof, no committed architecture result:
   (a) `Platform(model, topology="cim_topo_card").cim` params == m1_cim.json (dev_lat_us matches a default
       `CimTileModel()`);
   (b) build a CimTileModel from a tmp params file with a DIFFERENT `core_width`/`G_eff_Gmax_gops` and
       assert its `dev_lat_us` differs from the Metis one (proves params drive compute);
   (c) a SimConfig on a SYNTHETIC topology whose `cim_compute_params` != m1_cim.json is provenance-tagged
       `simulated` and `is_calibrated_anchor()` is False (monkeypatch `load_spec` / feed a spec dict; do
       NOT commit a fake topology spec file);
   (d) card/alpha/edge SimConfigs keep their current calibrated-anchor / provenance exactly
   → verify: `pytest tests/test_cim_params_swap.py` green.

## Final gate (Phase 2.5)
- `.venv/bin/pytest tests/` all green (2.4 tests + new swap test).
- All validators exit 0; `validation/reports/phase2/e2e_l4.json` byte-identical (0.1073/0.0649/0.0311) —
  the load-bearing proof the resolver did not perturb the Metis path.
- `phase2/topology_ab.json` byte-identical (card/alpha/edge unchanged).
- `docs/report/phase1-site/build.py --strict` green (14 pages).
- `grep -rn cim_compute_params simulator/ --include=*.py` now returns reader(s) — `platform.py` (and
  `config.py` for the provenance guard) — the field is no longer dead metadata.

Outputs: edited `platform.py`, `config.py`, `cim_topo_*.json` (_doc only), `CONTEXT.md`; new
`tests/test_cim_params_swap.py`; this plan `docs/plans/phase-2.5.md`.

## Out of scope
- Committing an example non-Metis architecture as a result/figure.
- Deriving the SRAM residency knee from the SRAM spec (audit follow-up).
- Per-architecture energy coefficients (audit follow-up).
- Rewiring `tools/analysis/recompose_e2e.py` / `tools/plotting/phase1_figs.py` `CimTileModel()` calls
  (outside the runtime swap surface + validator path).
