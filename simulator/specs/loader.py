"""Spec loader for Phase 1.2 swappable component specs (simulator/specs/*.json).

A spec is a JSON file: numeric/structural fields + a `provenance` map (field -> source &
honesty-tag string). Swapping a model = swapping a spec file; the engine is unchanged.
Honesty tags follow the repo discipline: `measured` / `calibrated` (fit to our silicon) /
`simulated` / `assumption` / `borrowed`. `load_spec(name)` parses the file; `provenance()`
reads a field's tag (or the whole map). No validation here beyond existence — each engine
asserts the fields it needs.
"""
import json
from pathlib import Path

_SPECS = Path(__file__).parent


def load_spec(name):
    """Load simulator/specs/<name>.json -> dict (carries a 'provenance' map)."""
    p = _SPECS / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"spec not found: {p}")
    return json.loads(p.read_text())


def provenance(spec, field=None):
    """Provenance/honesty tag for `field` (or the whole map if field is None)."""
    prov = spec.get("provenance", {})
    return prov if field is None else prov.get(field)
