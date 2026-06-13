"""Guards for the report metric layer (tools/report/_metrics.py).

These make it impossible for a report number to be hand-mistyped or to silently drift from its
committed JSON source: every report-chapter {{key}} must resolve, and the agent-facing
findings gate-summary literals must equal the current _metrics values.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/report"))
import _metrics  # noqa: E402

SITE = ROOT / "docs/report/phase1-site/src"
FINDINGS = ROOT / "docs/phase1.1-findings.md"


def test_load_resolves_all_sources():
    """Every curated key's JSON path resolves (load raises otherwise)."""
    m = _metrics.load()
    assert len(m) > 0
    assert all(isinstance(v, str) and v != "" for v in m.values())


def test_known_values():
    m = _metrics.load()
    assert m["cim.decode_median_pct"] == "2.7"
    assert m["recompose.err_8b_pct"] == "9.5"
    assert m["recompose.meas_8b"] == "2.70"
    assert m["mem.pcie_floor_us"] == "911"


def test_all_site_placeholders_resolve():
    """No phase1-site page may carry an unresolved {{key}} (substitute raises if so).
    (Repointed from the retired old markdown report to the hand-coded site.)"""
    m = _metrics.load()
    for p in sorted(SITE.glob("*.src.html")):
        _metrics.substitute(p.read_text(), m)  # raises KeyError on any unknown {{key}}


def test_substitute_raises_on_unknown():
    try:
        _metrics.substitute("x {{does.not.exist}} y")
    except KeyError:
        return
    raise AssertionError("substitute should raise on an unknown placeholder")


def test_findings_block_in_sync():
    """Agent-facing findings gate-summary is generated (build_findings.py) into a marker region;
    assert the committed file equals what regeneration would produce, i.e. it has NOT drifted from
    the JSON. If this fails, run `./.venv/bin/python tools/report/build_findings.py`."""
    import build_findings
    changed = build_findings.apply(write=False)
    assert not any(changed.values()), f"findings out of sync with JSON: {changed} — run build_findings.py"


def test_findings_gate_literals_match_metrics():
    """Belt-and-suspenders: the headline findings rows carry the current _metrics values."""
    m = _metrics.load()
    t = FINDINGS.read_text()
    assert f"median **{m['cim.decode_median_pct']}%**, p95 **{m['cim.decode_p95_pct']}%**" in t
    assert f"median **{m['gpu.attn_median_pct']}%**, p95 **{m['gpu.attn_p95_pct']}%**" in t
    assert f"**{m['recompose.err_8b_pct']}%** ({m['recompose.pred_8b']} vs {m['recompose.meas_8b']})" in t


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all passed")
