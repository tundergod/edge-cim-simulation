"""Build the hand-coded Phase-1 multi-page report site (build artifact).

Each unit is one page authored as src/<stem>.src.html (inner content only). This script:
  1. resolves every {{key}} from committed JSON via tools/report/_metrics.py — FAILS on any
     unresolved placeholder, so narrative/table numbers cannot be hand-mistyped (B1/S1);
  2. renders the honesty-chip row from chips.json (each chip's on-state + source path are
     committed there, so the audit checks chips against a table, not prose) (B2);
  3. mtime-checks every referenced figure against figs.json sources — warns (or --strict fails)
     if a data JSON is newer than its PNG (S2);
  4. wraps each page with the shared sidebar nav (active page highlighted) + <head>.

Run:  ./.venv/bin/python docs/report/phase1-site/build.py          (warn on stale figs)
      ./.venv/bin/python docs/report/phase1-site/build.py --strict (fail on stale figs)
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools/report"))
import _metrics  # noqa: E402 — single source of truth for {{key}} numbers

SRC = HERE / "src"
CHIPS = json.loads((HERE / "chips.json").read_text())
FIGS = json.loads((HERE / "figs.json").read_text())

# Page registry: stem -> (mcode, nav label, group, status). status 'todo' = not yet authored
# (rendered as a disabled nav item). Order is the reading order.
PAGES = [
    ("00-overview",      "",        "概覽",        "導論",   "done"),
    ("01-readiness",     "",        "就緒矩陣",     "導論",   "done"),
    ("02-cim",           "M1",      "CIM 計算核",   "各單元",  "done"),
    ("03-memory",        "M2",      "記憶體",       "各單元",  "done"),
    ("04-cpu",           "M4·CPU",  "A76 CPU",     "各單元",  "done"),
    ("05-gpu",           "M4·GPU",  "Mali GPU",    "各單元",  "done"),
    ("06-npu",           "M4·NPU",  "RKNPU2",      "各單元",  "done"),
    ("07-workload",      "M5",      "Workload",    "跨單元",  "done"),
    ("08-energy-e2e",    "M7",      "能量 + E2E",   "跨單元",  "done"),
    ("09-phase2-preview","M3/M6",   "Phase 2 結果", "跨單元",  "done"),
    ("10-phase2-walkthrough","M3/M6","Phase 2 逐步詳解","跨單元","done"),
    ("10-gaps",          "",        "缺口 / GO-NOGO","結論",   "done"),
    ("11-sources",       "",        "來源",         "結論",   "done"),
    ("12-thermal",       "M8",      "溫度 (熱)",    "各單元",  "done"),
]
CHIP_ORDER = ["calibrated", "fitted", "simulated", "assumption", "borrowed"]

HEAD = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Edge-CIM Phase 1</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Noto+Serif+TC:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
</head><body>
{sidebar}
<div class="main"><div class="page">
{body}
</div></div></body></html>"""


def render_sidebar(active):
    rows, seen = [], set()
    for stem, mc, label, grp, status in PAGES:
        if grp not in seen:
            rows.append(f'<li class="grp">{grp}</li>')
            seen.add(grp)
        cls = "active" if stem == active else ("todo" if status == "todo" else "")
        href = f"{stem}.html" if status == "done" or stem == active else "#"
        mc_html = f'<span class="mc">{mc}</span>' if mc else '<span class="mc"></span>'
        rows.append(f'<li><a class="{cls}" href="{href}">{mc_html}<span>{label}</span></a></li>')
    return ('<nav class="sidebar"><div class="brand">Edge-CIM · Phase 1'
            '<small>元件建模與驗證 · 進 Phase 2 前</small></div>'
            f'<nav><ul>{"".join(rows)}</ul></nav></nav>')


def render_chips(stem):
    spec = {c["label"]: c for c in CHIPS.get(stem, [])}
    out = []
    for label in CHIP_ORDER:
        c = spec.get(label)
        if not c:  # page declares no chip of this kind -> render as off/unknown
            out.append(f'<span class="chip off" title="未宣告">{label}</span>')
            continue
        klass = "chip on" if c.get("on") else "chip off"
        if c.get("warn"):
            klass += " warn"
        title = c.get("src", "").replace('"', "&quot;")
        out.append(f'<span class="{klass}" title="{title}">{label}</span>')
    return f'<div class="chips">{"".join(out)}</div>'


def check_figs(stem, body, strict):
    """mtime-guard every figure the page references against figs.json sources."""
    stale = []
    for m in re.finditer(r'src="(?:\.\./)+figures/(phase[0-9.\-a-z]+)/([A-Za-z0-9_]+)\.png"', body):
        name = m.group(2)
        png = ROOT / "docs/figures" / m.group(1) / f"{name}.png"
        meta = FIGS.get(name)
        if not png.exists():
            stale.append(f"  [{stem}] MISSING png: {png.relative_to(ROOT)}")
            continue
        if not meta:
            stale.append(f"  [{stem}] {name}: not in figs.json (add script+sources)")
            continue
        png_mt = png.stat().st_mtime
        for s in meta["sources"]:
            sp = ROOT / s
            if sp.exists() and sp.stat().st_mtime > png_mt:
                stale.append(f"  [{stem}] STALE {name}.png  (source newer: {s} — re-run {meta['script']})")
    return stale


def main():
    strict = "--strict" in sys.argv
    metrics = _metrics.load()
    built, all_stale = [], []
    for stem, *_ in PAGES:
        src = SRC / f"{stem}.src.html"
        if not src.exists():
            continue
        title = next(f"{mc} {lb}".strip() for s, mc, lb, *_ in PAGES if s == stem)
        body = _metrics.substitute(src.read_text(), metrics)   # fail-loud on unresolved {{key}}
        body = body.replace("<!--CHIPS-->", render_chips(stem))
        all_stale += check_figs(stem, body, strict)
        html = HEAD.format(title=title, sidebar=render_sidebar(stem), body=body)
        (HERE / f"{stem}.html").write_text(html)
        built.append(stem)

    print(f"built {len(built)} page(s): {', '.join(built)}")
    if all_stale:
        print("figure-staleness:")
        print("\n".join(all_stale))
        if strict:
            sys.exit("stale figures (--strict)")
    else:
        print("figure-staleness: all fresh")


if __name__ == "__main__":
    main()
