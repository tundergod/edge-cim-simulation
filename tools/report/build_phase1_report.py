"""Build the consolidated Phase 1 long-form HTML + PDF report (build artifact).

Stitches docs/report/phase1/chapters/*.md (the single consolidated Phase-1 report that absorbs
1.1/1.2/1.3) into one polished, self-contained HTML — figures from docs/figures/phase1.{1,2,3}/
are base64-embedded; per-chapter markdown footnotes render as the numbered source references —
then prints to PDF with headless Chrome (PDF is the primary reading artifact).

Run: ./.venv/bin/python tools/report/build_phase1_report.py
"""
import base64
import re
import subprocess
from pathlib import Path

import markdown as md

ROOT = Path(__file__).resolve().parents[2]
CH = ROOT / "docs/report/phase1/chapters"
FIG = ROOT / "docs/figures"
OUT = ROOT / "docs/report/phase1/index.html"
PDF = ROOT / "docs/report/phase1/phase1-report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

ORDER = ["00-intro", "01-readiness-matrix", "02-cim", "03-memory", "04-cpu",
         "05-gpu", "06-npu", "07-workload-energy-e2e", "08-integration-layer-m3-m6",
         "09-gaps-gonogo", "99-sources"]

# group header shown in the TOC before this stem
GROUP = {
    "00-intro": "導論",
    "01-readiness-matrix": "全局總覽",
    "02-cim": "各單元（CIM / 記憶體 / CPU / GPU / NPU）",
    "07-workload-energy-e2e": "跨單元與整合層",
    "09-gaps-gonogo": "結論",
    "99-sources": "附錄",
}


def embed_figs(html):
    """Replace ../../../figures/phase1.{1,2,3}/NAME.png src with base64 data URIs (any of the
    three phase figure dirs)."""
    def repl(m):
        phase, name = m.group(1), m.group(2)
        p = FIG / phase / f"{name}.png"
        if not p.exists():
            return m.group(0)
        b = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:image/png;base64,{b}"'
    return re.sub(r'src="(?:\.\./)+figures/(phase1\.[123])/([^"]+)\.png"', repl, html)


def main():
    conv = md.Markdown(extensions=["tables", "fenced_code", "sane_lists", "footnotes"])
    toc, body = [], []
    for stem in ORDER:
        conv.reset()
        text = (CH / f"{stem}.md").read_text()
        m = re.search(r"^#\s+(.+)$", text, re.M)
        title = m.group(1).strip() if m else stem
        anchor = f"ch-{stem}"
        html = embed_figs(conv.convert(text))
        if stem in GROUP:
            toc.append(f'<li class="toc-h">{GROUP[stem]}</li>')
        # TOC label: code · short name (drop the parenthetical / bold markers)
        mt = re.match(r"(.+?)\s*[—–]\s*(.+)", title)
        if mt:
            code = mt.group(1).strip()
            rest = re.sub(r"（[^）]*）|\*\*", "", mt.group(2)).strip()
            label = f"{code} · {rest}"[:26]
        else:
            label = title[:26]
        toc.append(f'<li><a href="#{anchor}">{label}</a></li>')
        body.append(f'<section class="chapter" id="{anchor}">{html}</section>')

    toc_html = "\n".join(toc)
    body_html = "\n".join(body)

    HTML = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 1 總報告 — 元件建模與驗證（進入 Phase 2 前的完整 review）</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Noto+Serif+TC:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{
  --paper:#f6f2ea; --ink:#17150f; --soft:#5b554a; --line:#ddd5c5;
  --cim:#0072B2; --warn:#C45A12; --ok:#1b7f5a;
  --serif:'Spectral','Noto Serif TC',Georgia,serif;
  --cjk:'Noto Serif TC','Spectral',Georgia,serif;
  --mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,monospace;
}}
*{{box-sizing:border-box}}
html{{-webkit-print-color-adjust:exact;print-color-adjust:exact;scroll-behavior:smooth}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--cjk);
  font-size:15px;line-height:1.78;}}
.sidebar{{position:fixed;top:0;left:0;width:258px;height:100vh;overflow-y:auto;
  background:#efe9dd;border-right:1px solid var(--line);padding:26px 18px;font-size:12.5px}}
.sidebar .brand{{font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--cim);font-weight:600;margin-bottom:14px}}
.sidebar ul{{list-style:none;margin:0;padding:0}}
.sidebar li a{{color:var(--soft);text-decoration:none;display:block;padding:3px 6px;border-radius:3px;line-height:1.4}}
.sidebar li a:hover{{background:#e3dccb;color:var(--ink)}}
.toc-h{{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--cim);margin:14px 0 5px;font-weight:600}}
.main{{margin-left:258px;max-width:880px;padding:52px 56px 90px}}
.chapter{{padding-bottom:30px;margin-bottom:18px}}
.chapter+.chapter{{border-top:1px solid var(--line);padding-top:8px}}
h1{{font-family:var(--serif);font-size:30px;line-height:1.18;font-weight:600;margin:.5em 0 .4em;letter-spacing:-.01em}}
h2{{font-size:13px;font-family:var(--mono);letter-spacing:.13em;text-transform:uppercase;
  margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line);color:var(--cim)}}
h3{{font-size:18px;font-weight:600;margin:24px 0 6px;font-family:var(--serif)}}
p{{margin:.55em 0}}
a{{color:var(--cim)}}
strong{{font-weight:600}}
blockquote{{border-left:3px solid var(--cim);background:#fbf6ec;margin:16px 0;padding:10px 18px;font-size:14px;color:#3a352b}}
blockquote blockquote{{border-left-color:var(--warn);background:#fff4ea}}
code{{font-family:var(--mono);font-size:12.5px;background:#ece5d6;padding:1px 5px;border-radius:3px}}
pre{{background:#211e17;color:#f0eede;padding:14px 16px;border-radius:5px;overflow-x:auto;font-size:12px;line-height:1.55}}
pre code{{background:none;color:inherit;padding:0}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0}}
th{{text-align:left;font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--soft);border-bottom:1.5px solid var(--ink);padding:7px 9px;vertical-align:bottom}}
td{{padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}}
img{{display:block;max-width:560px;width:100%;margin:14px auto 4px;border:1px solid var(--line);background:#fff;border-radius:4px}}
hr{{border:none;border-top:1px solid var(--line);margin:26px 0}}
sup[id^="fnref"]{{font-size:.72em;font-weight:600}}
.footnote{{font-size:11.5px;color:var(--soft);border-top:1px solid var(--line);margin-top:28px;padding-top:6px}}
.footnote ol{{padding-left:20px;margin:6px 0}}
.footnote li{{margin:3px 0;line-height:1.5}}
.footnote code{{font-size:11px}}
.cover{{border-bottom:3px solid var(--ink);padding-bottom:26px;margin-bottom:10px}}
.cover .eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--cim);font-weight:600}}
.cover h1{{font-size:40px;margin:.2em 0 .15em}}
.cover .lede{{font-size:18px;color:var(--soft);font-style:italic;max-width:60ch;margin:0}}
.cover .meta{{font-family:var(--mono);font-size:11px;color:var(--soft);margin-top:14px;display:flex;gap:18px;flex-wrap:wrap}}
@page{{margin:15mm 14mm}}
@media print{{
  .sidebar{{display:none}}
  .main{{margin-left:0;max-width:100%;padding:0}}
  .chapter{{break-before:page}}
  .cover{{break-after:page}}
  h2,h3{{break-after:avoid}} img{{break-before:avoid;break-inside:avoid}} table,pre,blockquote{{break-inside:avoid}}
}}
</style></head><body>

<nav class="sidebar">
  <div class="brand">Edge-CIM · Phase 1 總報告</div>
  <ul>{toc_html}</ul>
</nav>

<div class="main">
  <header class="cover">
    <div class="eyebrow">Edge-CIM Simulation · Phase 1 · 進入 Phase 2 前的完整 review</div>
    <h1>元件建模與驗證 — Phase 1 總報告</h1>
    <p class="lede">把 1.1 / 1.2 / 1.3 合併為單一份、自給自足的回顧：每個元件模擬什麼、模型從哪來、驗證到什麼程度、缺什麼，以及一個帶條件的 GO/NO-GO 判定。不誇大、不軟化。</p>
    <div class="meta"><span>Phase 1 consolidated</span><span>decode-calibrated</span><span>誠實標籤：calibrated / simulated / assumption / borrowed</span><span>數字皆可回查 committed artifact</span></div>
  </header>
  {body_html}
</div>
</body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML)
    print(f"wrote {OUT} ({len(HTML)//1024} KB, {len(ORDER)} chapters, figures base64-embedded)")

    if Path(CHROME).exists():
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", OUT.as_uri()],
                       capture_output=True, timeout=120)
        if PDF.exists():
            print(f"wrote {PDF} ({PDF.stat().st_size // 1024} KB)")
    else:
        print("(Chrome not found — HTML only; PDF skipped)")


if __name__ == "__main__":
    main()
