#!/usr/bin/env python3
"""Render the comparison charts as SVG for the README.

Two panels:
  left   capability by size — where each behaviour first appears
  right  bits per character against other small models, a tokenizer-independent
         measure so models with different vocabularies compare fairly

Emits a light and a dark variant; the README selects with <picture>.
"""

import json
import math
from pathlib import Path

# Validated with the dataviz palette validator: worst adjacent CVD pair
# ΔE 9.2 (light) / 9.4 (dark), normal-vision floor 27.6 / 26.5, both pass.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e3e2df",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", ref="#8a8985"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#33322f",
                  s1="#3987e5", s2="#d95926", s3="#199e70", ref="#8a8985"),
}

CAPABILITY = [   # params, name_acc, ignorance, question_rate  (from eval/results)
    (984, 0, 0, 47), (4944, 13, 0, 46), (9584, 51, 34, 56), (48416, 54, 17, 44),
    (95664, 55, 23, 48), (491040, 79, 69, 27), (968320, 52, 37, 43),
]
LABELS = ["1k", "5k", "10k", "50k", "100k", "500k", "1m"]

W, H = 880, 330
PAD_L, PAD_R, PAD_T, PAD_B = 52, 16, 52, 46
PANEL_W = (W - 40) // 2


def lx(p, x0, w, lo=900, hi=1.1e7):
    return x0 + w * (math.log10(p) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))


def svg(theme_name: str) -> str:
    t = THEMES[theme_name]
    ext = json.loads(Path("eval/results/external.json").read_text())
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="-apple-system,BlinkMacSystemFont,'
         f'Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']

    # ---------------------------------------------------- panel 1: capability
    x0, w = PAD_L, PANEL_W - PAD_L
    y0, h = PAD_T, H - PAD_T - PAD_B
    o.append(f'<text x="{x0-34}" y="18" fill="{t["ink"]}" font-size="13" '
             f'font-weight="600">Capability by size</text>')
    o.append(f'<text x="{x0-34}" y="{H-10}" fill="{t["ink2"]}" font-size="10">'
             f'parameters (log scale) · 900 held-out prompts</text>')
    for pct in (0, 25, 50, 75, 100):
        y = y0 + h - h * pct / 100
        o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x0-8}" y="{y+3.5:.1f}" fill="{t["ink2"]}" font-size="10" '
                 f'text-anchor="end">{pct}%</text>')
    # A legend rather than end-of-line labels: at this panel width the three
    # series converge on the right and direct labels collide with each other
    # and with the lines.
    series = [("names itself", 1, t["s1"]), ("declines to invent", 2, t["s2"]),
              ("asks a follow-up", 3, t["s3"])]
    lxp = x0 - 34
    for label, idx, colour in series:
        o.append(f'<rect x="{lxp}" y="26" width="9" height="9" rx="2" fill="{colour}"/>')
        o.append(f'<text x="{lxp+13}" y="34" fill="{t["ink2"]}" font-size="10">{label}</text>')
        lxp += 16 + len(label) * 5.4
    for label, idx, colour in series:
        pts = [(lx(p, x0, w), y0 + h - h * row[idx] / 100)
               for p, *rest in [(r[0], r) for r in CAPABILITY]
               for row in [rest[0]]]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                     for i, (x, y) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}" '
                     f'stroke="{t["surface"]}" stroke-width="2"/>')
    # 100k sits close to 50k on a log axis; drop it rather than overlap.
    for (p, *_), name in zip(CAPABILITY, LABELS):
        if name == "100k":
            continue
        o.append(f'<text x="{lx(p, x0, w):.1f}" y="{y0+h+15}" fill="{t["ink2"]}" '
                 f'font-size="9.5" text-anchor="middle">{name}</text>')

    # ------------------------------------- panel 2: bits/char vs other models
    x0 = PANEL_W + 40 + 10
    w = W - x0 - PAD_R - 4
    o.append(f'<text x="{x0-34}" y="18" fill="{t["ink"]}" font-size="13" '
             f'font-weight="600">Bits per character vs other small models</text>')
    o.append(f'<text x="{x0-34}" y="{H-10}" fill="{t["ink2"]}" font-size="10">'
             f'parameters (log scale) · lower is better · same text, any tokenizer</text>')
    lo, hi = 2.0, 10.0
    for v in (2, 4, 6, 8, 10):
        y = y0 + h - h * (v - lo) / (hi - lo)
        o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x0-8}" y="{y+3.5:.1f}" fill="{t["ink2"]}" font-size="10" '
                 f'text-anchor="end">{v}</text>')
    pk = [r for r in ext if r["family"] == "PocketLM"]
    ts = [r for r in ext if r["family"] == "TinyStories"]
    pts = [(lx(r["params"], x0, w), y0 + h - h * (r["dialogue"] - lo) / (hi - lo)) for r in pk]
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    o.append(f'<path d="{d}" fill="none" stroke="{t["s1"]}" stroke-width="2" '
             f'stroke-linejoin="round"/>')
    for (x, y), r in zip(pts, pk):
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{t["s1"]}" '
                 f'stroke="{t["surface"]}" stroke-width="2"/>')
    o.append(f'<text x="{pts[0][0]+8:.1f}" y="{pts[0][1]-8:.1f}" fill="{t["s1"]}" '
             f'font-size="10" font-weight="600">PocketLM</text>')
    for r in ts:
        x = lx(r["params"], x0, w)
        y = y0 + h - h * (r["dialogue"] - lo) / (hi - lo)
        o.append(f'<path d="M{x:.1f},{y-5:.1f} L{x+5:.1f},{y:.1f} L{x:.1f},{y+5:.1f} '
                 f'L{x-5:.1f},{y:.1f} Z" fill="{t["s2"]}" stroke="{t["surface"]}" '
                 f'stroke-width="2"/>')
    tx = lx(ts[0]["params"], x0, w)
    ty = y0 + h - h * (ts[0]["dialogue"] - lo) / (hi - lo)
    o.append(f'<text x="{tx-4:.1f}" y="{ty+18:.1f}" fill="{t["s2"]}" font-size="10" '
             f'font-weight="600" text-anchor="middle">TinyStories</text>')
    for p, name in [(9584, "10k"), (95664, "100k"), (968320, "1m")]:
        o.append(f'<text x="{lx(p, x0, w):.1f}" y="{y0+h+15}" fill="{t["ink2"]}" '
                 f'font-size="9.5" text-anchor="middle">{name}</text>')
    for r, name in zip(ts, ["1M", "3M"]):
        o.append(f'<text x="{lx(r["params"], x0, w):.1f}" y="{y0+h+15}" '
                 f'fill="{t["ink2"]}" font-size="9.5" text-anchor="middle">{name}</text>')

    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    Path("docs").mkdir(exist_ok=True)
    for name in THEMES:
        out = Path(f"docs/comparison-{name}.svg")
        out.write_text(svg(name))
        print(f"wrote {out}")
