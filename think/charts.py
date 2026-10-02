"""README charts for v0.5.0: quality against imajev-4b and v0.4.0, and request6 latency on a log scale.

python think/charts.py docs/img
Numbers come from docs/think-35b.md (think/evals/collect.py) and the latency runs noted below.
"""
import math
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  base="#c3c2b7", series=["#2a78d6", "#eb6834", "#b5b4ab"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 base="#383835", series=["#3987e5", "#d95926", "#5d5c56"]),
}
X0, X1, LABEL_X = 220, 700, 208
FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'


def bar(x, y, w, t, fill, title):
    """Rounded data end, square at the baseline."""
    r = min(4, t / 2)
    return (f'<g><title>{title}</title><path d="M{X0},{y} H{x + w - r:.1f} A{r},{r} 0 0 1 {x + w:.1f},{y + r} '
            f'V{y + t - r} A{r},{r} 0 0 1 {x + w - r:.1f},{y + t} H{X0} Z" fill="{fill}"/></g>')


def frame(c, h, title, subtitle=None):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{h}" viewBox="0 0 760 {h}" {FONT}>',
         f'<rect width="760" height="{h}" fill="{c["surface"]}" rx="8"/>',
         f'<text x="20" y="28" font-size="16" font-weight="600" fill="{c["ink"]}">{title}</text>']
    if subtitle:
        s.append(f'<text x="20" y="46" font-size="12" fill="{c["ink2"]}">{subtitle}</text>')
    return s


def axis(c, top, bottom, ticks, pos):
    s = []
    for v, label in ticks:
        x = pos(v)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom}" stroke="{c["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{bottom + 16}" font-size="11" fill="{c["muted"]}" text-anchor="middle">{label}</text>')
    return s


def row_label(c, y, label, sub):
    return [f'<text x="{LABEL_X}" y="{y}" font-size="12.5" fill="{c["ink"]}" text-anchor="end">{label}</text>',
            f'<text x="{LABEL_X}" y="{y + 15}" font-size="11" fill="{c["muted"]}" text-anchor="end">{sub}</text>']


SERIES = ["peekaboolean-think-35b (v0.5.0)", "imajev-4b", "peekaboolean-450m (v0.4.0)"]
# accuracy; same requests to all three (imajev-4b as shipped). v0.4.0 takes one image, so two-image items count wrong.
GROUPS = [
    ("ImajevBench dev + cal", "254 items, its own scorer", (245 / 254, 203 / 254, 118 / 254)),
    ("RealJev", "2,100 real-image items", (0.784, 0.731, 0.565)),
    ("imajev's held-out exam", "1,779 real photos", (0.656, 0.580, 0.336)),
    ("JevBench hard", "111 items, text only", (103 / 111, 79 / 111, 47 / 111)),
]


def quality(c):
    pitch, t, gap, top = 56, 12, 2, 64
    bottom = top + pitch * len(GROUPS)
    h = bottom + 34
    s = frame(c, h, "Accuracy on the same requests (higher is better)")
    x = 20
    for name, fill in zip(SERIES, c["series"]):
        s.append(f'<rect x="{x}" y="42" width="12" height="12" rx="3" fill="{fill}"/>')
        s.append(f'<text x="{x + 18}" y="52" font-size="12" fill="{c["ink2"]}">{name}</text>')
        x += 18 + round(6.4 * len(name)) + 24
    pos = lambda v: X0 + (X1 - X0) * v
    s += axis(c, top, bottom, [(0, "0"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1, "100%")], pos)
    for i, (label, sub, vals) in enumerate(GROUPS):
        y0 = top + 7 + i * pitch
        s += row_label(c, y0 + 15, label, sub)
        for j, (name, v, fill) in enumerate(zip(SERIES, vals, c["series"])):
            s.append(bar(X0, y0 + j * (t + gap), (X1 - X0) * v, t, fill, f"{label}, {name}: {100 * v:.1f}%"))
            weight = ' font-weight="600"' if j == 0 else ""
            s.append(f'<text x="{pos(v) + 6:.1f}" y="{y0 + j * (t + gap) + 10}" font-size="11"{weight} '
                     f'fill="{c["ink"] if j == 0 else c["ink2"]}">{100 * v:.1f}%</text>')
    s.append(f'<line x1="{X0}" x2="{X0}" y1="{top}" y2="{bottom}" stroke="{c["base"]}" stroke-width="1"/>')
    return "\n".join(s + ["</svg>"]) + "\n"


# p95 of request6 (6 questions, 28 options). 450M rows: --mode auto, 512 px, measured 2026-09-30.
# think-35b rows: think/lat6.py on square.jpg, 2026-10-02; GPU = vLLM FP8 + jevsrv, 4 samples; Mac = jevmlx, 1 sample.
LATENCY = [("think-35b (v0.5.0)", "RTX PRO 6000, 4 samples", 74.27, 0),
           ("think-35b (v0.5.0)", "M1 Max, MLX 8-bit, 1 sample", 202.14, 0),
           ("450m (v0.4.0)", "M1 Max, LFM2.5-VL-450M", 0.264, 2),
           ("450m (v0.3.0)", "M1 Max, LFM2.5-VL-450M", 0.208, 2),
           ("500m (v0.2.0)", "M1 Max, SmolVLM-500M", 0.190, 2),
           ("450m (v0.4.0)", "RTX PRO 6000, bf16", 0.075, 2)]


def latency(c):
    rows = [r for r in LATENCY if r[2] is not None]
    pitch, t, top = 40, 14, 58
    bottom = top + pitch * len(rows)
    h = bottom + 34
    lo, hi = math.log10(0.05), math.log10(1000)
    pos = lambda v: X0 + (X1 - X0) * (math.log10(v) - lo) / (hi - lo)
    s = frame(c, h, "p95 latency, 6 questions / 28 options (log scale)",
              "think-35b reasons before every answer; the 450M models score options directly")
    s += axis(c, top, bottom, [(0.1, "0.1 s"), (1, "1 s"), (10, "10 s"), (100, "100 s"), (1000, "1000 s")], pos)
    bx = pos(0.5)
    s.append(f'<line x1="{bx:.1f}" x2="{bx:.1f}" y1="{top}" y2="{bottom}" stroke="{c["ink2"]}" '
             f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{bx + 4:.1f}" y="{bottom - 6}" font-size="11" fill="{c["ink2"]}">0.5 s budget of the 450M models</text>')
    for i, (label, sub, sec, k) in enumerate(rows):
        y0 = top + 8 + i * pitch
        w = pos(sec) - X0
        s += row_label(c, y0 + 11, label, sub)
        s.append(bar(X0, y0, w, t, c["series"][k], f"{label}, {sub}: {sec:g} s"))
        txt = f"{sec:.0f} s" if sec >= 10 else (f"{sec:.1f} s" if sec >= 1 else f"{sec * 1000:.0f} ms")
        s.append(f'<rect x="{X0 + w + 3:.1f}" y="{y0 - 1}" width="{7 * len(txt) + 6}" height="16" fill="{c["surface"]}"/>')  # over the budget line
        s.append(f'<text x="{X0 + w + 6:.1f}" y="{y0 + 11}" font-size="11.5" font-weight="600" fill="{c["ink"]}">{txt}</text>')
    s.append(f'<line x1="{X0}" x2="{X0}" y1="{top}" y2="{bottom}" stroke="{c["base"]}" stroke-width="1"/>')
    return "\n".join(s + ["</svg>"]) + "\n"


if __name__ == "__main__":
    for mode, c in THEMES.items():
        (OUT / f"quality-{mode}.svg").write_text(quality(c))
        (OUT / f"latency-{mode}.svg").write_text(latency(c))
