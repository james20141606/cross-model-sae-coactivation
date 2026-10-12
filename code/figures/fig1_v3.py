#!/usr/bin/env python3
"""Figure 1 v3 = hero composite (figstyle/PROTOCOL.md §2 Step 3).

Background: paper/figures/fig1_v3/round3_A.png -- Gemini (gemini-3-pro-image) schematic generated from
paper/figures/fig1_v3/prompt3.txt; it carries NO result numbers (whitelist: 200 / 500, 20,000, 200x) and
two EMPTY data slots. The slots are widened here to the panel margins (white rounded covers) and filled with
vector panels drawn from the committed receipts, never typed:

  slot a  nine cross-family C4 cells, mutual-NN z over each cell's own 200x permutation null
          results/cross_model_sae/c4_<a>__<b>.json ["mutual_z"]          (tab:c4_verdicts rows)
  slot d  share of the excess retained: lexical residualization (k=4096, cross-fitted)
          results/cross_model_sae/lextok20k_<cell>.json ["arms"]["4096"]["R_lex"]        (results.tex 0.933/0.942/0.918)
          and the leakage-clean token-aligned design
          results/cross_model_sae/tokaclean20k_<cell>.json ["s2_clean"]["R_tok_clean"]   (results.tex 0.925/0.986/0.977)

Slot geometry (PNG px, 3168 x 1344 = 576 dpi at 5.5 in): interiors measured by paper/figures/fig1_v3/measure_slots.py
(a: 918..1507 x 791..1235; d: 1666..2390 x 771..1235), panels from the tint-run scan (panel 2 = 824..1579,
panel 3 = 1612..2452, tint ends at row 1299). Output: paper/figures/fig1_v3/fig1_v3.{svg,pdf,png}; hero.save() runs
audit.py (0 errors required) and pdffonts (no Type 3). Run with a Python with matplotlib.
"""
from __future__ import annotations
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figstyle"))
import figstyle as fs                       # noqa: E402
from hero import Hero                        # noqa: E402
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402

V3 = ROOT / "paper" / "figures" / "fig1_v3"
RES = ROOT / "results" / "cross_model_sae"
PNG = V3 / "round3_A.png"
PX_PER_PT = 3168 / (5.5 * 72)                # 8.0 px per printed pt
M = 14                                       # panel inner margin, px
PANEL2, PANEL3, BOTTOM = (824, 1579), (1612, 2452), 1299
MA = 9                                       # slot a needs the extra 10 px: three 7 pt pair labels across
SLOT_A = (PANEL2[0] + MA, 791, PANEL2[1] - PANEL2[0] - 2 * MA, BOTTOM - M - 791)   # widened raster slot a
SLOT_D = (PANEL3[0] + M, 771, PANEL3[1] - PANEL3[0] - 2 * M, BOTTOM - M - 771)   # widened raster slot d
F = 7.0                                      # the one data font size (printed pt); slot letters 8 pt bold

CELLS = [("qwen3-1.7b__gemma2-2b", "Q1.7-G2"), ("gemma2-9b__mistral7b", "G9-M7"), ("llama3.2-1b__mistral7b", "L1-M7")]
NINE = [("c4_qwen3-1.7b__gemma2-2b", "Q1.7-G2"), ("c4_qwen3-8b__gemma2-2b", "Q8-G2"), ("c4_qwen3-8b__gemma2-9b", "Q8-G9"),
        ("c4_qwen3-1.7b__gemma2-9b", "Q1.7-G9"), ("c4_qwen3-8b__mistral7b", "Q8-M7"), ("c4_gemma2-9b__mistral7b", "G9-M7"),
        ("c4_gemma2-2b__mistral7b", "G2-M7"), ("c4_qwen3-1.7b__mistral7b", "Q1.7-M7"), ("c4_llama3.2-1b__mistral7b", "L1-M7")]


def load(name):
    d = json.loads((RES / f"{name}.json").read_text())
    assert not d.get("smoke"), f"{name}: refusing to plot a smoke receipt"
    return d


def main():
    h = Hero(PNG, width_in=fs.TEXTWIDTH_IN)
    plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})   # the page must stay exactly 5.5 in wide
    fig = h.fig

    def width_pt(s, size=F, weight="normal"):
        t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 72
        t.remove(); return w

    def cover(x, y, w, hh, r=20):
        fig.axes[0].add_patch(FancyBboxPatch((x, y), w, hh, boxstyle=f"round,pad=0,rounding_size={r}",
                                             facecolor="white", edgecolor="none", zorder=2))

    def slot_axes(x, y, w, hh):
        cover(x, y, w, hh)
        ax = h.slot(x, y, w, hh, cover=False, pad=0)
        ax.set_xlim(0, w / PX_PER_PT); ax.set_ylim(hh / PX_PER_PT, 0); ax.axis("off")
        return ax, w / PX_PER_PT, hh / PX_PER_PT

    # ---------------------------------------------------------------- slot a: nine tiles
    rows = sorted(((lab, load(n)["mutual_z"]) for n, lab in NINE), key=lambda t: -t[1])
    ax, W, H = slot_axes(*SLOT_A)
    gap, mg = 1.5, 0.75
    cw, ch = (W - 2 * mg - 2 * gap) / 3, (H - 2 * mg - 2 * gap) / 3
    wmax = max(max(width_pt(lab), width_pt(f"{z:.1f}")) for lab, z in rows)
    assert wmax + 2.8 <= cw, f"tile label {wmax:.1f} pt does not fit a {cw:.1f} pt tile"
    zmin, zmax = min(z for _, z in rows), max(z for _, z in rows)
    cmap = LinearSegmentedColormap.from_list("coral", [fs.tint_hex(fs.CORAL, 0.16), fs.CORAL, "#B5392A"])
    for k, (lab, z) in enumerate(rows):
        r, c = divmod(k, 3)
        x, y = mg + c * (cw + gap), mg + r * (ch + gap)
        t = (z - zmin) / (zmax - zmin)
        ax.add_patch(Rectangle((x, y), cw, ch, facecolor=cmap(t), edgecolor="none"))
        fg = fs.SURF if t > 0.5 else fs.INK
        ax.text(x + cw / 2, y + ch * 0.30, lab, ha="center", va="center", fontsize=F, color=fg)
        ax.text(x + cw / 2, y + ch * 0.70, f"{z:.1f}", ha="center", va="center", fontsize=F, color=fg)
    h.text(PANEL2[0] + M + 10, 786, "a", fontsize=8, fontweight="bold", color=fs.INK)   # slot letter, left of the raster heading

    # ---------------------------------------------------------------- slot d: two groups of three bars
    c_rows, d_rows = [], []
    for cell, lab in CELLS:
        dl, dt = load(f"lextok20k_{cell}"), load(f"tokaclean20k_{cell}")
        assert all(dt["a3_requirements"].values()), f"{cell}: A3 requirement failed in receipt"
        c_rows.append((lab, dl["arms"]["4096"]["R_lex"])); d_rows.append((lab, dt["s2_clean"]["R_tok_clean"]))
    ax, W, H = slot_axes(*SLOT_D)
    ax.text(3.0, 8.5, "d", fontsize=8, fontweight="bold", color=fs.INK, va="baseline")
    ax.add_line(Line2D([W - 10, W - 7.2, W - 2.5], [5.5, 8.2, 2.5], color=fs.TEAL, lw=1.2, solid_capstyle="round", solid_joinstyle="round"))
    x_lab, x0, scale, pitch, bar_h = 31.0, 34.0, 50.0, 7.2, 5.2
    y = 8.5
    for header, grp, col in (("lexical residualization", c_rows, fs.SEQ_TEAL[3]), ("token-aligned, leakage-clean", d_rows, fs.TEAL)):
        ax.text(14.0 if y < 10 else 3.0, y, header, fontsize=F, color=fs.INK, va="baseline")
        y += 5.1
        for lab, v in grp:
            ax.text(x_lab, y, lab, fontsize=F, color=fs.INK2, ha="right", va="center")
            ax.add_patch(Rectangle((x0, y - bar_h / 2), scale * v, bar_h, facecolor=col, edgecolor="none"))
            ax.text(x0 + scale * v + 2.0, y, f"{v:.2f}", fontsize=F, color=fs.INK, va="center")
            y += pitch
        y += 3.3
    bottom = y - 3.3 - pitch + bar_h / 2 + 1.0     # last bar's lower edge + margin
    assert bottom <= H, f"bar rows overflow slot d ({bottom:.1f} > {H:.1f} pt)"

    svg, pdf, png = h.save(V3 / "fig1_v3")
    print("tiles:", ["%s=%.1f" % r for r in rows])
    print("bars lexical:", ["%s=%.4f" % r for r in c_rows], " token:", ["%s=%.4f" % r for r in d_rows])
    print("slot a (px):", SLOT_A, " slot d (px):", SLOT_D, " tile label max %.1f pt in %.1f pt tiles" % (wmax, cw))


if __name__ == "__main__":
    main()
