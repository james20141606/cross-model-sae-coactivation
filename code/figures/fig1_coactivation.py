#!/usr/bin/env python3
"""Figure 1 (body): cross-model SAE coactivation survives the controls built to kill it.

Two layers, per the project figure rule (<internal-notes>, 2026-09-21):
  * SCHEMATIC layer -- panels 1, 3a/3b, 4 -- written straight to SVG with code/figstyle/glyphs.py.
    It carries no axes, ticks or data marks.
  * DATA layer -- two matplotlib panels plotted from the committed receipts, never typed:
      panel 2   nine cross-family C4 cells, mutual-NN z over each cell's own permutation null
                results/cross_model_sae/c4_<a>__<b>.json  ["mutual_z"]   (the rows of tab:c4_verdicts)
      panel 3c  share of the frozen-pair excess retained under cross-fitted lexical residualization
                results/cross_model_sae/lextok20k_<cell>.json  ["arms"]["4096"]["R_lex"]   (results.tex: 0.933/0.942/0.918)
      panel 3d  share of the token-level excess retained under the leakage-clean token-aligned design
                results/cross_model_sae/tokaclean20k_<cell>.json  ["s2_clean"]["R_tok_clean"] (results.tex: 0.925/0.986/0.977)
    The bold "retained" line in panel 3 is formatted from the same receipts.

Layout gate: compose.export() runs code/figstyle/audit.py (Chrome text metrics) and refuses to export on
any overlap / overflow / crossing / text < 7 pt at 5.5 in; PDF goes through headless Chrome (no Type 3).
Output: paper/figures/fig1_coactivation.{svg,pdf,png}; intermediates in paper/figures/fig1_v2/.
Run with a Python with matplotlib.
"""
from __future__ import annotations
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figstyle"))
import figstyle as fs                      # noqa: E402
import glyphs as G                          # noqa: E402
import compose                              # noqa: E402
import matplotlib                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle    # noqa: E402

RES = ROOT / "results" / "cross_model_sae"
OUT = ROOT / "paper" / "figures"
TMP = OUT / "fig1_v2"
TMP.mkdir(parents=True, exist_ok=True)

INDIGO, AMBER, CORAL, TEAL, GREY = fs.INDIGO, fs.AMBER, fs.CORAL, fs.TEAL, fs.GREY
INK, INK2, INK3, SURF = fs.INK, fs.INK2, fs.INK3, fs.SURF
T, B, K = 9.0, 7.5, 7.0            # the only three printed sizes: panel title (bold), label, note
LIGHT_TEAL = fs.SEQ_TEAL[3]
SAVE = dict(bbox_inches=None, pad_inches=0)   # exact figsize -> exact placement on the canvas

# the three representative cells, in the paper's order (strongest, median, weakest)
CELLS = [("qwen3-1.7b__gemma2-2b", "Q1.7–G2"), ("gemma2-9b__mistral7b", "G9–M7"), ("llama3.2-1b__mistral7b", "L1–M7")]
# the nine cross-family C4 cells of tab:c4_verdicts
NINE = [("c4_qwen3-1.7b__gemma2-2b", "Q1.7–G2"), ("c4_qwen3-8b__gemma2-2b", "Q8–G2"), ("c4_qwen3-8b__gemma2-9b", "Q8–G9"),
        ("c4_qwen3-1.7b__gemma2-9b", "Q1.7–G9"), ("c4_qwen3-8b__mistral7b", "Q8–M7"), ("c4_gemma2-9b__mistral7b", "G9–M7"),
        ("c4_gemma2-2b__mistral7b", "G2–M7"), ("c4_qwen3-1.7b__mistral7b", "Q1.7–M7"), ("c4_llama3.2-1b__mistral7b", "L1–M7")]


def load(name):
    d = json.loads((RES / f"{name}.json").read_text())
    assert not d.get("smoke"), f"{name}: refusing to plot a smoke receipt"
    return d


def text_width_pt(s, size=K, weight="normal"):
    """Real Times New Roman advance width, from matplotlib's renderer (points)."""
    f = plt.figure(figsize=(1, 1), dpi=72)
    w = f.text(0, 0, s, fontsize=size, fontweight=weight).get_window_extent(f.canvas.get_renderer()).width
    plt.close(f)
    return w


# ----------------------------------------------------------------------------- data layer
def panel_tiles():
    """Nine cross-family cells as a 3x3 tile grid: cell size from the widest label at 7 pt, colour monotone in z."""
    rows = sorted(((lab, load(n)["mutual_z"]) for n, lab in NINE), key=lambda t: -t[1])
    pad, gap = 3.0, 3.0
    wmax = max(max(text_width_pt(lab), text_width_pt(f"z = {z:.1f}")) for lab, z in rows)
    cw, ch = wmax + 2 * pad, 2 * K + 3 * pad + 1.0
    W, H = 3 * cw + 2 * gap, 3 * ch + 2 * gap
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    zmin, zmax = min(z for _, z in rows), max(z for _, z in rows)
    cmap = LinearSegmentedColormap.from_list("coral", [fs.tint_hex(CORAL, 0.16), CORAL, "#B5392A"])
    for k, (lab, z) in enumerate(rows):
        r, c = divmod(k, 3)
        x, y = c * (cw + gap), r * (ch + gap)
        t = (z - zmin) / (zmax - zmin)
        ax.add_patch(Rectangle((x, y), cw, ch, color=cmap(t), lw=0))
        fg = SURF if t > 0.5 else INK
        ax.text(x + cw / 2, y + pad + 0.5, lab, ha="center", va="top", fontsize=K, color=fg)
        ax.text(x + cw / 2, y + ch - pad - 0.5, f"z = {z:.1f}", ha="center", va="bottom", fontsize=K, color=fg)
    fig.savefig(TMP / "panel_tiles.svg", **SAVE); plt.close(fig)
    return rows, (W * compose.MPL_SCALE, H * compose.MPL_SCALE)


def panel_bars():
    """Share of the excess retained: (c) lexical residualization, (d) leakage-clean token-aligned; three cells."""
    c_rows, d_rows = [], []
    for cell, lab in CELLS:
        dl = load(f"lextok20k_{cell}"); dt = load(f"tokaclean20k_{cell}")
        assert all(dt["a3_requirements"].values()), f"{cell}: A3 requirement failed in receipt"
        c_rows.append((lab, dl["arms"]["4096"]["R_lex"])); d_rows.append((lab, dt["s2_clean"]["R_tok_clean"]))
    fig, ax = plt.subplots(figsize=(1.60, 0.84))
    ax.grid(False)
    ys = [5.75, 4.75, 3.75, 2.25, 1.25, 0.25]
    vals = [v for _, v in c_rows] + [v for _, v in d_rows]
    labs = [f"c · {l}" for l, _ in c_rows] + [f"d · {l}" for l, _ in d_rows]
    ax.barh(ys, vals, height=0.74, color=[LIGHT_TEAL] * 3 + [TEAL] * 3, edgecolor="none")
    for y, v in zip(ys, vals):
        ax.text(v + 0.025, y, f"{v:.2f}", va="center", ha="left", fontsize=K, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=K)
    ax.set_ylim(-0.3, 6.3); ax.set_xlim(0, 1.24); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"], fontsize=K)
    ax.tick_params(axis="both", length=2, pad=1.5)
    ax.set_xlabel("share of excess retained", fontsize=K, labelpad=2)
    fig.subplots_adjust(left=0.36, right=0.99, top=0.99, bottom=0.26)
    fig.savefig(TMP / "panel_bars.svg", **SAVE); plt.close(fig)
    return c_rows, d_rows, (1.60 * 72 * compose.MPL_SCALE, 0.84 * 72 * compose.MPL_SCALE)


# ----------------------------------------------------------------------- schematic layer
def lines(parent, x, y, items, size=B, pitch=None, **kw):
    pitch = pitch or round(G.pt(size) * 1.18)
    for i, s in enumerate(items):
        G.text(parent, x, y + i * pitch, s, size=size, **kw)


def header(svg, x, y, w, hue, number, title):
    G.element(svg, "circle", cx=x + 30, cy=y + 40, r=16, fill=SURF, stroke=hue, stroke_width=2.5)
    G.text(svg, x + 30, y + 47, str(number), size=B, color=hue, anchor="middle")
    lines(svg, x + 52, y + 50, title, size=T, weight="bold", pitch=42)


def card(svg, x, y, w, h, hue, dashed=False):
    r = G.element(svg, "rect", x=x, y=y, width=w, height=h, rx=12, fill=SURF, stroke=hue, stroke_width=2)
    if dashed:
        r.set("stroke-dasharray", "8 6")
    return r


def grid(svg, x, y, cols, rows, cell, gap, color, lit, lit_color=CORAL):
    """SAE dictionary as a cell grid; unlit cells wash 0.22 so they read on the panel tint. No numbers."""
    lit = set(lit)
    for r in range(rows):
        for c in range(cols):
            on = (c, r) in lit
            G.element(svg, "rect", x=x + c * (cell + gap), y=y + r * (cell + gap), width=cell, height=cell, rx=3,
                      fill=lit_color if on else fs.tint_hex(color, 0.22), stroke=color if on else "none", stroke_width=1.2)


def vline(svg, x, y1, y2, color=CORAL, width=2.5):
    G.element(svg, "path", d=f"M{x} {y1} L{x} {y2}", stroke=color, stroke_width=width, fill="none", stroke_linecap="round")


def schematic(tiles, bars, tiles_px, bars_px, W=1600, H=600):
    svg = G.canvas(W, H)
    py, ph = 16, H - 32
    P1 = (16, py, 310, ph); P2 = (338, py, 448, ph); P3 = (798, py, 494, ph); P4 = (1304, py, 280, ph)
    G.panel(svg, *P1, INDIGO); G.panel(svg, *P2, CORAL); G.panel(svg, *P3, TEAL); G.panel(svg, *P4, INK2, alpha=0.06)
    header(svg, P1[0], py, P1[2], INDIGO, 1, ["Paired features", "fire on the", "same documents"])
    header(svg, P2[0], py, P2[2], CORAL, 2, ["Nine cross-family", "cells on C4 text"])
    header(svg, P3[0], py, P3[2], TEAL, 3, ["Four controls built to", "kill it; all survived"])
    header(svg, P4[0], py, P4[2], INK2, 4, ["Shared", "response,", "not identity"])

    # ---- panel 1: dictionary A above, dictionary B below, the two documents between them
    cx = P1[0] + P1[2] / 2
    G.text(svg, cx, 178, "strict mutual-NN pairs", size=K, color=INK2, italic=True, anchor="middle")
    G.text(svg, cx, 208, "Model A · Gemma-2-9B", size=B, anchor="middle")
    gx, gy, cell, gap = 87, 220, 20, 4; pitch = cell + gap
    grid(svg, gx, gy, 8, 5, cell, gap, INDIGO, [(1, 4), (6, 4)])
    ga_bottom = gy + 5 * pitch - gap
    x_ob, x_cr = gx + 1 * pitch + cell / 2, gx + 6 * pitch + cell / 2
    cy1, ch1, cw1 = 360, 44, 104
    for xc, lab in ((x_ob, "obituary"), (x_cr, "cricket")):
        card(svg, xc - cw1 / 2, cy1, cw1, ch1, INK2)
        G.text(svg, xc, cy1 + ch1 / 2 + 9, lab, size=K, anchor="middle")
        vline(svg, xc, ga_bottom + 2, cy1 - 2)
    hx, hy = gx, 428
    for xc in (x_ob, x_cr):
        vline(svg, xc, cy1 + ch1 + 2, hy - 2)
    grid(svg, hx, hy, 8, 5, cell, gap, AMBER, [(1, 0), (6, 0)])
    G.text(svg, cx, hy + 5 * pitch - gap + 28, "Model B · Mistral-7B", size=B, anchor="middle")

    # ---- panel 2: tiles (data, placed by compose) + notes
    cx2 = P2[0] + P2[2] / 2
    tx, ty = cx2 - tiles_px[0] / 2, 132
    lines(svg, cx2, ty + tiles_px[1] + 36, ["each above its own", "200× permutation null"],
          size=K, color=INK2, italic=True, anchor="middle", pitch=31)
    lines(svg, cx2, 540, ["Q Qwen3 · G Gemma2 · M Mistral", "L Llama-3.2; digits = params (B)"], size=K, color=INK3, anchor="middle", pitch=31)

    # ---- panel 3: two schematic gates (a, b) + the data panel (c, d) + the verdict line
    x3, w3 = P3[0] + 14, 233
    for i, (letter, icons, label) in enumerate((("a", ("ruler", "tag"), ["length and coarse-", "topic matched null"]),
                                                ("b", ("boxes",), ["fine-topic null,", "200 / 500 TF-IDF"]))):
        x = x3 + i * w3; y, h = 114, 104
        card(svg, x, y, w3, h, TEAL)
        G.text(svg, x + 12, y + 24, letter, size=B, color=TEAL)
        G.check(svg, x + w3 - 34, y + 8, size=18, color=TEAL)
        for j, ic in enumerate(icons):
            G.icon(svg, ic, x + w3 / 2 - 16 + (j - (len(icons) - 1) / 2) * 48, y + 8, size=32, color=TEAL)
        lines(svg, x + w3 / 2, y + 66, label, size=K, anchor="middle", pitch=31)
    bx, by = x3, 230
    c_rows, d_rows = bars
    lo_c, hi_c = min(v for _, v in c_rows), max(v for _, v in c_rows)
    lo_d, hi_d = min(v for _, v in d_rows), max(v for _, v in d_rows)
    cx3 = P3[0] + P3[2] / 2
    lines(svg, cx3, by + bars_px[1] + 32, ["(c) residualization on 20,000 C4 docs", "(d) token-aligned, leakage-clean"],
          size=K, anchor="middle", pitch=31)
    G.text(svg, cx3, by + bars_px[1] + 32 + 2 * 31, f"{100 * lo_c:.0f}–{100 * hi_c:.0f}% · {lo_d:.2f}–{hi_d:.2f} retained",
           size=B, weight="bold", color=TEAL, anchor="middle")

    # ---- panel 4: shared response (yes) vs feature identity (no)
    x4, w4 = P4[0] + 14, P4[2] - 28
    y = 170; h = 176
    card(svg, x4, y, w4, h, TEAL)
    G.check(svg, x4 + 12, y + 10, size=18, color=TEAL)
    G.text(svg, x4 + w4 / 2 + 10, y + 34, "shared response", size=K, anchor="middle")
    G.icon(svg, "file-text", x4 + 30, y + 52, size=44, color=INK2)
    for k, hue in enumerate((INDIGO, AMBER)):
        G.element(svg, "rect", x=x4 + 156, y=y + 46 + k * 36, width=22, height=22, rx=4, fill=hue)
        G.element(svg, "path", d=f"M{x4 + 78} {y + 74} L{x4 + 154} {y + 57 + k * 36}", stroke=CORAL, stroke_width=2.4, fill="none")
    lines(svg, x4 + w4 / 2, y + 138, ["same documents drive", "a feature in each"], size=K, color=INK2, italic=True, anchor="middle")
    y = 362; h = 184
    card(svg, x4, y, w4, h, GREY, dashed=True)
    G.cross(svg, x4 + 14, y + 12, size=14, color=GREY)
    G.text(svg, x4 + w4 / 2 + 10, y + 34, "feature identity", size=K, color=GREY, anchor="middle")
    G.element(svg, "rect", x=x4 + 62, y=y + 58, width=22, height=22, rx=4, fill=INDIGO, opacity=0.45)
    G.text(svg, x4 + w4 / 2, y + 77, "=", size=T, color=GREY, anchor="middle")
    G.element(svg, "rect", x=x4 + w4 - 84, y=y + 58, width=22, height=22, rx=4, fill=AMBER, opacity=0.45)
    lines(svg, x4 + w4 / 2, y + 104, ["no compositional", "or syntactic", "sharing claimed"], size=K, color=GREY, italic=True, anchor="middle")
    return G.write(svg, TMP / "layout.svg"), (tx, ty), (bx, by)


def main():
    fs.use()
    tiles, tiles_px = panel_tiles()
    c_rows, d_rows, bars_px = panel_bars()
    layout, tpos, bpos = schematic(tiles, (c_rows, d_rows), tiles_px, bars_px)
    final = compose.place(layout, [(TMP / "panel_tiles.svg", tpos[0], tpos[1], None),
                                   (TMP / "panel_bars.svg", bpos[0], bpos[1], None)],
                          out=OUT / "fig1_coactivation.svg")
    print("tiles (nine cross-family cells, mutual_z, sorted):")
    for r in tiles: print("   %-8s z=%.2f" % r)
    print("bars  (c) R_lex @k=4096:", ["%s=%.4f" % r for r in c_rows])
    print("bars  (d) R_tok_clean:  ", ["%s=%.4f" % r for r in d_rows])
    pdf, png = compose.export(final)
    print("wrote", final, pdf, png)
    print(compose.check_fonts(pdf))


if __name__ == "__main__":
    main()
