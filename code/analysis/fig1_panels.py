#!/usr/bin/env python3
"""Figure 1 data panels, drawn from the committed receipts -- never typed.

Policy (2026-09-21): a generative image fixes only the LAYOUT of Figure 1; every panel
that carries a number is plotted here from the same receipts the checkers anchor:

  gate 3  lexical residualization   results/cross_model_sae/lextok20k_<cell>.json
          raw.excess (frozen-pair excess on raw features) vs
          arms.4096.pair_survival_real.excess (same pairs after residualization at the
          governing k=4096) and arms.4096.R_lex -- the "retain 92--94% of their excess"
          of the abstract / Sec. results.
  gate 4  token-aligned, leakage-clean results/cross_model_sae/tokaclean20k_<cell>.json
          s2_clean.R_tok_clean -- the "0.92--0.99 of the token-level excess" (A3-EXEC+A4).
  inset   nine cross-family cells    results/cross_model_sae/c4_<pair>.json, mutual_z
          (the nine cross-family rows of tab:c4_verdicts; all p at the 1/201 floor).

Each panel is written as an SVG at its final physical size (inches) with real-point fonts,
so nesting it into the 1600x1000 Figure-1 canvas at width=\textwidth reproduces those
point sizes exactly. Run with a Python with matplotlib.

  fig1_panels.py            -> paper/figures/fig1_v2/panel_{gate3,gate4,inset}.svg + values printed
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "cross_model_sae"
OUT = ROOT / "paper" / "figures" / "fig1_v2"

rcParams.update({
    "svg.fonttype": "none",            # real <text>, editable
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.linewidth": 0.5,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
})

# Figure-1 palette (shared with the schematic SVG)
INDIGO, TEAL, CORAL, AMBER, AMBER_DK, GREY = "#3B4CCA", "#0F766E", "#E8573F", "#F5C26B", "#C98F1E", "#9CA3AF"

# the three representative cells, in the paper's order (strongest, median, weakest)
CELLS = [("qwen3-1.7b__gemma2-2b", "Q1.7↔G2"),
         ("gemma2-9b__mistral7b", "G9↔M7"),
         ("llama3.2-1b__mistral7b", "L1↔M7")]

# the nine cross-family cells of tab:c4_verdicts (receipt basename, short label)
NINE = [("c4_qwen3-1.7b__gemma2-2b", "Q1.7↔G2"),
        ("c4_qwen3-8b__gemma2-2b", "Q8↔G2"),
        ("c4_qwen3-8b__gemma2-9b", "Q8↔G9"),
        ("c4_qwen3-1.7b__gemma2-9b", "Q1.7↔G9"),
        ("c4_qwen3-8b__mistral7b", "Q8↔M7"),
        ("c4_gemma2-9b__mistral7b", "G9↔M7"),
        ("c4_gemma2-2b__mistral7b", "G2↔M7"),
        ("c4_qwen3-1.7b__mistral7b", "Q1.7↔M7"),
        ("c4_llama3.2-1b__mistral7b", "L1↔M7")]


def load(name: str) -> dict:
    d = json.loads((RES / f"{name}.json").read_text())
    assert not d.get("smoke"), f"{name}: refusing to plot a smoke receipt"
    return d


def gate3() -> list[tuple]:
    """raw vs residualized frozen-pair excess, k=4096, three cells."""
    rows = []
    for cell, lab in CELLS:
        d = load(f"lextok20k_{cell}")
        arm = d["arms"]["4096"]
        rows.append((lab, d["raw"]["excess"], arm["pair_survival_real"]["excess"], arm["R_lex"]))
    fig, ax = plt.subplots(figsize=(0.66, 0.50))
    x = range(len(rows)); w = 0.36
    ax.bar([i - w / 2 for i in x], [r[1] for r in rows], w, color=AMBER, edgecolor="none")
    ax.bar([i + w / 2 for i in x], [r[2] for r in rows], w, color=AMBER_DK, edgecolor="none")
    for i, r in enumerate(rows):
        ax.text(i, max(r[1], r[2]) + 0.03, f"{100 * r[3]:.0f}%", ha="center", va="bottom",
                fontsize=5.6, fontweight="bold", color="#222")
    ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows], fontsize=4.8)
    ax.set_ylim(0, 0.95); ax.set_yticks([0, 0.4, 0.8]); ax.tick_params(axis="y", labelsize=4.6, length=1.5, pad=1)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_ylabel("pair excess", fontsize=4.8, labelpad=1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(0.02, 0.97, "raw", transform=ax.transAxes, fontsize=4.6, color=AMBER_DK, va="top")
    ax.text(0.02, 0.85, "resid.", transform=ax.transAxes, fontsize=4.6, color=AMBER_DK, va="top", fontweight="bold")
    fig.subplots_adjust(left=0.22, right=0.99, top=0.96, bottom=0.17)
    fig.savefig(OUT / "panel_gate3.svg"); plt.close(fig)
    return rows


def gate4() -> list[tuple]:
    """R_tok_clean per cell, leakage-clean design."""
    rows = []
    for cell, lab in CELLS:
        d = load(f"tokaclean20k_{cell}")
        assert all(d["a3_requirements"].values())
        rows.append((lab, d["s2_clean"]["R_tok_clean"]))
    fig, ax = plt.subplots(figsize=(0.66, 0.50))
    x = list(range(len(rows)))
    ax.bar(x, [r[1] for r in rows], 0.6, color=[INDIGO, TEAL, INDIGO], edgecolor="none")
    ax.axhline(1.0, color=GREY, lw=0.5, ls=(0, (2, 1.5)))
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 0.02, f"{r[1]:.2f}", ha="center", va="bottom", fontsize=5.6, fontweight="bold", color="#222")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=4.8)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(axis="y", labelsize=4.6, length=1.5, pad=1)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_ylabel(r"$R_{\mathrm{tok}}^{\mathrm{clean}}$", fontsize=5.0, labelpad=1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.96, bottom=0.17)
    fig.savefig(OUT / "panel_gate4.svg"); plt.close(fig)
    return rows


def inset() -> list[tuple]:
    """nine cross-family cells as z-coloured tiles, sorted by z."""
    rows = sorted(((lab, load(n)["mutual_z"]) for n, lab in NINE), key=lambda t: -t[1])
    zmin, zmax = min(r[1] for r in rows), max(r[1] for r in rows)
    fig, ax = plt.subplots(figsize=(1.22, 0.72))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("coral", ["#FBD9D2", CORAL])
    for k, (lab, z) in enumerate(rows):
        r, c = divmod(k, 3)
        ax.add_patch(plt.Rectangle((c, 2 - r), 0.94, 0.94, color=cmap((z - zmin) / (zmax - zmin)), lw=0))
        ax.text(c + 0.47, 2 - r + 0.60, lab, ha="center", va="center", fontsize=5.0, fontweight="bold", color="#222")
        ax.text(c + 0.47, 2 - r + 0.28, f"z = {z:.1f}", ha="center", va="center", fontsize=4.6, color="#222")
    ax.set_xlim(0, 2.94); ax.set_ylim(0, 2.94); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "panel_inset.svg"); plt.close(fig)
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    g3, g4, ins = gate3(), gate4(), inset()
    print("gate3 (cell, raw excess, residualized excess @k=4096, R_lex):")
    for r in g3: print("   %-9s raw=%.4f resid=%.4f R_lex=%.4f" % r)
    print("gate4 (cell, R_tok_clean):")
    for r in g4: print("   %-9s %.4f" % r)
    print("inset (nine cross-family cells, mutual_z, sorted):")
    for r in ins: print("   %-9s z=%.2f" % r)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("panel_*.svg"))))
