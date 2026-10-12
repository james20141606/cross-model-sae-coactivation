#!/usr/bin/env python3
"""Round-7 publication figures (all data read from released JSONs; colorblind-safe).

fig1_arc.pdf              : the paper's arc — three artifacts caught by controls, what survives.
fig_universality_matrix.pdf: model x model mutual-NN z heatmap (template corpus) + C4 verdicts.
fig_language_panel.pdf    : within-model cross-lingual z + cross-model per-language bars.

Usage:  python code/analysis/make_round7_figures.py   (from repo root)
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Okabe-Ito colorblind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}
# House style (<internal-notes> rule 4): Avenir Next typography,
# colorblind-safe hues, no default-matplotlib gray boxes, legible at column width.
plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Avenir Next",
    "mathtext.it": "Avenir Next:italic",
    "mathtext.bf": "Avenir Next:bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#222222", "text.color": "#222222",
})

RES = "results/cross_model_sae"


def mut_z(fname):
    fp = os.path.join(RES, fname)
    if not os.path.exists(fp):
        return None
    d = json.load(open(fp)); o = d["observed"]; n = d["null_mutual_nn"]
    sd = n.get("sd", 0) or 1e-9
    return (o["n_mutual_nn"] - n["mean"]) / sd


# ----------------------------------------------------------------- fig 1: the arc
def fig1_arc():
    fig, ax = plt.subplots(figsize=(9.6, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    rows = [
        (8.45, "Raw cross-lingual probe-attribution $r$\n(17 models, en/zh/ja/ko)",
         "layer-shuffle / concept-mismatch /\nmonolingual baseline",
         "depth-profile artifact\n(English-only Pythia tops the scale)", OI["red"]),
        (5.75, "Depth-corrected residual\n(multilingual $+0.059$ vs EN $+0.003$, $n{=}20$)",
         "scale the controls: 206 pairs,\n17 models, +4 EN-only families",
         "baseline artifact\n(GPT-2 largest; reversal, $p{=}0.79$)", OI["red"]),
        (3.05, "SAE feature matching flagship\n(Pythia-70M$\\leftrightarrow$Mistral $z{=}21.9$)",
         "natural-text (C4) corpus control\n+ strict reciprocal criterion",
         "corpus artifact\n(falls to null on C4, $z{=}-1.9$)", OI["red"]),
        (0.35, "Survives every control:", "",
         "capable-pair cross-family universality (C4 $z{=}13$–45, 3 families)\n"
         "+ graded cross-lingual sharing (ja$>$zh$>$ko; scale-dependent)", OI["green"]),
    ]
    for y, claim, control, verdict, col in rows:
        h = 1.9
        if control:
            ax.add_patch(FancyBboxPatch((0.1, y), 3.1, h, boxstyle="round,pad=0.08",
                                        fc="#F2F2F2", ec="#888888", lw=0.8))
            ax.text(1.65, y + h / 2, claim, ha="center", va="center", fontsize=7.6)
            ax.add_patch(FancyArrowPatch((3.35, y + h / 2), (4.05, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=13, color="#555555"))
            ax.add_patch(FancyBboxPatch((4.1, y), 2.55, h, boxstyle="round,pad=0.08",
                                        fc="#E8F0FE", ec=OI["blue"], lw=0.9))
            ax.text(5.38, y + h / 2, control, ha="center", va="center", fontsize=7.3, color=OI["blue"])
            ax.add_patch(FancyArrowPatch((6.8, y + h / 2), (7.5, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=13, color="#555555"))
            ax.add_patch(FancyBboxPatch((7.55, y), 2.35, h, boxstyle="round,pad=0.08",
                                        fc="#FDECEA", ec=col, lw=1.1))
            ax.text(8.72, y + h / 2, verdict, ha="center", va="center", fontsize=7.3, color=col)
        else:
            ax.add_patch(FancyBboxPatch((0.1, y), 9.8, 2.0, boxstyle="round,pad=0.08",
                                        fc="#E6F4EA", ec=col, lw=1.3))
            ax.text(1.6, y + 1.0, claim, ha="center", va="center", fontsize=8.4,
                    fontweight="bold", color=col)
            ax.text(6.0, y + 1.0, verdict, ha="center", va="center", fontsize=7.8, color="#1A5E20")
    ax.text(1.65, 10.0, "claimed signal", ha="center", fontsize=8, fontweight="bold")
    ax.text(5.38, 10.0, "control", ha="center", fontsize=8, fontweight="bold", color=OI["blue"])
    ax.text(8.72, 10.0, "verdict", ha="center", fontsize=8, fontweight="bold", color=OI["red"])
    fig.tight_layout()
    fig.savefig("paper/figures/fig1_arc.pdf", bbox_inches="tight")
    print("[wrote] paper/figures/fig1_arc.pdf")


# --------------------------------------------- fig 2: universality matrix + verdicts
def fig_matrix():
    models = ["Gemma2-2B", "Gemma2-9B", "Mistral-7B", "Qwen3-1.7B", "Qwen3-8B", "Pythia-70M", "GPT-2"]
    key = {("Gemma2-2B", "Gemma2-9B"): "gemma2-2b__gemma2-9b.json",
           ("Gemma2-2B", "Mistral-7B"): "gemma2-2b__mistral7b.json",
           ("Gemma2-9B", "Mistral-7B"): "gemma2-9b__mistral7b.json",
           ("Qwen3-1.7B", "Gemma2-2B"): "qwen3-1.7b__gemma2-2b.json",
           ("Qwen3-1.7B", "Mistral-7B"): "qwen3-1.7b__mistral7b.json",
           ("Pythia-70M", "Mistral-7B"): "pythia70m__mistral7b.json",
           ("Pythia-70M", "Gemma2-9B"): "pythia70m__gemma2-9b.json",
           ("Gemma2-2B", "GPT-2"): "gemma2-2b__gpt2.json",
           ("Gemma2-9B", "GPT-2"): "gemma2-9b__gpt2.json",
           ("Pythia-70M", "GPT-2"): "pythia70m__gpt2.json",
           ("Mistral-7B", "GPT-2"): "gpt2__mistral7b.json"}
    key[("Qwen3-8B", "Gemma2-2B")] = "c4_qwen3-8b__gemma2-2b.json"
    key[("Qwen3-1.7B", "Gemma2-9B")] = "c4_qwen3-1.7b__gemma2-9b.json"
    key[("Qwen3-8B", "Gemma2-9B")] = "c4_qwen3-8b__gemma2-9b.json"
    key[("Qwen3-8B", "Mistral-7B")] = "c4_qwen3-8b__mistral7b.json"
    key[("Qwen3-8B", "Qwen3-1.7B")] = "c4_qwen3-8b__qwen3-1.7b.json"
    c4 = {("Gemma2-2B", "Gemma2-9B"): mut_z("c4_gemma2-2b__gemma2-9b.json"),
          ("Qwen3-8B", "Gemma2-9B"): mut_z("c4_qwen3-8b__gemma2-9b.json"),
          ("Qwen3-8B", "Gemma2-2B"): mut_z("c4_qwen3-8b__gemma2-2b.json"),
          ("Qwen3-1.7B", "Gemma2-9B"): mut_z("c4_qwen3-1.7b__gemma2-9b.json"),
          ("Qwen3-8B", "Mistral-7B"): mut_z("c4_qwen3-8b__mistral7b.json"),
          ("Qwen3-8B", "Qwen3-1.7B"): mut_z("c4_qwen3-8b__qwen3-1.7b.json"),
          ("Gemma2-9B", "Mistral-7B"): mut_z("c4_gemma2-9b__mistral7b.json"),
          ("Gemma2-2B", "Mistral-7B"): mut_z("c4_gemma2-2b__mistral7b.json"),
          ("Qwen3-1.7B", "Gemma2-2B"): mut_z("c4_qwen3-1.7b__gemma2-2b.json"),
          ("Qwen3-1.7B", "Mistral-7B"): mut_z("c4_qwen3-1.7b__mistral7b.json"),
          ("Pythia-70M", "Mistral-7B"): mut_z("c4_pythia70m__mistral7b.json")}
    short = {"Gemma2-2B": "G2-2B", "Gemma2-9B": "G2-9B", "Mistral-7B": "M-7B",
             "Qwen3-1.7B": "Q3-1.7B", "Qwen3-8B": "Q3-8B", "Pythia-70M": "Py-70M", "GPT-2": "GPT-2"}
    n = len(models)
    M = np.full((n, n), np.nan)          # displayed value: C4 z where run, else template z
    is_c4 = np.zeros((n, n), bool)
    for (a, b), f in key.items():
        z = mut_z(f)
        if z is None:
            continue
        i, j = models.index(a), models.index(b)
        M[i, j] = M[j, i] = z
        if f.startswith("c4_"):
            is_c4[i, j] = is_c4[j, i] = True
    T = np.full((n, n), np.nan)          # template z kept to annotate verdict flips
    for (a, b), f in key.items():
        if not f.startswith("c4_"):
            z = mut_z(f)
            if z is not None:
                i, j = models.index(a), models.index(b)
                T[i, j] = T[j, i] = z
    for (a, b), cz in c4.items():
        if cz is None:
            continue
        i, j = models.index(a), models.index(b)
        M[i, j] = M[j, i] = cz
        is_c4[i, j] = is_c4[j, i] = True

    # designed at ~physical column size (included at 0.33\textwidth of 5.5in => ~1.82in)
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(1.95, 1.58))
    cmap = plt.cm.RdBu.copy(); cmap.set_bad("#EDEDF2")   # CVD-safe diverging; grey = unmeasured
    norm = TwoSlopeNorm(vmin=-5, vcenter=1.96, vmax=25)
    ax.imshow(np.ma.masked_invalid(np.clip(M, -5, 25)), cmap=cmap, norm=norm)
    ax.set_xticks(range(n)); ax.set_xticklabels([short[m] for m in models],
                                                rotation=50, ha="right", fontsize=6.0)
    ax.set_yticks(range(n)); ax.set_yticklabels([short[m] for m in models], fontsize=6.0)
    ax.tick_params(length=1.5, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "\u2014", ha="center", va="center", color="#999999", fontsize=4.2)
                continue
            if np.isnan(M[i, j]):
                continue
            z = M[i, j]
            dark = norm(np.clip(z, -5, 25)) > 0.72
            if is_c4[i, j]:
                fail = z <= 1.96
                if fail:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="///", ec="#B22222", lw=0.0))
                lab = f"{z:.1f}" + ("\u00d7" if fail else "")
                if fail and not np.isnan(T[i, j]) and T[i, j] > 1.96:
                    # verdict flip (the audited corpus artifact): show both regimes
                    ax.text(j, i - 0.22, f"({T[i, j]:.1f})", ha="center", va="center",
                            fontsize=3.8, fontstyle="italic", color="#444444")
                    ax.text(j, i + 0.2, lab, ha="center", va="center", fontsize=4.4,
                            fontweight="bold", color="#111111")
                else:
                    ax.text(j, i, lab, ha="center", va="center", fontsize=4.6,
                            fontweight="bold", color=("white" if dark else "#111111"))
            else:  # template-corpus-only cell: parentheses = explicit no-C4-verdict mark
                ax.text(j, i, f"({z:.1f})", ha="center", va="center", fontsize=4.2,
                        fontstyle="italic", color=("white" if dark else "#444444"))
    # annotation layer: THE comparisons that are the finding
    ax.set_ylim(6.5, -1.15)   # headroom above the grid for the block label
    ax.add_patch(plt.Rectangle((-0.48, -0.48), 4.96, 4.96, facecolor="none",
                               edgecolor="#0B5394", lw=0.9, zorder=6))
    ax.text(2.0, -0.85, "capable block: 8 cross-family C4-robust cells",
            ha="center", va="center", fontsize=5.0, color="#0B5394", fontweight="bold")
    ax.annotate("corpus artifact:\n21.9 to \u22121.9",
                xy=(2.5, 5.35), xytext=(3.62, 5.5), ha="center", va="center",
                fontsize=4.2, color="#B22222", fontstyle="italic",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#B22222"))
    fig.tight_layout(pad=0.2)
    fig.savefig("paper/figures/fig_universality_matrix.pdf", bbox_inches="tight")
    print("[wrote] paper/figures/fig_universality_matrix.pdf")


# --------------------------------------------- fig 3: language panel
def fig_language():
    # designed at ~physical size (included at 0.50\textwidth of 5.5in => ~2.75in wide)
    fig, axes = plt.subplots(1, 2, figsize=(3.05, 1.28))
    # left: within-model cross-lingual z
    langs = ["en\u2013zh", "en\u2013ja", "en\u2013ko"]
    g2b = [mut_z(f"xling_gemma2-2b_en_{l}.json") for l in ["zh", "ja", "ko"]]
    g9b = [mut_z(f"xling_gemma2-9b_en_{l}.json") for l in ["zh", "ja", "ko"]]
    x = np.arange(3); w = 0.36
    axes[0].bar(x - w / 2, g2b, w, color=OI["sky"], label="Gemma2-2B")
    axes[0].bar(x + w / 2, g9b, w, color=OI["blue"], label="Gemma2-9B")
    axes[0].axhline(1.96, ls="--", lw=0.7, color="#777777")
    axes[0].text(2.46, 2.3, "$z{=}1.96$", fontsize=4.2, color="#777777")
    axes[0].set_xticks(x); axes[0].set_xticklabels(langs, fontsize=7)
    axes[0].tick_params(axis="y", labelsize=6.5, length=1.5, pad=1)
    axes[0].tick_params(axis="x", length=1.5, pad=1)
    axes[0].set_ylabel("mutual-NN $z$", fontsize=7)
    axes[0].set_title("Within-model\ncross-lingual sharing", fontsize=7.4)
    axes[0].set_ylim(-2.5, 9.0)
    axes[0].legend(fontsize=5.0, frameon=False, loc="upper left", borderaxespad=0.1, handlelength=1.0, labelspacing=0.25)
    # right: cross-model per language
    cells = [("G2-2B$\\leftrightarrow$G2-9B", mut_z("gemma2-2b__gemma2-9b.json"), mut_z("zh_gemma2-2b__gemma2-9b.json")),
             ("G2-2B$\\leftrightarrow$Mistral", mut_z("gemma2-2b__mistral7b.json"), mut_z("zh_gemma2-2b__mistral7b.json"))]
    x2 = np.arange(len(cells))
    en_v = [c[1] for c in cells]; zh_v = [c[2] for c in cells]
    axes[1].bar(x2 - w / 2, en_v, w, color=OI["blue"], label="English")
    bars = axes[1].bar(x2 + w / 2, [v if v is not None else 0 for v in zh_v], w,
                       color=OI["orange"], label="Chinese")
    for xi, v in zip(x2 + w / 2, zh_v):
        if v is not None and v > 50:
            axes[1].text(xi, v * 1.25, f"$z{{=}}{v:.0f}$", ha="center",
                         fontsize=4.8, color="#222222")
    # symlog: shows z=101 unclipped while keeping the below-null bar readable
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_ylim(-8, 400)
    axes[1].set_yticks([-5, 0, 5, 10, 100])
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].minorticks_off()
    axes[1].axhline(1.96, ls="--", lw=0.7, color="#777777")
    axes[1].axhline(0, lw=0.5, color="#333333")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(["within-family\n2B$\\leftrightarrow$9B", "cross-family\n2B$\\leftrightarrow$Mistral"], fontsize=6.2)
    axes[1].tick_params(axis="y", labelsize=6.5, length=1.5, pad=1)
    axes[1].tick_params(axis="x", length=1.5, pad=1)
    axes[1].set_title("Cross-model sharing is\nlanguage-conditional", fontsize=7.4)
    axes[1].legend(fontsize=5.0, frameon=False, loc="upper right", borderaxespad=0.1, handlelength=1.0, labelspacing=0.25)
    # annotation layer: THE comparisons that are the finding
    axes[0].annotate("en\u2013ko: null in both", xy=(2.05, 1.2), xytext=(2.1, 5.6),
                     ha="center", va="center", fontsize=4.8, color="#555555",
                     fontstyle="italic",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#777777"))
    axes[1].annotate("cross-family collapses\non Chinese (below null)",
                     xy=(1.0 + w / 2, -3.0), xytext=(0.97, 12.0), ha="center",
                     va="center", fontsize=4.8, color="#B26A00", fontstyle="italic",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#B26A00"))
    fig.tight_layout(pad=0.25)
    fig.savefig("paper/figures/fig_language_panel.pdf", bbox_inches="tight")
    print("[wrote] paper/figures/fig_language_panel.pdf")


if __name__ == "__main__":
    # fig1_arc() intentionally NOT run: fig1_arc.pdf is now exported from
    # paper/figures/fig1_arc.drawio (house style, commit 09a4873); regenerating
    # here would overwrite it. Call fig1_arc() manually only as a fallback.
    fig_matrix(); fig_language()
