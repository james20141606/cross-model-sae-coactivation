#!/usr/bin/env python3
"""R15b house-style restyle of the two appendix data figures (reads receipts only).

fig_rosetta_sublinear.pdf : raw-metric scaling curves (app:scaling supplement, Fig 5)
fig_cross_model_sae.pdf   : loose-criterion confound scatter (app:greedy_scatter, Fig 6)

Style: <internal-notes> rule 4 — Avenir Next, colorblind-safe
palette, designed at ~physical included size so fonts are ~1:1 in print.
Data: results/rosetta_check/rosetta_sublinear_v0.json + results/cross_model_sae/SUMMARY.json
(read-only; numbers unchanged).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
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
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00"}


def rosetta():  # Fig 5 — included at 0.62\textwidth of 5.5in => ~3.4in wide
    d = json.load(open("results/rosetta_check/rosetta_sublinear_v0.json"))
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    fam_style = [("Qwen3", OI["blue"], "o"), ("Gemma3", OI["orange"], "s"),
                 ("Pythia", OI["green"], "^")]
    for fam, col, mk in fam_style:
        pts = sorted([(r["params"], r["r"]) for r in rows if r["family"] == fam])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, mk + "-", color=col, label=fam, ms=3.6, lw=1.1)
        for x, y in pts:
            ax.annotate(f"{x:.0f}B" if x >= 1 else f"{x}B", (x, y),
                        textcoords="offset points", xytext=(3, 3), fontsize=5, color=col)
    for label, col in [("Gemma3", OI["orange"]), ("Qwen3_excl14B", OI["blue"]),
                       ("Pythia", OI["green"])]:
        f = d["fits"][label]
        xs = np.array(sorted(f["params"])); b, c = f["exponent_b"], f["coef_c"]
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, c * xx ** b, "--", color=col, lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (B, log scale)", fontsize=7)
    ax.set_ylabel("cross-lingual probe-attribution mean $r$", fontsize=7)
    ax.tick_params(labelsize=6.5, length=2)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_title("Architecture gates scale (raw, depth-confounded metric)", fontsize=7.5)
    ax.legend(fontsize=6, loc="center left", frameon=False, handlelength=1.4)
    ax.grid(alpha=0.2, lw=0.4)
    g = [r["r"] for r in rows if r["model"] == "google/gemma-3-4b-it"][0]
    q = [r["r"] for r in rows if r["model"] == "Qwen/Qwen3-14B"][0]
    ax.annotate(f"{g/q:.0f}x gap\n@1/3.5 params", (4, 0.572),
                textcoords="offset points", xytext=(8, -20), fontsize=5.5,
                color=OI["orange"], fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color=OI["orange"], lw=0.7))
    fig.tight_layout(pad=0.3)
    fig.savefig("paper/figures/fig_rosetta_sublinear.pdf", bbox_inches="tight")
    print("[wrote] paper/figures/fig_rosetta_sublinear.pdf")


def confound_scatter():  # Fig 6 — included at 0.78\textwidth => ~4.3in wide
    rows = json.load(open("results/cross_model_sae/SUMMARY.json"))
    COL = {"within": OI["blue"], "cross-modern": OI["green"], "cross-GPT2": OI["verm"]}
    fig, ax = plt.subplots(1, 2, figsize=(4.4, 1.8))
    cats = ["within", "cross-modern", "cross-GPT2"]
    for xi, cat in enumerate(cats):
        zs = [r["mutual_nn_z"] for r in rows if r["category"] == cat]
        xs = [xi + (i - len(zs) / 2) * 0.06 for i in range(len(zs))]
        ax[0].scatter(xs, zs, c=COL[cat], s=14, label=cat)
    ax[0].axhline(1.96, color="grey", lw=0.5, ls="--")
    ax[0].set_xticks(range(len(cats)))
    ax[0].set_xticklabels(cats, fontsize=6)
    ax[0].tick_params(axis="y", labelsize=6, length=2)
    ax[0].set_ylabel("mutual-NN $z$ (excess / null SD)", fontsize=6.5)
    ax[0].set_title("reciprocal sharing: GPT-2 is the outlier", fontsize=7)
    ax[0].legend(fontsize=5, loc="upper right", frameon=False, handlelength=1.0)
    for r in rows:
        ax[1].scatter(r["greedy_excess"], r["mutual_nn_z"], c=COL[r["category"]], s=14)
    ax[1].axhline(1.96, color="grey", lw=0.5, ls="--")
    ax[1].set_xlabel("greedy-max excess (confounded)", fontsize=6.5)
    ax[1].set_ylabel("mutual-NN $z$ (valid)", fontsize=6.5)
    ax[1].tick_params(labelsize=6, length=2)
    ax[1].set_title("loose metric saturates; strict discriminates", fontsize=7)
    # annotation layer: THE comparison — saturation at zero sharing
    ax[1].annotate("saturates near +0.8\neven at zero sharing", xy=(0.79, -1.3),
                   xytext=(0.42, 8.5), ha="center", fontsize=5.5, color=OI["verm"],
                   fontstyle="italic",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color=OI["verm"]))
    fig.tight_layout(pad=0.3)
    fig.savefig("paper/figures/fig_cross_model_sae.pdf", bbox_inches="tight")
    print("[wrote] paper/figures/fig_cross_model_sae.pdf")


if __name__ == "__main__":
    rosetta()
    confound_scatter()
