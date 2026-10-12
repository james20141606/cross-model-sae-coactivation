#!/usr/bin/env python3
"""fig:language_panel -- within-model cross-lingual sharing (left) and cross-model sharing by language (right) (D class).

Receipts (read-only; `mutual_z`, cross-checked against observed / null_mutual_nn):
  left   results/cross_model_sae/xling_gemma2-{2b,9b}_en_{zh,ja,ko}.json
  right  results/cross_model_sae/{gemma2-2b__gemma2-9b, gemma2-2b__mistral7b}.json  (English)
         results/cross_model_sae/zh_{gemma2-2b__gemma2-9b, gemma2-2b__mistral7b}.json (Chinese)
Included at 0.62\textwidth = 3.41 in. Output: paper/figures/fig_language_panel.{svg,pdf}; gate: audit --width-in 3.41 -> 0.
"""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figstyle"))
import figstyle as fs     # noqa: E402
fs.use()
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402
plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})

WIDTH_IN = 0.62 * fs.TEXTWIDTH_IN
RES = ROOT / "results/cross_model_sae"
SIG = 1.96; F = 7


def z_of(name):
    d = json.loads((RES / f"{name}.json").read_text())
    assert not d.get("smoke"), name
    o, n = d["observed"], d["null_mutual_nn"]
    assert abs((o["n_mutual_nn"] - n["mean"]) / n["sd"] - d["mutual_z"]) < 1e-6, name
    return d["mutual_z"]


def key(ax, entries, x, y, dy=0.085):
    """Hand-drawn key: swatch + label in axes fraction (no legend machinery)."""
    for k, (col, lab) in enumerate(entries):
        ax.add_patch(Rectangle((x, y - k * dy - 0.018), 0.05, 0.036, transform=ax.transAxes, facecolor=col, edgecolor="none", clip_on=False))
        ax.text(x + 0.07, y - k * dy, lab, transform=ax.transAxes, fontsize=F, color=fs.INK, va="center", ha="left")


fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 1.75), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.5))
fig.subplots_adjust(left=0.15, right=0.975, top=0.86, bottom=0.24)
langs = ["zh", "ja", "ko"]
g2 = [z_of(f"xling_gemma2-2b_en_{l}") for l in langs]
g9 = [z_of(f"xling_gemma2-9b_en_{l}") for l in langs]
x = np.arange(3); w = 0.36
ax = axes[0]
ax.bar(x - w / 2, g2, w, color=fs.SEQ_TEAL[3])
ax.bar(x + w / 2, g9, w, color=fs.TEAL)
ax.axhline(SIG, ls=(0, (3, 2)), lw=0.7, color=fs.INK3)
ax.text(2.55, SIG + 0.35, "$z=1.96$", fontsize=F, color=fs.INK3, ha="right", va="bottom")
fs.zero_line(ax)
ax.set_xticks(x); ax.set_xticklabels([f"en–{l}" for l in langs], fontsize=F)
ax.tick_params(axis="y", labelsize=F)
ax.set_ylabel("mutual-NN $z$", fontsize=F)
ax.set_ylim(-2.8, 10.5)
ax.set_title("within-model, cross-lingual", fontsize=7.5)
key(ax, [(fs.SEQ_TEAL[3], "Gemma2-2B"), (fs.TEAL, "Gemma2-9B")], 0.04, 0.92)
cells = [("gemma2-2b__gemma2-9b", "within-family\n2B–9B"), ("gemma2-2b__mistral7b", "cross-family\n2B–Mistral")]
en = [z_of(c) for c, _ in cells]; zh = [z_of("zh_" + c) for c, _ in cells]
ax = axes[1]
x2 = np.arange(2)
ax.bar(x2 - w / 2, en, w, color=fs.INDIGO)
ax.bar(x2 + w / 2, zh, w, color=fs.AMBER)
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(-8, 900); ax.set_yticks([-5, 0, 5, 10, 100]); ax.minorticks_off()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.axhline(SIG, ls=(0, (3, 2)), lw=0.7, color=fs.INK3)
fs.zero_line(ax)
ax.set_xticks(x2); ax.set_xticklabels([lab for _, lab in cells], fontsize=F)
ax.tick_params(axis="y", labelsize=F)
ax.set_ylabel("mutual-NN $z$ (symlog)", fontsize=F)
ax.set_title("cross-model by language", fontsize=7.5)
key(ax, [(fs.INDIGO, "English"), (fs.AMBER, "Chinese")], 0.55, 0.92)
fs.direct_label(ax, x2[0] + w / 2, zh[0], f"$z={zh[0]:.0f}$", dy=2, size=F)
ax.text(1.0, 40, "below null\non Chinese", fontsize=F, color=fs.INK2, style="italic", ha="center", va="center")
fs.save(fig, ROOT / "paper/figures/fig_language_panel.svg")
print("left  Gemma2-2B", ["%.2f" % v for v in g2], " Gemma2-9B", ["%.2f" % v for v in g9])
print("right English", ["%.2f" % v for v in en], " Chinese", ["%.2f" % v for v in zh])
print("wrote paper/figures/fig_language_panel.{svg,pdf}")
