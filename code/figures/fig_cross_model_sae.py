#!/usr/bin/env python3
"""fig:greedy_scatter -- the loose matching criterion is confounded; the strict one discriminates (D class).

Receipt (read-only): results/cross_model_sae/SUMMARY.json (templated concept corpus; one row per pair,
fields greedy_excess, mutual_nn_z, category in {within, cross-modern, cross-GPT2}).
Included at 0.78\textwidth = 4.29 in. Output: paper/figures/fig_cross_model_sae.{svg,pdf}; gate: audit --width-in 4.29 -> 0.
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figstyle"))
import figstyle as fs     # noqa: E402
fs.use()
import matplotlib.pyplot as plt   # noqa: E402
plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})

WIDTH_IN = 0.78 * fs.TEXTWIDTH_IN
rows = json.loads((ROOT / "results/cross_model_sae/SUMMARY.json").read_text())
CATS = ["within", "cross-modern", "cross-GPT2"]
COL = {"within": fs.INDIGO, "cross-modern": fs.TEAL, "cross-GPT2": fs.CORAL}
TICK = {"within": "within-\nfamily", "cross-modern": "cross-family,\nmodern", "cross-GPT2": "GPT-2\npairs"}
SIG = 1.96; F = 7

fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 1.95), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.42))
fig.subplots_adjust(left=0.12, right=0.975, top=0.86, bottom=0.24)
ax = axes[0]
for xi, cat in enumerate(CATS):
    zs = sorted(r["mutual_nn_z"] for r in rows if r["category"] == cat)
    xs = [xi + (k - (len(zs) - 1) / 2) * 0.09 for k in range(len(zs))]
    ax.scatter(xs, zs, s=16, color=COL[cat], zorder=3)
ax.axhline(SIG, ls=(0, (3, 2)), lw=0.7, color=fs.INK3)
ax.text(-0.55, SIG + 0.9, "$z=1.96$", fontsize=F, color=fs.INK3, ha="left", va="bottom")
fs.zero_line(ax)
ax.set_xticks(range(3)); ax.set_xticklabels([TICK[c] for c in CATS], fontsize=F)
ax.tick_params(axis="y", labelsize=F)
ax.set_ylabel("mutual-NN $z$ (strict)", fontsize=F)
ax.set_title("reciprocal sharing by pair class", fontsize=8)
ax.set_xlim(-0.6, 2.6)
ax = axes[1]
for r in rows:
    ax.scatter(r["greedy_excess"], r["mutual_nn_z"], s=16, color=COL[r["category"]], zorder=3)
ax.axhline(SIG, ls=(0, (3, 2)), lw=0.7, color=fs.INK3)
fs.zero_line(ax)
ax.set_xlabel("greedy-max excess (loose criterion)", fontsize=F)
ax.set_ylabel("mutual-NN $z$ (strict)", fontsize=F)
ax.tick_params(axis="both", labelsize=F)
ax.set_title("loose saturates, strict discriminates", fontsize=8)
gpt2m = next(r for r in rows if r["pair"] == "gpt2__mistral7b")
ax.annotate("saturates near $+0.8$ with\nzero reciprocal sharing", xy=(gpt2m["greedy_excess"], gpt2m["mutual_nn_z"] + 0.6),
            xytext=(0.0, 12.0), ha="left", va="center", fontsize=F, color=fs.CORAL, style="italic",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.CORAL, shrinkB=3))
ax.set_xlim(-0.02, 0.92); ax.set_ylim(-3.5, 24)
fs.save(fig, ROOT / "paper/figures/fig_cross_model_sae.svg")
for cat in CATS:
    print(cat, ["%.2f" % r["mutual_nn_z"] for r in rows if r["category"] == cat])
print("wrote paper/figures/fig_cross_model_sae.{svg,pdf}")
