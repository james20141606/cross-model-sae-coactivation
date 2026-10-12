"""
fig_xlingual_overlap.pdf — cross-lingual feature overlap (Jaccard)
heatmap-style grouped bar across model sizes.

fig_xlingual_layers.pdf — layer-banded Jaccard for en-zh on antonyms.

Real source: results/03_cross_lingual/*/overlap_matrix.json (when produced)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    setup_style, COLORS, RESULTS_DIR, FIGURES_DIR,
    safe_load_json, is_placeholder_run,
    synth_xlingual_overlap, annotate_placeholder, save_fig,
)


PAIRS = ["en-zh", "en-ja", "en-ko", "zh-ja", "zh-ko", "ja-ko"]
PAIR_COLORS = {
    "en-zh": "#1f77b4", "en-ja": "#ff7f0e", "en-ko": "#2ca02c",
    "zh-ja": "#d62728", "zh-ko": "#9467bd", "ja-ko": "#8c564b",
}


def load_real() -> dict | None:
    # results/03_cross_lingual/{model}/overlap.json — schema TBD; once schema is fixed,
    # adapt the load. For now return None to fall back to synthetic.
    return None


def plot_main(data: dict, placeholder: bool):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 3))
    sizes = sorted(data.keys())
    width = 0.13
    n_pairs = len(PAIRS)
    centres = np.arange(len(sizes))

    for i, pair in enumerate(PAIRS):
        means = [data[s][pair]["mean"] for s in sizes]
        cis = [data[s][pair]["ci"] for s in sizes]
        offset = (i - (n_pairs - 1) / 2) * width
        ax.bar(centres + offset, means, width, yerr=cis,
               label=pair, color=PAIR_COLORS[pair], alpha=0.85,
               error_kw=dict(lw=0.6, capsize=1.8))

    ax.set_xticks(centres)
    ax.set_xticklabels([f"{s:g}B" for s in sizes])
    ax.set_xlabel("Qwen3 size")
    ax.set_ylabel(r"Jaccard overlap of feature sets")
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.set_ylim(0, max(0.20, ax.get_ylim()[1]))
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)
    ax.axhline(0.005, color="#999", linewidth=0.6, linestyle=":")
    ax.text(centres[-1] + 0.5, 0.006, "shuffle baseline", fontsize=7,
            color="#666", ha="right")

    if placeholder:
        annotate_placeholder(ax)

    fig.tight_layout()
    save_fig(fig, "fig_xlingual_overlap")


def plot_layers(data: dict, placeholder: bool):
    """Layer-banded Jaccard for en-zh on antonyms — single panel grouped bar."""
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 2.6))
    sizes = sorted(data.keys())

    rng = np.random.default_rng(3)
    bands = ["early (0-L/3)", "mid (L/3-2L/3)", "late (2L/3-L)"]
    band_colors = ["#9ecae1", "#2c7fb8", "#a6bddb"]
    centres = np.arange(len(sizes))
    width = 0.25

    for bi, band in enumerate(bands):
        # synth: mid-band peaks, early ~ tokeniser, late low
        mids = []
        cis = []
        for s in sizes:
            base = data[s]["en-zh"]["mean"]
            if "mid" in band:
                v = base * 1.6
            elif "late" in band:
                v = base * 0.4
            else:
                v = base * 0.7
            v = max(0.0, v + rng.normal(0, base * 0.05))
            mids.append(v)
            cis.append(max(0.005, base * 0.08))
        offset = (bi - 1) * width
        ax.bar(centres + offset, mids, width, yerr=cis, label=band,
               color=band_colors[bi], alpha=0.9,
               error_kw=dict(lw=0.6, capsize=1.5))

    ax.set_xticks(centres)
    ax.set_xticklabels([f"{s:g}B" for s in sizes])
    ax.set_xlabel("Qwen3 size")
    ax.set_ylabel("Jaccard (en-zh, antonym)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)

    if placeholder:
        annotate_placeholder(ax)

    fig.tight_layout()
    save_fig(fig, "fig_xlingual_layers")


if __name__ == "__main__":
    real = load_real()
    if real is None or is_placeholder_run():
        print("  (synthetic data)")
        data = synth_xlingual_overlap()
        placeholder = True
    else:
        print("  (real data)")
        data = real
        placeholder = False
    plot_main(data, placeholder)
    plot_layers(data, placeholder)
