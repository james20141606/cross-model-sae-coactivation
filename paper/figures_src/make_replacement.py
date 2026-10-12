"""
fig_replacement.pdf — replacement vs completeness scores across
all first-pass models. Higher = more interpretable computation.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    setup_style, COLORS, RESULTS_DIR,
    safe_load_json, is_placeholder_run, annotate_placeholder, save_fig,
)


def synth_replacement(seed: int = 6):
    rng = np.random.default_rng(seed)
    # Replacement score expected ~0.45-0.65; completeness ~0.6-0.85
    out = {
        "qwen3-0.6b":  (0.42 + rng.normal(0, 0.02), 0.60 + rng.normal(0, 0.02)),
        "qwen3-1.7b":  (0.46 + rng.normal(0, 0.02), 0.65 + rng.normal(0, 0.02)),
        "qwen3-4b":    (0.51 + rng.normal(0, 0.02), 0.71 + rng.normal(0, 0.02)),
        "qwen3-8b":    (0.55 + rng.normal(0, 0.02), 0.75 + rng.normal(0, 0.02)),
        "gemma3-4b":   (0.49 + rng.normal(0, 0.02), 0.68 + rng.normal(0, 0.02)),
        "gemma3-12b":  (0.53 + rng.normal(0, 0.02), 0.74 + rng.normal(0, 0.02)),
    }
    return out


def load_real():
    return None


def plot(data, placeholder):
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for m, (rep, comp) in data.items():
        family_color = COLORS["qwen"] if "qwen" in m else COLORS["gemma"]
        ax.scatter(rep, comp, s=45, color=family_color, edgecolors="white",
                   linewidth=0.5, zorder=3)
        ax.annotate(m, (rep, comp), xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5)

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.5, alpha=0.4, label="$y=x$")
    ax.set_xlim(0.30, 0.70); ax.set_ylim(0.50, 0.90)
    ax.set_xlabel("replacement score (token$\\to$logit through features)")
    ax.set_ylabel("completeness score (weighted non-error inputs)")
    ax.set_title("Interpretability scores per \\citet{ameisen2025attribution}")

    # legend by family
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["qwen"],
               markersize=7, label="Qwen3"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gemma"],
               markersize=7, label="Gemma3"),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="lower right")
    ax.grid(True, alpha=0.2, linewidth=0.4)

    if placeholder:
        annotate_placeholder(ax)

    fig.tight_layout()
    save_fig(fig, "fig_replacement")


if __name__ == "__main__":
    real = load_real()
    if real is None or is_placeholder_run():
        print("  (synthetic data)")
        data = synth_replacement()
        placeholder = True
    else:
        data = real
        placeholder = False
    plot(data, placeholder)
