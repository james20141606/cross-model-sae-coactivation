"""
fig_layer_distribution.pdf — heatmap of feature-node density across
relative layers, per (model, prompt-family).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    setup_style, RESULTS_DIR,
    safe_load_json, is_placeholder_run, annotate_placeholder, save_fig,
)


def synth_layer_dist(seed: int = 5):
    rng = np.random.default_rng(seed)
    n_bins = 12       # relative-layer bins (0..1 normalised)
    out = {}
    models = ["qwen3-0.6b", "qwen3-4b", "qwen3-8b", "gemma3-4b", "gemma3-12b"]
    for m in models:
        # Mid-layer peak for feature firing (echoes universal-semantic-core hypothesis)
        peak = 0.55 if "qwen" in m else 0.50
        widths = 0.20 if "qwen" in m else 0.22
        x = np.linspace(0, 1, n_bins) + 0.5 / n_bins
        density = np.exp(-((x - peak) ** 2) / (2 * widths ** 2))
        density += rng.normal(0, 0.05, n_bins)
        density = np.clip(density, 0, None)
        density /= density.sum()
        out[m] = density.tolist()
    return out


def load_real():
    return None


def plot(data, placeholder):
    setup_style()
    models = list(data.keys())
    matrix = np.array([data[m] for m in models])

    fig, ax = plt.subplots(figsize=(7, 2.6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    n_bins = matrix.shape[1]
    ax.set_xticks(np.linspace(0, n_bins - 1, 5))
    ax.set_xticklabels([f"{x:.2f}" for x in np.linspace(0, 1, 5)])
    ax.set_xlabel("relative layer index $\\ell / L$")
    ax.set_ylabel("model")
    ax.set_title("Feature-node density across relative depth (antonym prompts)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("normalised density", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if placeholder:
        annotate_placeholder(ax)

    fig.tight_layout()
    save_fig(fig, "fig_layer_distribution")


if __name__ == "__main__":
    real = load_real()
    if real is None or is_placeholder_run():
        print("  (synthetic data)")
        data = synth_layer_dist()
        placeholder = True
    else:
        data = real
        placeholder = False
    plot(data, placeholder)
