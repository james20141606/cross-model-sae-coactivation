"""
fig_thinking_compare.pdf — paired think vs. no_think structural stats
on Qwen3-4B and Qwen3-8B math prompts.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    setup_style, COLORS, RESULTS_DIR,
    safe_load_json, is_placeholder_run,
    synth_thinking_compare, annotate_placeholder, save_fig,
)


METRICS = [
    ("n_nodes",         r"$|V|$",           "feature node count"),
    ("n_edges",         r"$|E|$",           "edge count"),
    ("layer_span",      "layer span",       "max - min feature layer"),
    ("mid_layer_mass",  "mid-layer mass",   "fraction of influence in middle third"),
]


def load_real() -> dict | None:
    out = {}
    for size, name in [(4.0, "qwen3-4b"), (8.0, "qwen3-8b")]:
        d = safe_load_json(RESULTS_DIR / "04_thinking_mode" / name / "comparison.json")
        if d is None:
            return None
        out[size] = d
    return out


def plot(data: dict, placeholder: bool):
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.7), sharey=False)

    sizes = sorted(data.keys())
    width = 0.35
    centres = np.arange(len(sizes))

    for ax, (key, label, _) in zip(axes, METRICS):
        no_think_means = [np.mean(data[s]["no_think"][key]) for s in sizes]
        no_think_err   = [np.std(data[s]["no_think"][key]) / np.sqrt(len(data[s]["no_think"][key])) for s in sizes]
        think_means    = [np.mean(data[s]["think"][key]) for s in sizes]
        think_err      = [np.std(data[s]["think"][key]) / np.sqrt(len(data[s]["think"][key])) for s in sizes]

        ax.bar(centres - width / 2, no_think_means, width,
               yerr=no_think_err, label="no_think",
               color=COLORS["no_think"], alpha=0.9,
               error_kw=dict(lw=0.6, capsize=2))
        ax.bar(centres + width / 2, think_means, width,
               yerr=think_err, label="think",
               color=COLORS["think"], alpha=0.9,
               error_kw=dict(lw=0.6, capsize=2))
        ax.set_xticks(centres)
        ax.set_xticklabels([f"{s:g}B" for s in sizes])
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)

    axes[0].set_ylabel("metric value")
    axes[0].legend(frameon=False, loc="upper left")

    if placeholder:
        annotate_placeholder(axes[-1])

    fig.tight_layout()
    save_fig(fig, "fig_thinking_compare")


if __name__ == "__main__":
    real = load_real()
    if real is None or is_placeholder_run():
        print("  (synthetic data)")
        data = synth_thinking_compare()
        placeholder = True
    else:
        print("  (real data)")
        data = real
        placeholder = False
    plot(data, placeholder)
