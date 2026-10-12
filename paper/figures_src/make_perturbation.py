"""
fig_perturbation.pdf — top-feature ablation effect on target-token logit,
per prompt family, per model.

Real source: results/02_reproduce_anthropic/<model>/perturbation/*.json
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    setup_style, COLORS, RESULTS_DIR,
    safe_load_json, is_placeholder_run, annotate_placeholder, save_fig,
)


def synth_perturb(seed: int = 4):
    rng = np.random.default_rng(seed)
    out = {}
    families = ["antonym", "factual", "arithmetic", "hallucination"]
    models = ["qwen3-4b", "qwen3-8b", "gemma3-4b", "gemma3-12b"]
    for m in models:
        out[m] = {}
        for fam in families:
            # ablation should drop target logit by 1.5-3.0 nats on average
            base = -2.2 if "antonym" in fam or "arithmetic" in fam else -1.6
            vals = base + rng.normal(0, 0.35, 8)
            out[m][fam] = vals.tolist()
    return out


def load_real():
    return None  # populate when results land


def plot(data, placeholder):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 3))
    families = ["antonym", "factual", "arithmetic", "hallucination"]
    models = list(data.keys())
    width = 0.18
    centres = np.arange(len(families))

    for i, m in enumerate(models):
        means = [np.mean(data[m][fam]) for fam in families]
        sems = [np.std(data[m][fam]) / np.sqrt(len(data[m][fam])) for fam in families]
        offset = (i - (len(models) - 1) / 2) * width
        family_color = COLORS["qwen"] if "qwen" in m else COLORS["gemma"]
        alpha = 0.55 + 0.15 * i  # darker shades within family
        ax.bar(centres + offset, means, width, yerr=sems, label=m,
               color=family_color, alpha=min(alpha, 1.0),
               error_kw=dict(lw=0.6, capsize=2))

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(centres)
    ax.set_xticklabels(families)
    ax.set_ylabel(r"$\Delta_{\mathrm{ablate}}$ target-token logit (nats)")
    ax.set_title(r"Top-feature ablation: lower is stronger causal effect")
    ax.legend(ncol=2, frameon=False, loc="lower right")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)

    if placeholder:
        annotate_placeholder(ax)

    fig.tight_layout()
    save_fig(fig, "fig_perturbation")


if __name__ == "__main__":
    real = load_real()
    if real is None or is_placeholder_run():
        print("  (synthetic data)")
        data = synth_perturb()
        placeholder = True
    else:
        data = real
        placeholder = False
    plot(data, placeholder)
