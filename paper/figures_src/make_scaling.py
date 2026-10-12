"""
fig_scaling_curves.pdf — attribution-graph node count vs. parameter count
per family per prompt type, with power-law fits.

Real source (when available): results/05_cross_model/qwen_scaling/scaling_data.json
                              results/05_cross_model/gemma_scaling/scaling_data.json
Falls back to synthetic data otherwise.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    setup_style, COLORS, FIGURES_DIR, RESULTS_DIR,
    safe_load_json, is_placeholder_run,
    synth_scaling_data, annotate_placeholder, save_fig,
)


def power_law_fit(xs, ys):
    """Return (alpha, beta, R2) for y = alpha * x^beta."""
    log_x, log_y = np.log10(xs), np.log10(ys)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = intercept + slope * log_x
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return 10 ** intercept, slope, r2


def load_real() -> dict | None:
    qwen = safe_load_json(RESULTS_DIR / "05_cross_model" / "qwen_scaling" / "scaling_data.json")
    gemma = safe_load_json(RESULTS_DIR / "05_cross_model" / "gemma_scaling" / "scaling_data.json")
    if qwen is None or gemma is None:
        return None

    out = {"qwen": {}, "gemma": {}}
    for family, data in [("qwen", qwen), ("gemma", gemma)]:
        for entry in data:
            size = entry.get("params_b")
            out[family][size] = {}
            for r in entry.get("prompt_results", []):
                if "error" in r:
                    continue
                pt = r.get("prompt_type", "?")
                out[family][size].setdefault(pt, {"n_nodes": [], "n_edges": []})
                out[family][size][pt]["n_nodes"].append(r["n_nodes"])
                out[family][size][pt]["n_edges"].append(r["n_edges"])
    return out


def plot(data: dict, placeholder: bool):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    prompt_types = ["antonym", "factual", "arithmetic", "entity"]

    for ax_idx, family in enumerate(["qwen", "gemma"]):
        ax = axes[ax_idx]
        sizes = sorted(data[family].keys())
        for pt in prompt_types:
            xs, ys, ys_err = [], [], []
            for size in sizes:
                if pt not in data[family].get(size, {}):
                    continue
                vals = data[family][size][pt]["n_nodes"]
                if not vals:
                    continue
                xs.append(size)
                ys.append(np.mean(vals))
                ys_err.append(np.std(vals) / max(1, np.sqrt(len(vals))))
            if len(xs) < 2:
                continue
            xs = np.array(xs); ys = np.array(ys); ys_err = np.array(ys_err)
            ax.errorbar(xs, ys, yerr=ys_err, fmt="o-", label=pt,
                        color=COLORS.get(pt, "gray"), markersize=4, capsize=2)
            try:
                alpha, beta, r2 = power_law_fit(xs, ys)
                xx = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
                ax.plot(xx, alpha * xx ** beta, "--",
                        color=COLORS.get(pt, "gray"), alpha=0.45, linewidth=0.8)
            except Exception:
                pass

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Params (B)")
        if ax_idx == 0:
            ax.set_ylabel(r"$|V|$ (avg per prompt)")
        ax.set_title({"qwen": "Qwen3", "gemma": "Gemma3"}[family])
        ax.grid(True, which="both", alpha=0.2, linewidth=0.4)

    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    if placeholder:
        annotate_placeholder(axes[1])

    fig.tight_layout()
    save_fig(fig, "fig_scaling_curves")


if __name__ == "__main__":
    real = load_real()
    if real is None or is_placeholder_run():
        print("  (using synthetic placeholder data)")
        data = synth_scaling_data()
        placeholder = True
    else:
        print("  (using real results from ../../results/05_cross_model/)")
        data = real
        placeholder = False
    plot(data, placeholder)
