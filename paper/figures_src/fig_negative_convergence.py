"""ct paper Fig — convergence rate on three task families across model
scale and threshold. Companion visual for §results.negative tab:negative_2hop.

Data hardcoded from experiments_detail.md (11 sbatch outcomes):
  - antonyms (single-step factual recall)
  - multistep (2-hop factual chain)
  - hallucination (elicit known-false outputs)

Models: Qwen3-0.6B, Qwen3-4B, Qwen3-8B
Thresholds: τ ∈ {1e-2, 1e-3}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


# Convergence rate data from ct/experiments_detail.md jobs:
# 7838239 ct-exp02-antonyms-1p7b (thr=0.01) — but 1.7B not 0.6B; we focus 0.6B
# 7748699 ct-exp02 multistep 0.6B thr=0.01: 0/4
# 7794175 ct-multistep thr=0.001 0.6B: 0/4
# 7796281 ct-multistep-8b: 0/4
# 7749836 ct-exp02-4b OOM (treat as no data)
# 7753409 ct-exp02-4b H200 thr=0.01: 0/4 (3 fail+1 OOM)
# 7861548 ct-halluc-thr001 0.6B: 0/3
# 7760861 ct-hallucination 0.6B thr=0.01: 0/4
# 7861546 ct-antonyms-thr001 0.6B: 2/8
# 7736189 ct-exp02-antonyms 0.6B thr=0.01: 3/8

DATA = {
    # task: {threshold: {model: (converged, total)}}
    "antonyms": {
        "1e-2": {"0.6B": (3, 8), "4B": None, "8B": None},
        "1e-3": {"0.6B": (2, 8), "4B": None, "8B": None},
    },
    "multistep": {
        "1e-2": {"0.6B": (0, 4), "4B": (0, 4), "8B": (0, 4)},
        "1e-3": {"0.6B": (0, 4), "4B": None, "8B": None},
    },
    "hallucination": {
        "1e-2": {"0.6B": (0, 4), "4B": None, "8B": None},
        "1e-3": {"0.6B": (0, 3), "4B": None, "8B": None},
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-figure", required=True)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(DATA.keys())
    models = ["0.6B", "4B", "8B"]
    thresholds = ["1e-2", "1e-3"]
    n_groups = len(tasks) * len(thresholds)

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    x = np.arange(n_groups)
    palette = {"0.6B": "#37a", "4B": "#a37", "8B": "#7a3"}

    labels = []
    for ti, task in enumerate(tasks):
        for tj, thr in enumerate(thresholds):
            labels.append(f"{task}\n$\\tau={thr}$")

    for i, model in enumerate(models):
        ys = []
        for task in tasks:
            for thr in thresholds:
                cell = DATA[task][thr].get(model)
                if cell is None:
                    ys.append(np.nan)
                else:
                    conv, tot = cell
                    ys.append(conv / max(tot, 1))
        offsets = (i - 1) * width
        bars = ax.bar(x + offsets, [y if not np.isnan(y) else 0 for y in ys], width=width,
                      label=f"Qwen3-{model}", color=palette[model], alpha=0.85)
        # Annotate with conv/total
        for j, y in enumerate(ys):
            if np.isnan(y):
                ax.text(x[j] + offsets, 0.02, "n/a", ha="center", va="bottom", fontsize=7,
                        color="#888", rotation=0)
            else:
                task = tasks[j // len(thresholds)]
                thr = thresholds[j % len(thresholds)]
                conv, tot = DATA[task][thr][model]
                txt = f"{conv}/{tot}"
                ax.text(x[j] + offsets, max(y, 0.02) + 0.02, txt, ha="center", va="bottom",
                        fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("convergence rate (graphs converged / total prompts)")
    ax.set_ylim(0, 1.05)
    ax.set_title("circuit_tracer attribution-graph convergence rate by task × threshold × scale")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # Highlight 0% bars
    ax.axhline(0.0, color="#444", linewidth=0.6)

    plt.tight_layout()
    Path(args.out_figure).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_figure, dpi=150, bbox_inches="tight")
    print(f"saved → {args.out_figure}")


if __name__ == "__main__":
    main()
