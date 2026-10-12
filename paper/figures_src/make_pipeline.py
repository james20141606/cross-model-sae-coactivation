"""
fig_pipeline.pdf — schematic of the end-to-end attribution pipeline.
This is a hand-drawn block diagram that does not depend on any results.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _common import setup_style, save_fig


def plot():
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.axis("off")

    blocks = [
        ("Prompt\n(en/zh/ja/ko)",            "#cfe2f3"),
        ("ReplacementModel\n.from_pretrained", "#fce5cd"),
        ("attribute()\n+ prune_graph()",      "#fff2cc"),
        ("AdaptedGraph\n(nodes, edges)",      "#d9ead3"),
        ("Jaccard /\nstructure stats",        "#e6cce6"),
        ("Figures &\nTables",                 "#f4cccc"),
    ]
    n = len(blocks)
    box_w = 1.0 / n - 0.02
    for i, (txt, c) in enumerate(blocks):
        x = i * (1.0 / n) + 0.01
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.25), box_w, 0.5,
            boxstyle="round,pad=0.02", linewidth=0.6,
            facecolor=c, edgecolor="#444",
        ))
        ax.text(x + box_w / 2, 0.5, txt, ha="center", va="center", fontsize=8)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + box_w + 0.018, 0.5),
                xytext=(x + box_w + 0.001, 0.5),
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.8),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig(fig, "fig_pipeline")


if __name__ == "__main__":
    plot()
