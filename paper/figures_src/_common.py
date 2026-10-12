"""
Shared utilities for figure-generation scripts in this directory.

Each make_*.py script:
  1. Tries to load real results from ../../results/<exp>/.../*.json
  2. Falls back to synthetic placeholder data so the figure compiles
     even before the cluster has produced anything
  3. Writes its PDF to ../figures/<name>.pdf

Run all figures via:
  cd paper/figures_src && python make_all.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---- styling --------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


COLORS = {
    "qwen": "#1f77b4",
    "gemma": "#d62728",
    "antonym": "#2ca02c",
    "factual": "#ff7f0e",
    "arithmetic": "#9467bd",
    "entity": "#8c564b",
    "think": "#e377c2",
    "no_think": "#7f7f7f",
}


# ---- data loading helpers ------------------------------------

def safe_load_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def is_placeholder_run() -> bool:
    """True if no real results exist yet — figures should fall back to synthetic."""
    metadata_globs = list(RESULTS_DIR.rglob("metadata.json"))
    return len(metadata_globs) == 0


# ---- synthetic data generators (placeholder mode) ------------

QWEN_SIZES_B = [0.6, 1.7, 4.0, 8.0]      # first-pass (no 14B)
GEMMA_SIZES_B = [4.0, 12.0]               # first-pass (no 27B)


def synth_scaling_data(seed: int = 0) -> dict:
    """Mock scaling data resembling expected qualitative pattern."""
    rng = np.random.default_rng(seed)
    out = {"qwen": {}, "gemma": {}}
    prompt_types = ["antonym", "factual", "arithmetic", "entity"]

    # Qwen: power-law n_nodes ~ N^0.4
    for size in QWEN_SIZES_B:
        out["qwen"][size] = {}
        for pt in prompt_types:
            base = 30 * (size ** 0.4)
            jitter = rng.normal(0, base * 0.08, 6)
            out["qwen"][size][pt] = {
                "n_nodes": (base + jitter).tolist(),
                "n_edges": ((base * 4) + rng.normal(0, base * 0.4, 6)).tolist(),
            }

    # Gemma: slightly different exponent ~0.45
    for size in GEMMA_SIZES_B:
        out["gemma"][size] = {}
        for pt in prompt_types:
            base = 28 * (size ** 0.45)
            jitter = rng.normal(0, base * 0.10, 6)
            out["gemma"][size][pt] = {
                "n_nodes": (base + jitter).tolist(),
                "n_edges": ((base * 3.8) + rng.normal(0, base * 0.4, 6)).tolist(),
            }

    return out


def synth_xlingual_overlap(seed: int = 1) -> dict:
    """Mock cross-lingual Jaccard matrix per (model, lang_pair)."""
    rng = np.random.default_rng(seed)
    pairs = [("en", "zh"), ("en", "ja"), ("en", "ko"),
             ("zh", "ja"), ("zh", "ko"), ("ja", "ko")]
    out = {}
    for size in QWEN_SIZES_B:
        out[size] = {}
        for pair in pairs:
            # zh-ja typologically closer -> higher overlap
            base = 0.04 + 0.03 * np.log10(size + 1)
            if "zh" in pair and "ja" in pair:
                base += 0.05
            if "ja" in pair and "ko" in pair:
                base += 0.04
            mean = max(0.0, base + rng.normal(0, 0.005))
            ci = max(0.005, abs(rng.normal(0.012, 0.003)))
            out[size]["{}-{}".format(*pair)] = {"mean": mean, "ci": ci}
    return out


def synth_thinking_compare(seed: int = 2) -> dict:
    """Mock think vs no_think structural stats per Qwen size."""
    rng = np.random.default_rng(seed)
    out = {}
    for size in [4.0, 8.0]:
        n_base = 50 * (size ** 0.4)
        out[size] = {
            "no_think": {
                "n_nodes": (n_base + rng.normal(0, 4, 8)).tolist(),
                "n_edges": (n_base * 3.5 + rng.normal(0, 12, 8)).tolist(),
                "layer_span": (rng.normal(14, 1.0, 8)).tolist(),
                "mid_layer_mass": (rng.normal(0.42, 0.03, 8)).tolist(),
            },
            "think": {
                # 30-50% more nodes, longer span, more mid-layer mass
                "n_nodes": (n_base * 1.4 + rng.normal(0, 5, 8)).tolist(),
                "n_edges": (n_base * 5.0 + rng.normal(0, 14, 8)).tolist(),
                "layer_span": (rng.normal(18.5, 1.2, 8)).tolist(),
                "mid_layer_mass": (rng.normal(0.55, 0.03, 8)).tolist(),
            },
        }
    return out


def annotate_placeholder(ax):
    """Stamp a small badge on a synthetic-data figure."""
    ax.text(
        0.99, 0.02, "PLACEHOLDER (synthetic)",
        ha="right", va="bottom", transform=ax.transAxes,
        fontsize=7, color="#cc3333", style="italic", alpha=0.7,
    )


def save_fig(fig, name: str):
    out = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
