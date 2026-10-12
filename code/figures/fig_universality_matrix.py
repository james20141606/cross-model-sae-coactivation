#!/usr/bin/env python3
"""fig:cross_model_sae -- model x model mutual-NN sharing (z over the permutation null) (D class).

Receipts (read-only; `mutual_z`, cross-checked against observed / null_mutual_nn):
  C4 natural-text verdict (bold)                 results/cross_model_sae/c4_<a>__<b>.json
  template corpus only (italic, parenthesised)   results/cross_model_sae/<a>__<b>.json
The 6 x 6 capable block holds the nine cross-family C4-robust cells (+ two within-family C4 cells);
Pythia-70M<->Mistral-7B is the corpus artefact (template 21.9, C4 -1.9, drawn as both); GPT-2 cells are
template-only (its C4 cells are underpowered, tab:gpt2_c4). Grey = unmeasured.
Included at 0.58\textwidth = 3.19 in (was a 0.36-width minipage: 8 x 8 cells cannot carry 7 pt labels there).
Output: paper/figures/fig_universality_matrix.{svg,pdf}; gate: figstyle/audit.py --width-in 3.19 -> 0 errors.
"""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figstyle"))
import figstyle as fs     # noqa: E402
fs.use()
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402
plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})

WIDTH_IN = 0.58 * fs.TEXTWIDTH_IN
RES = ROOT / "results/cross_model_sae"
MODELS = ["Gemma2-2B", "Gemma2-9B", "Mistral-7B", "Qwen3-1.7B", "Qwen3-8B", "Llama-3.2-1B", "Pythia-70M", "GPT-2"]
SHORT = {"Gemma2-2B": "G2-2B", "Gemma2-9B": "G2-9B", "Mistral-7B": "M-7B", "Qwen3-1.7B": "Q3-1.7B",
         "Qwen3-8B": "Q3-8B", "Llama-3.2-1B": "L3-1B", "Pythia-70M": "Py-70M", "GPT-2": "GPT-2"}
TEMPLATE = {("Gemma2-2B", "Gemma2-9B"): "gemma2-2b__gemma2-9b", ("Gemma2-2B", "Mistral-7B"): "gemma2-2b__mistral7b",
            ("Gemma2-9B", "Mistral-7B"): "gemma2-9b__mistral7b", ("Qwen3-1.7B", "Gemma2-2B"): "qwen3-1.7b__gemma2-2b",
            ("Qwen3-1.7B", "Mistral-7B"): "qwen3-1.7b__mistral7b", ("Pythia-70M", "Mistral-7B"): "pythia70m__mistral7b",
            ("Pythia-70M", "Gemma2-9B"): "pythia70m__gemma2-9b", ("Gemma2-2B", "GPT-2"): "gemma2-2b__gpt2",
            ("Gemma2-9B", "GPT-2"): "gemma2-9b__gpt2", ("Pythia-70M", "GPT-2"): "pythia70m__gpt2",
            ("Mistral-7B", "GPT-2"): "gpt2__mistral7b"}
C4 = {("Gemma2-2B", "Gemma2-9B"): "c4_gemma2-2b__gemma2-9b", ("Qwen3-8B", "Gemma2-9B"): "c4_qwen3-8b__gemma2-9b",
      ("Qwen3-8B", "Gemma2-2B"): "c4_qwen3-8b__gemma2-2b", ("Qwen3-1.7B", "Gemma2-9B"): "c4_qwen3-1.7b__gemma2-9b",
      ("Qwen3-8B", "Mistral-7B"): "c4_qwen3-8b__mistral7b", ("Qwen3-8B", "Qwen3-1.7B"): "c4_qwen3-8b__qwen3-1.7b",
      ("Gemma2-9B", "Mistral-7B"): "c4_gemma2-9b__mistral7b", ("Gemma2-2B", "Mistral-7B"): "c4_gemma2-2b__mistral7b",
      ("Qwen3-1.7B", "Gemma2-2B"): "c4_qwen3-1.7b__gemma2-2b", ("Qwen3-1.7B", "Mistral-7B"): "c4_qwen3-1.7b__mistral7b",
      ("Pythia-70M", "Mistral-7B"): "c4_pythia70m__mistral7b", ("Llama-3.2-1B", "Mistral-7B"): "c4_llama3.2-1b__mistral7b"}
SIG = 1.96


def z_of(name):
    d = json.loads((RES / f"{name}.json").read_text())
    assert not d.get("smoke"), name
    o, n = d["observed"], d["null_mutual_nn"]
    assert abs((o["n_mutual_nn"] - n["mean"]) / n["sd"] - d["mutual_z"]) < 1e-6, name
    return d["mutual_z"]


n = len(MODELS)
Z = np.full((n, n), np.nan); T = np.full((n, n), np.nan); is_c4 = np.zeros((n, n), bool)
for (a, b), f in TEMPLATE.items():
    i, j = MODELS.index(a), MODELS.index(b); T[i, j] = T[j, i] = Z[i, j] = Z[j, i] = z_of(f)
for (a, b), f in C4.items():
    i, j = MODELS.index(a), MODELS.index(b); Z[i, j] = Z[j, i] = z_of(f); is_c4[i, j] = is_c4[j, i] = True

fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.05))
fig.subplots_adjust(left=0.16, right=0.985, top=0.93, bottom=0.235)
ax.grid(False)
cmap = LinearSegmentedColormap.from_list("teal", [fs.SEQ_TEAL[0], fs.SEQ_TEAL[5], fs.SEQ_TEAL[7]])
norm = Normalize(vmin=0, vmax=45)
F = 7
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        z = Z[i, j]
        if np.isnan(z):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=fs.GRID, edgecolor=fs.SURF, lw=0.6))
            continue
        fail = z <= SIG
        face = fs.tint_hex(fs.CORAL, 0.18) if fail else cmap(norm(min(z, 45)))
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=face, edgecolor=fs.SURF, lw=0.6,
                               alpha=1.0 if is_c4[i, j] else 0.55))
        dark = (not fail) and norm(min(z, 45)) > 0.55 and is_c4[i, j]
        col = fs.SURF if dark else fs.INK
        if is_c4[i, j]:
            if fail and not np.isnan(T[i, j]) and T[i, j] > SIG:   # the corpus artefact: both regimes in one cell
                ax.text(j, i - 0.22, f"({T[i, j]:.1f})", ha="center", va="center", fontsize=F, style="italic", color=fs.INK2)
                ax.text(j, i + 0.24, f"{z:.1f}×", ha="center", va="center", fontsize=F, fontweight="bold", color=fs.CORAL)
            else:
                ax.text(j, i, f"{z:.1f}" + ("×" if fail else ""), ha="center", va="center", fontsize=F,
                        fontweight="bold", color=fs.CORAL if fail else col)
        else:
            ax.text(j, i, f"({z:.1f})", ha="center", va="center", fontsize=F, style="italic",
                    color=fs.CORAL if fail else fs.INK2)
ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5); ax.set_aspect("equal")
ax.set_xticks(range(n)); ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=90, fontsize=F)
ax.set_yticks(range(n)); ax.set_yticklabels([SHORT[m] for m in MODELS], fontsize=F)
ax.tick_params(length=0, pad=2)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.add_patch(Rectangle((-.5, -.5), 6, 6, facecolor="none", edgecolor=fs.INDIGO, lw=1.0, zorder=6))
ax.text(2.5, -0.85, "capable block: nine cross-family C4-robust cells", ha="center", va="center", fontsize=F,
        color=fs.INDIGO, fontweight="bold", clip_on=False)
fig.text(0.5, 0.035, "bold = C4 verdict · (italic) = template only · × = fails null · grey = unmeasured",
         ha="center", va="center", fontsize=F, color=fs.INK2)
fs.save(fig, ROOT / "paper/figures/fig_universality_matrix.svg")
nine = [(a, b, Z[MODELS.index(a), MODELS.index(b)]) for (a, b) in C4 if a.split("-")[0] != b.split("-")[0] and "Pythia" not in a]
print("nine cross-family C4 cells:", len(nine)); [print("   %-12s %-12s z=%.1f" % t) for t in sorted(nine, key=lambda t: -t[2])]
print("wrote paper/figures/fig_universality_matrix.{svg,pdf}")
