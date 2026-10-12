"""Figure 1 (2026-10-10 redraw): the control ladder.

One message: every control leaves the cross-model shared excess almost untouched -- the retention
ratio R of eq. (5) (control / same-sized reference arm) sits near R = 1 ("control removes nothing")
and far from R = 0 ("control explains all of it").

Rows and receipts (no number is typed here; every mark is read from these files):
  length deciles, topic k=10   all nine cells   results/ninecell_20260926/cluster_18577359/A_*.json
                                                point.len.R / point.top10.R   (Table 14)
  topic k=200, k=500           three cells      same A_*.json point.top{200,500}.R, 20-partition
                                                calibration (Table 1, revision R1)
  lexical: mutual NN / Lan et al. rule / Procrustes
                               three cells      results/cross_model_sae/{,lanpair_,proclex_}
                                                lextok20k_<cell>.json arms.4096.R_lex (Table 1)
  current word (token level)   three cells      results/cross_model_sae/tokaclean20k_<cell>.json
                                                s2_clean.R_tok_clean (Table 1)
No intervals are drawn: Table 1 prints none, and the document-bootstrap CI was withdrawn
(results/f1_bootstrap_diag_20261004/, 2026-10-05).
The leaky first-design R_tok is omitted, as in Table 1's verdict rows.

Run: python3 code/figures/fig1_control_ladder_20261010.py
"""
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/figstyle"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fig1_rc import apply, C
import compose

apply(8.0)
plt.rcParams.update({"savefig.bbox": None, "xtick.labelsize": 8, "ytick.labelsize": 8})
RES = REPO / "results/cross_model_sae"
NINE_DIR = REPO / "results/ninecell_20260926/cluster_18577359"
OUT = REPO / "paper/figures/fig1_20261010"
OUT.mkdir(parents=True, exist_ok=True)
inputs = []


def load(path):
    inputs.append({"path": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return json.loads(path.read_text())


NINE = ["qwen3-1.7b__gemma2-2b", "qwen3-1.7b__gemma2-9b", "qwen3-1.7b__mistral7b",
        "qwen3-8b__gemma2-2b", "qwen3-8b__gemma2-9b", "qwen3-8b__mistral7b",
        "gemma2-2b__mistral7b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
REPS = ["qwen3-1.7b__gemma2-2b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
NAME = {REPS[0]: "Qwen3-1.7B–Gemma-2-2B", REPS[1]: "Gemma-2-9B–Mistral-7B",
        REPS[2]: "Llama-3.2-1B–Mistral-7B"}
MARK = {REPS[0]: "o", REPS[1]: "s", REPS[2]: "^"}
MS = {REPS[0]: 4.6, REPS[1]: 4.2, REPS[2]: 5.0}

# ------------------------------------------------------------------ data
ROWS = [  # (label, key)
    ("Length deciles", "len"),
    ("Topic strata, k = 10", "top10"),
    ("Topic strata, k = 200", "top200"),
    ("Topic strata, k = 500", "top500"),
    ("Lexical, mutual NN", "lex"),
    ("Lexical, Lan et al. rule", "lan"),
    ("Lexical, Procrustes", "proc"),
    ("Current word (tokens)", "tok"),
]
vals = {k: {} for _, k in ROWS}
for s in NINE:
    a = load(NINE_DIR / f"A_{s}.json")
    for key in ("len", "top10", "top200", "top500"):
        if key not in a["point"]:          # k = 200/500 strata were run on the three cells only
            assert s not in REPS, (s, key)
            continue
        pt = a["point"][key]
        assert pt["calibration"]["n_rep"] == 20, (s, key)
        if key in ("len", "top10") or s in REPS:
            vals[key][s] = pt["R"]
for s in REPS:
    vals["lex"][s] = load(RES / f"lextok20k_{s}.json")["arms"]["4096"]["R_lex"]
    vals["lan"][s] = load(RES / f"lanpair_lextok20k_{s}.json")["arms"]["4096"]["R_lex"]
    vals["proc"][s] = load(RES / f"proclex_lextok20k_{s}.json")["arms"]["4096"]["R_lex"]
    t = load(RES / f"tokaclean20k_{s}.json")
    assert all(t["a3_requirements"].values())
    vals["tok"][s] = t["s2_clean"]["R_tok_clean"]
allR = [v for d in vals.values() for v in d.values()]
repR = [v for k, d in vals.items() for s, v in d.items() if s in REPS]
r_min = min(allR)
s_min, k_min = min(((s, k) for k, d in vals.items() for s in d), key=lambda sk: vals[sk[1]][sk[0]])

# y positions: one blank half-row between the four control families
GAP_AFTER = {"len", "top500", "proc"}
ys, y = {}, 0.0
for _, k in ROWS:
    ys[k] = y
    y += 1.0 + (0.55 if k in GAP_AFTER else 0.0)
y_max = y - 1.0

# ------------------------------------------------------------------ figure
W, H = 5.5, 1.9
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.275, 0.20, 0.705, 0.775])
XL = (-0.03, 1.10)
ax.set_xlim(*XL)
ax.set_ylim(y_max + 0.6, -1.15)

# references: R = 0 (control explains all of the excess) and R = 1 (control removes nothing)
ax.axvline(0.0, color=C.REF, lw=0.6, zorder=1)
ax.axvline(1.0, color=C.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
# family separators: thin grey rules across the data region
for k in GAP_AFTER:
    ax.axhline(ys[k] + 0.775, color=C.GRID, lw=0.6, zorder=0)

off = {REPS[0]: -0.24, REPS[1]: 0.0, REPS[2]: 0.24}
for _, k in ROWS:
    for s, v in vals[k].items():
        if s in REPS:
            ax.plot(v, ys[k] + off[s], MARK[s], ms=MS[s], color=C.ACC, mec="white", mew=0.4, zorder=4)
        else:
            ax.plot(v, ys[k], "o", ms=3.4, mfc="white", mec=C.ACC, mew=0.7, zorder=3)

ax.set_yticks([ys[k] for _, k in ROWS])
ax.set_yticklabels([lab for lab, _ in ROWS])
ax.tick_params(axis="y", length=0, pad=4)
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel("Retention ratio R  (excess left by the control / by a same-sized reference arm)", labelpad=3)

# direct labels on the two references (placed in the empty region, clear of the marks)
ax.text(0.015, ys["len"], "R = 0: control explains all shared excess", ha="left", va="center",
        fontsize=8, color=C.REF)
ax.text(0.985, -0.85, "R = 1: control removes nothing", ha="right", va="center", fontsize=8, color=C.INK)
# one numeric annotation: the lowest retention ratio anywhere on the ladder
ax.annotate(f"lowest R = {r_min:.3f}", xy=(r_min, ys[k_min] + off.get(s_min, 0.0)),
            xytext=(0.80, ys[k_min] + off.get(s_min, 0.0)), ha="right", va="center", fontsize=8,
            color=C.INK, arrowprops=dict(arrowstyle="-", color=C.REF, lw=0.6, shrinkA=1, shrinkB=3))

# key for the marks, inside the empty part of the data region (rows 'Topic k=200'..'Current word')
kx, ky0, dy = 0.03, ys["top500"] + 0.15, 0.95
key = [(Line2D([], [], marker=MARK[s], ls="", color=C.ACC, mec="white", ms=MS[s]), NAME[s]) for s in REPS]
key.append((Line2D([], [], marker="o", ls="", mfc="white", mec=C.ACC, mew=0.7, ms=3.4),
            "the other six cells (top two rows)"))
for i, (h, txt) in enumerate(key):
    yy = ky0 + i * dy
    ax.plot(kx, yy, marker=h.get_marker(), ls="", ms=h.get_markersize(), mfc=h.get_markerfacecolor(),
            mec=h.get_markeredgecolor(), mew=h.get_markeredgewidth(), color=h.get_color(), clip_on=False)
    ax.text(kx + 0.03, yy, txt, ha="left", va="center", fontsize=8, color=C.INK)

svg = OUT / "fig1.svg"
fig.savefig(svg)
plt.close(fig)
pdf, png = compose.export(svg, width_in=W)
(OUT / "fig1.json").write_text(json.dumps({
    "script": "code/figures/fig1_control_ladder_20261010.py", "inputs": inputs,
    "R_min_all": r_min, "R_min_at": [s_min, k_min], "R_min_three_cells": min(repR),
    "n_marks": len(allR), "width_in": W, "height_in": H}, indent=1))
print(pdf, png, "R_min", r_min, s_min, k_min, "n", len(allR))
