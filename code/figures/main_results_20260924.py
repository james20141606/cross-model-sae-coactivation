"""Body result figure (2026-09-24 overhaul): one figure replaces main_heldout + main_controls.

  a  nine held-out C4 cells: frozen-pair excess on the evaluation half vs pairs re-selected
     there (dumbbell), ratio r printed per row          <- heldout_*.json
  b  control ladder on the three representative cells: every retention ratio R of eq. (5)
     on one axis                                        <- ninecell_20260926/cluster_18577359/A_*.json
                                                           point.top{200,500}.R (revision R1,
                                                           2026-10-07: 20-partition calibration;
                                                           the submitted figure read the one-partition
                                                           covctl_k{200,500}_c4_*.json),
                                                           lextok20k_*.json (k=4096),
                                                           lanpair_/proclex_lextok20k_*_k4096.json,
                                                           tokaclean20k_*.json
  c  random-dictionary control: reciprocal count vs its own permutation null, trained SAE vs
     size/sparsity-matched random directions; dashed line = K^2/(2K-1) of Proposition 1(a)
                                                        <- c4_*.json, rand_c4_*.json
No number is typed here; every mark is read from the receipts listed above.
Run: python3 code/figures/main_results_20260924.py
"""
import json, sys, hashlib, pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/figstyle"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs
import compose

fs.use()
plt.rcParams.update({"savefig.bbox": None, "axes.grid": False})
RES = REPO / "results/cross_model_sae"
OUT = REPO / "paper/figures/results_20260924"
OUT.mkdir(parents=True, exist_ok=True)
inputs = []


def load(name, base=None):
    p = (base or RES) / name
    inputs.append({"path": str(p.relative_to(REPO)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return json.loads(p.read_text())


SHORT = {"qwen3-1.7b": "Q1.7", "qwen3-8b": "Q8", "gemma2-2b": "G2", "gemma2-9b": "G9",
         "mistral7b": "M7", "llama3.2-1b": "L1"}
NINE = ["qwen3-1.7b__gemma2-2b", "qwen3-8b__gemma2-2b", "qwen3-8b__gemma2-9b",
        "qwen3-1.7b__gemma2-9b", "qwen3-8b__mistral7b", "gemma2-9b__mistral7b",
        "gemma2-2b__mistral7b", "qwen3-1.7b__mistral7b", "llama3.2-1b__mistral7b"]
REPS = ["qwen3-1.7b__gemma2-2b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
BAND = {REPS[0]: "strong", REPS[1]: "median", REPS[2]: "weak"}
CCOL = {REPS[0]: fs.INDIGO, REPS[1]: fs.AMBER, REPS[2]: fs.VIOLET}
CMARK = {REPS[0]: "o", REPS[1]: "s", REPS[2]: "^"}
CMS = {REPS[0]: 4.0, REPS[1]: 3.8, REPS[2]: 4.8}
MINUS = "−"


def lab(stem):
    a, b = stem.split("__")
    return f"{SHORT[a]}–{SHORT[b]}"


# ------------------------------------------------------------------ data
held = {}
for s in NINE:
    d = load(f"heldout_{s}.json")
    assert d["corpora_disjoint_asserted"] and d["n_perm"] == 2000
    held[s] = (d["held_out"]["excess"], d["in_sample_eval_own_pairs"]["excess"],
               d["ratio_heldout_over_insample"])
order = sorted(NINE, key=lambda s: -held[s][0])

RUNGS = ["Topic strata, k = 200", "Topic strata, k = 500", "Lexical: mutual NN",
         "Lexical: Lan et al. rule", "Lexical: Procrustes", "Current word (clean)"]
ladder = {s: [] for s in REPS}
for s in REPS:
    # R1 (2026-10-07): topic-strata R from the registered post-submission run, whose size-matched
    # calibration averages 20 random partitions (the covctl receipts used one; Table 1 caption).
    a = load(f"A_{s}.json", REPO / "results/ninecell_20260926/cluster_18577359")
    for k in (200, 500):
        assert a["point"][f"top{k}"]["calibration"]["n_rep"] == 20
        ladder[s].append(a["point"][f"top{k}"]["R"])
    ladder[s].append(load(f"lextok20k_{s}.json")["arms"]["4096"]["R_lex"])
    for pre in ("lanpair", "proclex"):
        ladder[s].append(load(f"{pre}_lextok20k_{s}.json")["arms"]["4096"]["R_lex"])
    t = load(f"tokaclean20k_{s}.json")
    assert all(t["a3_requirements"].values())
    ladder[s].append(t["s2_clean"]["R_tok_clean"])

rand = {}
for s in REPS:
    c, r = load(f"c4_{s}.json"), load(f"rand_c4_{s}.json")
    rand[s] = ((c["observed"]["n_mutual_nn"], c["null_mutual_nn"]["mean"], c["null_mutual_nn"]["sd"]),
               (r["observed"]["n_mutual_nn"], r["null_mutual_nn"]["mean"], r["null_mutual_nn"]["sd"]))
    assert c["topk"] == r["topk"] == 1500
K = 1500
chance = K * K / (2 * K - 1)

# ------------------------------------------------------------------ figure
W, H = 5.5, 2.3
fig = plt.figure(figsize=(W, H))
axa = fig.add_axes([0.10, 0.19, 0.22, 0.60])
axb = fig.add_axes([0.51, 0.19, 0.215, 0.60])
axc = fig.add_axes([0.80, 0.19, 0.185, 0.60])
TICK = fs.SIZE_TICK

# a -- held-out dumbbells
for y, s in enumerate(order):
    h, e, r = held[s]
    axa.plot([h, e], [y, y], color=fs.GRID, lw=1.6, zorder=1, solid_capstyle="round")
    axa.plot(e, y, "o", ms=4.2, mfc="white", mec=fs.GREY, mew=0.9, zorder=2)
    axa.plot(h, y, "o", ms=4.2, color=fs.TEAL, mec="white", mew=0.5, zorder=3)
    axa.text(0.845, y, f"{r:.2f}", va="center", ha="left", fontsize=TICK, color=fs.INK2)
    if s in CCOL:
        # 2026-09-24 final review: these identify the three representative cells of (b)/(c);
        # they sit in the tick-label gutter (x < 0 in axes units), not on the data axis.
        axa.plot(-0.035, y, CMARK[s], ms=CMS[s] - 0.3, color=CCOL[s], mec="white", mew=0.4,
                 clip_on=False, transform=axa.get_yaxis_transform())
axa.text(0.845, -1.05, "r", ha="left", va="center", fontsize=TICK, color=fs.INK2)
axa.set_xlim(0.45, 0.91)
axa.set_ylim(len(order) - 0.4, -1.2)
axa.set_yticks(range(len(order)))
axa.set_yticklabels([lab(s) for s in order])
axa.set_xticks([0.5, 0.6, 0.7, 0.8])
axa.set_xlabel("Excess mean pair correlation")
axa.tick_params(axis="y", length=0, pad=8)
axa.spines["left"].set_visible(False)
axa.grid(axis="x", color=fs.GRID, lw=0.5)
fig.legend([Line2D([], [], marker="o", ls="", color=fs.TEAL, mec="white"),
            Line2D([], [], marker="o", ls="", mfc="white", mec=fs.GREY)],
           ["frozen, held-out", "re-selected"], loc="center left", bbox_to_anchor=(0.0, 0.875),
           bbox_transform=fig.transFigure,
           ncol=2, handletextpad=0.3, columnspacing=0.8, fontsize=TICK)

# b -- control ladder
axb.axhspan(1.5, 4.5, color=fs.tint_hex(fs.GREY, 0.10), lw=0, zorder=0)
axb.axvline(1.0, color=fs.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
off = {REPS[0]: -0.2, REPS[1]: 0.0, REPS[2]: 0.2}
for s in REPS:
    for y, v in enumerate(ladder[s]):
        axb.plot(v, y + off[s], CMARK[s], ms=CMS[s], color=CCOL[s], mec="white", mew=0.4, zorder=3)
axb.set_xlim(0.86, 1.08)
axb.set_xticks([0.9, 1.0])
axb.set_ylim(len(RUNGS) - 0.45, -1.2)
axb.set_yticks(range(len(RUNGS)))
axb.set_yticklabels(RUNGS)
axb.tick_params(axis="y", length=0)
axb.spines["left"].set_visible(False)
axb.grid(axis="x", color=fs.GRID, lw=0.5)
axb.set_xlabel("Retention ratio R")
axb.text(1.004, -1.05, "no removal", ha="left", va="center", fontsize=TICK, color=fs.INK2)

# c -- random-dictionary control
xs = {0: "trained\nSAE", 1: "random\ndirections"}
dx = {REPS[0]: -0.22, REPS[1]: 0.0, REPS[2]: 0.22}
axc.axhline(chance, color=fs.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
axc.text(1.45, chance + 18, "i.i.d. chance", ha="right", va="bottom", fontsize=TICK, color=fs.INK2)
for s in REPS:
    for g in (0, 1):
        obs, mu, sd = rand[s][g]
        x = g + dx[s]
        axc.add_patch(matplotlib.patches.Rectangle((x - 0.08, mu - 2 * sd), 0.16, 4 * sd,
                                                   color=fs.GREY, alpha=0.55, lw=0, zorder=2))
        axc.plot([x, x], [mu, obs], color=CCOL[s], lw=0.9, zorder=2)
        axc.plot(x, obs, CMARK[s], ms=CMS[s], color=CCOL[s], mec="white", mew=0.4, zorder=3)
axc.set_xlim(-0.5, 1.5)
axc.set_ylim(0, 900)
axc.set_xticks([0, 1])
axc.set_xticklabels([xs[0], xs[1]])
axc.set_yticks([0, 300, 600, 900])
axc.set_ylabel("Reciprocal pairs M", labelpad=2)
axc.grid(axis="y", color=fs.GRID, lw=0.5)
axc.tick_params(axis="x", length=0)

fig.text(0.005, 0.965, "a", fontsize=fs.SIZE_PANEL, fontweight="bold", va="top")
fig.text(0.025, 0.965, "Nine cells, held-out half", fontsize=fs.SIZE_LABEL, va="top", color=fs.INK)
fig.text(0.345, 0.965, "b", fontsize=fs.SIZE_PANEL, fontweight="bold", va="top")
fig.text(0.365, 0.965, "Control ladder, three cells", fontsize=fs.SIZE_LABEL, va="top", color=fs.INK)
fig.text(0.745, 0.965, "c", fontsize=fs.SIZE_PANEL, fontweight="bold", va="top")
fig.text(0.765, 0.965, "Random dictionaries", fontsize=fs.SIZE_LABEL, va="top", color=fs.INK)
handles = [Line2D([], [], marker=CMARK[s], ls="", color=CCOL[s], mec="white", ms=CMS[s] + 0.3) for s in REPS]
fig.legend(handles, [f"{lab(s)} ({BAND[s]})" for s in REPS], loc="center left",
           bbox_to_anchor=(0.345, 0.875), ncol=3, handletextpad=0.2, columnspacing=0.9, fontsize=TICK)

svg = OUT / "main_results.svg"
fig.savefig(svg)
plt.close(fig)
pdf, png = compose.export(svg, width_in=W)
(OUT / "main_results.json").write_text(json.dumps({
    "script": "code/figures/main_results_20260924.py", "inputs": inputs,
    "chance_Kiid": chance, "width_in": W, "height_in": H}, indent=1))
print(pdf, png)
