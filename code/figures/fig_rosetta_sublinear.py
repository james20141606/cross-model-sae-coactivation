#!/usr/bin/env python3
"""fig:rosetta_sublinear -- raw cross-lingual probe-attribution mean r vs parameter count, three families (D class).

Receipts (read-only):
  results/rosetta_check/rosetta_sublinear_v0.json     rows (per-model r), fits (power-law c, b per family)
  results/rosetta_check/rosetta_slope_bootstrap.json  10,000-resample bootstrap 95% CI on the exponent b
Cross-check: each family's fitted exponent must equal its bootstrap point estimate. Qwen3 uses the all-five-
checkpoint fit (Qwen3_incl14B), the one the caption quotes (b = -0.32).
Included at 0.62\textwidth = 3.41 in; figsize is that width so every font is its printed size (>= 7 pt).
Output: paper/figures/fig_rosetta_sublinear.{svg,pdf}; gate: figstyle/audit.py --width-in 3.41 -> 0 errors.
"""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figstyle"))
import figstyle as fs     # noqa: E402
fs.use()
import matplotlib.pyplot as plt   # noqa: E402
plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})

WIDTH_IN = 0.62 * fs.TEXTWIDTH_IN
d = json.loads((ROOT / "results/rosetta_check/rosetta_sublinear_v0.json").read_text())
bs = json.loads((ROOT / "results/rosetta_check/rosetta_slope_bootstrap.json").read_text())["families"]
rows = d["rows"]
FAM = [("Pythia", fs.INDIGO, "o", "Pythia", "Pythia"), ("Gemma3", fs.AMBER, "s", "Gemma3", "Gemma3"),
       ("Qwen3", fs.VIOLET, "^", "Qwen3_incl14B", "Qwen3")]

fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.4))
fig.subplots_adjust(left=0.165, right=0.985, top=0.975, bottom=0.165)
fs.zero_line(ax, 0.0)
labels = []
for fam, col, mk, fitkey, bskey in FAM:
    pts = sorted((r["params"], r["r"]) for r in rows if r["family"] == fam)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    f = d["fits"][fitkey]; b = bs[bskey]
    assert abs(f["exponent_b"] - b["b"]) < 1e-6, (fam, f["exponent_b"], b["b"])
    xx = np.geomspace(min(f["params"]), max(f["params"]), 60)
    ax.plot(xx, f["coef_c"] * xx ** f["exponent_b"], ls="--", color=col, lw=0.9, alpha=0.75, zorder=2)
    ax.plot(xs, ys, marker=mk, color=col, lw=1.3, ms=4.6, zorder=3)
    lo, hi = b["ci95"]
    labels.append((fam, col, f"{fam}: b = {b['b']:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {b['n']}"))
    print("%-7s b=%+.3f  95%% CI [%+.2f, %+.2f]  n=%d  fit range %s" % (fam, b["b"], lo, hi, b["n"], f["params"]))
# family key: text in the data-free top-left corner (no legend handles), one line per family
for k, (fam, col, s) in enumerate(labels):
    ax.text(0.062, 1.0 - 0.085 * k, s, fontsize=7, color=col, va="center", ha="left")
ax.text(0.2, -0.085, "Pythia-70M: r < 0, excluded from its fit", fontsize=7, color=fs.INK3, va="center", ha="left")
ax.set_xscale("log"); ax.set_xlim(0.055, 45); ax.set_ylim(-0.14, 1.06)
ax.set_xlabel("parameters (B, log scale)")
ax.set_ylabel("cross-lingual mean r (raw metric)")
fs.save(fig, ROOT / "paper/figures/fig_rosetta_sublinear.svg")
print("wrote paper/figures/fig_rosetta_sublinear.{svg,pdf}")
