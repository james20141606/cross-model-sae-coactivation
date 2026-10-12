#!/usr/bin/env python3
"""plan-CT-2b: Paired-bootstrap 95% CIs for the Rosetta power-law slope b.

Background
----------
`rosetta_sublinear_check.py` fits a power law r = c * N^b (OLS on log-log) for the
within-family cross-lingual probe-attribution alignment (mean_pairwise_r) vs model
parameter count N, per family. That script reports the OLS *point* estimates only
(exponent_b, coef_c, r2) and writes them to results/rosetta_check/rosetta_sublinear_v0.json.

This script adds the missing uncertainty quantification: a nonparametric bootstrap
over the (params, r) points within each family, reporting the 95% CI of the slope b.
Pure CPU / numpy, no GPU, no SSH. Reads ONLY the per-model JSONs in
results/probe_attribution_v2/ (same source as the point-estimate script) so the CIs
are reproducible from released artifacts.

Output: results/rosetta_check/rosetta_slope_bootstrap.json with, per family,
the OLS point estimate b, R^2, n, and the bootstrap 95% CI [lo, hi] of b.
"""
import json, glob, math
import numpy as np

V2 = "results/probe_attribution_v2"
PARAMS = {
    "Qwen/Qwen3-0.6B": 0.6, "Qwen/Qwen3-1.7B": 1.7, "Qwen/Qwen3-4B": 4.0,
    "Qwen/Qwen3-8B": 8.0, "Qwen/Qwen3-14B": 14.0,
    "google/gemma-3-1b-it": 1.0, "google/gemma-3-4b-it": 4.0,
    "google/gemma-3-12b-it": 12.0, "google/gemma-3-27b-it": 27.0,
    "EleutherAI/pythia-70m-deduped": 0.07, "EleutherAI/pythia-160m-deduped": 0.16,
    "EleutherAI/pythia-410m-deduped": 0.41, "EleutherAI/pythia-1b-deduped": 1.0,
    "EleutherAI/pythia-1.4b-deduped": 1.4,
}
N_BOOT = 10000
SEED = 1  # matches the bootstrap-seed convention in tab_hyperparams.tex

def family_of(m):
    return "Qwen3" if "Qwen" in m else ("Pythia" if "pythia" in m.lower() else "Gemma3")

def load_rows():
    rows = []
    for fp in sorted(glob.glob(f"{V2}/*.json")):
        d = json.load(open(fp))
        m = d["model"]
        rows.append((family_of(m), PARAMS[m], float(d["mean_pairwise_r"])))
    return rows

def slope(x, y):
    """OLS slope of log r on log N. Returns nan if x has no spread."""
    if np.unique(x).size < 2:
        return float("nan")
    b, _logc = np.polyfit(np.log(x), np.log(y), 1)
    return b

def fit_family(pts):
    pts = sorted(pts)
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    b, logc = np.polyfit(np.log(x), np.log(y), 1)
    yhat = b * np.log(x) + logc
    ss_res = np.sum((np.log(y) - yhat) ** 2)
    ss_tot = np.sum((np.log(y) - np.log(y).mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    n = len(pts)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        bb = slope(x[idx], y[idx])
        if not math.isnan(bb):
            boot.append(bb)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(b=float(b), r2=float(r2), n=n,
                ci95=[float(lo), float(hi)], n_valid_boot=int(boot.size),
                params=[p[0] for p in pts], r=[p[1] for p in pts])

def main():
    rows = load_rows()
    out = {"method": f"nonparametric point bootstrap, {N_BOOT} resamples, seed={SEED}, "
                      "OLS slope of log(mean_pairwise_r) on log(params); only r>0 points used",
           "families": {}}
    # Only positive-r points enter the log-log fit (mirrors rosetta_sublinear_check.py).
    for fam in ["Pythia", "Gemma3", "Qwen3"]:
        pts = [(p, r) for (f, p, r) in rows if f == fam and r > 0]
        out["families"][fam] = fit_family(pts)
    # Qwen3 sensitivity: excluding the 14B floor point.
    q_excl = [(p, r) for (f, p, r) in rows if f == "Qwen3" and r > 0 and p != 14.0]
    out["families"]["Qwen3_excl14B"] = fit_family(q_excl)

    for fam, d in out["families"].items():
        print(f"[{fam:14s}] b={d['b']:+.3f}  R^2={d['r2']:.3f}  n={d['n']}  "
              f"95%CI[{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]")
    json.dump(out, open("results/rosetta_check/rosetta_slope_bootstrap.json", "w"), indent=2)
    print("\nwrote results/rosetta_check/rosetta_slope_bootstrap.json")

if __name__ == "__main__":
    main()
