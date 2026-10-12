#!/usr/bin/env python3
"""Aggregate cross_model_causal.py outputs with degeneracy-robust statistics.

The per-pair Pearson correlation between two features' per-prompt ablation-Delta-loss
vectors is unstable when single-feature effects are sparse (vectors dominated by one
prompt -> correlations pile up at exactly +-1). We therefore report, alongside the
stored derangement-null mean/z/p:
  - median per-pair correlation (robust to +-1 outliers),
  - a SIGN TEST: fraction of matched pairs with positive causal-corr and a two-sided
    binomial p vs 0.5 (does the match predict same-direction causal effect more often
    than chance?),
  - a non-degenerate mean excluding |r|>0.99 pairs,
  - the mean single-feature ablation effect (sanity: are these features causal at all?).
"""
import json, glob, os, sys, math
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results/cross_model_causal"


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    from math import comb
    def pmf(i):
        return comb(n, i) * p**i * (1 - p)**(n - i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        bn = os.path.basename(f).replace(".json", "")
        if bn.startswith("smoke") or bn == "CAUSAL_SUMMARY":
            continue
        d = json.load(open(f))
        if not isinstance(d, dict):
            continue
        pp = np.array(d.get("matched_causal_corr_per_pair", []), float)
        nd = pp[np.abs(pp) <= 0.99]
        pos = int((pp > 0).sum()); n = len(pp)
        rows.append({
            "pair": bn,
            "within": bn.count("gemma2-") == 2,
            "matched_mean": d["matched_causal_corr_mean"],
            "null_mean": d["null_mean"], "z": d["z"], "perm_p": d["p"],
            "median_pp": float(np.median(pp)) if n else 0.0,
            "frac_pos": pos / n if n else 0.0,
            "sign_p": binom_two_sided(pos, n),
            "n_pairs": n, "n_degenerate": int((np.abs(pp) > 0.99).sum()),
            "nondegen_mean": float(nd.mean()) if len(nd) else float("nan"),
            "nondegen_n": int(len(nd)),
            "abl_A": d["mean_abl_effect_A"], "abl_B": d["mean_abl_effect_B"],
        })
    rows.sort(key=lambda r: (not r["within"], -r["z"]))
    hdr = ("pair", "cls", "matched", "null", "permP", "median", "frac+", "signP", "degen", "ndMean", "ablA", "ablB")
    print("%-28s %-6s %8s %7s %6s %7s %6s %6s %6s %7s %7s %7s" % hdr)
    for r in rows:
        print("%-28s %-6s %+8.3f %+7.3f %6.3f %+7.3f %6.2f %6.3f %3d/%-2d %+7.3f %+7.3f %+7.3f" % (
            r["pair"], "WITHIN" if r["within"] else "cross", r["matched_mean"], r["null_mean"],
            r["perm_p"], r["median_pp"], r["frac_pos"], r["sign_p"], r["n_degenerate"], r["n_pairs"],
            r["nondegen_mean"], r["abl_A"], r["abl_B"]))
    json.dump(rows, open(os.path.join(RES, "CAUSAL_SUMMARY.json"), "w"), indent=2)
    print(f"\n[wrote] {os.path.join(RES, 'CAUSAL_SUMMARY.json')}")


if __name__ == "__main__":
    main()
