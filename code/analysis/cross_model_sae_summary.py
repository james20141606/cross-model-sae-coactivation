#!/usr/bin/env python3
"""Aggregate cross_model_sae_match.py outputs into a summary table + a figure.

Reads results/cross_model_sae/*.json (excluding smoke_*), classifies each pair as
within-family or cross-family, and reports the three matching statistics with their
permutation nulls. The headline contrast is the strict reciprocal mutual-NN
excess-over-null; greedy-max and Hungarian are reported to show they are confounded
(saturated / inconsistent).

Usage:
  python cross_model_sae_summary.py [results_dir]
Writes:
  results/cross_model_sae/SUMMARY.json
  paper/figures/fig_cross_model_sae.pdf  (if matplotlib available)
"""
import json, glob, os, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "results/cross_model_sae"

# capability tier for interpretation (params, rough): gpt2=0.12B, gemma-2-2b=2B,
# mistral-7b=7B, gemma-2-9b=9B. "within" = same base family (both gemma-2).
FAMILY = {"gemma2-2b": "gemma2", "gemma2-9b": "gemma2", "gpt2": "gpt2", "mistral7b": "mistral"}
SCALE = {"gemma2-2b": 2.0, "gemma2-9b": 9.0, "gpt2": 0.12, "mistral7b": 7.0}


def parse_pair(name):
    # e.g. "gemma2-2b__gpt2" or "gemma2-2bL6__gemma2-9bL10"
    a, b = name.split("__")
    a0 = a.split("L")[0]; b0 = b.split("L")[0]
    return a0, b0


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        bn = os.path.basename(f).replace(".json", "")
        if bn.startswith("smoke") or bn in ("SUMMARY",):
            continue
        d = json.load(open(f)); o = d["observed"]
        a0, b0 = parse_pair(bn)
        within = FAMILY.get(a0) == FAMILY.get(b0)
        has_gpt2 = "gpt2" in (a0, b0)
        nmean = d["null_mutual_nn"]["mean"]; nsd = d["null_mutual_nn"].get("sd", 0) or 1e-9
        mut_excess = o["n_mutual_nn"] - nmean
        rows.append({
            "pair": bn, "within_family": within, "has_gpt2": has_gpt2,
            "category": "within" if within else ("cross-GPT2" if has_gpt2 else "cross-modern"),
            "scale_min": min(SCALE.get(a0, 0), SCALE.get(b0, 0)),
            "greedy_excess": d["greedy_excess_over_null"],
            "hungarian_obs": o["hungarian_mean"],
            "hungarian_null": d.get("null_hungarian_mean") or 0.0,
            "mutual_nn": o["n_mutual_nn"], "mutual_nn_null": nmean,
            "mutual_nn_excess": mut_excess, "mutual_nn_z": mut_excess / nsd,
            "mutual_nn_ratio": o["n_mutual_nn"] / max(nmean, 1e-9), "mutual_p": d["mutual_p"],
            "N": d["N_prompts"], "topk": d["topk"],
        })
    rows.sort(key=lambda r: -r["mutual_nn_z"])

    print(f"{'pair':28s} {'category':12s} {'greedyX':>8s} {'hung':>7s} {'mutNN(null)':>13s} {'z':>6s} {'ratio':>6s} {'p':>6s}")
    for r in rows:
        print("%-28s %-12s %+8.3f %7.3f %5d(%5.1f) %6.1f %6.2f %6.3f" % (
            r["pair"], r["category"], r["greedy_excess"], r["hungarian_obs"],
            r["mutual_nn"], r["mutual_nn_null"], r["mutual_nn_z"], r["mutual_nn_ratio"], r["mutual_p"]))

    json.dump(rows, open(os.path.join(RES, "SUMMARY.json"), "w"), indent=2)
    print(f"\n[wrote] {os.path.join(RES, 'SUMMARY.json')}")

    # figure: (L) mutual-NN z by category — within & cross-modern share, GPT-2 pairs don't;
    #         (R) greedy excess (confounded, saturates) vs mutual-NN z (valid).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        COL = {"within": "#1f77b4", "cross-modern": "#2ca02c", "cross-GPT2": "#d62728"}
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
        cats = ["within", "cross-modern", "cross-GPT2"]
        for xi, cat in enumerate(cats):
            zs = [r["mutual_nn_z"] for r in rows if r["category"] == cat]
            xs = [xi + (i - len(zs) / 2) * 0.06 for i in range(len(zs))]
            ax[0].scatter(xs, zs, c=COL[cat], s=55, label=cat)
        ax[0].axhline(1.96, color="grey", lw=0.6, ls="--")
        ax[0].set_xticks(range(len(cats))); ax[0].set_xticklabels(cats, fontsize=8)
        ax[0].set_ylabel("mutual-NN z (excess / null SD)")
        ax[0].set_title("reciprocal sharing: GPT-2 is the outlier")
        for r in rows:
            ax[1].scatter(r["greedy_excess"], r["mutual_nn_z"], c=COL[r["category"]], s=55)
        ax[1].axhline(1.96, color="grey", lw=0.6, ls="--")
        ax[1].set_xlabel("greedy-max excess (confounded)")
        ax[1].set_ylabel("mutual-NN z (valid)")
        ax[1].set_title("loose metric saturates; strict metric discriminates")
        ax[0].legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        out = "paper/figures/fig_cross_model_sae.pdf"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        print(f"[wrote] {out}")
    except Exception as e:
        print(f"[fig skipped] {e}")


if __name__ == "__main__":
    main()
