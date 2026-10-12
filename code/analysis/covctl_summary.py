#!/usr/bin/env python3
"""Summarise the covariate-controlled nulls against the pre-registered decision rule.

The rule was fixed in experiments_detail.md BEFORE any k=200/k=500 receipt existed, and
amended (also before any receipt existed) to calibrate the block-size effect:

  R := bow_matched_excess / bow_size_matched_random_excess      (per cell)
  SURVIVES  R >= 0.70 at BOTH k=200 and k=500 in >=2 of 3 cells, bow p at the floor
  PARTIAL   0.30 <= R < 0.70
  FAILS     R < 0.30 in >=2 of 3 cells

Why not bow/baseline: permuting within strata approaches the identity as blocks shrink, so
the excess decays MECHANICALLY with finer partitions (at block size 1 it is exactly 0 by
construction). bow_size_matched_random holds block sizes fixed and randomises membership,
so it carries no lexical information and isolates that mechanical component.

Primary statistic is the EXCESS (obs - null_mean), not z: finer strata also shrink the null
SD, which inflates z for a reason that has nothing to do with the effect.

    python3 code/analysis/covctl_summary.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results/cross_model_sae")
ARMS = ["baseline_random", "length_matched", "bow_matched",
        "bow_size_matched_random", "scramble"]


def load(path):
    d = json.load(open(path))
    m = re.match(r"covctl_k(\d+)_c4_(.+)\.json", os.path.basename(path))
    k = int(m.group(1)) if m else d.get("n_bow_clusters_eff")
    cell = m.group(2) if m else os.path.basename(path)[len("covctl_c4_"):-len(".json")]
    return k, cell, d


def main():
    files = sorted(glob.glob(os.path.join(RES, "covctl_*c4_*.json")))
    if not files:
        print("no covctl receipts found")
        return 1
    rows = []
    for f in files:
        k, cell, d = load(f)
        b, w = d["baseline_random"], d["bow_matched"]
        c = d.get("bow_size_matched_random")
        raw = w["mutual_excess"] / b["mutual_excess"] if b["mutual_excess"] else float("nan")
        R = (w["mutual_excess"] / c["mutual_excess"]
             if c and c["mutual_excess"] else float("nan"))
        rows.append((cell, k, d, raw, R))
    rows.sort(key=lambda r: (r[0], r[1]))

    print("%-22s %4s %5s %6s  %8s %8s   %s" %
          ("cell", "k", "nperm", "blksz", "raw ret.", "R (lex)", "excess: base/len/bow/sizerand/scram"))
    for cell, k, d, raw, R in rows:
        arms = " ".join("%+7.1f" % d[a]["mutual_excess"] for a in ARMS if a in d)
        print("%-22s %4s %5d %6.1f  %8.3f %8.3f   %s"
              % (cell, k, d["n_perm"], d.get("mean_bow_block_size", float("nan")), raw, R, arms))

    print("\nper-arm detail (obs / null mean / excess / z / p):")
    for cell, k, d, raw, R in rows:
        print("  %s  k=%s  (n_bow_eff=%s, mean block %.1f docs)"
              % (cell, k, d.get("n_bow_clusters_eff"), d.get("mean_bow_block_size", float("nan"))))
        for a in ARMS:
            if a not in d:
                continue
            r = d[a]
            print("     %-24s obs=%5d null=%8.1f excess=%+8.1f (%+5.1f%% of %d) z=%7.1f p=%.4f"
                  % (a, r["obs_mutual_nn"], r["null_mutual_mean"], r["mutual_excess"],
                     r["mutual_excess_frac"] * 100, d["topk"], r["mutual_z"], r["mutual_p"]))

    # verdict against the pre-registered (amended) rule
    print("\n---- pre-registered decision rule (amended: R = bow / bow_size_matched_random) ----")
    verdicts = {}
    for k_target in (200, 500):
        sel = [(c, d, R) for c, k, d, raw, R in rows if k == k_target]
        if not sel:
            continue
        floors = all(d["bow_matched"]["mutual_p"] <= 1.0 / (d["n_perm"] + 1) + 1e-12
                     for _, d, _ in sel)
        n_surv = sum(1 for _, _, R in sel if R >= 0.70)
        n_fail = sum(1 for _, _, R in sel if R < 0.30)
        verdicts[k_target] = ("SURVIVES" if (n_surv >= 2 and floors)
                              else "FAILS" if n_fail >= 2 else "PARTIAL")
        print("  k=%-4d %d/%d cells R>=0.70, %d/%d R<0.30, bow p at floor: %s  -> %s"
              % (k_target, n_surv, len(sel), n_fail, len(sel), floors, verdicts[k_target]))
    if len(verdicts) == 2:
        print("  OVERALL: %s"
              % ("SURVIVES" if all(v == "SURVIVES" for v in verdicts.values())
                 else "FAILS" if any(v == "FAILS" for v in verdicts.values())
                 else "PARTIAL"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
