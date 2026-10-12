#!/usr/bin/env python3
"""Merge per-k lextok chunk receipts into the canonical per-cell receipt.

Cell 0 had to be chunked by k because two full-cell attempts were destroyed by
cluster-wide admin scancel sweeps (288 and 38 jobs) purely for being the longest-running
job. Chunking is statistically identical -- each k is an independent arm and the
k-independent quantities are recomputed from the same cached features with the same seed --
but "identical" is a claim, so it is CHECKED rather than assumed:

  the frozen mutual-NN pair set, its size, and the raw pair excess/z must agree EXACTLY
  across every chunk. Three independently scheduled jobs agreeing bit-for-bit on those is
  what licenses treating the arms as one run. Any disagreement aborts the merge.

    python3 code/analysis/merge_lextok_chunks.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results/cross_model_sae")


def main():
    chunks = {}
    for f in sorted(glob.glob(os.path.join(RES, "lextok20k_*_k[0-9]*.json"))):
        m = re.match(r"lextok20k_(.+)_k(\d+)\.json", os.path.basename(f))
        chunks.setdefault(m.group(1), []).append((int(m.group(2)), f))
    if not chunks:
        print("no chunk receipts found")
        return 1

    rc = 0
    for cell, items in chunks.items():
        items.sort()
        base = None
        merged = None
        for k, f in items:
            d = json.load(open(f))
            key = (d["frozen_pairs"]["n"], tuple(d["frozen_pairs"]["feat_a"]),
                   tuple(d["frozen_pairs"]["feat_b"]),
                   round(d["raw"]["excess"], 12), round(d["raw"]["obs_mean_pair_corr"], 12))
            if base is None:
                base, merged = key, dict(d)
                merged["arms"] = {}
                merged["merged_from_chunks"] = []
            elif key != base:
                print("ABORT %s: chunk k=%d disagrees with the first chunk on a "
                      "k-INDEPENDENT quantity (frozen pairs or raw excess)" % (cell, k))
                rc = 1
                break
            merged["arms"].update(d["arms"])
            merged["merged_from_chunks"].append(os.path.basename(f))
        else:
            merged["any_arm_gate_passed"] = any(
                a["adequacy"]["gate_passed"] for a in merged["arms"].values())
            out = os.path.join(RES, "lextok20k_%s.json" % cell)
            json.dump(merged, open(out, "w"), indent=2)
            # read back what was written rather than trusting the write
            chk = json.load(open(out))
            ok = (sorted(chk["arms"], key=int) == [str(k) for k, _ in items]
                  and chk["frozen_pairs"]["n"] == merged["frozen_pairs"]["n"])
            print("%s: merged arms %s from %d chunks | k-independent cross-check PASS | "
                  "read-back %s" % (cell, sorted(chk["arms"], key=int), len(items),
                                    "ok" if ok else "FAILED"))
            print("   frozen_pairs=%d raw_excess=%+.6f  (identical in all chunks)"
                  % (chk["frozen_pairs"]["n"], chk["raw"]["excess"]))
            if not ok:
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
