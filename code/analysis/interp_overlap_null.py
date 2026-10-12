#!/usr/bin/env python3
"""Derangement null for the top-document-overlap audit (CPU only, no GPU).

The paper's "auditable by reading" claim reports that matched mutual-NN feature pairs
share max-activating C4 documents (e.g. mean 3.35/5 for Gemma-2-9B<->Mistral-7B). That
number had NO null: the matched features are topical, and two *topical* sets of features
could share top documents at a non-trivial rate under an arbitrary pairing (a handful of
long/generic C4 documents can dominate many features' top-5).

This script builds that null from the stored receipts: hold the k A-side top-doc sets and
the k B-side top-doc sets fixed, and re-pair them under a DERANGEMENT (no pair keeps its
own partner). The statistic is the same one the paper reports: mean |top5_A ∩ top5_B| and
the fraction of pairs sharing >= 1 and >= 3 documents.

The derangement primitive is the one already used for the cross-lingual residual split
(code/analysis/probe_residual_by_langpair.py:35).

Usage:
  python interp_overlap_null.py                       # all interp*.json receipts
  python interp_overlap_null.py results/.../interp_x.json
Writes results/cross_model_sae/INTERP_OVERLAP_NULL.json.
"""
import json, glob, os, sys
import numpy as np

N_DER = int(os.environ.get("N_DER", 2000))
SEED = int(os.environ.get("SEED", 0))


def derange(n, rng):
    """Random derangement of 0..n-1 (identical primitive to probe_residual_by_langpair.py)."""
    if n < 2:
        return np.arange(n)
    for _ in range(50):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return np.roll(np.arange(n), 1)


def stats(docs_a, docs_b, perm):
    """Overlap statistics for pairing A_i with B_{perm[i]}."""
    ov = np.array([len(docs_a[i] & docs_b[perm[i]]) for i in range(len(docs_a))], float)
    return ov.mean(), float((ov >= 1).mean()), float((ov >= 3).mean())


def run(path, rng, n_top=None, tie_aware=True):
    """n_top: recount using only the top-n documents per side (n<=5 as stored). Exposes the
    free parameter in the statistic's DEFINITION -- steering found that refitting its law on
    a different endpoint flipped which law won, so an endpoint chosen by convention has to be
    shown not to be load-bearing.

    tie_aware: drop slots whose activation is exactly 0. SAE activations are sparse, and
    argsort breaks ties by index, so two models can select the SAME arbitrary zero-activation
    documents and manufacture overlap. Audit found this is real but tiny: 8 zero slots in
    1100, all in the held-out arm, contributing exactly 1 overlap count (mean 2.35 -> 2.30).
    Reported tie-aware by default so the number cannot drift back."""
    d = json.load(open(path))
    pairs = d["pairs"]

    def sel(p, side):
        xs = p[side]
        if n_top:
            xs = xs[:n_top]
        if tie_aware:
            xs = [x for x in xs if x["act"] > 0]
        return set(x["doc"] for x in xs)

    docs_a = [sel(p, "top_docs_a") for p in pairs]
    docs_b = [sel(p, "top_docs_b") for p in pairs]
    k = len(pairs)
    ident = np.arange(k)
    obs_mean, obs_ge1, obs_ge3 = stats(docs_a, docs_b, ident)

    null_mean = np.empty(N_DER)
    null_ge1 = np.empty(N_DER)
    null_ge3 = np.empty(N_DER)
    for t in range(N_DER):
        perm = derange(k, rng)
        null_mean[t], null_ge1[t], null_ge3[t] = stats(docs_a, docs_b, perm)

    # one-sided empirical p with the +1 correction (observed included in the reference set)
    p = float((np.sum(null_mean >= obs_mean) + 1) / (N_DER + 1))
    sd = float(null_mean.std(ddof=1))
    z = float((obs_mean - null_mean.mean()) / sd) if sd > 1e-12 else float("inf")
    # how concentrated is the corpus? a few documents dominating would inflate the null
    from collections import Counter
    cnt = Counter()
    for s in docs_a + docs_b:
        cnt.update(s)
    top_doc_share = cnt.most_common(1)[0][1] / sum(cnt.values())

    return {
        "receipt": os.path.basename(path),
        "model_a": d.get("model_a"), "model_b": d.get("model_b"),
        "n_docs": d.get("n_docs"), "k_pairs": k, "n_derangements": N_DER, "seed": SEED,
        "tie_aware": tie_aware, "n_top": n_top or 5,
        "obs_mean_overlap": round(obs_mean, 4),
        "obs_frac_ge1": round(obs_ge1, 4), "obs_frac_ge3": round(obs_ge3, 4),
        "null_mean_overlap": round(float(null_mean.mean()), 4),
        "null_sd_overlap": round(sd, 4),
        "null_p95_overlap": round(float(np.percentile(null_mean, 95)), 4),
        "null_max_overlap": round(float(null_mean.max()), 4),
        "null_frac_ge1": round(float(null_ge1.mean()), 4),
        "null_frac_ge3": round(float(null_ge3.mean()), 4),
        "z": round(z, 2), "p_one_sided": round(p, 6),
        "p_floor": round(1.0 / (N_DER + 1), 6),
        "max_single_doc_share": round(float(top_doc_share), 4),
    }


def main():
    files = sys.argv[1:] or sorted(glob.glob("results/cross_model_sae/interp*.json"))
    rng = np.random.default_rng(SEED)
    out = [run(f, rng) for f in files]
    # free-parameter sensitivity: the "top-5 documents" endpoint is a convention
    sens = []
    for f in files:
        for n in (3, 5):
            r = run(f, np.random.default_rng(SEED), n_top=n)
            sens.append({"receipt": r["receipt"], "n_top": n,
                         "obs": r["obs_mean_overlap"], "null": r["null_mean_overlap"],
                         "z": r["z"], "p": r["p_one_sided"]})
    print("\n---- endpoint sensitivity (n_top) ----")
    for x in sens:
        print("  %-46s n_top=%d obs=%.3f null=%.4f z=%7.2f p=%.4f"
              % (x["receipt"], x["n_top"], x["obs"], x["null"], x["z"], x["p"]))
    for r in out:
        print(json.dumps(r))
    dest = "results/cross_model_sae/INTERP_OVERLAP_NULL.json"
    json.dump({"n_derangements": N_DER, "seed": SEED, "cells": out,
               "endpoint_sensitivity": sens},
              open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
