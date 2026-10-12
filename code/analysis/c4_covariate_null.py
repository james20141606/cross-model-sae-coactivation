#!/usr/bin/env python3
"""C4 cross-model SAE-sharing: COVARIATE-CONTROLLED permutation null + token-scramble.

Addresses the shared-document-covariate confound flagged in review: the standard
permutation null (cross_model_sae_match.match_and_null) permutes B row order fully
at random, destroying ALL prompt structure -- so the excess-over-null absorbs any
sharing that is merely driven by document-level covariates (doc length, topic /
bag-of-words) that BOTH models track. This script recomputes the SAME mutual-NN /
greedy statistic under covariate-MATCHED nulls (permute B only WITHIN covariate
strata, so the null itself reproduces the covariate structure) and under a
token-order-scramble destructive control (re-extract activations on word-shuffled
docs: preserves length + bag-of-words, destroys word order).

Reads nothing cached -- re-extracts pooled SAE features (GPU) exactly as the paper
did, then does all null analysis in pure numpy/sklearn (CPU). Reuses the proven
extraction + statistics core from cross_model_sae_match.py.

Controls (statistic = mutual-NN count, and greedy_max_mean; excess = obs - null_mean):
  baseline_random   : random global perm  (reproduces the published null)
  length_matched    : perm B within doc-length deciles
  bow_matched       : perm B within TF-IDF/KMeans topic clusters
  scramble          : re-extract on word-shuffled docs, random global perm
Interpretation: real order-dependent universality SURVIVES length/bow-matched nulls
and COLLAPSES under scramble; a pure bag-of-words/length covariate artifact does the
opposite (dies under matched nulls and/or survives scramble).
"""
import argparse, json, os, sys, random
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_model_sae_match import (compute_pooled_features, topk_features,
                                    _zscore_cols, _match_stats)


def _mutual_and_greedy(C):
    a2b = C.argmax(1); b2a = C.argmax(0)
    n_mut = int((b2a[a2b] == np.arange(C.shape[0])).sum())
    return float(C.max(1).mean()), n_mut


def strata_perm(rng, strata):
    """Return a permutation of range(N) that only moves indices within their stratum."""
    N = len(strata); perm = np.arange(N)
    for lab in np.unique(strata):
        idx = np.where(strata == lab)[0]
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def null_under(Za, Zb, strata, n_perm, seed):
    """Observed + null mutual-NN & greedy under a (possibly stratified) permutation."""
    N = Za.shape[0]
    C = (Za.T @ Zb) / N
    obs_greedy, obs_mut = _mutual_and_greedy(C)
    rng = np.random.default_rng(seed)
    g, m = [], []
    for _ in range(n_perm):
        perm = rng.permutation(N) if strata is None else strata_perm(rng, strata)
        Cp = (Za.T @ Zb[perm]) / N
        gg, mm = _mutual_and_greedy(Cp)
        g.append(gg); m.append(mm)
    g = np.array(g); m = np.array(m)
    def z(o, a): s = a.std(); return float((o - a.mean()) / s) if s > 1e-12 else float("inf")
    def p(o, a): return float((np.sum(a >= o) + 1) / (len(a) + 1))
    return {
        "obs_mutual_nn": obs_mut, "obs_greedy_max_mean": obs_greedy,
        "null_mutual_mean": float(m.mean()), "null_mutual_sd": float(m.std()),
        "null_greedy_mean": float(g.mean()), "null_greedy_sd": float(g.std()),
        "mutual_excess": float(obs_mut - m.mean()),
        "mutual_excess_frac": float((obs_mut - m.mean()) / Za.shape[1]),
        "mutual_z": z(obs_mut, m), "mutual_p": p(obs_mut, m),
        "greedy_excess": float(obs_greedy - g.mean()),
        "greedy_z": z(obs_greedy, g), "greedy_p": p(obs_greedy, g),
    }


def size_matched_random_strata(strata, seed):
    """Strata with the SAME block-size multiset as `strata` but random membership.

    Necessary calibration for FINE partitions. Permuting within strata approaches the
    identity as blocks shrink, so the null mean rises toward the observed count and the
    excess decays MECHANICALLY -- at block size 1 the excess is exactly 0 by construction,
    whatever the effect is. A drop in bow-matched excess is therefore uninterpretable on
    its own. This arm holds block sizes fixed and destroys only the lexical *content* of
    the blocks, so:
      bow_matched ~= bow_size_matched_random  -> the attenuation is mechanical (block size)
      bow_matched <<  bow_size_matched_random -> the extra drop is genuine lexical structure
    Same logic the paper already applies to the lexical-residualization test, where a random
    subspace of equal rank removed as much as the real lexical one.
    """
    rng = np.random.default_rng(seed)
    sizes = [int((strata == lab).sum()) for lab in np.unique(strata)]
    order = rng.permutation(len(strata))
    out = np.empty(len(strata), dtype=int)
    i = 0
    for lab, s in enumerate(sizes):
        out[order[i:i + s]] = lab
        i += s
    return out


def build_strata(texts, n_len_bins, n_bow_clusters, seed):
    wc = np.array([len(t.split()) for t in texts])
    # length deciles (quantile bins; unique edges to avoid degenerate bins)
    edges = np.unique(np.quantile(wc, np.linspace(0, 1, n_len_bins + 1)))
    len_strata = np.digitize(wc, edges[1:-1])
    # bag-of-words / topic clusters
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    X = TfidfVectorizer(max_features=4000, stop_words="english").fit_transform(texts)
    km = KMeans(n_clusters=n_bow_clusters, n_init=4, random_state=seed).fit(X)
    return len_strata, km.labels_, wc


def scramble_prompts(prompts, seed):
    r = random.Random(seed)
    out = []
    for p in prompts:
        q = dict(p); w = p["prompt"].split(); r.shuffle(w); q["prompt"] = " ".join(w)
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True); ap.add_argument("--sae_a_release", required=True); ap.add_argument("--sae_a_id", required=True)
    ap.add_argument("--model_b", required=True); ap.add_argument("--sae_b_release", required=True); ap.add_argument("--sae_b_id", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--lang", default="en")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_len_bins", type=int, default=10)
    ap.add_argument("--n_bow_clusters", type=int, default=10)
    ap.add_argument("--n_size_matched_reps", type=int, default=20,
                    help="independently sampled size-matched random partitions; the "
                         "calibrator is their mean excess (one draw is not enough)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts)]
    # reproduce compute_pooled_features ordering to align strata with Phi rows
    sel = [p for p in prompts if args.lang is None or p.get("lang") == args.lang]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    if args.max_prompts: sel = sel[:args.max_prompts]
    texts = [p["prompt"] for p in sel]

    print(f"[extract] REAL activations ({len(texts)} prompts)", flush=True)
    Pa = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id, prompts, lang=args.lang, max_prompts=args.max_prompts)
    Pb = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id, prompts, lang=args.lang, max_prompts=args.max_prompts)
    n = min(Pa.shape[0], Pb.shape[0]); Pa, Pb, texts = Pa[:n], Pb[:n], texts[:n]
    Pa, ia = topk_features(Pa, args.topk); Pb, ib = topk_features(Pb, args.topk)
    Za, _ = _zscore_cols(Pa); Zb, _ = _zscore_cols(Pb)
    N = min(Za.shape[0], Zb.shape[0]); Za, Zb = Za[:N], Zb[:N]

    len_strata, bow_strata, wc = build_strata(texts[:N], args.n_len_bins, args.n_bow_clusters, args.seed)
    res = {"cell": os.path.basename(args.output), "model_a": args.model_a, "model_b": args.model_b,
           "sae_a": f"{args.sae_a_release}::{args.sae_a_id}", "sae_b": f"{args.sae_b_release}::{args.sae_b_id}",
           "N_prompts": int(N), "topk": args.topk, "n_perm": args.n_perm,
           "wc_min_med_max": [int(wc.min()), int(np.median(wc)), int(wc.max())],
           "n_len_bins_eff": int(len(np.unique(len_strata))), "n_bow_clusters_eff": int(len(np.unique(bow_strata)))}

    print("[null] baseline_random / length_matched / bow_matched / bow_size_matched_random", flush=True)
    res["baseline_random"] = null_under(Za, Zb, None, args.n_perm, args.seed)
    res["length_matched"]  = null_under(Za, Zb, len_strata, args.n_perm, args.seed + 1)
    res["bow_matched"]     = null_under(Za, Zb, bow_strata, args.n_perm, args.seed + 2)
    # calibration for the block-size effect (see size_matched_random_strata docstring):
    # same block sizes, random membership, so it carries no lexical information.
    # A SINGLE sampled partition is the known weak point of this arm -- one draw could be
    # unrepresentative -- so sample several independently and report the spread. The
    # headline calibrator is the MEAN excess over draws; `reps` records every draw.
    reps = []
    for r in range(args.n_size_matched_reps):
        rs = size_matched_random_strata(bow_strata, args.seed + 1234 + 97 * r)
        reps.append(null_under(Za, Zb, rs, args.n_perm, args.seed + 3 + r))
    exc = np.array([x["mutual_excess"] for x in reps], float)
    res["bow_size_matched_random"] = dict(
        reps[0],
        mutual_excess=float(exc.mean()),
        n_reps=len(reps),
        excess_mean=float(exc.mean()), excess_sd=float(exc.std(ddof=1)) if len(exc) > 1 else 0.0,
        excess_min=float(exc.min()), excess_max=float(exc.max()),
        note=("mutual_excess is the MEAN over n_reps independently sampled size-matched "
              "partitions; the other fields are from the first draw"),
    )
    res["bow_size_matched_random_reps"] = [
        {"rep": i, "mutual_excess": x["mutual_excess"], "null_mutual_mean": x["null_mutual_mean"],
         "mutual_z": x["mutual_z"], "mutual_p": x["mutual_p"]} for i, x in enumerate(reps)]
    res["mean_bow_block_size"] = float(len(bow_strata) / len(np.unique(bow_strata)))

    print("[extract] SCRAMBLED activations (word-shuffled docs)", flush=True)
    scr = scramble_prompts(prompts, args.seed)
    Sa = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id, scr, lang=args.lang, max_prompts=args.max_prompts)
    Sb = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id, scr, lang=args.lang, max_prompts=args.max_prompts)
    ns = min(Sa.shape[0], Sb.shape[0]); Sa, Sb = Sa[:ns], Sb[:ns]
    Sa, _ = topk_features(Sa, args.topk); Sb, _ = topk_features(Sb, args.topk)
    Sza, _ = _zscore_cols(Sa); Szb, _ = _zscore_cols(Sb)
    ns = min(Sza.shape[0], Szb.shape[0]); Sza, Szb = Sza[:ns], Szb[:ns]
    res["scramble"] = null_under(Sza, Szb, None, args.n_perm, args.seed)

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    b, l, w, s = res["baseline_random"], res["length_matched"], res["bow_matched"], res["scramble"]
    c = res["bow_size_matched_random"]
    print("\n==== SUMMARY (mutual-NN excess over null) ====")
    for name, r in [("baseline_random", b), ("length_matched", l), ("bow_matched", w),
                    ("bow_size_matched_random", c), ("scramble", s)]:
        print(f"  {name:16s} obs={r['obs_mutual_nn']:4d} null={r['null_mutual_mean']:7.1f} "
              f"excess={r['mutual_excess']:+7.1f} ({r['mutual_excess_frac']*100:+5.1f}% of {args.topk}) z={r['mutual_z']:6.1f} p={r['mutual_p']:.4f}")
    ratio = s['mutual_excess'] / b['mutual_excess'] if b['mutual_excess'] else float('nan')
    print(f"\n  scramble/baseline excess ratio = {ratio:.3f}")
    if b['mutual_excess']:
        print(f"  mean bow block size = {res['mean_bow_block_size']:.1f} docs")
        print(f"  retention  bow_matched/baseline              = {w['mutual_excess']/b['mutual_excess']:.3f}")
        print(f"  retention  bow_size_matched_random/baseline  = {c['mutual_excess']/b['mutual_excess']:.3f}  <- mechanical floor")
        if c['mutual_excess']:
            print(f"  LEXICAL-SPECIFIC retention (bow / size-matched) = {w['mutual_excess']/c['mutual_excess']:.3f}")
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
