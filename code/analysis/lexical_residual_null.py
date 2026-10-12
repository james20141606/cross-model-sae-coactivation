#!/usr/bin/env python3
"""C4 cross-model SAE-sharing: LEXICAL-RESIDUALIZED permutation null.

The missing experiment named by the 2026-07-15 audit. `c4_covariate_null.py`
established that the cross-model mutual-NN excess survives a doc-LENGTH-matched
null (~2.5% of the excess) and a coarse k=10 TOPIC-matched null (~0%), but does
NOT collapse under word-order scramble -- and it cannot, because the statistic
MEAN-POOLS SAE activations over tokens and is order-insensitive by construction.
So the surviving overlap is compatible with two models sharing *unigram/lexical*
detectors. Lexical composition is itself a shared-document covariate; a 10-cluster
KMeans partition is far too coarse to absorb it.

This script asks the sharper question: does the sharing survive REGRESSING THE
LEXICAL SIGNAL OUT of each feature's per-document activation vector?

METHOD
  1. Pooled SAE features Phi_a, Phi_b [N x n_feat] on the SAME C4 docs -- reusing
     compute_pooled_features/topk_features from cross_model_sae_match.py verbatim
     (identical extraction path to the covctl receipt). Cached to .npz so the
     k-sweep never re-runs the GPU.
  2. Lexical design matrix L = TruncatedSVD_k( TF-IDF(max_features=4000,
     stop_words="english") ) -- the same TF-IDF config build_strata() already uses.
  3. CROSS-FITTED residualization: KFold(5, shuffle, seed) over documents; for each
     fold fit multi-output OLS L[train] -> Phi[train], predict the held-out fold,
     residual = actual - predicted. The SAME folds for both models. Cross-fitting
     is essential: in-sample OLS would over-remove signal and bias the test toward
     killing the effect.
  4. Re-run the unchanged mutual-NN statistic + 200-perm global null ON RESIDUALS.

CONTROLS
  raw               : no residualization (must reproduce the covctl baseline)
  lexical           : residualize on real L
  lexical_shuffled  : residualize on ROW-PERMUTED L (same k, same spectrum, doc
                      correspondence destroyed). CALIBRATION -- projecting out ANY
                      k-dim subspace removes some signal, so a drop under `lexical`
                      is only evidence of a lexical explanation if it is much
                      larger than the drop under `lexical_shuffled`.

Interpretation: excess survives with large z -> sharing is not merely shared
unigram detectors. Excess collapses toward 0 while lexical_shuffled does not ->
the cross-model "universality" is substantially shared lexical statistics.

Usage (GPU, one job does the whole k-sweep after a single extraction):
  python lexical_residual_null.py \
    --model_a google/gemma-2-2b --sae_a_release gemma-scope-2b-pt-res-canonical \
    --sae_a_id layer_12/width_16k/canonical \
    --model_b mistralai/Mistral-7B-v0.1 --sae_b_release mistral-7b-res-wg \
    --sae_b_id blocks.16.hook_resid_pre \
    --prompts data/c4_slice.jsonl --lang en --topk 1500 --n_perm 200 \
    --svd_k 64,128,256 --output results/cross_model_sae/lexres_c4_gemma2-2b__mistral7b.json
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_model_sae_match import compute_pooled_features, topk_features, _zscore_cols
from c4_covariate_null import null_under


def build_lexical(texts, k, seed, max_features=4000):
    """L = TruncatedSVD_k(TF-IDF(texts)), fit on ALL texts.

    GLOBAL-FIT variant only. The representation sees the held-out docs' text, so it
    is NOT cross-fitted; used solely for the explicitly-labelled `lexical_globalfit`
    sensitivity arm. The headline arm uses fold_lexical() below.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    X = TfidfVectorizer(max_features=max_features, stop_words="english").fit_transform(texts)
    k_eff = min(k, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=k_eff, random_state=seed)
    L = svd.fit_transform(X)
    return np.asarray(L, np.float64), k_eff, float(svd.explained_variance_ratio_.sum())


def fold_lexical(texts_tr, texts_te, k, seed, max_features=4000):
    """Fit TF-IDF+SVD on TRAINING-fold texts only; apply to the held-out fold.

    The nuisance REPRESENTATION is cross-fitted, not just the regression: the
    vectorizer vocabulary/idf and the SVD basis never see held-out documents.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    Xtr = vec.fit_transform(texts_tr)
    Xte = vec.transform(texts_te)
    k_eff = min(k, Xtr.shape[1] - 1, Xtr.shape[0] - 1)
    svd = TruncatedSVD(n_components=k_eff, random_state=seed)
    Ltr = np.asarray(svd.fit_transform(Xtr), np.float64)
    Lte = np.asarray(svd.transform(Xte), np.float64)
    return Ltr, Lte, k_eff, float(svd.explained_variance_ratio_.sum())


def crossfit_residualize(P, texts, k, n_folds, seed, shuffle_L=False,
                         L_global=None, max_features=4000):
    """Out-of-fold residuals of every column of P regressed on a lexical design.

    L_global=None  -> representation fitted INSIDE each training fold (headline).
    L_global=array -> use a precomputed globally-fitted L (sensitivity arm).
    shuffle_L=True -> row-permute L within train and within test independently:
                      same k, same spectrum, doc<->lexical correspondence destroyed.

    Returns (residuals [N x F], per-feature OOF R^2 array [F], fold_ids [N], k_eff, evr).
    fold_ids is REQUIRED by the caller: the null must permute WITHIN fold, because
    each fold's residuals carry that fold's own regression idiosyncrasy, and both
    models share the folds -- a global permutation would destroy fold alignment that
    the observed statistic enjoys, letting fold structure masquerade as sharing.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    P = np.asarray(P, np.float64)
    R = np.empty_like(P)
    fold_ids = np.empty(P.shape[0], dtype=int)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    k_eff, evr = k, float("nan")
    for fi, (tr, te) in enumerate(kf.split(P)):
        fold_ids[te] = fi
        if L_global is not None:
            Ltr, Lte = L_global[tr], L_global[te]
        else:
            Ltr, Lte, k_eff, evr = fold_lexical([texts[i] for i in tr], [texts[i] for i in te],
                                                k, seed, max_features)
        if shuffle_L:
            rs = np.random.default_rng(seed + 1000 + fi)
            Ltr = Ltr[rs.permutation(Ltr.shape[0])]
            Lte = Lte[rs.permutation(Lte.shape[0])]
        reg = LinearRegression(fit_intercept=True).fit(Ltr, P[tr])
        R[te] = P[te] - reg.predict(Lte)
    ss_res = (R ** 2).sum(0)
    ss_tot = ((P - P.mean(0, keepdims=True)) ** 2).sum(0)
    ok = ss_tot > 1e-12
    r2 = np.full(P.shape[1], np.nan)
    r2[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return R, r2[ok], fold_ids, k_eff, evr


def crossfit_ridge_residualize(P, texts, n_folds, seed, alpha,
                               max_features=20000, min_df=2):
    """POST-HOC arm. Residualize on the FULL TF-IDF space via cross-fitted ridge.

    Why this exists: the pre-registered TF-IDF->SVD_k design came back with a median
    out-of-fold R^2 of ~0.00 on Mistral (only 32-48% of features even positive), i.e.
    the lexical proxy removed almost NOTHING, so its "sharing survives" verdict is
    vacuous rather than evidence. Truncating to <=256 SVD components discards exactly
    the fine-grained rare-word detail that a shared *unigram detector* would fire on
    (the k=256 basis captures only 37% of the TF-IDF variance).

    Ridge on the untruncated vocabulary can represent specific-word detectors, so it
    is a strictly more powerful test of the lexical hypothesis. alpha is selected by
    out-of-fold R^2 -- a LEXICAL-FIT criterion that never looks at the sharing
    statistic -- so this is not outcome-tuning; every alpha is reported regardless.
    """
    from sklearn.linear_model import Ridge
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import KFold
    P = np.asarray(P, np.float64)
    R = np.empty_like(P)
    fold_ids = np.empty(P.shape[0], dtype=int)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    vocab_sizes = []
    for fi, (tr, te) in enumerate(kf.split(P)):
        fold_ids[te] = fi
        vec = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=min_df)
        Xtr = vec.fit_transform([texts[i] for i in tr]).toarray()
        Xte = vec.transform([texts[i] for i in te]).toarray()
        vocab_sizes.append(Xtr.shape[1])
        reg = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky").fit(Xtr, P[tr])
        R[te] = P[te] - reg.predict(Xte)
        del Xtr, Xte
    ss_res = (R ** 2).sum(0)
    ss_tot = ((P - P.mean(0, keepdims=True)) ** 2).sum(0)
    ok = ss_tot > 1e-12
    r2 = np.full(P.shape[1], np.nan)
    r2[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return R, r2[ok], fold_ids, int(np.mean(vocab_sizes))


def r2_summary(r2):
    """OOF R^2 DISTRIBUTION across features -- a near-zero R^2 would mean the
    residualization removed nothing and any 'survives' verdict would be vacuous."""
    r = np.asarray(r2, float); r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0}
    return {"n": int(r.size), "mean": float(r.mean()), "median": float(np.median(r)),
            "q25": float(np.percentile(r, 25)), "q75": float(np.percentile(r, 75)),
            "min": float(r.min()), "max": float(r.max()),
            "frac_positive": float((r > 0).mean())}


def run_null(Pa, Pb, n_perm, seed, topk, strata=None):
    """z-score -> mutual-NN + permutation null. Mirrors the covctl path exactly.

    strata=None      -> global permutation (matches the covctl receipt; correct for
                        the un-residualized `raw` arm, which has no fold structure).
    strata=fold_ids  -> permute WITHIN fold, so the null reproduces the fold-block
                        structure that cross-fitted residuals impose on BOTH models.
    """
    Za, _ = _zscore_cols(Pa); Zb, _ = _zscore_cols(Pb)
    N = min(Za.shape[0], Zb.shape[0]); Za, Zb = Za[:N], Zb[:N]
    st = None if strata is None else np.asarray(strata)[:N]
    r = null_under(Za, Zb, st, n_perm, seed)
    r["Ka"] = int(Za.shape[1]); r["Kb"] = int(Zb.shape[1])
    r["null_type"] = "global_perm" if strata is None else "within_fold_perm"
    # excess as a fraction of the nominal pool, so it is comparable to the receipt
    r["mutual_excess_frac"] = float(r["mutual_excess"] / topk)
    return r


# ----------------------------- self-test (no GPU) -----------------------------
def selftest():
    """Synthetic two-sided power check: the test must be able to return EITHER verdict.

    CASE 1 (pure lexical): both models' features are deterministic functions of L.
      -> residualizing on L MUST destroy the sharing (excess -> ~0).
    CASE 2 (lexical + shared latent orthogonal to L): a genuine non-lexical shared
      factor on top of the lexical drive.
      -> residualizing on L MUST preserve the sharing (excess stays large).

    If CASE 2 collapsed too, the method would kill everything and a collapse on real
    data would be a method artefact rather than a finding. Both must pass.
    """
    rng = np.random.default_rng(0)
    N, K, k = 400, 120, 16
    L = rng.normal(size=(N, k))
    Wa = rng.normal(size=(k, K)); Wb = rng.normal(size=(k, K))

    def resid(P, Lg):   # synthetic arms pass an explicit design matrix
        R, r2, fid, _, _ = crossfit_residualize(P, None, k, 5, 0, L_global=Lg)
        return R, r2, fid

    Pa1 = L @ Wa + 0.1 * rng.normal(size=(N, K))
    Pb1 = L @ Wb + 0.1 * rng.normal(size=(N, K))
    raw1 = run_null(Pa1, Pb1, 100, 0, K)
    Ra1, r2a, fid = resid(Pa1, L); Rb1, r2b, _ = resid(Pb1, L)
    res1 = run_null(Ra1, Rb1, 100, 0, K, strata=fid)
    print(f"CASE1 pure-lexical : raw excess={raw1['mutual_excess']:+7.1f} z={raw1['mutual_z']:6.1f}"
          f" -> resid excess={res1['mutual_excess']:+7.1f} z={res1['mutual_z']:6.1f}"
          f" (oofR2 med {np.median(r2a):.2f}/{np.median(r2b):.2f})")

    S = rng.normal(size=(N, 8))
    S = S - L @ np.linalg.lstsq(L, S, rcond=None)[0]      # make S orthogonal to L
    Va = rng.normal(size=(8, K)); Vb = rng.normal(size=(8, K))
    Pa2 = L @ Wa + 3.0 * (S @ Va) + 0.1 * rng.normal(size=(N, K))
    Pb2 = L @ Wb + 3.0 * (S @ Vb) + 0.1 * rng.normal(size=(N, K))
    raw2 = run_null(Pa2, Pb2, 100, 0, K)
    Ra2, _, fid2 = resid(Pa2, L); Rb2, _, _ = resid(Pb2, L)
    res2 = run_null(Ra2, Rb2, 100, 0, K, strata=fid2)
    print(f"CASE2 lex+shared   : raw excess={raw2['mutual_excess']:+7.1f} z={raw2['mutual_z']:6.1f}"
          f" -> resid excess={res2['mutual_excess']:+7.1f} z={res2['mutual_z']:6.1f}")

    # CASE 3 -- the fold-alignment leak (audit defect 1) and its fix.
    # Pa3, Pb3 are INDEPENDENT noise: there is NO cross-model sharing whatsoever, so
    # the honest excess is 0. Cross-fitting on shared folds still stamps each fold's
    # own regression idiosyncrasy into BOTH models' residuals. A global permutation
    # destroys that fold alignment -> spurious excess. Within-fold permutation
    # reproduces it -> excess returns to ~0.
    Pa3 = rng.normal(size=(N, K)); Pb3 = rng.normal(size=(N, K))
    Ra3, _, fid3 = resid(Pa3, L); Rb3, _, _ = resid(Pb3, L)
    leak = run_null(Ra3, Rb3, 100, 0, K)                    # buggy: global perm
    fixed = run_null(Ra3, Rb3, 100, 0, K, strata=fid3)      # fixed: within-fold perm
    print(f"CASE3 no-sharing   : global-perm null excess={leak['mutual_excess']:+7.1f} z={leak['mutual_z']:6.1f}"
          f"  | within-fold null excess={fixed['mutual_excess']:+7.1f} z={fixed['mutual_z']:6.1f}")

    ok1 = res1["mutual_excess"] < 0.25 * raw1["mutual_excess"]
    ok2 = res2["mutual_excess"] > 0.5 * raw2["mutual_excess"]
    ok3 = abs(fixed["mutual_z"]) < 3.0
    print(f"\nPASS case1 (lexical sharing destroyed)        = {ok1}")
    print(f"PASS case2 (real sharing preserved)          = {ok2}")
    print(f"PASS case3 (within-fold null kills fold leak) = {ok3}")
    ok = ok1 and ok2 and ok3
    print("VERDICT:", "pipeline discriminates correctly" if ok else "PIPELINE BROKEN")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True); ap.add_argument("--sae_a_release", required=True); ap.add_argument("--sae_a_id", required=True)
    ap.add_argument("--model_b", required=True); ap.add_argument("--sae_b_release", required=True); ap.add_argument("--sae_b_id", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--lang", default="en")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--svd_k", default="64,128,256", help="comma-separated SVD component counts")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--tfidf_max_features", type=int, default=4000)
    ap.add_argument("--ridge_alphas", default="", help="post-hoc ridge-on-full-TFIDF arm, e.g. 1,10,100,1000")
    ap.add_argument("--ridge_max_features", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="npz path to cache pooled features")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.svd_k.split(",")]

    prompts = [json.loads(l) for l in open(args.prompts)]
    # reproduce compute_pooled_features ordering so texts[i] is Phi row i
    sel = [p for p in prompts if args.lang is None or p.get("lang") == args.lang]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    if args.max_prompts: sel = sel[:args.max_prompts]
    texts = [p["prompt"] for p in sel]

    if args.cache and os.path.exists(args.cache):
        print(f"[cache] loading pooled features <- {args.cache}", flush=True)
        z = np.load(args.cache); Pa, Pb = z["Pa"], z["Pb"]
    else:
        print(f"[extract] pooled SAE features ({len(texts)} docs)", flush=True)
        Pa = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id, prompts, lang=args.lang, max_prompts=args.max_prompts)
        Pb = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id, prompts, lang=args.lang, max_prompts=args.max_prompts)
        if args.cache:
            Path(os.path.dirname(args.cache)).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cache, Pa=Pa, Pb=Pb)
            print(f"[cache] saved -> {args.cache}", flush=True)

    n = min(Pa.shape[0], Pb.shape[0], len(texts))
    Pa, Pb, texts = Pa[:n], Pb[:n], texts[:n]
    Pa, _ = topk_features(Pa, args.topk); Pb, _ = topk_features(Pb, args.topk)
    print(f"[data] N={n} Pa={Pa.shape} Pb={Pb.shape}", flush=True)

    res = {"cell": os.path.basename(args.output), "model_a": args.model_a, "model_b": args.model_b,
           "sae_a": f"{args.sae_a_release}::{args.sae_a_id}", "sae_b": f"{args.sae_b_release}::{args.sae_b_id}",
           "N_prompts": int(n), "topk": args.topk, "n_perm": args.n_perm,
           "n_folds": args.n_folds, "svd_k_grid": ks, "seed": args.seed,
           "tfidf_max_features": args.tfidf_max_features}

    # POSITIVE CONTROL: the un-residualized arm must reproduce the covctl receipt
    # (obs 689, null 469.8, excess +219.2, z=22.3). If it does not, the extraction
    # or statistic has drifted and NOTHING downstream is trustworthy.
    print("[null] raw (no residualization) -- positive control vs covctl receipt", flush=True)
    res["raw"] = run_null(Pa, Pb, args.n_perm, args.seed, args.topk)
    res["baseline_receipt"] = {"obs_mutual_nn": 689, "null_mutual_mean": 469.8,
                               "mutual_excess": 219.2, "mutual_z": 22.3,
                               "source": "results/cross_model_sae/covctl_c4_gemma2-2b__mistral7b.json"}
    if args.max_prompts is None and args.topk == 1500:
        d_obs = abs(res["raw"]["obs_mutual_nn"] - 689)
        d_exc = abs(res["raw"]["mutual_excess"] - 219.2)
        res["baseline_reproduced"] = bool(d_obs <= 5 and d_exc <= 15.0)
        print(f"[control] raw obs={res['raw']['obs_mutual_nn']} (receipt 689, d={d_obs}) "
              f"excess={res['raw']['mutual_excess']:+.1f} (receipt +219.2, d={d_exc:.1f}) "
              f"-> reproduced={res['baseline_reproduced']}", flush=True)
        if not res["baseline_reproduced"]:
            print("[control][WARN] baseline NOT reproduced -- downstream numbers are suspect", flush=True)

    res["lexical"] = {}; res["lexical_shuffled"] = {}; res["lexical_globalfit"] = {}
    for k in ks:
        # ---- headline arm: representation fitted INSIDE training folds ----
        Ra, r2a, fid, k_eff, evr = crossfit_residualize(Pa, texts, k, args.n_folds, args.seed,
                                                        max_features=args.tfidf_max_features)
        Rb, r2b, fidb, _, _ = crossfit_residualize(Pb, texts, k, args.n_folds, args.seed,
                                                   max_features=args.tfidf_max_features)
        assert np.array_equal(fid, fidb), "both models must share folds"
        print(f"[lex] k={k} (eff {k_eff}) per-fold tfidf-SVD explained_var={evr:.3f}", flush=True)
        r = run_null(Ra, Rb, args.n_perm, args.seed, args.topk, strata=fid)
        r.update({"svd_k": k, "svd_k_eff": k_eff, "svd_explained_var": evr,
                  "oof_r2_a": r2_summary(r2a), "oof_r2_b": r2_summary(r2b)})
        # quantify the fold-alignment leak ON REAL DATA: the same residuals judged by
        # the (incorrect) global permutation. Difference = size of audit defect 1 here.
        rgp = run_null(Ra, Rb, args.n_perm, args.seed, args.topk, strata=None)
        r["fold_leak_check_global_perm"] = {"mutual_excess": rgp["mutual_excess"],
                                            "mutual_z": rgp["mutual_z"],
                                            "excess_inflation_vs_within_fold":
                                                float(rgp["mutual_excess"] - r["mutual_excess"])}
        res["lexical"][str(k)] = r
        print(f"  [lexical k={k}] obs={r['obs_mutual_nn']} null={r['null_mutual_mean']:.1f}"
              f" excess={r['mutual_excess']:+.1f} z={r['mutual_z']:.1f}"
              f" | oofR2 med a={r2_summary(r2a)['median']:.3f} b={r2_summary(r2b)['median']:.3f}", flush=True)

        # ---- calibration: same k-dim subspace removed, doc<->lexical link destroyed ----
        Sa, s2a, sfid, _, _ = crossfit_residualize(Pa, texts, k, args.n_folds, args.seed,
                                                   shuffle_L=True, max_features=args.tfidf_max_features)
        Sb, s2b, _, _, _ = crossfit_residualize(Pb, texts, k, args.n_folds, args.seed,
                                                shuffle_L=True, max_features=args.tfidf_max_features)
        rs = run_null(Sa, Sb, args.n_perm, args.seed, args.topk, strata=sfid)
        rs.update({"svd_k": k, "svd_k_eff": k_eff,
                   "oof_r2_a": r2_summary(s2a), "oof_r2_b": r2_summary(s2b)})
        res["lexical_shuffled"][str(k)] = rs
        print(f"  [shuffled k={k}] obs={rs['obs_mutual_nn']} null={rs['null_mutual_mean']:.1f}"
              f" excess={rs['mutual_excess']:+.1f} z={rs['mutual_z']:.1f}", flush=True)

        # ---- SENSITIVITY (explicitly labelled, NOT the headline): globally-fitted
        # representation. TF-IDF+SVD sees held-out docs' text -> representation not
        # cross-fitted. Reported only to quantify how much that choice matters.
        Lg, kg, evg = build_lexical(texts, k, args.seed, args.tfidf_max_features)
        Ga, g2a, gfid, _, _ = crossfit_residualize(Pa, None, k, args.n_folds, args.seed, L_global=Lg)
        Gb, g2b, _, _, _ = crossfit_residualize(Pb, None, k, args.n_folds, args.seed, L_global=Lg)
        rg = run_null(Ga, Gb, args.n_perm, args.seed, args.topk, strata=gfid)
        rg.update({"svd_k": k, "svd_k_eff": kg, "svd_explained_var": evg,
                   "oof_r2_a": r2_summary(g2a), "oof_r2_b": r2_summary(g2b),
                   "NOTE": "global-fit representation (not cross-fitted) -- sensitivity only"})
        res["lexical_globalfit"][str(k)] = rg
        print(f"  [globalfit k={k}] obs={rg['obs_mutual_nn']} null={rg['null_mutual_mean']:.1f}"
              f" excess={rg['mutual_excess']:+.1f} z={rg['mutual_z']:.1f}  (sensitivity)", flush=True)

    # ---- POST-HOC arm (NOT pre-registered): ridge on the untruncated TF-IDF ----
    # Added because the pre-registered SVD arm proved under-powered (median OOF R^2
    # ~0.00 on model B). Reported for every alpha; alpha selected by OOF R^2 only.
    res["lexical_ridge_fulltfidf"] = {}
    if args.ridge_alphas:
        for a in [float(x) for x in args.ridge_alphas.split(",")]:
            Ra, r2a, fid, vsz = crossfit_ridge_residualize(Pa, texts, args.n_folds, args.seed, a,
                                                           args.ridge_max_features)
            Rb, r2b, _, _ = crossfit_ridge_residualize(Pb, texts, args.n_folds, args.seed, a,
                                                       args.ridge_max_features)
            r = run_null(Ra, Rb, args.n_perm, args.seed, args.topk, strata=fid)
            r.update({"alpha": a, "vocab_size": vsz,
                      "oof_r2_a": r2_summary(r2a), "oof_r2_b": r2_summary(r2b),
                      "NOTE": "POST-HOC arm, not pre-registered; added after the SVD arm's "
                              "OOF R^2 showed it removed ~no lexical signal"})
            res["lexical_ridge_fulltfidf"][str(a)] = r
            print(f"  [ridge alpha={a:g} V={vsz}] obs={r['obs_mutual_nn']} null={r['null_mutual_mean']:.1f}"
                  f" excess={r['mutual_excess']:+.1f} z={r['mutual_z']:.1f}"
                  f" | oofR2 med a={r2_summary(r2a)['median']:+.3f} b={r2_summary(r2b)['median']:+.3f}", flush=True)

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)

    print("\n==== SUMMARY (mutual-NN, excess over 200-perm null) ====")
    b = res["raw"]
    print(f"  {'raw (no resid)':24s} obs={b['obs_mutual_nn']:4d} null={b['null_mutual_mean']:7.1f}+-{b['null_mutual_sd']:4.1f}"
          f" excess={b['mutual_excess']:+7.1f} ({b['mutual_excess_frac']*100:+5.1f}%) z={b['mutual_z']:6.1f}")
    print(f"  {'covctl receipt':24s} obs= 689 null=  469.8+- 9.8 excess= +219.2 (+14.6%) z=  22.3   <- must match `raw`")
    for k in ks:
        for tag in ("lexical", "lexical_shuffled", "lexical_globalfit"):
            r = res[tag][str(k)]
            keep = r["mutual_excess"] / b["mutual_excess"] if b["mutual_excess"] else float("nan")
            r2m = r["oof_r2_a"].get("median", float("nan"))
            print(f"  {tag+' k='+str(k):24s} obs={r['obs_mutual_nn']:4d} null={r['null_mutual_mean']:7.1f}+-{r['null_mutual_sd']:4.1f}"
                  f" excess={r['mutual_excess']:+7.1f} ({r['mutual_excess_frac']*100:+5.1f}%) z={r['mutual_z']:6.1f}"
                  f" retained={keep*100:6.1f}% oofR2med={r2m:+.3f}")
    for a, r in res.get("lexical_ridge_fulltfidf", {}).items():
        keep = r["mutual_excess"] / b["mutual_excess"] if b["mutual_excess"] else float("nan")
        print(f"  {'ridge-fullTFIDF a='+a:24s} obs={r['obs_mutual_nn']:4d} null={r['null_mutual_mean']:7.1f}+-{r['null_mutual_sd']:4.1f}"
              f" excess={r['mutual_excess']:+7.1f} ({r['mutual_excess_frac']*100:+5.1f}%) z={r['mutual_z']:6.1f}"
              f" retained={keep*100:6.1f}% oofR2med={r['oof_r2_a']['median']:+.3f}  [POST-HOC]")
    print("\n==== OOF R^2 DISTRIBUTION across features -- how much lexical signal was REMOVED ====")
    print("  (near-zero R^2 => residualization did nothing => a 'survives' verdict is VACUOUS)")
    for k in ks:
        for m in ("a", "b"):
            s = res["lexical"][str(k)][f"oof_r2_{m}"]
            print(f"  SVD  k={k:3d} model {m.upper()}: n={s['n']:4d} median={s['median']:+.3f}"
                  f" IQR=[{s['q25']:+.3f},{s['q75']:+.3f}] min={s['min']:+.3f} max={s['max']:+.3f}"
                  f" frac>0={s['frac_positive']:.2f}")
    for a, r in res.get("lexical_ridge_fulltfidf", {}).items():
        for m in ("a", "b"):
            s = r[f"oof_r2_{m}"]
            print(f"  RIDGE a={a:>6s} model {m.upper()}: n={s['n']:4d} median={s['median']:+.3f}"
                  f" IQR=[{s['q25']:+.3f},{s['q75']:+.3f}] min={s['min']:+.3f} max={s['max']:+.3f}"
                  f" frac>0={s['frac_positive']:.2f}")
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
