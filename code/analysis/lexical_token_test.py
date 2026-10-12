#!/usr/bin/env python3
"""Does the cross-model sharing survive removing the LEXICAL/UNIGRAM signal?

Implements the design pre-registered in experiments_detail.md (commit 6848d4c) BEFORE
this file existed. Do not change the thresholds here without amending that entry; they are
duplicated as constants below so a reader can diff them against the pre-registration.

WHAT THIS FIXES relative to lexical_residual_null.py (the earlier, inconclusive run):
  1. FIRST-STAGE ADEQUACY GATE (hard). The earlier run's lexical model explained almost
     nothing (median OOF R^2 ~0.037 / ~0.013), so its "sharing survives" verdict was
     vacuous -- an intervention that removes nothing cannot exclude anything. Here the
     first stage must beat a SHUFFLED design of identical spectrum, per-feature, or the
     arm is marked adequacy_failed and the process exits non-zero.
  2. FROZEN ORIGINAL PAIRS. The earlier run rematched on residuals and compared totals, so
     a similar total recomputed from DIFFERENT pairs looked like survival. Here the
     mutual-NN pair set M is computed on the RAW features, frozen, recorded in the receipt,
     and the test asks whether THOSE pairs still co-activate.
  3. CALIBRATED RATIO. Projecting out any k-dim subspace removes some signal, so the
     primary statistic is R_lex = retention_real / retention_shuffled, not retention_real.
     (The same correction changed the verdict of the covariate control.)
  4. n_perm >= 2000, and a within-fold null -- cross-fitting stamps fold structure into
     BOTH models, and a global permutation would let fold alignment masquerade as sharing.

    python lexical_token_test.py --selftest      # no GPU, must pass before any real run

--pairing selects which pair set is frozen and tested (pre-registered at 3237ecb):
  mutual      reciprocal argmax (this paper's rule). DEFAULT; behaviour byte-identical
              to the pre-3237ecb file, which every committed lextok20k_* receipt used.
  lan_argmax  Lan et al. (arXiv:2410.06981) 1A-to-ManyB rule, quoted verbatim below.
              Transplanting a PUBLISHED pairing rule into this same control is the
              point: it asks how much of *its* excess survives lexical residualization.
Both pair sets are always computed and recorded, and the reimplementation validity
gate (G1/G2/G3) is always evaluated, whichever one is under test.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- pre-registered constants (commit 6848d4c) -------------------------------
GATE_FRAC_BETTER = 0.75   # A1: frac(R2_real > R2_shuffled) per feature, both models
GATE_MEDIAN_DR2 = 0.02    # A2: median(R2_real) - median(R2_shuffled), both models
SURVIVE_R = 0.70          # R_lex >= this in >=2 of 3 cells -> SURVIVES
COLLAPSE_R = 0.30         # R_lex <  this in >=2 of 3 cells -> COLLAPSES

# ---- pre-registered constants for the Lan-pairing arm (commit 3237ecb) -------
# Their rule, verbatim from arXiv:2410.06981v4 sec.3 "Highest Activation Correlation".
# Stored in every receipt so a reader can check the transplant against the source.
LAN_RULE_QUOTE = ("for each feature in SAE A, we find its highest correlated feature in "
                  "SAE B (as this allows for many-to-1 mappings, this is termed 1A-to-ManyB)")
LAN_RULE_SOURCE = "Lan et al., arXiv:2410.06981v4, Sec. 3 (Highest Activation Correlation)"
# G2 interval: 1-1 fraction implied by their Tables 5 and 6 (Pythia-70m x Pythia-160m,
# 55 layer pairs, 32768-feature SAEs) = Table6[i]/Table5[i]. min/max over all 55 cells.
LAN_T56_MIN, LAN_T56_MAX = 0.1593, 0.6324


def _zcols(P):
    P = np.asarray(P, np.float64)
    mu = P.mean(0, keepdims=True)
    sd = P.std(0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (P - mu) / sd


def mutual_nn_pairs(Za, Zb):
    """Frozen pair set M: reciprocal argmax over the correlation matrix."""
    C = (Za.T @ Zb) / Za.shape[0]
    a2b = C.argmax(1)
    b2a = C.argmax(0)
    ia = np.where(b2a[a2b] == np.arange(C.shape[0]))[0]
    jb = a2b[ia]
    return ia, jb, C


def lan_argmax_pairs(Za, Zb, C=None):
    """Lan et al.'s 1A-to-ManyB rule: EVERY A-feature takes its highest-correlated
    B-feature. Many-to-1 by construction -- that is the documented property, not a defect.
    Two lines, because their rule is two sentences (LAN_RULE_QUOTE)."""
    if C is None:
        C = (Za.T @ Zb) / Za.shape[0]
    return np.arange(C.shape[0]), C.argmax(1), C


PROC_RANK = 64      # registered a priori (PREREGISTRATION_procrustes_baseline.md)
PROC_SPLIT_SEED = 0


def procrustes_pairs(Za, Zb, train_rows):
    """Mutual-NN pairs in an ORTHOGONALLY ALIGNED coordinate system, fitted on a
    DISJOINT document split (PREREGISTRATION_procrustes_baseline.md).

    The rotation account of arXiv:2605.24577 predicts that if the cross-model
    correspondence is a rotation of a shared subspace, pairing after an orthogonal
    alignment should find it at least as well as raw correlation pairing. The
    alignment is fitted over the SHARED DOCUMENT axis, so it assumes no feature
    correspondence: with Ua, Ub the top-r left singular bases of the two models'
    train-half feature matrices, Q = U V^T from the SVD of Ub^T Ua is the
    closed-form orthogonal map taking B's document basis onto A's. Feature
    coordinates are each feature's train-half document vector projected onto those
    bases; B's are carried through Q; pairs are mutual NN by cosine there.

    Every quantity here comes from train_rows only. The caller evaluates the frozen
    pairs on the complementary rows.
    """
    A, B = Za[train_rows], Zb[train_rows]
    r = min(PROC_RANK, A.shape[1], B.shape[1], A.shape[0])
    Ua = np.linalg.svd(A, full_matrices=False)[0][:, :r]
    Ub = np.linalg.svd(B, full_matrices=False)[0][:, :r]
    u, _, vt = np.linalg.svd(Ub.T @ Ua)
    Q = u @ vt                                   # orthogonal, B-basis -> A-basis
    Ca = Ua.T @ A                                # r x K_A feature coordinates
    Cb = Q.T @ (Ub.T @ B)                        # r x K_B, aligned into A's frame
    Ca = Ca / (np.linalg.norm(Ca, axis=0, keepdims=True) + 1e-12)
    Cb = Cb / (np.linalg.norm(Cb, axis=0, keepdims=True) + 1e-12)
    C = Ca.T @ Cb                                # cosine in the aligned space
    a2b = C.argmax(1)
    b2a = C.argmax(0)
    ia = np.where(b2a[a2b] == np.arange(C.shape[0]))[0]
    return ia, a2b[ia], C


def split_rows(n, seed=PROC_SPLIT_SEED):
    """Fixed 50/50 document split. Registered; not swept."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def reimpl_gate(ia_lan, jb_lan, ia_mut, jb_mut, K_A, K_B):
    """Reimplementation validity gate, read BEFORE any R_lex (pre-registered at 3237ecb).

    G1  mapping is many-to-1                     (Lan sec.3 and App. J)
    G2  1-1 fraction inside Lan's Tables 5+6 interval
    G3  mutual-NN pair set STRICTLY nested in the argmax pair set (implementation
        correctness: a reciprocal rule is by definition a subset of the argmax rule, so a
        failure means one of the two is miscoded)

    G2 is NECESSARY, NOT SUFFICIENT and the receipt says so: an arbitrary argmax assignment
    gives 1-1 fraction -> (1-1/K_B)^(K_A-1) ~ 1/e, which lies inside Lan's interval. It
    rules out gross miscoding (a reciprocal or Hungarian rule scores exactly 1.000) and
    confirms the rule is of the right family; it does not certify numerical fidelity to
    their tables, which is unachievable from the paper (their tables are token-granularity,
    same-tokenizer, and post-filter). See the pre-registration for the full statement.
    """
    counts = np.bincount(np.asarray(jb_lan), minlength=K_B)
    frac11 = float((counts[jb_lan] == 1).mean())
    n_distinct = int(np.unique(jb_lan).size)
    mut = set(zip(np.asarray(ia_mut).tolist(), np.asarray(jb_mut).tolist()))
    lan = set(zip(np.asarray(ia_lan).tolist(), np.asarray(jb_lan).tolist()))
    g1 = bool(n_distinct < K_A)
    g2 = bool(LAN_T56_MIN <= frac11 <= LAN_T56_MAX)
    g3 = bool(mut.issubset(lan) and len(mut) < len(lan))
    return {
        "lan_rule_quote": LAN_RULE_QUOTE, "lan_rule_source": LAN_RULE_SOURCE,
        "K_A": int(K_A), "K_B": int(K_B),
        "n_pairs_argmax": int(len(lan)), "n_pairs_mutual": int(len(mut)),
        "n_distinct_targets": n_distinct,
        "frac_one_to_one": frac11,
        "one_over_e_reference": float((1.0 - 1.0 / K_B) ** (K_A - 1)),
        "lan_table56_interval": [LAN_T56_MIN, LAN_T56_MAX],
        "G1_many_to_one": g1, "G2_frac11_in_lan_interval": g2, "G3_mutual_nested": g3,
        "G2_is_necessary_not_sufficient": True,
        "passed": bool(g1 and g2 and g3),
    }


def pair_stat(Za, Zb, ia, jb, perm=None):
    """Mean correlation over the FROZEN pairs (i_m, j_m), optionally permuting B's rows."""
    B = Zb if perm is None else Zb[perm]
    A = Za[:, ia]
    Bs = B[:, jb]
    return float((A * Bs).sum(0).mean() / Za.shape[0])


def within_fold_perm(rng, fold_ids):
    perm = np.arange(len(fold_ids))
    for f in np.unique(fold_ids):
        idx = np.where(fold_ids == f)[0]
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def pair_null(Za, Zb, ia, jb, n_perm, seed, fold_ids=None):
    obs = pair_stat(Za, Zb, ia, jb)
    rng = np.random.default_rng(seed)
    N = Za.shape[0]
    vals = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(N) if fold_ids is None else within_fold_perm(rng, fold_ids)
        vals[t] = pair_stat(Za, Zb, ia, jb, perm)
    sd = vals.std(ddof=1)
    return {
        "n_pairs": int(len(ia)),
        "obs_mean_pair_corr": obs,
        "null_mean": float(vals.mean()), "null_sd": float(sd),
        "excess": float(obs - vals.mean()),
        "z": float((obs - vals.mean()) / sd) if sd > 1e-15 else float("inf"),
        "p": float((np.sum(vals >= obs) + 1) / (n_perm + 1)),
        "null_type": "global_perm" if fold_ids is None else "within_fold_perm",
    }


def fold_design(texts_tr, texts_te, k, seed, max_features):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    Xtr = vec.fit_transform(texts_tr)
    Xte = vec.transform(texts_te)
    k_eff = int(min(k, Xtr.shape[1] - 1, Xtr.shape[0] - 1))
    svd = TruncatedSVD(n_components=k_eff, random_state=seed)
    Ltr = np.asarray(svd.fit_transform(Xtr), np.float64)
    Lte = np.asarray(svd.transform(Xte), np.float64)
    return Ltr, Lte, k_eff, float(svd.explained_variance_ratio_.sum())


def kfold_indices(n, n_folds, seed):
    """Deterministic shuffled K-fold. Depends only on (n, n_folds, seed), so BOTH models
    and BOTH designs get identical folds -- required for the paired adequacy comparison
    and for the within-fold null to line up across models."""
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(np.arange(n), te, assume_unique=False), np.sort(te))
            for te in np.array_split(idx, n_folds)]


def _ols_fit_predict(Xtr, Ytr, Xte):
    """Multi-output OLS with intercept, numpy only (identical to sklearn's
    LinearRegression(fit_intercept=True)). Kept dependency-free so --selftest runs
    anywhere; sklearn is needed only to BUILD the TF-IDF/SVD design."""
    xm, ym = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
    beta, *_ = np.linalg.lstsq(Xtr - xm, Ytr - ym, rcond=None)
    return (Xte - xm) @ beta + ym


def residualize(P, texts, k, n_folds, seed, shuffle_design, max_features, L_global=None):
    """Cross-fitted residuals + per-feature OOF R^2 aligned to ALL columns (NaN if degenerate).

    Alignment matters: the adequacy gate is a PAIRED per-feature comparison between the
    real and shuffled designs, so both must return arrays of the same length in the same
    order. (The older helper dropped degenerate columns, which silently broke pairing.)
    """
    P = np.asarray(P, np.float64)
    R = np.empty_like(P)
    fold_ids = np.empty(P.shape[0], dtype=int)
    k_eff, evr = k, float("nan")
    for fi, (tr, te) in enumerate(kfold_indices(P.shape[0], n_folds, seed)):
        fold_ids[te] = fi
        if L_global is not None:
            Ltr, Lte = L_global[tr], L_global[te]
        else:
            Ltr, Lte, k_eff, evr = fold_design([texts[i] for i in tr], [texts[i] for i in te],
                                               k, seed, max_features)
        if shuffle_design:
            rs = np.random.default_rng(seed + 1000 + fi)
            Ltr = Ltr[rs.permutation(Ltr.shape[0])]
            Lte = Lte[rs.permutation(Lte.shape[0])]
        R[te] = P[te] - _ols_fit_predict(Ltr, P[tr], Lte)
    ss_res = (R ** 2).sum(0)
    ss_tot = ((P - P.mean(0, keepdims=True)) ** 2).sum(0)
    r2 = np.full(P.shape[1], np.nan)
    ok = ss_tot > 1e-12
    r2[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return R, r2, fold_ids, k_eff, evr


def adequacy(r2_real, r2_shuf, label):
    """Paired per-feature adequacy of the lexical first stage. Pre-registered thresholds."""
    m = np.isfinite(r2_real) & np.isfinite(r2_shuf)
    rr, rs = r2_real[m], r2_shuf[m]
    frac = float((rr > rs).mean()) if rr.size else 0.0
    dmed = float(np.median(rr) - np.median(rs)) if rr.size else 0.0
    passed = bool(frac >= GATE_FRAC_BETTER and dmed >= GATE_MEDIAN_DR2)
    return {"side": label, "n_features": int(rr.size),
            "median_r2_real": float(np.median(rr)) if rr.size else float("nan"),
            "median_r2_shuffled": float(np.median(rs)) if rr.size else float("nan"),
            "median_delta_r2": dmed, "frac_real_beats_shuffled": frac,
            "gate_frac_threshold": GATE_FRAC_BETTER, "gate_dr2_threshold": GATE_MEDIAN_DR2,
            "passed": passed}


# --------------------------------- self-test ---------------------------------
def selftest():
    """The test must be able to return EITHER verdict, and the gate must be able to fail."""
    # numpy 2.0.2 on this host emits spurious divide/overflow RuntimeWarnings from BLAS
    # matmul while returning correct finite values (verified separately). Silence only
    # that noise so a real numerical problem is still visible as a nan/inf in the results.
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    rng = np.random.default_rng(0)
    N, K, k = 600, 100, 12
    L = rng.normal(size=(N, k))
    Wa, Wb = rng.normal(size=(k, K)), rng.normal(size=(k, K))
    ok = {}

    def arms(Pa, Pb, pairing="mutual"):
        Za, Zb = _zcols(Pa), _zcols(Pb)
        ia, jb, _ = (lan_argmax_pairs if pairing == "lan_argmax" else mutual_nn_pairs)(Za, Zb)
        raw = pair_null(Za, Zb, ia, jb, 200, 0)
        out = {}
        for shuf in (False, True):
            Ra, r2a, fid, _, _ = residualize(Pa, None, k, 5, 0, shuf, 4000, L_global=L)
            Rb, r2b, _, _, _ = residualize(Pb, None, k, 5, 0, shuf, 4000, L_global=L)
            r = pair_null(_zcols(Ra), _zcols(Rb), ia, jb, 200, 0, fold_ids=fid)
            out["shuf" if shuf else "real"] = (r, r2a, r2b)
        ret_r = out["real"][0]["excess"] / raw["excess"]
        ret_s = out["shuf"][0]["excess"] / raw["excess"]
        return raw, out, ret_r, ret_s, (ret_r / ret_s if ret_s else float("nan"))

    # CASE 1 -- purely lexical sharing: R_lex must COLLAPSE.
    Pa = L @ Wa + 0.1 * rng.normal(size=(N, K))
    Pb = L @ Wb + 0.1 * rng.normal(size=(N, K))
    raw, out, rr, rs, R1 = arms(Pa, Pb)
    ok["case1_collapses"] = R1 < COLLAPSE_R
    print(f"CASE1 pure-lexical   raw_excess={raw['excess']:.4f} ret_real={rr:.3f} "
          f"ret_shuf={rs:.3f} R_lex={R1:.3f}  (want < {COLLAPSE_R})")

    # CASE 2 -- genuine shared latent orthogonal to L: R_lex must SURVIVE.
    S = rng.normal(size=(N, 8))
    S = S - L @ np.linalg.lstsq(L, S, rcond=None)[0]
    Va, Vb = rng.normal(size=(8, K)), rng.normal(size=(8, K))
    Pa2 = L @ Wa + 3.0 * (S @ Va) + 0.1 * rng.normal(size=(N, K))
    Pb2 = L @ Wb + 3.0 * (S @ Vb) + 0.1 * rng.normal(size=(N, K))
    raw2, out2, rr2, rs2, R2 = arms(Pa2, Pb2)
    ok["case2_survives"] = R2 >= SURVIVE_R
    print(f"CASE2 lex+shared     raw_excess={raw2['excess']:.4f} ret_real={rr2:.3f} "
          f"ret_shuf={rs2:.3f} R_lex={R2:.3f}  (want >= {SURVIVE_R})")

    # CASE 3 -- adequacy gate must PASS when the design is real and FAIL when the
    # activations are unrelated to it (an inert first stage must not license a verdict).
    Ra, r2a_real, _, _, _ = residualize(Pa, None, k, 5, 0, False, 4000, L_global=L)
    _, r2a_shuf, _, _, _ = residualize(Pa, None, k, 5, 0, True, 4000, L_global=L)
    g_good = adequacy(r2a_real, r2a_shuf, "informative")
    Pn = rng.normal(size=(N, K))          # independent of L
    _, r2n_real, _, _, _ = residualize(Pn, None, k, 5, 0, False, 4000, L_global=L)
    _, r2n_shuf, _, _, _ = residualize(Pn, None, k, 5, 0, True, 4000, L_global=L)
    g_bad = adequacy(r2n_real, r2n_shuf, "inert")
    ok["case3_gate_passes_informative"] = g_good["passed"]
    ok["case3_gate_fails_inert"] = not g_bad["passed"]
    print(f"CASE3 gate informative passed={g_good['passed']} "
          f"(frac={g_good['frac_real_beats_shuffled']:.2f} dR2={g_good['median_delta_r2']:.3f})")
    print(f"CASE3 gate inert       passed={g_bad['passed']} "
          f"(frac={g_bad['frac_real_beats_shuffled']:.2f} dR2={g_bad['median_delta_r2']:.3f})")

    # CASE 5 -- the SAME two-sided power check under Lan et al.'s 1A-to-ManyB pairing.
    # Load-bearing: without it, a collapse under their rule could be an artifact of the
    # pairing rule itself rather than of lexical structure. The rule must be able to
    # return BOTH verdicts too, or the arm cannot discriminate anything.
    _, _, _, _, R1L = arms(Pa, Pb, pairing="lan_argmax")
    _, _, _, _, R2L = arms(Pa2, Pb2, pairing="lan_argmax")
    ok["case5_lan_pairing_collapses_on_pure_lexical"] = R1L < COLLAPSE_R
    ok["case5_lan_pairing_survives_on_shared_latent"] = R2L >= SURVIVE_R
    print(f"CASE5 lan-argmax     pure-lexical R_lex={R1L:.3f} (want < {COLLAPSE_R}) | "
          f"lex+shared R_lex={R2L:.3f} (want >= {SURVIVE_R})")

    # CASE 7 -- the Procrustes-aligned pairing (PREREGISTRATION_procrustes_baseline.md)
    # must also return BOTH verdicts, and its alignment must be fitted on the TRAIN half
    # only. A pairing rule that cannot collapse on pure-lexical data would certify a
    # rotation artifact as survival, which is exactly the failure 2605.24577 documents.
    def arms_proc(Pa_, Pb_):
        Za_, Zb_ = _zcols(Pa_), _zcols(Pb_)
        tr, te = split_rows(Za_.shape[0])
        ia_, jb_, _ = procrustes_pairs(Za_, Zb_, tr)          # fit: train rows only
        Za_te, Zb_te = Za_[te], Zb_[te]                        # evaluate: held-out rows
        raw_ = pair_null(Za_te, Zb_te, ia_, jb_, 200, 0)
        got = {}
        for shuf_ in (False, True):
            Ra_, _, fid_, _, _ = residualize(Pa_[te], None, k, 5, 0, shuf_, 4000,
                                             L_global=L[te])
            Rb_, _, _, _, _ = residualize(Pb_[te], None, k, 5, 0, shuf_, 4000,
                                          L_global=L[te])
            got[shuf_] = pair_null(_zcols(Ra_), _zcols(Rb_), ia_, jb_, 200, 0,
                                   fold_ids=fid_)
        rr_ = got[False]["excess"] / raw_["excess"]
        rs_ = got[True]["excess"] / raw_["excess"]
        return rr_ / rs_ if rs_ else float("nan"), len(ia_)

    R1P, n1P = arms_proc(Pa, Pb)
    R2P, n2P = arms_proc(Pa2, Pb2)
    ok["case7_procrustes_collapses_on_pure_lexical"] = R1P < COLLAPSE_R
    ok["case7_procrustes_survives_on_shared_latent"] = R2P >= SURVIVE_R
    print(f"CASE7 procrustes     pure-lexical R_lex={R1P:.3f} (want < {COLLAPSE_R}) | "
          f"lex+shared R_lex={R2P:.3f} (want >= {SURVIVE_R})  pairs={n1P}/{n2P}")

    # CASE 6 -- the reimplementation validity gate must DISCRIMINATE, not rubber-stamp.
    # On the argmax rule it must pass; fed a reciprocal pair set as if it were the argmax
    # rule it must FAIL (frac_1-1 == 1.000, outside Lan's interval, and not strictly
    # nested). A gate that cannot fail would certify any pairing rule as "theirs".
    # Deliberately unstructured inputs: the gate is a STRUCTURAL check on the mapping, and
    # unstructured data isolates that structure (argmax targets ~ uniform, so frac_1-1
    # lands near the 1/e reference) instead of entangling it with planted signal.
    Zc, Zd = _zcols(rng.normal(size=(N, K))), _zcols(rng.normal(size=(N, K)))
    im, jm, Cc = mutual_nn_pairs(Zc, Zd)
    il, jl, _ = lan_argmax_pairs(Zc, Zd, C=Cc)
    g_lan = reimpl_gate(il, jl, im, jm, Zc.shape[1], Zd.shape[1])
    g_mut = reimpl_gate(im, jm, im, jm, Zc.shape[1], Zd.shape[1])   # reciprocal as if argmax
    ok["case6_gate_passes_argmax"] = g_lan["passed"]
    ok["case6_gate_fails_reciprocal"] = not g_mut["passed"]
    print(f"CASE6 gate argmax     passed={g_lan['passed']} "
          f"(frac_1-1={g_lan['frac_one_to_one']:.3f} n_distinct={g_lan['n_distinct_targets']} "
          f"1/e ref={g_lan['one_over_e_reference']:.3f})")
    print(f"CASE6 gate reciprocal passed={g_mut['passed']} "
          f"(frac_1-1={g_mut['frac_one_to_one']:.3f}) <- must be False")

    print()
    for kk, v in ok.items():
        print(f"PASS {kk} = {v}")
    good = all(ok.values())
    print("VERDICT:", "pipeline discriminates correctly" if good else "PIPELINE BROKEN")
    return 0 if good else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True); ap.add_argument("--sae_a_release", required=True); ap.add_argument("--sae_a_id", required=True)
    ap.add_argument("--model_b", required=True); ap.add_argument("--sae_b_release", required=True); ap.add_argument("--sae_b_id", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--lang", default="en")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--svd_k", default="256,1024,4096")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--tfidf_max_features", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pairing", choices=["mutual", "lan_argmax", "procrustes"],
                    default="mutual",
                    help="which pair set to freeze and test. 'mutual' (default) is this "
                         "paper's reciprocal rule and is byte-identical to the pre-3237ecb "
                         "behaviour; 'lan_argmax' is Lan et al.'s published 1A-to-ManyB rule.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from cross_model_sae_match import compute_pooled_features, topk_features
    ks = [int(x) for x in args.svd_k.split(",")]

    prompts = [json.loads(l) for l in open(args.prompts)]
    sel = [p for p in prompts if args.lang is None or p.get("lang") == args.lang]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    if args.max_prompts:
        sel = sel[:args.max_prompts]
    texts = [p["prompt"] for p in sel]

    if args.cache and os.path.exists(args.cache):
        print(f"[cache] load <- {args.cache}", flush=True)
        z = np.load(args.cache); Pa, Pb = z["Pa"], z["Pb"]
    else:
        print(f"[extract] {len(texts)} docs", flush=True)
        Pa = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id,
                                     prompts, lang=args.lang, max_prompts=args.max_prompts)
        Pb = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id,
                                     prompts, lang=args.lang, max_prompts=args.max_prompts)
        Pa, _ = topk_features(Pa, args.topk)
        Pb, _ = topk_features(Pb, args.topk)
        if args.cache:
            Path(os.path.dirname(args.cache)).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cache, Pa=Pa, Pb=Pb)
            print(f"[cache] saved -> {args.cache}", flush=True)
    if Pa.shape[1] > args.topk:
        Pa, _ = topk_features(Pa, args.topk)
    if Pb.shape[1] > args.topk:
        Pb, _ = topk_features(Pb, args.topk)

    n = min(Pa.shape[0], Pb.shape[0], len(texts))
    Pa, Pb, texts = Pa[:n], Pb[:n], texts[:n]
    print(f"[data] N={n} Pa={Pa.shape} Pb={Pb.shape}", flush=True)

    Za, Zb = _zcols(Pa), _zcols(Pb)
    # Both pair sets are always computed: the validity gate compares them, and recording
    # both makes the transplant auditable from the receipt alone.
    ia_m, jb_m, C = mutual_nn_pairs(Za, Zb)
    ia_l, jb_l, _ = lan_argmax_pairs(Za, Zb, C=C)
    gate = reimpl_gate(ia_l, jb_l, ia_m, jb_m, Za.shape[1], Zb.shape[1])
    print(f"[reimpl-gate] argmax pairs={gate['n_pairs_argmax']} mutual={gate['n_pairs_mutual']} "
          f"distinct targets={gate['n_distinct_targets']} frac_1-1={gate['frac_one_to_one']:.4f} "
          f"(Lan T5/T6 interval {LAN_T56_MIN}-{LAN_T56_MAX}, 1/e ref "
          f"{gate['one_over_e_reference']:.4f})", flush=True)
    print(f"[reimpl-gate] G1={gate['G1_many_to_one']} G2={gate['G2_frac11_in_lan_interval']} "
          f"G3={gate['G3_mutual_nested']} -> {'PASS' if gate['passed'] else 'FAIL'}", flush=True)

    # Branch 5: the gate is read BEFORE any R_lex exists, and a failure stops the arm here
    # rather than producing a number that would be attributed to Lan et al.'s rule.
    if args.pairing == "lan_argmax" and not gate["passed"]:
        Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
        json.dump({"cell": os.path.basename(args.output), "pairing": args.pairing,
                   "pairing_prereg_commit": "3237ecb", "reimpl_gate": gate,
                   "branch": "B5_reimplementation_does_not_reproduce", "arms": {}},
                  open(args.output, "w"), indent=2)
        print("[GATE] reimplementation validity gate FAILED -- Branch 5. No R_lex is "
              "computed or read; no comparison is licensed.", file=sys.stderr)
        return 3

    proc_meta = None
    if args.pairing == "procrustes":
        # Fit on the train half; from here on EVERY quantity -- raw excess, the
        # residualization, the null -- is computed on the held-out half only, so the
        # alignment can never see the data its pairs are scored on.
        tr, te = split_rows(Za.shape[0])
        ia, jb, _ = procrustes_pairs(Za, Zb, tr)
        Pa, Pb, texts = Pa[te], Pb[te], [texts[i] for i in te]
        Za, Zb = _zcols(Pa), _zcols(Pb)
        n = len(te)
        proc_meta = {"rank": PROC_RANK, "split_seed": PROC_SPLIT_SEED,
                     "n_train_docs": int(len(tr)), "n_eval_docs": int(len(te)),
                     "fit_on": "train half only", "evaluated_on": "held-out half"}
        print(f"[procrustes] alignment fitted on {len(tr)} docs (rank {PROC_RANK}); "
              f"everything below is the held-out {len(te)} docs", flush=True)
    else:
        ia, jb = (ia_l, jb_l) if args.pairing == "lan_argmax" else (ia_m, jb_m)
    print(f"[pairs] pairing={args.pairing}: frozen pair set M has {len(ia)} pairs", flush=True)
    raw = pair_null(Za, Zb, ia, jb, args.n_perm, args.seed)
    print(f"[raw]  obs={raw['obs_mean_pair_corr']:.4f} null={raw['null_mean']:.4f} "
          f"excess={raw['excess']:+.4f} z={raw['z']:.1f} p={raw['p']:.4f}", flush=True)

    res = {"cell": os.path.basename(args.output),
           "model_a": args.model_a, "model_b": args.model_b,
           "sae_a": f"{args.sae_a_release}::{args.sae_a_id}",
           "sae_b": f"{args.sae_b_release}::{args.sae_b_id}",
           "N_docs": int(n), "topk": args.topk, "n_perm": args.n_perm,
           "n_folds": args.n_folds, "svd_k_grid": ks, "seed": args.seed,
           "tfidf_max_features": args.tfidf_max_features,
           "prereg_commit": "6848d4c",
           "pairing": args.pairing,
           "pairing_prereg_commit": {"lan_argmax": "3237ecb",
                                      "procrustes": "3be9b7d"}.get(args.pairing),
           "procrustes": proc_meta,
           "reimpl_gate": gate,
           "gate_thresholds": {"frac_real_beats_shuffled": GATE_FRAC_BETTER,
                               "median_delta_r2": GATE_MEDIAN_DR2},
           "frozen_pairs": {"n": int(len(ia)),
                            "feat_a": [int(x) for x in ia],
                            "feat_b": [int(x) for x in jb]},
           "pairs_mutual": {"n": int(len(ia_m)), "feat_a": [int(x) for x in ia_m],
                            "feat_b": [int(x) for x in jb_m]},
           "pairs_lan_argmax": {"n": int(len(ia_l)), "feat_a": [int(x) for x in ia_l],
                                "feat_b": [int(x) for x in jb_l]},
           "raw": raw, "arms": {}}

    any_pass = False
    for k in ks:
        print(f"\n[k={k}] residualizing (real and shuffled designs)", flush=True)
        Ra, r2a_r, fid, k_eff, evr = residualize(Pa, texts, k, args.n_folds, args.seed,
                                                 False, args.tfidf_max_features)
        Rb, r2b_r, _, _, _ = residualize(Pb, texts, k, args.n_folds, args.seed,
                                         False, args.tfidf_max_features)
        Sa, r2a_s, _, _, _ = residualize(Pa, texts, k, args.n_folds, args.seed,
                                         True, args.tfidf_max_features)
        Sb, r2b_s, _, _, _ = residualize(Pb, texts, k, args.n_folds, args.seed,
                                         True, args.tfidf_max_features)

        ga = adequacy(r2a_r, r2a_s, "model_a")
        gb = adequacy(r2b_r, r2b_s, "model_b")
        gate_ok = ga["passed"] and gb["passed"]
        any_pass |= gate_ok
        print(f"[k={k}] adequacy A: frac={ga['frac_real_beats_shuffled']:.3f} "
              f"dR2={ga['median_delta_r2']:+.4f} pass={ga['passed']} | "
              f"B: frac={gb['frac_real_beats_shuffled']:.3f} "
              f"dR2={gb['median_delta_r2']:+.4f} pass={gb['passed']} -> GATE "
              f"{'PASS' if gate_ok else 'FAIL'}", flush=True)

        real = pair_null(_zcols(Ra), _zcols(Rb), ia, jb, args.n_perm, args.seed, fold_ids=fid)
        shuf = pair_null(_zcols(Sa), _zcols(Sb), ia, jb, args.n_perm, args.seed, fold_ids=fid)
        ret_r = real["excess"] / raw["excess"] if raw["excess"] else float("nan")
        ret_s = shuf["excess"] / raw["excess"] if raw["excess"] else float("nan")
        R_lex = ret_r / ret_s if ret_s not in (0.0, float("nan")) else float("nan")
        print(f"[k={k}] pair survival real excess={real['excess']:+.4f} z={real['z']:.1f} "
              f"p={real['p']:.4f} | shuffled excess={shuf['excess']:+.4f}", flush=True)
        print(f"[k={k}] retention real={ret_r:.3f} shuffled={ret_s:.3f}  R_lex={R_lex:.3f}",
              flush=True)

        res["arms"][str(k)] = {
            "svd_k_requested": k, "svd_k_effective": int(k_eff),
            "explained_variance_ratio": evr,
            "adequacy": {"model_a": ga, "model_b": gb, "gate_passed": gate_ok},
            "pair_survival_real": real, "pair_survival_shuffled": shuf,
            "retention_real": ret_r, "retention_shuffled": ret_s, "R_lex": R_lex,
            "interpretable": gate_ok,
        }

    res["any_arm_gate_passed"] = bool(any_pass)
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    # read back what we just wrote rather than trusting the write
    chk = json.load(open(args.output))
    assert chk["frozen_pairs"]["n"] == len(ia), "receipt read-back mismatch on frozen pairs"
    print(f"\n[done] -> {args.output} (read-back ok: {len(chk['arms'])} arms, "
          f"{chk['frozen_pairs']['n']} frozen pairs)")

    if not any_pass:
        print("[GATE] first-stage adequacy FAILED at every k -- nothing downstream is "
              "interpretable. Exiting non-zero per the pre-registered rule.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
