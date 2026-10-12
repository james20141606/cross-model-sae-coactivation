#!/usr/bin/env python3
"""Selection-clean token-aligned/order-sensitive rerun (AMENDMENT A3-EXEC).

Implements PREREGISTRATION_token_aligned.md AMENDMENT A3-EXEC (commit b0b07cd) EXACTLY.
The four A3 requirements, each visibly enforced and receipted:

  (a) DISJOINT selection/evaluation: 50/50 doc split by numpy default_rng(seed 42),
      registered before running. Pairs = verbatim lexical_token_test.mutual_nn_pairs on
      the SPLIT-A rows of the md5-pinned pooled caches (_zcols z-scoring, same candidate
      pool of top-1500 columns/side). Every statistic and every null draw uses SPLIT-B
      occurrences only. Split sha256s go into the receipt.
  (b) GENUINE joint document+type FE: float64 alternating projections (doc-centering,
      type-centering per sweep) until max |per-doc residual mean| AND max |per-type
      residual mean| < 1e-8 on both sides; max 500 sweeps; non-convergence is a hard
      error (no receipt). Sweep counts + final deviations receipted.
  (c) IDENTICAL residualization under every null draw: each N3 derangement permutes side
      B's RAW retained rows, then the FULL alternating-projection residualization is
      re-run on the permuted matrix (same tolerance) before the statistic.
      (N1 permutes within documents, which commutes exactly with per-doc centering, so
      re-centering per N1 draw is mathematically the identity -- stated in A3-EXEC E3.)
  (d) ONE common scored-pair mask: G2(S2) support AND residual sd > 1e-8 on both sides
      under BOTH the joint-FE and the doc-centering transformation; numerator and
      denominator of R_tok_clean are asserted to score exactly this mask;
      n_pairs_scored + mask sha256 written to the receipt.

Bars unchanged from the original registration (S1: p at floor AND r >= 0.05;
S2: R_tok_clean >= 0.30 + floor survive / < 0.10 or in-band collapse).
Omitted per A3-EXEC E6: bigram, max-pool, S2c arms.

--selftest (no GPU, no dumps) must show, on planted synthetic dumps run through the
FULL cell path (split -> selection on A -> G0 mapping -> evaluation on B):
  planted unigram pairs  -> S1 passes, S2-clean collapses;
  planted context pairs  -> S1 passes, S2-clean survives;
  planted doc-topic pairs-> S1 at null;
  the A3 counterexample docs [A,A,B]/[A,B,B], f(A)=1,f(B)=0 residualizes to EXACTLY 0
  under the joint FE (the sequential-centering defect visibly closed);
  + split determinism/disjointness, + mask-identity assertion, + requirement markers.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokalign_analysis import (                                   # committed machinery
    G0_MIN_PAIR_FRAC, G1_MAX_DOC_FAIL, G2_MIN_OCC, G2_MIN_PAIRS, N_PERM, S2_TYPE_MIN,
    _synth_acts, _synth_side, _dense_to_side, _t, align_side, build_occurrences,
    dump_integrity, g1_report, group_mean_center, load_side, null_summary, run_g0,
    within_group_derangement, within_group_perm, word_matrices, word_spans)
from lexical_token_test import _zcols, mutual_nn_pairs            # verbatim selection

# ---- A3-EXEC registered constants (PREREGISTRATION_token_aligned.md, b0b07cd) ----
PREREG = "df330c6"
A3EXEC = "b0b07cd"
SPLIT_SEED = 42
FE_TOL = 1e-8
FE_MAX_SWEEPS = 500
SD_MIN = 1e-8            # common-mask residual-sd floor (both transforms, both sides)


def doc_split(n_docs, seed=SPLIT_SEED):
    """Registered 50/50 split: selection on A, evaluation on B."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_docs)
    a = np.sort(perm[: n_docs // 2])
    b = np.sort(perm[n_docs // 2:])
    return a.astype(np.int64), b.astype(np.int64)


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ======================= joint doc+type FE (requirement b) =======================
def _group_absmean_max(X, g, n_groups):
    torch = _t()
    S = torch.zeros((n_groups, X.shape[1]), dtype=X.dtype, device=X.device)
    S.index_add_(0, g, X)
    c = torch.zeros(n_groups, dtype=X.dtype, device=X.device)
    c.index_add_(0, g, torch.ones_like(g, dtype=X.dtype))
    mean = S / torch.clamp(c, min=1).unsqueeze(1)
    return float(mean.abs().max())


def joint_fe(X, doc_ids, type_ids, n_docs, n_types, tol=FE_TOL,
             max_sweeps=FE_MAX_SWEEPS):
    """Alternating projections to the joint doc+type FE residual, float64.
    Returns (residual, n_sweeps, max_dev). Non-convergence -> hard error."""
    torch = _t()
    assert X.dtype == torch.float64, "joint FE is registered in float64"
    for sweep in range(1, max_sweeps + 1):
        X = group_mean_center(X, doc_ids, n_docs)
        X = group_mean_center(X, type_ids, n_types)
        dev = max(_group_absmean_max(X, doc_ids, n_docs),
                  _group_absmean_max(X, type_ids, n_types))
        if dev < tol:
            return X, sweep, dev
    raise AssertionError(
        f"joint FE did not converge in {max_sweeps} sweeps (max dev {dev:.3e} "
        f">= tol {tol:g}) -- registered hard error, no receipt")


# ========================= masked correlation (req. d) =========================
def masked_mean_r(Ar, Br, mask):
    """Mean per-pair Pearson r over EXACTLY `mask` (residual series, ~zero-mean).
    Norms clamped at 1e-12 (mask already excludes sd<SD_MIN columns)."""
    torch = _t()
    s = torch.zeros(Ar.shape[1], dtype=Ar.dtype, device=Ar.device)
    na = torch.zeros_like(s); nb = torch.zeros_like(s)
    for c0 in range(0, Ar.shape[1], 256):
        sl = slice(c0, c0 + 256)
        s[sl] = (Ar[:, sl] * Br[:, sl]).sum(0)
        na[sl] = torch.sqrt(torch.clamp((Ar[:, sl] ** 2).sum(0), min=0))
        nb[sl] = torch.sqrt(torch.clamp((Br[:, sl] ** 2).sum(0), min=0))
    r = s / torch.clamp(na * nb, min=1e-12)
    return float(r[mask].mean()), r


def col_sd(X):
    torch = _t()
    n = X.shape[0]
    return torch.sqrt(torch.clamp((X ** 2).sum(0), min=0) / max(n, 1))


# ============================== clean statistics ==============================
def run_clean_stats(Am, Bm, occ, cfg):
    """A3-EXEC statistics on SPLIT-B occurrence matrices [n_occ x P] (already
    filtered to split-B docs by the caller). Returns the stats sub-dict."""
    torch = _t()
    dev = cfg["device"]
    n_perm, seed = cfg["n_perm"], cfg["seed"]
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    A = torch.tensor(Am, device=dev); B = torch.tensor(Bm, device=dev)
    doc = torch.tensor(occ["doc"], device=dev)
    n_docs = int(doc.max()) + 1 if len(occ["doc"]) else 1
    res = {}

    # ---- G2(S1) support on split-B occurrences ----
    sup1 = ((A > 0) | (B > 0)).float().sum(0)
    mask1 = sup1 >= cfg["g2_min_occ"]
    res["g2_s1"] = {"n_pairs_in": int(A.shape[1]), "n_pairs_pass": int(mask1.sum()),
                    "min_occ": cfg["g2_min_occ"],
                    "testable": bool(int(mask1.sum()) >= cfg["g2_min_pairs"])}

    # ---- S1 vs N1 (doc-centering; N1 commutes with it, A3-EXEC E3) ----
    Ac = group_mean_center(A, doc, n_docs)
    Bc = group_mean_center(B, doc, n_docs)
    sd_ok1 = (col_sd(Ac) > SD_MIN) & (col_sd(Bc) > SD_MIN)
    m1 = mask1 & sd_ok1
    obs1, _ = masked_mean_r(Ac, Bc, m1)
    null1 = np.empty(n_perm)
    for t in range(n_perm):
        pi = within_group_perm(doc, gen, dev)
        null1[t], _ = masked_mean_r(Ac, Bc[pi], m1)
    res["s1"] = {"obs_mean_pair_r": obs1,
                 "null_n1": null_summary(obs1, null1, n_perm),
                 "excess": float(obs1 - null1.mean()),
                 "n_pairs_scored": int(m1.sum())}
    # ---- N2 direction check (unchanged, reported only) ----
    posbin = torch.tensor(np.minimum(
        (10 * occ["widx"].astype(np.int64)) // np.maximum(occ["nwords"], 1), 9),
        device=dev)
    null2 = np.empty(n_perm)
    for t in range(n_perm):
        pi = within_group_perm(posbin, gen, dev)
        null2[t], _ = masked_mean_r(Ac, Bc[pi], m1)
    res["s1"]["null_n2"] = null_summary(obs1, null2, n_perm)
    res["s1"]["excess_n2"] = float(obs1 - null2.mean())

    # ---- S2-clean: retained types (>= S2_TYPE_MIN within SPLIT-B occ, nonempty) ----
    tc = np.bincount(occ["type"], minlength=int(occ["type"].max()) + 1
                     if len(occ["type"]) else 1)
    names = occ["type_names"]
    nonempty = np.array([names[i] != "" for i in range(len(tc))])
    keep_type = (tc >= cfg["type_min"]) & nonempty
    ret = np.where(keep_type[occ["type"]])[0]
    ridx = torch.tensor(ret, device=dev)
    tr = torch.tensor(occ["type"][ret], device=dev)
    dr = doc[ridx]
    tr_u, tr_c = torch.unique(tr, return_inverse=True)
    res["s2_support"] = {"n_types_retained": int(keep_type.sum()),
                         "n_occ_retained": int(len(ret)),
                         "n_occ_total": int(A.shape[0])}
    Ar = A[ridx].double(); Br = B[ridx].double()          # float64 per E2
    sup2 = ((Ar > 0) | (Br > 0)).float().sum(0)
    g2_mask = sup2 >= cfg["g2_min_occ"]
    res["g2_s2"] = {"n_pairs_in": int(A.shape[1]), "n_pairs_pass": int(g2_mask.sum()),
                    "min_occ": cfg["g2_min_occ"],
                    "testable": bool(int(g2_mask.sum()) >= cfg["g2_min_pairs"])}

    # observed joint-FE residuals (requirement b)
    Afe, swA, devA = joint_fe(Ar, dr, tr_c, n_docs, len(tr_u))
    Bfe, swB, devB = joint_fe(Br, dr, tr_c, n_docs, len(tr_u))
    # doc-centering transform (denominator reference)
    Adc = group_mean_center(Ar, dr, n_docs)
    Bdc = group_mean_center(Br, dr, n_docs)

    # ---- ONE common scored-pair mask (requirement d) ----
    cmask = (g2_mask
             & (col_sd(Afe) > SD_MIN) & (col_sd(Bfe) > SD_MIN)
             & (col_sd(Adc) > SD_MIN) & (col_sd(Bdc) > SD_MIN))
    cmask_np = cmask.cpu().numpy().astype(np.uint8)
    mask_sha = _sha(cmask_np)

    def scored(mask):
        """Assert this statistic scores EXACTLY the common mask."""
        assert int(mask.sum()) == int(cmask.sum()) and bool((mask == cmask).all()), \
            "numerator/denominator scored-pair masks differ -- A3 requirement d"
        return mask

    # ---- S2-clean vs N3-clean (re-residualized per draw; requirement c) ----
    obs2, _ = masked_mean_r(Afe, Bfe, scored(cmask))
    null3 = np.empty(n_perm)
    sweeps = np.empty(n_perm, np.int32)
    for t in range(n_perm):
        pi = within_group_derangement(tr_c, gen, dev)
        Bp, sw, _ = joint_fe(Br[pi], dr, tr_c, n_docs, len(tr_u))
        sweeps[t] = sw
        null3[t], _ = masked_mean_r(Afe, Bp, scored(cmask))
    res["s2_clean"] = {
        "obs_mean_pair_r": obs2, "null_n3": null_summary(obs2, null3, n_perm),
        "excess": float(obs2 - null3.mean()),
        "fe_convergence": {"tol": FE_TOL, "max_sweeps": FE_MAX_SWEEPS,
                           "obs_sweeps_a": swA, "obs_sweeps_b": swB,
                           "obs_max_dev_a": devA, "obs_max_dev_b": devB,
                           "null_sweeps_mean": float(sweeps.mean()),
                           "null_sweeps_max": int(sweeps.max())},
        "null_reresidualized": True}

    # ---- denominator: S1 on the SAME mask & S2-retained occurrences vs N1 ----
    obs_d, _ = masked_mean_r(Adc, Bdc, scored(cmask))
    null_d = np.empty(n_perm)
    for t in range(n_perm):
        pi = within_group_perm(dr, gen, dev)
        null_d[t], _ = masked_mean_r(Adc, Bdc[pi], scored(cmask))
    res["s2_clean"]["s1_on_s2_support"] = {
        "obs_mean_pair_r": obs_d, "null_n1": null_summary(obs_d, null_d, n_perm),
        "excess": float(obs_d - null_d.mean())}
    num = res["s2_clean"]["excess"]
    den = res["s2_clean"]["s1_on_s2_support"]["excess"]
    res["s2_clean"]["R_tok_clean"] = float(num / den) if den > 0 else float("nan")
    den_n = res["s1"]["excess"]
    res["s2_clean"]["R_tok_clean_naive"] = (float(num / den_n) if den_n > 0
                                            else float("nan"))
    res["common_mask"] = {
        "n_pairs_scored": int(cmask.sum()), "sha256": mask_sha,
        "definition": ("G2(S2) support AND residual sd > 1e-8 both sides under BOTH "
                       "the joint-FE and the doc-centering transformation; asserted "
                       "identical for R_tok_clean numerator and denominator")}
    return res


# ============================== full cell driver ==============================
def run_cell(sa, sb, texts, Pa_c, Pb_c, meta, cfg):
    """Full A3-EXEC path for one cell: split -> selection(A) -> G0 -> G1 ->
    occurrences filtered to split B -> clean stats. Returns (receipt, exit_code)."""
    n_docs = len(texts)
    a_idx, b_idx = doc_split(n_docs)
    out = {"cell": f"tokaclean20k_{meta['tag']}.json", "tag": meta["tag"],
           "prereg_commit": PREREG, "a3_exec_commit": A3EXEC,
           "smoke": bool(meta.get("smoke", False)),
           "model_a": meta.get("model_a"), "model_b": meta.get("model_b"),
           "sae_a": meta.get("sae_a"), "sae_b": meta.get("sae_b"),
           "n_docs": int(n_docs), "n_perm": cfg["n_perm"], "seed": cfg["seed"],
           "device": cfg["device"],
           "split": {"seed": SPLIT_SEED, "n_a": int(len(a_idx)),
                     "n_b": int(len(b_idx)), "sha256_a": _sha(a_idx),
                     "sha256_b": _sha(b_idx)}}

    # ---- AMENDMENT A4(b): candidate universe screened on split A ONLY, over the
    # FULL dictionary carried by the dumps; selection then runs inside that universe.
    K = min(1500, sa["pooled"].shape[1], sb["pooled"].shape[1])
    var_a = np.asarray(sa["pooled"][a_idx], np.float64).var(axis=0)
    var_b = np.asarray(sb["pooled"][a_idx], np.float64).var(axis=0)
    univ_a = np.sort(np.argsort(-var_a)[:K]).astype(np.int64)
    univ_b = np.sort(np.argsort(-var_b)[:K]).astype(np.int64)
    Za = _zcols(np.asarray(sa["pooled"][np.ix_(a_idx, univ_a)], np.float64))
    Zb = _zcols(np.asarray(sb["pooled"][np.ix_(a_idx, univ_b)], np.float64))
    ia, jb, _ = mutual_nn_pairs(Za, Zb)
    feat_a4_a = univ_a[np.asarray(ia, np.int64)]
    feat_a4_b = univ_b[np.asarray(jb, np.int64)]
    out["selection"] = {"procedure": "A4: top-K-by-variance universe computed on split-A "
                                     "pooled rows of the FULL dictionary per side, then "
                                     "mutual_nn_pairs (_zcols) inside it",
                        "universe_k": int(K),
                        "n_selected_pairs": int(len(feat_a4_a))}

    # ---- G0 (full-corpus identity anchoring, A3-EXEC E1) ----
    # A4: the pairs are A-derived over dump feature ids directly. Dump-integrity is
    # the HARD gate; overlap with the frozen cached pairs is INFORMATIONAL (A4(b)).
    feat_a, feat_b = feat_a4_a, feat_a4_b
    ia_ = dump_integrity(sa, np.unique(feat_a), None)
    ib_ = dump_integrity(sb, np.unique(feat_b), None)
    g0_pass = bool(ia_["passed"] and ib_["passed"])
    pair_ok = np.ones(len(feat_a), bool)
    overlap = {"computed": False}
    if Pa_c is not None and Pb_c is not None:
        g0a, map_a, _, _ = run_g0(sa, Pa_c, np.arange(Pa_c.shape[1]), cfg["device"])
        g0b, map_b, _, _ = run_g0(sb, Pb_c, np.arange(Pb_c.shape[1]), cfg["device"])
        Zca = _zcols(Pa_c[a_idx]); Zcb = _zcols(Pb_c[a_idx])
        cia, cjb, _ = mutual_nn_pairs(Zca, Zcb)
        frozen_pairs = set(zip(map_a[np.asarray(cia, np.int64)].tolist(),
                               map_b[np.asarray(cjb, np.int64)].tolist()))
        new_pairs = set(zip(feat_a.tolist(), feat_b.tolist()))
        overlap = {"computed": True,
                   "n_frozen_style_pairs": len(frozen_pairs),
                   "n_a4_pairs": len(new_pairs),
                   "n_shared": len(frozen_pairs & new_pairs),
                   "note": "informational per A4(b); not gated"}
    print(f"[G0-A4] dump-integrity A={ia_['passed']} B={ib_['passed']} "
          f"pairs={len(feat_a)} overlap={overlap.get('n_shared', 'n/a')} "
          f"-> {'PASS' if g0_pass else 'FAIL'}", flush=True)
    out["gates"] = {"G0": {"dump_integrity_a": ia_, "dump_integrity_b": ib_,
                           "n_selected_pairs": int(len(feat_a)),
                           "overlap_with_frozen": overlap,
                           "passed": g0_pass}}

    # ---- G1 (unchanged, full corpus) ----
    alA = align_side(texts, sa["meta"]); alB = align_side(texts, sb["meta"])
    g1a = g1_report(texts, sa["meta"], alA); g1b = g1_report(texts, sb["meta"], alB)
    frac_fail = float(1.0 - (alA["qa_ok"] & alB["qa_ok"]).mean())
    g1_pass = bool(g1a["passed"] and g1b["passed"] and frac_fail <= G1_MAX_DOC_FAIL)
    print(f"[G1] A wordcov={g1a['word_coverage']:.3f} B wordcov="
          f"{g1b['word_coverage']:.3f} doc-fail={frac_fail:.4f} -> "
          f"{'PASS' if g1_pass else 'FAIL'}", flush=True)
    out["gates"]["G1"] = {"side_a": g1a, "side_b": g1b,
                          "frac_docs_failed": frac_fail, "passed": g1_pass}
    if not (g0_pass and g1_pass):
        out["gates"]["G2"] = {"s1": {"testable": False, "n_pairs_pass": 0,
                                     "n_pairs_in": int(len(pair_ok))},
                              "s2": {"testable": False, "n_pairs_pass": 0,
                                     "n_pairs_in": int(len(pair_ok))}}
        return out, 3

    # ---- occurrences, filtered to SPLIT B ONLY (requirement a) ----
    occ = build_occurrences(texts, alA, alB)
    in_b = np.isin(occ["doc"], b_idx)
    assert not np.isin(occ["doc"][in_b], a_idx).any(), \
        "evaluation occurrence in split A -- A3 requirement a violated"
    occ_b = {"doc": occ["doc"][in_b], "widx": occ["widx"][in_b],
             "nwords": occ["nwords"][in_b], "type": occ["type"][in_b],
             "prev": occ["prev"][in_b], "type_names": occ["type_names"]}
    out["alignment"] = {"n_occurrences_total": int(len(occ["doc"])),
                        "n_occurrences_split_b": int(len(occ_b["doc"])),
                        "n_word_types": int(len(occ["type_names"]))}
    print(f"[align] {len(occ['doc'])} occurrences total, "
          f"{len(occ_b['doc'])} in split B", flush=True)

    keep = pair_ok
    fa, fb = feat_a[keep], feat_b[keep]
    ufa, inv_a = np.unique(fa, return_inverse=True)
    ufb, inv_b = np.unique(fb, return_inverse=True)
    # A4(a): gate the token stream to split-B documents, and audit whether the
    # pre-A4 radix aliasing could have bitten this cell's data at all.
    R = int(occ_b["widx"].max()) + 2
    a_wid = occ["widx"][np.isin(occ["doc"], a_idx)]
    n_alias_tok = 0
    if len(a_wid):
        arisk = occ["doc"][np.isin(occ["doc"], a_idx)][a_wid >= R] + (
            a_wid[a_wid >= R] // R)
        n_alias_tok = int(np.isin(arisk, b_idx).sum())
    out["aliasing_audit"] = {"radix": R,
                             "max_a_widx": int(a_wid.max()) if len(a_wid) else -1,
                             "n_aliasing_a_occurrences": n_alias_tok}
    print(f"[A4-audit] radix={R} max_a_widx={out['aliasing_audit']['max_a_widx']} "
          f"aliasing_occurrences={n_alias_tok}", flush=True)
    Am_u, _ = word_matrices(sa, alA, occ_b, ufa, allowed_docs=b_idx)
    Bm_u, _ = word_matrices(sb, alB, occ_b, ufb, allowed_docs=b_idx)
    Am, Bm = Am_u[:, inv_a], Bm_u[:, inv_b]
    print(f"[matrices] A {Am.shape} B {Bm.shape} (split-B occurrences only)",
          flush=True)

    res = run_clean_stats(Am, Bm, occ_b, cfg)
    out["gates"]["G2"] = {"s1": res.pop("g2_s1"), "s2": res.pop("g2_s2")}
    out["alignment"].update(res.pop("s2_support"))
    out.update(res)
    out["a3_requirements"] = {
        "disjoint_selection_eval": True,
        "joint_fe_converged": True,
        "null_reresidualized": bool(out["s2_clean"]["null_reresidualized"]),
        "common_mask_asserted": True}
    out["a4_requirements"] = {
        "token_stream_gated_to_b": True,
        "universe_screened_on_a_only": True,
        "aliasing_audited": True}
    nt = (not out["gates"]["G2"]["s1"]["testable"]
          or not out["gates"]["G2"]["s2"]["testable"])
    return out, (2 if nt else 0)


# ================================= selftest =================================
def _counterexample_check():
    """A3's non-commuting counterexample must residualize to EXACTLY 0 (<= tol)
    under the joint FE: docs [A,A,B] and [A,B,B], f(A)=1, f(B)=0."""
    torch = _t()
    doc = torch.tensor([0, 0, 0, 1, 1, 1])
    typ = torch.tensor([0, 0, 1, 0, 1, 1])
    x = torch.tensor([[1.], [1.], [0.], [1.], [0.], [0.]], dtype=torch.float64)
    r, sweeps, dev = joint_fe(x, doc, typ, 2, 2)
    resid = float(r.abs().max())
    # the sequential S2c centering leaves nonzero residuals on this input:
    seq = group_mean_center(group_mean_center(x, doc, 2), typ, 2)
    seq_resid = float(seq.abs().max())
    return resid, seq_resid, sweeps


def selftest():
    import torch  # noqa
    rng = np.random.default_rng(0)
    ok = {}

    # ---- the A3 counterexample: joint FE closes it, sequential does not ----
    resid, seq_resid, sweeps = _counterexample_check()
    ok["counterexample_joint_fe_zero"] = resid <= FE_TOL
    ok["counterexample_sequential_nonzero"] = seq_resid > 1e-3
    print(f"counterexample   joint-FE max|resid|={resid:.2e} (want <= {FE_TOL:g}, "
          f"{sweeps} sweeps) | sequential={seq_resid:.4f} (want > 0: the S2c defect)")

    # ---- split determinism + disjointness ----
    a1, b1 = doc_split(20000); a2, b2 = doc_split(20000)
    ok["split_deterministic"] = bool((a1 == a2).all() and (b1 == b2).all())
    ok["split_disjoint_covering"] = (len(np.intersect1d(a1, b1)) == 0
                                     and len(a1) + len(b1) == 20000)
    print(f"split seed 42    n_a={len(a1)} n_b={len(b1)} "
          f"sha_a={_sha(a1)[:12]} sha_b={_sha(b1)[:12]} disjoint="
          f"{ok['split_disjoint_covering']}")

    # ---- planted cases through the FULL cell path ----
    n_docs, n_words, P, D = 240, 30, 24, 200
    vocab = [f"w{i:03d}" if i % 2 == 0 else f"wordform{i:03d}" for i in range(60)]
    probs = np.array([1.0 / (i + 1) for i in range(60)]); probs /= probs.sum()
    docs = [[str(v) for v in rng.choice(vocab, n_words, p=probs)]
            for _ in range(n_docs)]
    texts = [" ".join(ws) for ws in docs]
    cfg = {"n_perm": 200, "seed": 0, "device": "cpu",
           "g2_min_occ": 15, "g2_min_pairs": 8, "type_min": 10}
    floor = 1.0 / (cfg["n_perm"] + 1)

    def run_case(kind):
        mA = _synth_side(docs, split_long=False)
        mB = _synth_side(docs, split_long=True)
        A = _synth_acts(mA, docs, kind, P, D, struct_seed=42, noise_seed=1)
        B = _synth_acts(mB, docs, kind, P, D, struct_seed=42, noise_seed=2)
        sa, sb = _dense_to_side(mA, A), _dense_to_side(mB, B)
        # candidate pool = the planted columns' pooled vectors (float64, exact)
        Pa_c = np.asarray(sa["pooled"][:, :P], np.float64)
        Pb_c = np.asarray(sb["pooled"][:, :P], np.float64)
        meta = {"tag": f"selftest_{kind}", "smoke": True}
        return run_cell(sa, sb, texts, Pa_c, Pb_c, meta, cfg)

    rA, rcA = run_case("unigram")
    s1, s2 = rA["s1"], rA["s2_clean"]
    ok["A_full_path_ok"] = rcA == 0
    ok["A_s1_passes"] = (s1["null_n1"]["p"] <= floor * 1.001
                         and s1["obs_mean_pair_r"] >= 0.05)
    ok["A_s2_clean_collapses"] = (np.isnan(s2["R_tok_clean"])
                                  or s2["R_tok_clean"] < 0.10
                                  or (s2["null_n3"]["ci95"][0] <= s2["obs_mean_pair_r"]
                                      <= s2["null_n3"]["ci95"][1]))
    ok["A_mask_receipted"] = ("sha256" in rA["common_mask"]
                              and rA["common_mask"]["n_pairs_scored"] > 0)
    ok["A_requirements_marked"] = all(rA["a3_requirements"].values())
    print(f"CASE A unigram   S1 obs={s1['obs_mean_pair_r']:.3f} "
          f"p={s1['null_n1']['p']:.4f} | S2-clean excess={s2['excess']:+.4f} "
          f"R_tok_clean={s2['R_tok_clean']:.3f} (want S1 pass, collapse) "
          f"n_scored={rA['common_mask']['n_pairs_scored']} "
          f"sel={rA['selection']['n_selected_pairs']}")

    rB, rcB = run_case("context")
    s1, s2 = rB["s1"], rB["s2_clean"]
    ok["B_full_path_ok"] = rcB == 0
    ok["B_s1_passes"] = (s1["null_n1"]["p"] <= floor * 1.001
                         and s1["obs_mean_pair_r"] >= 0.05)
    ok["B_s2_clean_survives"] = (s2["R_tok_clean"] >= 0.30
                                 and s2["null_n3"]["p"] <= floor * 1.001)
    print(f"CASE B context   S1 obs={s1['obs_mean_pair_r']:.3f} "
          f"p={s1['null_n1']['p']:.4f} | S2-clean excess={s2['excess']:+.4f} "
          f"R_tok_clean={s2['R_tok_clean']:.3f} p={s2['null_n3']['p']:.4f} "
          f"(want both rungs pass) fe_sweeps null max="
          f"{s2['fe_convergence']['null_sweeps_max']}")

    rC, rcC = run_case("doctopic")
    s1 = rC["s1"]
    band = s1["null_n1"]["ci95"]
    ok["C_s1_at_null"] = band[0] <= s1["obs_mean_pair_r"] <= band[1]
    print(f"CASE C doctopic  S1 obs={s1['obs_mean_pair_r']:.4f} "
          f"band=[{band[0]:.4f},{band[1]:.4f}] (want inside; rc={rcC})")

    print()
    for k, v in ok.items():
        print(f"PASS {k} = {v}")
    good = all(ok.values())
    print("SELFTEST", "PASS" if good else "FAIL",
          "(A3-EXEC pipeline enforces all four requirements and can return "
          "every registered outcome)" if good else "")
    return 0 if good else 1


# =================================== main ===================================
def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--dir_a", required=True); ap.add_argument("--dir_b", required=True)
    ap.add_argument("--pairs_receipt", required=True,
                    help="original lextok20k receipt (model/sae provenance only; "
                         "its frozen pairs are NOT used)")
    ap.add_argument("--cache_npz", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    from tokalign_extract import load_docs
    sa = load_side(args.dir_a); sb = load_side(args.dir_b)
    assert sa["n_docs"] == sb["n_docs"], "sides extracted different doc counts"
    n_docs = sa["n_docs"]
    if not args.smoke:
        assert n_docs == 20000, f"full run requires 20000 docs, got {n_docs}"
    texts = load_docs(args.corpus, "en", n_docs)
    rec = json.load(open(args.pairs_receipt))
    z = np.load(args.cache_npz)
    Pa_c, Pb_c = z["Pa"][:n_docs], z["Pb"][:n_docs]
    print(f"[in] cell={args.cell} n_docs={n_docs} cache Pa{Pa_c.shape} "
          f"Pb{Pb_c.shape} device={dev} n_perm={args.n_perm}", flush=True)
    meta = {"tag": args.cell, "smoke": bool(args.smoke),
            "model_a": rec["model_a"], "model_b": rec["model_b"],
            "sae_a": rec["sae_a"], "sae_b": rec["sae_b"]}
    cfg = {"n_perm": args.n_perm, "seed": args.seed, "device": dev,
           "g2_min_occ": G2_MIN_OCC, "g2_min_pairs": G2_MIN_PAIRS,
           "type_min": S2_TYPE_MIN}
    out, rc = run_cell(sa, sb, texts, Pa_c, Pb_c, meta, cfg)
    out["elapsed_s"] = round(time.time() - t0, 1)
    if args.smoke:
        d, b = os.path.split(args.output)
        if not b.startswith("smoke_"):
            args.output = os.path.join(d, "smoke_" + b)
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    if rc == 0:
        s1, s2 = out["s1"], out["s2_clean"]
        print(f"[S1] obs={s1['obs_mean_pair_r']:.4f} p={s1['null_n1']['p']:.4f} "
              f"excess={s1['excess']:+.4f} | N2 excess={s1['excess_n2']:+.4f}",
              flush=True)
        print(f"[S2-clean] obs={s2['obs_mean_pair_r']:.4f} "
              f"p_N3={s2['null_n3']['p']:.4f} excess={s2['excess']:+.4f} "
              f"R_tok_clean={s2['R_tok_clean']:.3f} "
              f"(naive {s2['R_tok_clean_naive']:.3f}) "
              f"n_scored={out['common_mask']['n_pairs_scored']} "
              f"mask={out['common_mask']['sha256'][:12]}", flush=True)
    print(f"[done] rc={rc} -> {args.output} ({out['elapsed_s']}s)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
