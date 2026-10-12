#!/usr/bin/env python3
"""Pooled-level split-half check for the three representative cells (app:threats (iv)).

QUESTION. The verdict-bearing lexical arm (lexical_token_test.py) freezes mutual-NN pairs
on ALL 20,000 documents and scores those same pairs on the same documents. Does the pooled
sharing of A-selected pairs survive on documents the selection never saw?

DESIGN. One registered 50/50 document split (numpy default_rng(seed)). On split A: z-score
per column, reciprocal argmax -> frozen pair set M_A (lexical_token_test.mutual_nn_pairs,
verbatim). On split B: z-score per column USING SPLIT-B STATISTICS ONLY, mean pair
correlation of M_A, and a B-only row-permutation null (n_perm draws; the pair set is fixed
so this is pair_null's global_perm branch restricted to B). The in-sample comparator is the
same statistic with M_B selected on B and scored on B, so held-out / in-sample is a ratio
of two numbers computed on the identical documents. Both halves are also run with the
roles swapped (select on B, score on A) so the reader sees both directions.

The candidate universe is the top-1500 columns already carried by the md5-pinned caches
(the same universe the lextok20k receipts used); it is NOT re-screened per half, because
the lexical arm under audit did not re-screen either -- this is a split-half of THAT arm.
(The token-aligned A4 rerun does re-screen; see tokalign_clean_analysis.py.)

    python3 pooled_splithalf.py --cache cache/lextok20k_<cell>.npz --cell <cell> \
        --seed 42 --n_perm 2000 --output results/cross_model_sae/splithalf20k_<cell>.json
    python3 pooled_splithalf.py --selftest
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexical_token_test import _zcols, mutual_nn_pairs   # verbatim selection rule


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def split_docs(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[: n // 2]).astype(np.int64), np.sort(perm[n // 2:]).astype(np.int64)


def pair_stat(Za, Zb, ia, jb):
    """Mean over frozen pairs of corr(Za[:,i], Zb[:,j]); Z are column-z-scored on these rows."""
    return float((Za[:, ia] * Zb[:, jb]).sum(0).mean() / Za.shape[0])


def pair_null(Za, Zb, ia, jb, n_perm, seed):
    rng = np.random.default_rng(seed)
    obs = pair_stat(Za, Zb, ia, jb)
    # Pre-slice the pair columns once: permuting rows of an [n x n_pairs] gather is the same
    # statistic as permuting rows of the full [n x 1500] matrix (the permutation acts on
    # documents; column choice commutes with it) and is ~25x cheaper (39 vs 945 ms/draw).
    A = np.ascontiguousarray(Za[:, ia]); B = np.ascontiguousarray(Zb[:, jb]); n = A.shape[0]
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = float((A * B[rng.permutation(n)]).sum(0).mean() / n)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    sd = float(null.std())
    return {"obs_mean_pair_corr": obs, "null_mean": float(null.mean()), "null_sd": sd,
            "excess": float(obs - null.mean()),
            "z": float((obs - null.mean()) / sd) if sd > 1e-12 else float("inf"),
            "p": p, "n_perm": int(n_perm), "null_type": "global_perm_within_half"}


def one_direction(P_a, P_b, sel_idx, eval_idx, n_perm, seed):
    """Select on sel_idx rows, score on eval_idx rows (held-out) and on sel_idx rows
    (in-sample, pairs selected and scored on the same half)."""
    Za_s, Zb_s = _zcols(P_a[sel_idx]), _zcols(P_b[sel_idx])
    ia, jb, _ = mutual_nn_pairs(Za_s, Zb_s)
    ia = np.asarray(ia, np.int64); jb = np.asarray(jb, np.int64)
    Za_e, Zb_e = _zcols(P_a[eval_idx]), _zcols(P_b[eval_idx])
    held = pair_null(Za_e, Zb_e, ia, jb, n_perm, seed)
    insample = pair_null(Za_s, Zb_s, ia, jb, n_perm, seed + 1)
    # pairs selected ON the eval half, scored on it: the in-sample comparator at equal n
    Za_e2, Zb_e2 = Za_e, Zb_e
    ia2, jb2, _ = mutual_nn_pairs(Za_e2, Zb_e2)
    ia2 = np.asarray(ia2, np.int64); jb2 = np.asarray(jb2, np.int64)
    insample_same_docs = pair_null(Za_e2, Zb_e2, ia2, jb2, n_perm, seed + 2)
    overlap = len(set(zip(ia.tolist(), jb.tolist())) & set(zip(ia2.tolist(), jb2.tolist())))
    return {
        "n_pairs_selected": int(len(ia)),
        "n_pairs_selected_on_eval_half": int(len(ia2)),
        "n_pairs_shared_between_halves": int(overlap),
        "held_out": held,
        "in_sample_selection_half": insample,
        "in_sample_eval_half_own_pairs": insample_same_docs,
        "ratio_heldout_over_insample_same_docs": float(
            held["excess"] / insample_same_docs["excess"]) if insample_same_docs["excess"] else None,
        "ratio_heldout_over_insample_selection_half": float(
            held["excess"] / insample["excess"]) if insample["excess"] else None,
    }


def run(P_a, P_b, seed, n_perm, meta):
    n = P_a.shape[0]
    a, b = split_docs(n, seed)
    assert len(np.intersect1d(a, b)) == 0 and len(a) + len(b) == n
    t0 = time.time()
    out = dict(meta)
    out.update({"n_docs": int(n), "topk": int(P_a.shape[1]), "split": {
        "seed": int(seed), "n_a": int(len(a)), "n_b": int(len(b)),
        "sha256_a": _sha(a), "sha256_b": _sha(b)}, "n_perm": int(n_perm)})
    out["select_A_score_B"] = one_direction(P_a, P_b, a, b, n_perm, seed * 10 + 1)
    out["select_B_score_A"] = one_direction(P_a, P_b, b, a, n_perm, seed * 10 + 2)
    r1 = out["select_A_score_B"]["ratio_heldout_over_insample_same_docs"]
    r2 = out["select_B_score_A"]["ratio_heldout_over_insample_same_docs"]
    out["summary"] = {
        "mean_ratio_heldout_over_insample": float(np.mean([r1, r2])),
        "min_heldout_p": float(min(out["select_A_score_B"]["held_out"]["p"],
                                   out["select_B_score_A"]["held_out"]["p"])),
        "note": ("ratio = held-out excess of A-selected pairs on B / excess of B-selected "
                 "pairs on B (same documents, same n). 1.0 = no selection optimism at the "
                 "pooled level; the in-sample number is the one the lextok20k arm reports."),
    }
    out["runtime_s"] = float(time.time() - t0)
    return out


def selftest():
    rng = np.random.default_rng(0)
    N, K, S = 4000, 300, 40
    L = rng.standard_normal((N, S))
    Pa = np.hstack([L + 0.5 * rng.standard_normal((N, S)), rng.standard_normal((N, K - S))])
    Pb = np.hstack([L[:, rng.permutation(S)] + 0.5 * rng.standard_normal((N, S)),
                    rng.standard_normal((N, K - S))])
    r = run(Pa, Pb, 42, 200, {"cell": "selftest_planted"})
    ok1 = r["select_A_score_B"]["held_out"]["p"] < 0.01 and r["summary"]["mean_ratio_heldout_over_insample"] > 0.6
    Pa2 = rng.standard_normal((N, K)); Pb2 = rng.standard_normal((N, K))
    r2 = run(Pa2, Pb2, 42, 200, {"cell": "selftest_noise"})
    # noise: in-sample selection still finds spurious reciprocal pairs with positive corr;
    # held-out must sit at the null (p not small) -- this is the optimism the check exists for
    ok2 = r2["select_A_score_B"]["held_out"]["p"] > 0.05 and \
        r2["select_A_score_B"]["in_sample_eval_half_own_pairs"]["excess"] > \
        r2["select_A_score_B"]["held_out"]["excess"]
    print(f"[selftest planted] held p={r['select_A_score_B']['held_out']['p']:.4f} "
          f"ratio={r['summary']['mean_ratio_heldout_over_insample']:.3f} -> {ok1}")
    print(f"[selftest noise]   held p={r2['select_A_score_B']['held_out']['p']:.4f} "
          f"held excess={r2['select_A_score_B']['held_out']['excess']:+.4f} "
          f"insample excess={r2['select_A_score_B']['in_sample_eval_half_own_pairs']['excess']:+.4f} -> {ok2}")
    print("SELFTEST", "PASS" if ok1 and ok2 else "FAIL")
    return 0 if ok1 and ok2 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache"); ap.add_argument("--cell")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--output")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    for req in ("cache", "cell", "output"):
        if getattr(a, req) is None:
            ap.error(f"--{req} required")
    z = np.load(a.cache)
    Pa, Pb = np.asarray(z["Pa"], np.float64), np.asarray(z["Pb"], np.float64)
    md5 = hashlib.md5(open(a.cache, "rb").read()).hexdigest()
    meta = {"cell": a.cell, "cache": os.path.basename(a.cache), "cache_md5": md5,
            "selection_rule": "lexical_token_test.mutual_nn_pairs on _zcols (verbatim)",
            "universe": "top-1500 columns carried by the pinned cache (not re-screened per half)"}
    out = run(Pa, Pb, a.seed, a.n_perm, meta)
    Path(os.path.dirname(a.output)).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2)
    s = out["summary"]; d1 = out["select_A_score_B"]; d2 = out["select_B_score_A"]
    print(f"[{a.cell}] A->B held={d1['held_out']['excess']:.4f} (p={d1['held_out']['p']:.4f}) "
          f"insample(B,own)={d1['in_sample_eval_half_own_pairs']['excess']:.4f} "
          f"ratio={d1['ratio_heldout_over_insample_same_docs']:.3f} | "
          f"B->A ratio={d2['ratio_heldout_over_insample_same_docs']:.3f} | "
          f"mean ratio={s['mean_ratio_heldout_over_insample']:.3f} min p={s['min_heldout_p']:.4f} "
          f"| {out['runtime_s']:.0f}s -> {a.output}")


if __name__ == "__main__":
    main()
