#!/usr/bin/env python3
"""Go/no-go: zero-shot cross-model concept-probe transfer via matched SAE features.

If matched features are a genuine shared interface, a concept classifier trained on model A's
matched-feature activations should classify concepts in model B WITHOUT any training on B ---
using only the unsupervised mutual-NN matching as the shared coordinate system. A non-matched
(permuted-pairing) control must fail. This converts "features are shared" into "the shared
features are a usable cross-model interface".

METHOD:
  1. Pooled matched-feature activations for A and B on the labelled concept prompts
     (prompts_v3_multi, 9 families), top-K by variance, z-scored per model.
  2. Shared basis = mutual-NN matched pairs (position m = m-th matched pair).
  3. Train logistic regression on A's matched vectors (all prompts) -> test on B's matched
     vectors (train-A/test-B transfer accuracy).
  4. Controls: (a) non-matched pairing (permute A<->B pairing) -> chance; (b) within-model
     5-fold CV on A and on B (ceilings); (c) majority-class baseline.

selftest (CPU): planted shared concept structure -> matched transfer >> non-matched.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def _z(X, eps=1e-6):
    X = np.nan_to_num(np.asarray(X, np.float64))
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    keep = (sd > eps).ravel()
    return np.clip((X[:, keep] - mu[:, keep]) / sd[:, keep], -30, 30), np.where(keep)[0]


def mutual_nn(PhiA, PhiB, topk):
    ia = np.argsort(PhiA.var(0))[::-1][:topk]; ib = np.argsort(PhiB.var(0))[::-1][:topk]
    Za, ka = _z(PhiA[:, ia]); Zb, kb = _z(PhiB[:, ib])
    ia, ib = ia[ka], ib[kb]
    C = (Za.T @ Zb) / Za.shape[0]
    a2b = C.argmax(1); b2a = C.argmax(0)
    recip = np.where(b2a[a2b] == np.arange(C.shape[0]))[0]
    return ia[recip], ib[a2b[recip]]        # matched index lists (aligned by position)


def _ridge_fit(X, y, C, lam=1.0):
    """Closed-form one-hot ridge classifier (numpy only). Returns weight matrix W [d+1 x C]."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    Y = np.eye(C)[y]
    d = Xb.shape[1]
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(d), Xb.T @ Y)
    return W


def _ridge_pred(X, W):
    return (np.hstack([X, np.ones((X.shape[0], 1))]) @ W).argmax(1)


def _cv_acc(X, y, C, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); accs = []
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        W = _ridge_fit(X[tr], y[tr], C)
        accs.append((_ridge_pred(X[te], W) == y[te]).mean())
    return float(np.mean(accs))


def transfer_acc(XA, XB, y, seed=0, n_perm=200):
    rng = np.random.default_rng(seed)
    C = int(y.max() + 1)
    # per-model z-score of matched activations (shared pattern, model-specific scale)
    ZA, _ = _z(XA); ZB, _ = _z(XB)
    K = min(ZA.shape[1], ZB.shape[1]); ZA, ZB = ZA[:, :K], ZB[:, :K]
    W = _ridge_fit(ZA, y, C)
    matched = float((_ridge_pred(ZB, W) == y).mean())
    null = []
    for _ in range(n_perm):                         # non-matched: permute B's columns
        null.append(float((_ridge_pred(ZB[:, rng.permutation(K)], W) == y).mean()))
    null = np.array(null)
    maj = float(np.bincount(y).max() / len(y))
    p = float((np.sum(null >= matched) + 1) / (len(null) + 1))
    return dict(transfer_acc=matched, nonmatched_mean=float(null.mean()),
                nonmatched_ci=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                transfer_p=p, cv_A=_cv_acc(ZA, y, C), cv_B=_cv_acc(ZB, y, C),
                majority=maj, K=int(K), n=int(len(y)))


def selftest():
    rng = np.random.default_rng(0)
    n, K, C = 180, 60, 9
    y = np.repeat(np.arange(C), n // C)
    centres = rng.standard_normal((C, K)) * 2.5
    latent = centres[y] + rng.standard_normal((n, K))            # shared concept code
    XA = np.hstack([latent + 0.4 * rng.standard_normal((n, K)), rng.standard_normal((n, 40))])
    XB = np.hstack([latent * 1.7 + 0.4 * rng.standard_normal((n, K)), rng.standard_normal((n, 40))])
    # matched features = the first K (planted); simulate matching returning them aligned
    r = transfer_acc(XA[:, :K], XB[:, :K], y, n_perm=100)
    print("[selftest] transfer=%.3f nonmatched=%.3f p=%.4f (cvA=%.3f cvB=%.3f maj=%.3f)" %
          (r["transfer_acc"], r["nonmatched_mean"], r["transfer_p"], r["cv_A"], r["cv_B"], r["majority"]))
    ok = r["transfer_acc"] > 0.6 and r["transfer_p"] < 0.02 and r["transfer_acc"] > r["nonmatched_mean"] + 0.3
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model_a"); ap.add_argument("--sae_a_release"); ap.add_argument("--sae_a_id")
    ap.add_argument("--model_b"); ap.add_argument("--sae_b_release"); ap.add_argument("--sae_b_id")
    ap.add_argument("--prompts", default="../../data/prompts_v3_multi.jsonl")
    ap.add_argument("--lang", default="en"); ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--match_prompts", default=None,
                    help="fit the mutual-NN matching on this SEPARATE corpus (e.g. the C4 "
                         "slice) instead of the eval prompts --- hardening against "
                         "same-corpus circularity; eval stays on --prompts")
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    from cross_model_sae_match import compute_pooled_features
    prompts = [json.loads(l) for l in open(args.prompts)]
    sel = [p for p in prompts if p.get("lang") == args.lang]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    fams = sorted(set(p["family"] for p in sel))
    fam_idx = {f: i for i, f in enumerate(fams)}
    y = np.array([fam_idx[p["family"]] for p in sel])

    if args.match_prompts:
        # one forward pass per model over [match corpus | eval prompts]; split after.
        # pair_id offset keeps the eval prompts AFTER the match docs under the
        # pair_id sort inside compute_pooled_features.
        mdocs = [json.loads(l) for l in open(args.match_prompts)]
        mdocs = [p for p in mdocs if p.get("lang") == args.lang]
        mdocs.sort(key=lambda p: p.get("pair_id", 0))
        combo = mdocs + [dict(p, pair_id=p.get("pair_id", 0) + 10_000_000) for p in sel]
        nm = len(mdocs)
        A_all = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id, combo, lang=args.lang)
        B_all = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id, combo, lang=args.lang)
        assert A_all.shape[0] == len(combo) and B_all.shape[0] == len(combo)
        MA, PhiA = A_all[:nm], A_all[nm:]
        MB, PhiB = B_all[:nm], B_all[nm:]
        n = min(PhiA.shape[0], PhiB.shape[0], len(y)); PhiA, PhiB, y = PhiA[:n], PhiB[:n], y[:n]
        ma, mb = mutual_nn(MA, MB, args.topk)
        print(f"[match] fitted on {nm} independent docs ({args.match_prompts})", flush=True)
    else:
        PhiA = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id, prompts, lang=args.lang)
        PhiB = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id, prompts, lang=args.lang)
        n = min(PhiA.shape[0], PhiB.shape[0], len(y)); PhiA, PhiB, y = PhiA[:n], PhiB[:n], y[:n]
        ma, mb = mutual_nn(PhiA, PhiB, args.topk)
    XA, XB = PhiA[:, ma], PhiB[:, mb]
    print(f"[matched] {len(ma)} mutual-NN features; {n} prompts, {len(fams)} families", flush=True)
    res = transfer_acc(XA, XB, y)
    res.update(model_a=args.model_a, model_b=args.model_b, families=fams,
               n_matched_features=int(len(ma)),
               match_corpus=(args.match_prompts or "eval_prompts"))
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    print("[done] A->B transfer=%.3f | non-matched=%.3f (p=%.4f) | cvA=%.3f cvB=%.3f | majority=%.3f -> %s" %
          (res["transfer_acc"], res["nonmatched_mean"], res["transfer_p"], res["cv_A"],
           res["cv_B"], res["majority"], args.output))


if __name__ == "__main__":
    main()
