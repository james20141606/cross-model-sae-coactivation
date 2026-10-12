#!/usr/bin/env python3
"""Audit of the Lan et al. protocol under our control battery.

Lan et al. (arXiv:2410.06981; a preprint, NOT an ICLR paper -- verified against the
source, and references.bib has it right) claim 'analogous feature universality': pair SAE features across
models by max activation correlation (greedy), then compare the PAIRED DECODER SPACES
with SVCCA/RSA (rotation-invariant), against a random-pairing baseline. Their protocol
has no artifact controls (no permutation-calibrated pairing, no corpus control, no
capability cells). We implement their statistic faithfully and apply our battery:

  cells   : capable pair (gemma-9b<->mistral) | corpus-artifact pair (pythia70m<->mistral)
            | tooling-null pair (gemma-2b<->gpt2)     [what our metric adjudicated]
  corpora : templated concepts (en) vs C4 natural text
  baselines: (a) random pairing (theirs); (b) PERMUTED-activation pairing (ours: permute
            B's prompt rows before computing the pairing correlations -- destroys genuine
            correspondence; if SVCCA/RSA stays high, the statistic reflects dictionary
            geometry, not model correspondence).

Verdicts of interest: does Lan-SVCCA/RSA (i) stay high under permuted pairing, (ii) stay
high on the pair our C4 control exposed as a corpus artifact, (iii) track our capable/null
distinction? Any of (i)/(ii) demonstrated = 'published metric, unmeasured confounds'.

Selftest (CPU): planted rotation -> true pairing beats random & permuted; noise -> all equal.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def _z(X, eps=1e-6):
    X = np.nan_to_num(np.asarray(X, np.float64))
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True)
    keep = (sd > eps).ravel()
    return np.clip((X[:, keep] - mu[:, keep]) / sd[:, keep], -50, 50), np.where(keep)[0]


def svcca(X, Y, c=20):
    """mean canonical correlation between top-c PCs of X and Y (rows = paired samples)."""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    c = min(c, X.shape[0] - 1, X.shape[1], Y.shape[1])
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    Xr = Ux[:, :c] * Sx[:c]; Yr = Uy[:, :c] * Sy[:c]
    qx, _ = np.linalg.qr(Xr - Xr.mean(0)); qy, _ = np.linalg.qr(Yr - Yr.mean(0))
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def rsa(X, Y):
    """Pearson r between upper triangles of the cosine-similarity matrices."""
    def cosmat(M):
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        return Mn @ Mn.T
    iu = np.triu_indices(X.shape[0], 1)
    a = cosmat(X)[iu]; b = cosmat(Y)[iu]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def lan_statistics(PhiA, PhiB, WdecA, WdecB, topk=512, n_base=20, seed=0):
    """Greedy activation-corr pairing -> SVCCA/RSA of paired decoder rows, with
    (a) random-pairing and (b) permuted-activation-pairing baselines."""
    rng = np.random.default_rng(seed)
    ia = np.argsort(PhiA.var(0))[::-1][:topk]
    ib = np.argsort(PhiB.var(0))[::-1][:topk]
    Za, ka = _z(PhiA[:, ia]); Zb, kb = _z(PhiB[:, ib])
    ia, ib = ia[ka], ib[kb]
    C = (Za.T @ Zb) / Za.shape[0]
    pair = C.argmax(1)                                # greedy (Lan), NOT reciprocal
    DA = WdecA[ia]; DB = WdecB[ib[pair]]
    out = {"svcca": svcca(DA, DB), "rsa": rsa(DA, DB), "K": int(len(ia))}

    rand_s, rand_r = [], []
    for _ in range(n_base):
        pr = rng.permutation(len(ib))[:len(ia)] if len(ib) >= len(ia) else rng.integers(0, len(ib), len(ia))
        rand_s.append(svcca(DA, WdecB[ib[pr]])); rand_r.append(rsa(DA, WdecB[ib[pr]]))
    out["svcca_random_pairing"] = float(np.mean(rand_s))
    out["rsa_random_pairing"] = float(np.mean(rand_r))

    perm_s, perm_r = [], []
    for _ in range(n_base):
        Cp = (Za.T @ Zb[rng.permutation(Zb.shape[0])]) / Za.shape[0]
        pp = Cp.argmax(1)
        perm_s.append(svcca(DA, WdecB[ib[pp]])); perm_r.append(rsa(DA, WdecB[ib[pp]]))
    out["svcca_permuted_pairing"] = float(np.mean(perm_s))
    out["rsa_permuted_pairing"] = float(np.mean(perm_r))
    return out


def selftest():
    rng = np.random.default_rng(0)
    N, K, d = 300, 200, 64
    latents = rng.standard_normal((N, K))
    PhiA = latents + 0.2 * rng.standard_normal((N, K))
    permK = rng.permutation(K)
    PhiB = latents[:, permK] + 0.2 * rng.standard_normal((N, K))
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    WdecA = rng.standard_normal((K, d))
    WdecB = (WdecA @ R)[permK] + 0.05 * rng.standard_normal((K, d))
    r = lan_statistics(PhiA, PhiB, WdecA, WdecB, topk=K, n_base=10)
    print("[planted] svcca=%.3f (rand %.3f, perm %.3f) rsa=%.3f (rand %.3f, perm %.3f)" %
          (r["svcca"], r["svcca_random_pairing"], r["svcca_permuted_pairing"],
           r["rsa"], r["rsa_random_pairing"], r["rsa_permuted_pairing"]))
    ok1 = r["svcca"] > r["svcca_random_pairing"] + 0.1 and r["rsa"] > r["rsa_random_pairing"] + 0.2
    ok1 &= r["svcca"] > r["svcca_permuted_pairing"] + 0.1
    Phi2 = rng.standard_normal((N, K)); W2 = rng.standard_normal((K, d))
    r2 = lan_statistics(PhiA, Phi2, WdecA, W2, topk=K, n_base=10)
    print("[noise]   svcca=%.3f (rand %.3f, perm %.3f) rsa=%.3f" %
          (r2["svcca"], r2["svcca_random_pairing"], r2["svcca_permuted_pairing"], r2["rsa"]))
    ok2 = abs(r2["svcca"] - r2["svcca_random_pairing"]) < 0.08 and abs(r2["rsa"]) < 0.1
    print("SELFTEST", "PASS" if (ok1 and ok2) else "FAIL")
    return 0 if (ok1 and ok2) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model_a"); ap.add_argument("--sae_a_release"); ap.add_argument("--sae_a_id")
    ap.add_argument("--model_b"); ap.add_argument("--sae_b_release"); ap.add_argument("--sae_b_id")
    ap.add_argument("--prompts_templates", default="../../data/prompts_v3_multi.jsonl")
    ap.add_argument("--prompts_c4", default="../../data/c4_slice.jsonl")
    ap.add_argument("--max_prompts", type=int, default=2000)
    ap.add_argument("--corpora", default="templates,c4")
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    from cross_model_sae_match import compute_pooled_features
    import importlib, cross_model_sae_match as xm

    def get_sae_dec(model_name, release, sid):
        import torch
        from sae_lens import SAE
        if release.startswith("qwenpt:"):
            from qwen_sae_adapter import load_qwen_sae
            sae = load_qwen_sae(release[len("qwenpt:"):], sid, "cpu")
        else:
            sae = SAE.from_pretrained(release, sid, device="cpu")
            if isinstance(sae, tuple):
                sae = sae[0]
        return sae.W_dec.detach().float().cpu().numpy()

    prompts_t = [json.loads(l) for l in open(args.prompts_templates)]
    prompts_c = [json.loads(l) for l in open(args.prompts_c4)]
    WdecA = get_sae_dec(args.model_a, args.sae_a_release, args.sae_a_id)
    WdecB = get_sae_dec(args.model_b, args.sae_b_release, args.sae_b_id)
    res = {"model_a": args.model_a, "model_b": args.model_b, "topk": args.topk}
    todo = [("templates", prompts_t), ("c4", prompts_c)]
    todo = [t for t in todo if t[0] in args.corpora.split(",")]
    for corpus, prompts in todo:
        PhiA = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id,
                                       prompts, lang="en", max_prompts=args.max_prompts)
        PhiB = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id,
                                       prompts, lang="en", max_prompts=args.max_prompts)
        n = min(PhiA.shape[0], PhiB.shape[0]); PhiA, PhiB = PhiA[:n], PhiB[:n]
        res[corpus] = lan_statistics(PhiA, PhiB, WdecA, WdecB, topk=args.topk)
        print(f"[{corpus}] {json.dumps(res[corpus])}", flush=True)
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
