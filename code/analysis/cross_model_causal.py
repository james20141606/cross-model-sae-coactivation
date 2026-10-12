#!/usr/bin/env python3
"""Causal cross-model feature-transfer test.

Upgrades the correlational mutual-NN matching (cross_model_sae_match.py) to a CAUSAL
claim of shared computation. For a matched feature pair (f_A in model A, f_B in model B,
identified by reciprocal best-buddy / mutual-NN), we ask: does ablating f_A in model A
disrupt the SAME prompts that ablating f_B disrupts in model B? If matched features are
causally responsible for the same inputs across two independently-trained models, that is
evidence of shared COMPUTATION, not merely shared representational geometry.

METHOD:
  1. Build the pooled-feature correlation matrix (as in cross_model_sae_match) and take the
     top-K reciprocal best-buddy (mutual-NN) feature pairs by correlation.
  2. For each model, per prompt, measure each matched feature's CAUSAL EFFECT by ablation:
     register a forward hook on the layer whose output is the SAE's residual, subtract that
     feature's decoder contribution (h <- h - act_f * W_dec[f]), and record the change in the
     model's own teacher-forced next-token loss on that prompt. This yields a per-prompt
     causal-effect vector c[f] over the N prompts.
  3. For each matched pair (f_A, f_B) compute Pearson corr(c_A[f_A], c_B[f_B]) over prompts:
     do the two features hurt the same prompts? Headline = mean over the K matched pairs.
  4. NULL: corr(c_A[f_A], c_B[f_{sigma(B)}]) for random derangements sigma of the B-features
     (the same features, mismatched). If matched >> null, the correlational match predicts a
     shared causal role.

Efficiency: (K+1) forward passes per model over a prompt subset; chunked, <90 min.

Usage:
  python cross_model_causal.py \
    --model_a google/gemma-2-2b --sae_a_release gemma-scope-2b-pt-res-canonical --sae_a_id layer_12/width_16k/canonical \
    --model_b google/gemma-2-9b --sae_b_release gemma-scope-9b-pt-res-canonical --sae_b_id layer_20/width_16k/canonical \
    --prompts data/prompts_v3_multi.jsonl --lang en --max_prompts 100 --topk 1500 --k_pairs 20 --n_perm 1000 \
    --output results/cross_model_causal/gemma2-2b__gemma2-9b.json
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np


def _zscore_cols(X, eps=1e-6):
    X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True)
    keep = (sd > eps).ravel()
    Xz = np.clip(np.nan_to_num((X[:, keep] - mu[:, keep]) / sd[:, keep]), -50, 50)
    return np.ascontiguousarray(Xz), np.where(keep)[0]


def _layer_list(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model; ok = True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        if ok:
            return obj, path
    raise RuntimeError("could not locate transformer layer list")


def _hook_index(cfg, sae_id, hook_layer):
    name = str(getattr(cfg, "hook_name", "")) + " " + str(sae_id)
    # hidden_states[hs_idx] is the residual we matched on; it equals the OUTPUT of
    # decoder layer (hs_idx-1). resid_post of block N -> hs_idx N+1 -> layer module N.
    hs_idx = hook_layer + 1 if "resid_post" in name else hook_layer
    layer_module_idx = max(hs_idx - 1, 0)
    return hs_idx, layer_module_idx


def load(model_name, sae_release, sae_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    sae = SAE.from_pretrained(sae_release, sae_id, device=str(next(mdl.parameters()).device))
    if isinstance(sae, tuple):
        sae = sae[0]
    dev = next(mdl.parameters()).device
    sae = sae.to(dev)
    cfg = getattr(sae, "cfg", sae)
    hl = getattr(cfg, "hook_layer", None)
    if hl is None:
        for cand in (getattr(cfg, "hook_name", None), str(sae_id)):
            nums = re.findall(r"\d+", str(cand or ""))
            if nums:
                hl = int(nums[0]); break
    hs_idx, layer_idx = _hook_index(cfg, sae_id, hl)
    layers, _ = _layer_list(mdl)
    return tok, mdl, sae, dev, hs_idx, layer_idx, layers


def pooled_features(tok, mdl, sae, dev, hs_idx, texts):
    import torch
    feats = []
    for t in texts:
        inp = tok(t, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = mdl(**inp, output_hidden_states=True).hidden_states[hs_idx]
            acts = sae.encode(h.to(sae.W_enc.dtype))
        feats.append(acts[0].mean(0).float().cpu().numpy())
    return np.vstack(feats)


def causal_vectors(tok, mdl, sae, dev, layer_module, feat_ids, texts):
    """Per matched feature, per prompt: increase in teacher-forced next-token loss when the
    feature's decoder contribution is ablated at layer_module's output."""
    import torch
    import torch.nn.functional as F

    def prompt_loss(text, hook=None):
        inp = tok(text, return_tensors="pt").to(dev)
        ids = inp["input_ids"]
        handle = layer_module.register_forward_hook(hook) if hook else None
        try:
            with torch.no_grad():
                logits = mdl(**inp).logits
        finally:
            if handle:
                handle.remove()
        # teacher-forced CE over the sequence
        lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        return float(-lp[torch.arange(tgt.shape[0]), tgt].mean())

    Wdec = sae.W_dec  # [d_sae, d_model]

    def make_hook(f):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts = sae.encode(h.to(sae.W_enc.dtype))           # [1,T,d_sae]
            contrib = acts[..., f:f + 1] * Wdec[f]              # [1,T,d_model]
            h2 = h - contrib.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    clean = np.array([prompt_loss(t) for t in texts])
    C = np.zeros((len(feat_ids), len(texts)))
    for i, f in enumerate(feat_ids):
        f = int(f)
        abl = np.array([prompt_loss(t, hook=make_hook(f)) for t in texts])
        C[i] = abl - clean
    return C  # [K, N] causal effect per feature per prompt


def pearson_rows(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 1e-9 and db > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a"); ap.add_argument("--sae_a_release"); ap.add_argument("--sae_a_id")
    ap.add_argument("--model_b"); ap.add_argument("--sae_b_release"); ap.add_argument("--sae_b_id")
    ap.add_argument("--prompts", required=True); ap.add_argument("--lang", default="en")
    ap.add_argument("--max_prompts", type=int, default=100)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--k_pairs", type=int, default=20)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts)]
    texts = [p["prompt"] for p in prompts if p.get("lang") == args.lang][:args.max_prompts]
    print(f"[prompts] {len(texts)}", flush=True)

    print(f"[load A] {args.model_a}", flush=True)
    tokA, mdlA, saeA, devA, hsA, liA, layersA = load(args.model_a, args.sae_a_release, args.sae_a_id)
    print(f"[load B] {args.model_b}", flush=True)
    tokB, mdlB, saeB, devB, hsB, liB, layersB = load(args.model_b, args.sae_b_release, args.sae_b_id)

    PhiA = pooled_features(tokA, mdlA, saeA, devA, hsA, texts)
    PhiB = pooled_features(tokB, mdlB, saeB, devB, hsB, texts)
    # top-k by variance, then correlation matrix, then mutual-NN pairs
    ia = np.argsort(PhiA.var(0))[::-1][:args.topk]; ib = np.argsort(PhiB.var(0))[::-1][:args.topk]
    Za, ka = _zscore_cols(PhiA[:, ia]); Zb, kb = _zscore_cols(PhiB[:, ib])
    ia, ib = ia[ka], ib[kb]
    Cmat = (Za.T @ Zb) / Za.shape[0]
    a2b = Cmat.argmax(1); b2a = Cmat.argmax(0)
    recip = np.where(b2a[a2b] == np.arange(Cmat.shape[0]))[0]
    pair_corr = Cmat[recip, a2b[recip]]
    order = recip[np.argsort(pair_corr)[::-1]][:args.k_pairs]
    feats_a = ia[order]; feats_b = ib[a2b[order]]
    print(f"[matched] {len(order)} mutual-NN pairs (top corr {Cmat[order[0],a2b[order[0]]]:.3f})", flush=True)

    print("[causal A] ablating matched features ...", flush=True)
    CA = causal_vectors(tokA, mdlA, saeA, devA, layersA[liA], feats_a, texts)
    print("[causal B] ablating matched features ...", flush=True)
    CB = causal_vectors(tokB, mdlB, saeB, devB, layersB[liB], feats_b, texts)

    # matched causal consistency
    matched = np.array([pearson_rows(CA[i], CB[i]) for i in range(len(order))])
    rng = np.random.default_rng(args.seed)
    null = []
    K = len(order)
    for _ in range(args.n_perm):
        perm = rng.permutation(K)
        if K > 1:
            while np.any(perm == np.arange(K)):
                perm = rng.permutation(K)
        null.append(np.mean([pearson_rows(CA[i], CB[perm[i]]) for i in range(K)]))
    null = np.array(null)
    obs = float(matched.mean())
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    z = float((obs - null.mean()) / null.std()) if null.std() > 1e-9 else float("inf")

    out = {
        "model_a": args.model_a, "model_b": args.model_b,
        "n_prompts": len(texts), "k_pairs": int(K), "n_perm": args.n_perm,
        "matched_causal_corr_mean": obs,
        "matched_causal_corr_per_pair": matched.tolist(),
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
        "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
        "excess_over_null": obs - float(null.mean()), "p": p, "z": z,
        "frac_pairs_positive": float((matched > 0).mean()),
        "mean_abl_effect_A": float(CA.mean()), "mean_abl_effect_B": float(CB.mean()),
    }
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"[done] matched causal-corr={obs:+.3f} null={null.mean():+.3f} excess={out['excess_over_null']:+.3f} "
          f"p={p:.4f} z={z:.1f} frac+={out['frac_pairs_positive']:.2f} -> {args.output}")


if __name__ == "__main__":
    main()
