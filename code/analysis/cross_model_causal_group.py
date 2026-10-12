#!/usr/bin/env python3
"""Properly-powered cross-model causal test: MULTI-FEATURE GROUP ablation.

The single-feature ablation probe (cross_model_causal.py) is underpowered: one SAE
feature's ablation moves loss by ~0.01-0.06 nats, so per-prompt Delta-loss vectors are
sparse and per-pair correlations degenerate to +-1. Here we ablate the TOP-K MATCHED
features JOINTLY in each model (a dense, large intervention), giving one per-prompt
Delta-loss vector per model, and ask whether the matched groups disrupt the SAME prompts:

  obs   = corr( dA_matched , dB_matched )   over N prompts
  null  = corr( dA_matched , dB_random  )   for many size-K random feature groups in B

where dX = per-prompt (clean_loss - group_ablated_loss). The random-group null controls
for the generic "hard prompts take bigger hits from any ablation" covariate, so
obs >> null means the MATCHED features specifically target the same computation across
models -> cross-model causal shared computation. Group ablation is dense, so no +-1
degeneration (we also report the degeneration-proof correlation directly).

Usage:
  python cross_model_causal_group.py \
    --model_a google/gemma-2-9b --sae_a_release gemma-scope-9b-pt-res-canonical --sae_a_id layer_20/width_16k/canonical \
    --model_b mistralai/Mistral-7B-v0.1 --sae_b_release mistral-7b-res-wg --sae_b_id blocks.16.hook_resid_pre \
    --prompts data/prompts_v3_multi.jsonl --lang en --max_prompts 120 --topk 1500 --k_group 30 --n_perm 200 \
    --output results/cross_model_causal_group/gemma2-9b__mistral7b.json
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
            return obj
    raise RuntimeError("no layer list")


def load(model_name, sae_release, sae_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    dev = next(mdl.parameters()).device
    sae = SAE.from_pretrained(sae_release, sae_id, device=str(dev))
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(dev)
    cfg = getattr(sae, "cfg", sae)
    hl = getattr(cfg, "hook_layer", None)
    if hl is None:
        for cand in (getattr(cfg, "hook_name", None), str(sae_id)):
            nums = re.findall(r"\d+", str(cand or ""))
            if nums:
                hl = int(nums[0]); break
    name = str(getattr(cfg, "hook_name", "")) + " " + str(sae_id)
    hs_idx = hl + 1 if "resid_post" in name else hl
    layer_module = _layer_list(mdl)[max(hs_idx - 1, 0)]
    return tok, mdl, sae, dev, hs_idx, layer_module


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


def clean_and_group_losses(tok, mdl, sae, dev, layer_module, texts, groups):
    """Return clean per-prompt loss and, for each feature group, the per-prompt
    ablated-minus-clean Delta-loss vector. `groups` = list of index-arrays."""
    import torch
    import torch.nn.functional as F
    Wdec = sae.W_dec

    def loss_vec(hook=None):
        out = []
        handle = layer_module.register_forward_hook(hook) if hook else None
        try:
            for t in texts:
                inp = tok(t, return_tensors="pt").to(dev)
                ids = inp["input_ids"]
                with torch.no_grad():
                    logits = mdl(**inp).logits
                lp = F.log_softmax(logits[0, :-1].float(), -1)
                tgt = ids[0, 1:]
                out.append(float(-lp[torch.arange(tgt.shape[0]), tgt].mean()))
        finally:
            if handle:
                handle.remove()
        return np.array(out)

    def make_hook(idx):
        idx_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts = sae.encode(h.to(sae.W_enc.dtype))            # [1,T,d_sae]
            contrib = acts[..., idx_t] @ Wdec[idx_t]            # [1,T,d_model]
            h2 = h - contrib.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    clean = loss_vec(None)
    deltas = [loss_vec(make_hook(g)) - clean for g in groups]
    return clean, deltas


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 1e-9 and db > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a"); ap.add_argument("--sae_a_release"); ap.add_argument("--sae_a_id")
    ap.add_argument("--model_b"); ap.add_argument("--sae_b_release"); ap.add_argument("--sae_b_id")
    ap.add_argument("--prompts", required=True); ap.add_argument("--lang", default="en")
    ap.add_argument("--max_prompts", type=int, default=120)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--k_group", type=int, default=30)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    prompts = [json.loads(l) for l in open(args.prompts)]
    texts = [p["prompt"] for p in prompts if p.get("lang") == args.lang][:args.max_prompts]
    print(f"[prompts] {len(texts)}", flush=True)

    tokA, mdlA, saeA, devA, hsA, lmA = load(args.model_a, args.sae_a_release, args.sae_a_id)
    tokB, mdlB, saeB, devB, hsB, lmB = load(args.model_b, args.sae_b_release, args.sae_b_id)
    PhiA = pooled_features(tokA, mdlA, saeA, devA, hsA, texts)
    PhiB = pooled_features(tokB, mdlB, saeB, devB, hsB, texts)

    ia = np.argsort(PhiA.var(0))[::-1][:args.topk]; ib = np.argsort(PhiB.var(0))[::-1][:args.topk]
    Za, ka = _zscore_cols(PhiA[:, ia]); Zb, kb = _zscore_cols(PhiB[:, ib])
    ia, ib = ia[ka], ib[kb]
    Cmat = (Za.T @ Zb) / Za.shape[0]
    a2b = Cmat.argmax(1); b2a = Cmat.argmax(0)
    recip = np.where(b2a[a2b] == np.arange(Cmat.shape[0]))[0]
    order = recip[np.argsort(Cmat[recip, a2b[recip]])[::-1]][:args.k_group]
    grpA = ia[order]                       # matched group in A
    grpB_matched = ib[a2b[order]]          # matched group in B
    K = len(order)
    print(f"[matched group] K={K} features", flush=True)

    # model A: clean + matched-group ablation -> dA
    _, [dA] = clean_and_group_losses(tokA, mdlA, saeA, devA, lmA, texts, [grpA])
    # model B: matched group + n_perm random size-K groups (from B's top-k feature pool)
    pool = ib
    rand_groups = [rng.choice(pool, size=K, replace=False) for _ in range(args.n_perm)]
    _, dB_list = clean_and_group_losses(tokB, mdlB, saeB, devB, lmB, texts, [grpB_matched] + rand_groups)
    dB_matched = dB_list[0]; dB_rand = dB_list[1:]

    obs = pear(dA, dB_matched)
    null = np.array([pear(dA, d) for d in dB_rand])
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    z = float((obs - null.mean()) / null.std()) if null.std() > 1e-9 else float("inf")

    out = {
        "model_a": args.model_a, "model_b": args.model_b,
        "n_prompts": len(texts), "k_group": int(K), "n_perm": args.n_perm,
        "matched_group_corr": obs,
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
        "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
        "excess_over_null": obs - float(null.mean()), "p": p, "z": z,
        "mean_abl_dA": float(dA.mean()), "mean_abl_dB": float(dB_matched.mean()),
        "std_dA": float(dA.std()), "std_dB": float(dB_matched.std()),
    }
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"[done] group causal: matched_corr={obs:+.3f} null={null.mean():+.3f} "
          f"excess={out['excess_over_null']:+.3f} p={p:.4f} z={z:.1f} "
          f"| dA_effect={dA.mean():+.3f} dB_effect={dB_matched.mean():+.3f} nats -> {args.output}")


if __name__ == "__main__":
    main()
