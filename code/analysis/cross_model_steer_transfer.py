#!/usr/bin/env python3
"""Cross-model steering-transfer probe: do matched features PROMOTE the same tokens?

The 'key next step' named in Limitations: an activation-level test of shared computation
that escapes both confounds that sank the ablation probes:
  * single-feature ablation underpower / degeneration  -> steering ADDS a large, controlled
    activation (beta x per-feature natural scale), a dense intervention;
  * prompt-fragility axis (any strong ablation hurts the same prompts) -> we compare the
    IDENTITY of promoted tokens, not loss magnitudes.

METHOD. For each mutual-NN matched feature pair (f_A, f_B) -- read from the released
interpretation JSON so the probe targets exactly the pairs discussed in the paper -- steer
each feature in ITS OWN model: h <- h + beta * s_f * W_dec[f] at the SAE's layer, where
s_f is the feature's max pooled activation on C4 (its natural scale). On each of a few
neutral prompts, record Delta-logits = steered - clean at the last position; take the
top-N promoted tokens, decode to strings (lowercased, stripped) -- tokenizer-robust.

STATISTIC. overlap(k) = |top-N promoted strings of f_A^k  ∩  f_B^k|. Null: derange the
pairing (f_A^k vs f_B^{sigma(k)}) -> mean overlap under 2000 derangements. If matched
features implement the same computation, matched overlap >> deranged overlap.

Usage (GPU):
  python cross_model_steer_transfer.py \
    --interp results/cross_model_sae/interp_gemma2-9b__mistral7b.json \
    --model_a google/gemma-2-9b --sae_a_release gemma-scope-9b-pt-res-canonical --sae_a_id layer_20/width_16k/canonical \
    --model_b mistralai/Mistral-7B-v0.1 --sae_b_release mistral-7b-res-wg --sae_b_id blocks.16.hook_resid_pre \
    --prompts data/c4_slice.jsonl --beta 8.0 --n_top 30 \
    --output results/cross_model_sae/steer_gemma2-9b__mistral7b.json

Selftest (CPU, no GPU): python cross_model_steer_transfer.py --selftest
"""
import argparse, json, os, sys, re
from pathlib import Path
import numpy as np


# ------------------------- CPU statistic (selftested) -------------------------
def overlap_stats(tokens_a, tokens_b, n_perm=2000, seed=0):
    """tokens_a/b: list over pairs of sets of promoted token strings."""
    K = len(tokens_a)
    matched = np.array([len(tokens_a[k] & tokens_b[k]) for k in range(K)], float)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(K)
        while K > 1 and np.any(perm == np.arange(K)):
            perm = rng.permutation(K)
        null.append(np.mean([len(tokens_a[k] & tokens_b[perm[k]]) for k in range(K)]))
    null = np.array(null)
    obs = float(matched.mean())
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    z = float((obs - null.mean()) / null.std()) if null.std() > 1e-9 else float("inf")
    return {"matched_mean_overlap": obs, "matched_per_pair": matched.tolist(),
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p": p, "z": z, "n_pairs": K,
            "frac_pairs_any_overlap": float((matched > 0).mean())}


def selftest():
    rng = np.random.default_rng(0)
    vocab = [f"tok{i}" for i in range(500)]
    K, N = 20, 30
    # planted: matched pairs share ~1/3 of their promoted tokens
    ta, tb = [], []
    for k in range(K):
        shared = set(rng.choice(vocab, 10, replace=False))
        ta.append(shared | set(rng.choice(vocab, N - 10, replace=False)))
        tb.append(shared | set(rng.choice(vocab, N - 10, replace=False)))
    r = overlap_stats(ta, tb, n_perm=500)
    print("[selftest:planted] matched=%.2f null=%.2f z=%.1f p=%.4f" %
          (r["matched_mean_overlap"], r["null_mean"], r["z"], r["p"]))
    ok1 = r["p"] < 0.01 and r["matched_mean_overlap"] > r["null_mean"] + 3
    # null case: no shared structure
    ta2 = [set(rng.choice(vocab, N, replace=False)) for _ in range(K)]
    tb2 = [set(rng.choice(vocab, N, replace=False)) for _ in range(K)]
    r2 = overlap_stats(ta2, tb2, n_perm=500)
    print("[selftest:noise]   matched=%.2f null=%.2f p=%.4f (expect ns)" %
          (r2["matched_mean_overlap"], r2["null_mean"], r2["p"]))
    ok2 = r2["p"] > 0.05
    print("SELFTEST", "PASS" if (ok1 and ok2) else "FAIL")
    return 0 if (ok1 and ok2) else 1


# ------------------------- GPU steering -------------------------
NEUTRAL_PROMPTS = ["The", "Today,", "I think that", "Here is the news:"]


def load(model_name, sae_release, sae_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    dev = next(mdl.parameters()).device
    if sae_release.startswith("qwenpt:"):
        from qwen_sae_adapter import load_qwen_sae
        sae = load_qwen_sae(sae_release[len("qwenpt:"):], sae_id, dev)
    else:
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
    # locate layer module whose OUTPUT is hidden_states[hs_idx]
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = mdl; ok = True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        if ok:
            layers = obj; break
    layer_module = layers[max(hs_idx - 1, 0)]
    return tok, mdl, sae, dev, hs_idx, layer_module


def promoted_tokens(tok, mdl, sae, dev, layer_module, feat_scale_pairs, beta, n_top):
    """For each (feature_id, scale): steer h += beta*scale*W_dec[f] at layer output on the
    neutral prompts; return set of top-N promoted token strings (mean Delta-logits)."""
    import torch
    Wdec = sae.W_dec
    out_sets = []
    # clean logits per prompt (reused)
    clean = []
    for p in NEUTRAL_PROMPTS:
        inp = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            clean.append(mdl(**inp).logits[0, -1].float())
    for f, scale in feat_scale_pairs:
        vec = (beta * float(scale)) * Wdec[int(f)]
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h + vec.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        deltas = []
        handle = layer_module.register_forward_hook(hook)
        try:
            for i, p in enumerate(NEUTRAL_PROMPTS):
                inp = tok(p, return_tensors="pt").to(dev)
                with torch.no_grad():
                    steered = mdl(**inp).logits[0, -1].float()
                deltas.append((steered - clean[i]).cpu().numpy())
        finally:
            handle.remove()
        d = np.mean(deltas, axis=0)
        top = np.argsort(d)[::-1][:n_top]
        strs = set()
        for t in top:
            sdec = tok.decode([int(t)]).strip().lower()
            if sdec:
                strs.add(sdec)
        out_sets.append(strs)
        print(f"  [steer] f={f} scale={scale:.2f} top: {sorted(list(strs))[:8]}", flush=True)
    return out_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--interp"); ap.add_argument("--prompts")
    ap.add_argument("--model_a"); ap.add_argument("--sae_a_release"); ap.add_argument("--sae_a_id")
    ap.add_argument("--model_b"); ap.add_argument("--sae_b_release"); ap.add_argument("--sae_b_id")
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--n_top", type=int, default=30)
    ap.add_argument("--k_pairs", type=int, default=20)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    for req in ("interp", "model_a", "sae_a_release", "sae_a_id", "model_b",
                "sae_b_release", "sae_b_id", "prompts", "output"):
        if getattr(args, req) is None:
            ap.error(f"--{req} required")

    interp = json.load(open(args.interp))
    pairs = interp["pairs"][:args.k_pairs]
    # natural scale per feature = its top pooled activation from the interp top-docs
    fa = [(p["feature_a"], p["top_docs_a"][0]["act"]) for p in pairs]
    fb = [(p["feature_b"], p["top_docs_b"][0]["act"]) for p in pairs]
    print(f"[pairs] {len(pairs)} matched pairs from {args.interp}", flush=True)

    print(f"[load A] {args.model_a}", flush=True)
    tokA, mdlA, saeA, devA, _, lmA = load(args.model_a, args.sae_a_release, args.sae_a_id)
    setsA = promoted_tokens(tokA, mdlA, saeA, devA, lmA, fa, args.beta, args.n_top)
    del mdlA, saeA
    import torch; torch.cuda.empty_cache()
    print(f"[load B] {args.model_b}", flush=True)
    tokB, mdlB, saeB, devB, _, lmB = load(args.model_b, args.sae_b_release, args.sae_b_id)
    setsB = promoted_tokens(tokB, mdlB, saeB, devB, lmB, fb, args.beta, args.n_top)

    res = overlap_stats(setsA, setsB, n_perm=args.n_perm)
    res.update({"model_a": args.model_a, "model_b": args.model_b, "beta": args.beta,
                "n_top": args.n_top, "neutral_prompts": NEUTRAL_PROMPTS,
                "promoted_a": [sorted(s) for s in setsA],
                "promoted_b": [sorted(s) for s in setsB],
                "pair_features": [[p["feature_a"], p["feature_b"]] for p in pairs]})
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(f"[done] matched overlap={res['matched_mean_overlap']:.2f}/{args.n_top} "
          f"null={res['null_mean']:.2f} z={res['z']:.1f} p={res['p']:.4f} "
          f"frac_any={res['frac_pairs_any_overlap']:.2f} -> {args.output}")


if __name__ == "__main__":
    main()
