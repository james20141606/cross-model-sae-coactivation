"""V2 probe-attribution: expanded n=20/lang prompt set, bootstrap CIs, Gemma-compatible.

Inputs:
  --model HF_ID       (Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B}, google/gemma-3-{4b-it,12b-it,27b-it}, etc.)
  --prompts JSONL     (data/prompts_expanded_v2.jsonl)
  --output JSON       (results/probe_attribution_v2/<model>.json)

Reports:
  - per-pair Pearson r per layer
  - bootstrap 95% CI per pair (5000 resamples)
  - permutation-test p-value per pair (5000 perms vs language-shuffled null)
  - mean pairwise r over 6 language pairs

Schema mirrors v1 for downstream compatibility.
"""
import argparse, json, sys, os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def logit_lens_per_layer(model, tok, prompt, target, device="cuda"):
    inputs = tok(prompt, return_tensors="pt").to(device)
    tgt_ids = tok(" " + target, add_special_tokens=False).input_ids
    if not tgt_ids:
        tgt_ids = tok(target, add_special_tokens=False).input_ids
    if not tgt_ids:
        return None
    tgt_id = tgt_ids[0]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states
    # Locate final norm + lm_head across architectures (Qwen3, Gemma3, etc.)
    final_ln = None
    for path in [("model", "norm"), ("model", "final_layernorm"),
                  ("language_model", "model", "norm"), ("language_model", "norm"),
                  ("model", "language_model", "norm")]:
        obj = model
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        if ok:
            final_ln = obj
            break
    if final_ln is None and hasattr(model, "norm"):
        final_ln = model.norm
    if final_ln is None:
        final_ln = lambda x: x
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    per_layer = {}
    for i, h in enumerate(hidden):
        last_hidden = h[:, -1, :]
        with torch.no_grad():
            normed = final_ln(last_hidden)
            logits = lm_head(normed)
        per_layer[i] = float(logits[0, tgt_id])
    return per_layer


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_ci(x, y, n_boot=5000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(x)
    if n < 3:
        return (None, None)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rs.append(pearson(np.array(x)[idx], np.array(y)[idx]))
    lo = float(np.percentile(rs, 100 * alpha / 2))
    hi = float(np.percentile(rs, 100 * (1 - alpha / 2)))
    return (lo, hi)


def permutation_p(x, y, n_perm=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = abs(pearson(x, y))
    if obs == 0.0:
        return 1.0
    n = len(x)
    cnt = 0
    y_arr = np.asarray(y)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if abs(pearson(x, y_arr[perm])) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--n_perm", type=int, default=5000)
    args = ap.parse_args()

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    mdl.eval()
    device = next(mdl.parameters()).device

    prompts_by_pair = defaultdict(dict)
    with open(args.prompts) as f:
        for line in f:
            r = json.loads(line)
            prompts_by_pair[r["pair_id"]][r["lang"]] = (r["prompt"], r["target"])
    pair_ids = sorted(prompts_by_pair.keys())
    langs = ["en", "zh", "ja", "ko"]
    print(f"[data] {len(pair_ids)} pairs, langs={langs}")

    results = {pid: {} for pid in pair_ids}
    for pid in pair_ids:
        for lang in langs:
            if lang not in prompts_by_pair[pid]:
                continue
            prompt, target = prompts_by_pair[pid][lang]
            ll = logit_lens_per_layer(mdl, tok, prompt, target, device=device)
            if ll is not None:
                results[pid][lang] = ll
        print(f"[pair={pid}] done")

    # Gemma3 / multimodal configs nest under .text_config
    cfg = mdl.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    n_layers = cfg.num_hidden_layers + 1
    layers = list(range(n_layers))
    mid_band = list(range(n_layers // 4, 3 * n_layers // 4))

    lang_pairs = [(a, b) for i, a in enumerate(langs) for b in langs[i+1:]]
    pair_stats = {}
    rng = np.random.default_rng(42)
    for (la, lb) in lang_pairs:
        xs, ys = [], []
        for pid in pair_ids:
            if la not in results[pid] or lb not in results[pid]:
                continue
            for L in mid_band:
                xs.append(results[pid][la].get(L, 0.0))
                ys.append(results[pid][lb].get(L, 0.0))
        if len(xs) < 3:
            continue
        r = pearson(xs, ys)
        ci_lo, ci_hi = bootstrap_ci(xs, ys, n_boot=args.n_boot, rng=rng)
        p = permutation_p(xs, ys, n_perm=args.n_perm, rng=rng)
        pair_stats[f"{la}-{lb}"] = {
            "n_pairs": sum(1 for pid in pair_ids if la in results[pid] and lb in results[pid]),
            "n_observations": len(xs),
            "pearson_r": r,
            "ci95": [ci_lo, ci_hi],
            "perm_p": p,
        }

    rs = [v["pearson_r"] for v in pair_stats.values()]
    mean_r = float(np.mean(rs)) if rs else 0.0
    n_positive = sum(1 for r in rs if r > 0)

    out = {
        "model": args.model,
        "n_layers": n_layers,
        "mid_layer_band": mid_band,
        "n_pairs_total": len(pair_ids),
        "n_lang_pairs": len(pair_stats),
        "lang_pairs": pair_stats,
        "mean_pairwise_r": mean_r,
        "n_positive_pairs": n_positive,
        "method": "logit_lens_v2_with_bootstrap_perm",
    }

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] mean_r={mean_r:.4f} ({n_positive}/{len(pair_stats)} positive) → {args.output}")


if __name__ == "__main__":
    main()
