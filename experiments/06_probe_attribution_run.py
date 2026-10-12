"""V1 pull-forward: probe-based attribution that does NOT require iterative
graph convergence. Per-layer logit-lens contribution to target token.

The CT v0 paper's headline result is that the iterative attribution-graph
procedure of Ameisen et al. systematically fails to converge on Qwen3
0.6B-8B at default thresholds (negative finding). The abstract pre-registers
"probe-based attribution and activation patching" as v1 alternatives.

This script provides a minimal v1-ready logit-lens probe that runs the same
antonym/capital-city prompts the attribution graph fails on. For each prompt,
we compute per-layer logit-lens contribution to the target token and
sweep across en/zh/ja/ko. Output mirrors the attribution-graph JSON schema
so paper §results can swap in once measured.
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Minimal antonym prompts (would expand to match config.CROSS_LINGUAL_PROMPTS)
ANTONYM_PROMPTS = {
    "en": [("The opposite of hot is", "cold"),
           ("The opposite of big is", "small"),
           ("The opposite of fast is", "slow"),
           ("The opposite of dark is", "light"),
           ("The opposite of high is", "low")],
    "zh": [("热的反义词是", "冷"),
           ("大的反义词是", "小"),
           ("快的反义词是", "慢"),
           ("黑暗的反义词是", "光明"),
           ("高的反义词是", "低")],
    "ja": [("熱いの反対は", "冷たい"),
           ("大きいの反対は", "小さい"),
           ("速いの反対は", "遅い")],
    "ko": [("뜨거운의 반대는", "차가운"),
           ("큰의 반대는", "작은"),
           ("빠른의 반대는", "느린")],
}


def logit_lens_per_layer(model, tok, prompt, target, device="cuda"):
    """Compute per-layer logit-lens contribution to target token logit.

    Returns: dict {layer_idx -> target_logit_at_layer}
    """
    inputs = tok(prompt, return_tensors="pt").to(device)
    # Get target token id; first token of target string after space.
    tgt_ids = tok(" " + target, add_special_tokens=False).input_ids
    if not tgt_ids:
        tgt_ids = tok(target, add_special_tokens=False).input_ids
    if not tgt_ids:
        return None
    tgt_id = tgt_ids[0]

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states is tuple (embedding + per-layer outputs)
    hidden = out.hidden_states  # length = num_layers + 1
    # final layer norm + lm_head as the "logit lens"
    final_ln = model.model.norm if hasattr(model.model, "norm") else model.norm
    lm_head = model.lm_head

    per_layer = {}
    for i, h in enumerate(hidden):
        # last position only
        last_hidden = h[:, -1, :]  # (1, d)
        with torch.no_grad():
            normed = final_ln(last_hidden)
            logits = lm_head(normed)
        tgt_logit = float(logits[0, tgt_id])
        per_layer[i] = tgt_logit
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out_dir", default="<REPO_ROOT>/results/probe_attribution")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace("/", "_")

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    mdl.eval()
    device = next(mdl.parameters()).device

    results = {"model": args.model, "n_layers": mdl.config.num_hidden_layers, "per_lang": {}}
    for lang, prompts in ANTONYM_PROMPTS.items():
        print(f"[lang={lang}]")
        per_prompt = []
        for prompt, target in prompts:
            ll = logit_lens_per_layer(mdl, tok, prompt, target, device=device)
            if ll is None:
                continue
            per_prompt.append({"prompt": prompt, "target": target, "per_layer": ll})
            # Mid-band contribution
            n_layers = len(ll) - 1  # exclude embedding
            mid_lo, mid_hi = n_layers // 3, 2 * n_layers // 3
            mid_contrib = ll[mid_hi] - ll[mid_lo]
            print(f"   {prompt[:40]} -> {target}: mid-band contribution = {mid_contrib:+.3f}")
        results["per_lang"][lang] = per_prompt

    # Cross-lingual Jaccard: at each layer band, which top-K logit-shifters overlap across languages?
    # (kept minimal at v0; v1 will expand)
    cross_lang_overlap = {}
    for lang_a in results["per_lang"]:
        for lang_b in results["per_lang"]:
            if lang_a >= lang_b:
                continue
            # Compute pairwise correlation of per-layer mid-band contributions
            a = [p["per_layer"][len(p["per_layer"]) // 2] for p in results["per_lang"][lang_a]]
            b = [p["per_layer"][len(p["per_layer"]) // 2] for p in results["per_lang"][lang_b]]
            n = min(len(a), len(b))
            if n < 2:
                continue
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
            cross_lang_overlap[f"{lang_a}-{lang_b}"] = {"mid_layer_corr": corr, "n": n}
    results["cross_lingual_mid_layer_correlations"] = cross_lang_overlap

    out_path = Path(args.out_dir) / f"probe_attribution_{model_tag}.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"[DONE] wrote {out_path}")
    print(f"   model={args.model}, n_layers={results['n_layers']}")
    print(f"   cross-lingual mid-layer correlations: {cross_lang_overlap}")


if __name__ == "__main__":
    main()
