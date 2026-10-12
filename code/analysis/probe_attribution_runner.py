#!/usr/bin/env python3
"""Phase-2 construct-validity runner for the cross-lingual probe-attribution metric.

This REBUILDS the generator of results/probe_attribution_v2/*.json (the original
experiments/probe_attribution_v2.py on the cluster) as a clean, in-repo module, and adds
the construct-validity controls.

METHOD (matches the original generator exactly so it reproduces the JSONs):
  * For each (concept pair_id, language) run ONE forward pass and read the
    logit-lens TARGET-TOKEN LOGIT at every layer: project layer hidden_states[L]
    (last token) through the model's final norm + lm_head, take logits[target_id].
    (NB: the headline "metric" is a logit-lens *logit value* per layer, NOT a
    gradient norm, and the backbone is HF AutoModelForCausalLM, NOT TransformerLens.
    The task brief misremembered both; reproduction is only possible against the
    real method, so we match the real method and document the discrepancy.)
  * mid-band = [n_layers//4, 3*n_layers//4), n_layers = num_hidden_layers + 1.
  * For each of the 6 language pairs, POOL all (pair_id x mid-band-layer)
    observations into one vector per language and take a single Pearson r.
    mean_pairwise_r = mean over the 6 language pairs.  This reproduces the JSONs.

CONSTRUCT-VALIDITY CONTROLS (cheap, computed from the saved raw matrix):
  real        : the headline pooled r (validation target).
  layer_shuffle: permute the mid-band LAYER order of lang B independently within
                 each concept pair (lang A untouched). Destroys the layer/depth
                 correspondence; PRESERVES each concept's per-pair mean level.
                 -> If r SURVIVES, the signal is per-concept (layer-independent),
                    consistent with semantics. If r COLLAPSES, the signal lived in
                    the shared depth profile (mid-layer hump of the logit-lens curve).
  mismatched_pair (= control c): align lang A's concept p to lang B's concept sigma(p)
                 (random derangement), keep layer order. Destroys concept alignment;
                 PRESERVES the depth profile shared by all concepts.
                 -> If r SURVIVES, the metric does NOT depend on matching concepts
                    => it is a depth-profile artifact. If r COLLAPSES to ~0, the
                    metric is semantic.
  full_shuffle : permute all pooled lang-B observations (sanity floor, ~0).

DECISION TABLE (per model):
  semantic / valid      : layer_shuffle preserved, mismatched_pair ~ 0.
  depth-profile artifact: mismatched_pair preserved, layer_shuffle collapses.

Usage:
  python probe_attribution_runner.py --model EleutherAI/pythia-410m-deduped \
      --prompts data/prompts_expanded_v2.jsonl \
      --output results/probe_attribution_controls/EleutherAI_pythia-410m-deduped.json
"""
import argparse, json, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- logit-lens (identical to the original generator) ----------
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
    final_ln = None
    for path in [("model", "norm"), ("model", "final_layernorm"),
                 ("language_model", "model", "norm"), ("language_model", "norm"),
                 ("model", "language_model", "norm")]:
        obj = model; ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        if ok:
            final_ln = obj; break
    if final_ln is None and hasattr(model, "norm"):
        final_ln = model.norm
    if final_ln is None:
        final_ln = lambda x: x
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    per_layer = {}
    for i, h in enumerate(hidden):
        last_hidden = h[:, -1, :]
        with torch.no_grad():
            logits = lm_head(final_ln(last_hidden))
        per_layer[i] = float(logits[0, tgt_id])
    return per_layer


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def derangement(n, rng, tries=1000):
    for _ in range(tries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    # fallback: cyclic shift (a valid derangement for n>=2)
    return np.roll(np.arange(n), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_rep", type=int, default=50, help="random draws per null control")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    mdl.eval()
    device = next(mdl.parameters()).device

    prompts_by_pair = defaultdict(dict)
    with open(args.prompts) as f:
        for line in f:
            r = json.loads(line)
            prompts_by_pair[r["pair_id"]][r["lang"]] = (r["prompt"], r["target"])
    pair_ids = sorted(prompts_by_pair.keys())
    langs = ["en", "zh", "ja", "ko"]

    # raw matrix M[pid][lang] = {layer: logit}
    M = {pid: {} for pid in pair_ids}
    for pid in pair_ids:
        for lang in langs:
            if lang not in prompts_by_pair[pid]:
                continue
            prompt, target = prompts_by_pair[pid][lang]
            ll = logit_lens_per_layer(mdl, tok, prompt, target, device=device)
            if ll is not None:
                M[pid][lang] = ll
    print("[forward] done", flush=True)

    cfg = mdl.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    n_layers = cfg.num_hidden_layers + 1
    mid_band = list(range(n_layers // 4, 3 * n_layers // 4))
    B = len(mid_band)

    lang_pairs = [(a, b) for i, a in enumerate(langs) for b in langs[i + 1:]]
    rng = np.random.default_rng(args.seed)

    pair_stats = {}
    for (la, lb) in lang_pairs:
        pids = [pid for pid in pair_ids if la in M[pid] and lb in M[pid]]
        if len(pids) < 2:
            continue
        # per-pid mid-band vectors
        A = {pid: np.array([M[pid][la].get(L, 0.0) for L in mid_band]) for pid in pids}
        Bv = {pid: np.array([M[pid][lb].get(L, 0.0) for L in mid_band]) for pid in pids}
        xs = np.concatenate([A[pid] for pid in pids])
        ys = np.concatenate([Bv[pid] for pid in pids])
        r_real = pearson(xs, ys)

        ls, mm, fl = [], [], []
        for _ in range(args.n_rep):
            # layer-shuffle: permute lb's layer order within each pid independently
            ys_ls = np.concatenate([Bv[pid][rng.permutation(B)] for pid in pids])
            ls.append(pearson(xs, ys_ls))
            # mismatched-pair: derange pid mapping for lb, keep layer order
            sig = derangement(len(pids), rng)
            ys_mm = np.concatenate([Bv[pids[sig[k]]] for k in range(len(pids))])
            mm.append(pearson(xs, ys_mm))
            # full shuffle floor
            fl.append(pearson(xs, rng.permutation(ys)))

        pair_stats[f"{la}-{lb}"] = {
            "n_pairs": len(pids),
            "n_observations": len(xs),
            "pearson_r": r_real,
            "layer_shuffle_r": float(np.mean(ls)),
            "layer_shuffle_sd": float(np.std(ls)),
            "mismatched_pair_r": float(np.mean(mm)),
            "mismatched_pair_sd": float(np.std(mm)),
            "full_shuffle_r": float(np.mean(fl)),
        }

    def mean_of(key):
        v = [s[key] for s in pair_stats.values()]
        return float(np.mean(v)) if v else 0.0

    out = {
        "model": args.model,
        "n_layers": n_layers,
        "mid_layer_band": mid_band,
        "n_pairs_total": len(pair_ids),
        "n_lang_pairs": len(pair_stats),
        "n_rep": args.n_rep,
        "seed": args.seed,
        "lang_pairs": pair_stats,
        "mean_pairwise_r": mean_of("pearson_r"),
        "mean_layer_shuffle_r": mean_of("layer_shuffle_r"),
        "mean_mismatched_pair_r": mean_of("mismatched_pair_r"),
        "mean_full_shuffle_r": mean_of("full_shuffle_r"),
        "n_positive_pairs": sum(1 for s in pair_stats.values() if s["pearson_r"] > 0),
        "method": "logit_lens_pooled_with_construct_validity_controls",
        # keep raw matrix for any later re-analysis (small: 20*4*n_layers floats)
        "raw_logit_lens": {str(pid): {lg: {str(L): M[pid][lg][L] for L in M[pid][lg]}
                                      for lg in M[pid]} for pid in pair_ids},
    }
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[done] {args.model}")
    print(f"  real mean_pairwise_r   = {out['mean_pairwise_r']:+.4f}")
    print(f"  layer_shuffle      mean = {out['mean_layer_shuffle_r']:+.4f}")
    print(f"  mismatched_pair    mean = {out['mean_mismatched_pair_r']:+.4f}")
    print(f"  full_shuffle floor mean = {out['mean_full_shuffle_r']:+.4f}")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
