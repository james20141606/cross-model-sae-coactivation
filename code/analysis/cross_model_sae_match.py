#!/usr/bin/env python3
"""Cross-model SAE feature matching with a permutation null.

POSITIVE-DIRECTION experiment for the circuit-tracing paper: instead of the
correlational logit-lens probe-attribution metric (shown to be a depth-profile
artifact in Sec. construct-validity), this measures whether two models learn
*matching SAE features* -- the cross-model universality quantity the field
actually rewards (Lan et al. arXiv:2410.06981 -- a preprint, not ICLR 2025 as this
line previously claimed; Bricken 2023; Dravid Rosetta 2023) --
and does so with this paper's construct-validity discipline: a permutation null,
because "matched features correlate highly" is meaningless on its own (the max of
K noise correlations is large, so a naive matched-correlation looks impressive
even between unrelated dictionaries).

METHOD (tokenizer-robust activation-correlation matching):
  1. Run a shared text corpus through model A and model B.
  2. At one layer per model (chosen by fractional depth), encode the residual
     stream through that model's SAE -> per-token feature activations; MEAN-POOL
     over each prompt's tokens to get one activation per (prompt, feature).
     Pooling at the prompt level sidesteps the cross-family tokenizer-grid
     mismatch (different tokenizers split text differently) -- each prompt yields
     one comparable scalar per feature.
  3. Keep the top-K most-active features per model (variance across prompts).
  4. C[i,j] = Pearson corr (over prompts) between A-feature i and B-feature j
     = (Za^T Zb)/N with column-z-scored pooled activations.
  5. Matching statistics:
       greedy_max_mean : mean_i max_j C[i,j]   (Lan-style greedy nearest)
       hungarian_mean  : mean matched corr under optimal 1-1 assignment
       mutual_nn       : best-buddies (reciprocal argmax) count + mean corr
  6. NULL: permute the prompt order of model B (row permutation of Zb) and
     recompute every statistic; repeat n_perm times. Report observed vs the null
     distribution (mean, 95%, empirical p, z). The depth-corrected/honest signal
     is (observed - null_mean): how much real cross-model structure survives the
     chance level of matching two large dictionaries.

The statistical core (match_and_null) is pure numpy/scipy and is verified by
`--selftest` (synthetic planted-match): no GPU/torch needed for that check.
GPU activation extraction (compute_pooled_features) reuses the proven HF forward
pattern from probe_attribution_runner.py plus sae-lens SAE.encode.

Usage (GPU):
  python cross_model_sae_match.py \
    --model_a google/gemma-2-2b --sae_a_release gemma-scope-2b-pt-res-canonical --sae_a_id layer_12/width_16k/canonical \
    --model_b Qwen/Qwen3-1.7B   --sae_b_release <qwen-sae-release>           --sae_b_id <layer_id> \
    --prompts data/prompts_v3_multi.jsonl --lang en --topk 1500 --n_perm 200 \
    --output results/cross_model_sae/gemma2-2b__qwen3-1.7b.json

Self-test (no GPU):
  python cross_model_sae_match.py --selftest

Matched random-SAE control (arXiv:2602.14111 random-baseline gate; see
random_sae_baseline.py): add --random_sae {random-dir,rotate} --random_sae_seed N to
replace both real SAEs with matched-dimension, matched-L0 random encoders (model A uses
--random_sae_seed, model B uses --random_sae_seed+10000, independently) before matching.
Default --random_sae none leaves behaviour byte-identical to the code above. Output is
tagged with a "random_sae" JSON key and the output filename is auto-prefixed "rand_" so
real and control receipts can never be confused.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np


# ----------------------------- statistics core (GPU-free, unit-tested) -----------------------------
def _zscore_cols(X, eps=1e-6):
    """z-score each column (feature) over rows (prompts). Robust for real SAE data:
    cast to float64, scrub non-finite, drop near-constant (dead) features, and clip
    extreme z-scores so a feature that is near-zero on all but one prompt cannot blow
    up the correlation matmul."""
    X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    keep = (sd > eps * (np.abs(mu) + eps)).ravel() & (sd > eps).ravel()
    Xz = (X[:, keep] - mu[:, keep]) / sd[:, keep]
    Xz = np.clip(np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
    return np.ascontiguousarray(Xz), keep


def _match_stats(C):
    """Given corr matrix C [Ka x Kb], return greedy/hungarian/mutual-NN summaries."""
    from scipy.optimize import linear_sum_assignment
    greedy_max = C.max(1)                      # best B-feature per A-feature
    greedy_max_mean = float(greedy_max.mean())
    # Hungarian one-to-one on the square min(Ka,Kb) sub-problem (maximize corr)
    ri, ci = linear_sum_assignment(-C)
    hung = C[ri, ci]
    hungarian_mean = float(hung.mean())
    # mutual nearest neighbours (best-buddies)
    a_to_b = C.argmax(1)
    b_to_a = C.argmax(0)
    recip = b_to_a[a_to_b] == np.arange(C.shape[0])
    n_mutual = int(recip.sum())
    mutual_mean = float(C[np.arange(C.shape[0])[recip], a_to_b[recip]].mean()) if n_mutual else 0.0
    return {
        "greedy_max_mean": greedy_max_mean,
        "hungarian_mean": hungarian_mean,
        "n_mutual_nn": n_mutual,
        "mutual_nn_mean_corr": mutual_mean,
    }


def match_and_null(Phi_a, Phi_b, n_perm=200, seed=0, n_perm_hungarian=0):
    """Core: observed matching stats + permutation-null distribution.

    Phi_a [N x Ka], Phi_b [N x Kb] : pooled feature activations on the SAME N prompts.
    Null permutes the prompt order of B (destroys cross-model prompt alignment).
    Hungarian is O(K^3); by default it is computed only for the observed stat and
    n_perm_hungarian (default 0) null draws -- the null uses the cheap greedy stat,
    which is the standard Lan-style permutation statistic.
    """
    rng = np.random.default_rng(seed)
    Za, _ = _zscore_cols(np.asarray(Phi_a, float))
    Zb, _ = _zscore_cols(np.asarray(Phi_b, float))
    N = Za.shape[0]
    assert Zb.shape[0] == N, "A and B must share the same prompts (rows)"
    C = (Za.T @ Zb) / N
    obs = _match_stats(C)

    null_greedy, null_mutual, null_hung = [], [], []
    for t in range(n_perm):
        perm = rng.permutation(N)
        Cp = (Za.T @ Zb[perm]) / N
        gm = float(Cp.max(1).mean())
        null_greedy.append(gm)
        a2b = Cp.argmax(1); b2a = Cp.argmax(0)
        null_mutual.append(int((b2a[a2b] == np.arange(Cp.shape[0])).sum()))
        if t < n_perm_hungarian:
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(-Cp)
            null_hung.append(float(Cp[ri, ci].mean()))

    ng = np.array(null_greedy)
    nm = np.array(null_mutual)

    def emp_p(obs_val, null_arr):
        return float((np.sum(null_arr >= obs_val) + 1) / (len(null_arr) + 1))

    def z(obs_val, null_arr):
        s = null_arr.std()
        return float((obs_val - null_arr.mean()) / s) if s > 1e-12 else float("inf")

    return {
        "N_prompts": int(N),
        "Ka": int(Za.shape[1]),
        "Kb": int(Zb.shape[1]),
        "n_perm": int(n_perm),
        "observed": obs,
        "null_greedy_max_mean": {"mean": float(ng.mean()), "sd": float(ng.std()),
                                 "ci95": [float(np.percentile(ng, 2.5)), float(np.percentile(ng, 97.5))]},
        "null_mutual_nn": {"mean": float(nm.mean()), "sd": float(nm.std()),
                           "ci95": [float(np.percentile(nm, 2.5)), float(np.percentile(nm, 97.5))]},
        # the load-bearing numbers: how much matching survives chance
        "greedy_excess_over_null": float(obs["greedy_max_mean"] - ng.mean()),
        "greedy_p": emp_p(obs["greedy_max_mean"], ng),
        "greedy_z": z(obs["greedy_max_mean"], ng),
        "mutual_excess_over_null": float(obs["n_mutual_nn"] - nm.mean()),
        "mutual_p": emp_p(obs["n_mutual_nn"], nm),
        "mutual_z": z(obs["n_mutual_nn"], nm),
        "null_hungarian_mean": (float(np.mean(null_hung)) if null_hung else None),
    }


def heldout_frozen_pairs(Pa_sel, Pb_sel, Pa_eval, Pb_eval, n_perm=200, seed=0):
    """Select reciprocal pairs on the SELECTION rows, score those frozen pairs on the
    EVALUATION rows (disjoint documents), with an evaluation-only permutation null.

    Answers the selection-optimism question for the pooled headline (app:threats (iv)):
    the in-sample comparator is the same statistic with pairs re-selected on the
    evaluation rows, so held-out/in-sample is a ratio on identical documents. z-scoring
    uses each split's own column statistics; the candidate universe is whatever columns the
    caller passes (topk_features on the selection rows, mirroring the headline)."""
    rng = np.random.default_rng(seed)
    Za_s, ka = _zscore_cols(np.asarray(Pa_sel, float))
    Zb_s, kb = _zscore_cols(np.asarray(Pb_sel, float))
    # restrict evaluation matrices to the columns that survived z-scoring on selection
    Pa_e = np.asarray(Pa_eval, float)[:, ka]; Pb_e = np.asarray(Pb_eval, float)[:, kb]
    Za_e, ka2 = _zscore_cols(Pa_e); Zb_e, kb2 = _zscore_cols(Pb_e)
    # columns dead on the eval split are dropped from BOTH sides consistently
    Za_s = Za_s[:, ka2]; Zb_s = Zb_s[:, kb2]
    C = (Za_s.T @ Zb_s) / Za_s.shape[0]
    a2b = C.argmax(1); b2a = C.argmax(0)
    recip = b2a[a2b] == np.arange(C.shape[0])
    ia = np.arange(C.shape[0])[recip]; jb = a2b[recip]
    n_e = Za_e.shape[0]

    def stat(Zb_rows):
        return float((Za_e[:, ia] * Zb_rows[:, jb]).sum(0).mean() / n_e)

    obs = stat(Zb_e)
    null = np.array([stat(Zb_e[rng.permutation(n_e)]) for _ in range(n_perm)])
    # in-sample comparator: pairs re-selected on the evaluation rows themselves
    Ce = (Za_e.T @ Zb_e) / n_e
    a2b_e = Ce.argmax(1); b2a_e = Ce.argmax(0)
    recip_e = b2a_e[a2b_e] == np.arange(Ce.shape[0])
    ia_e = np.arange(Ce.shape[0])[recip_e]; jb_e = a2b_e[recip_e]
    obs_e = float((Za_e[:, ia_e] * Zb_e[:, jb_e]).sum(0).mean() / n_e)
    null_e = np.array([float((Za_e[:, ia_e] * Zb_e[rng.permutation(n_e)][:, jb_e]).sum(0).mean() / n_e)
                       for _ in range(n_perm)])
    shared = len(set(zip(ia.tolist(), jb.tolist())) & set(zip(ia_e.tolist(), jb_e.tolist())))
    sd = float(null.std()); sd_e = float(null_e.std())
    hx = obs - null.mean(); ix = obs_e - null_e.mean()
    return {
        "n_sel_docs": int(Za_s.shape[0]), "n_eval_docs": int(n_e),
        "n_pairs_selected": int(len(ia)), "n_pairs_reselected_on_eval": int(len(ia_e)),
        "n_pairs_shared": int(shared), "n_perm": int(n_perm),
        "held_out": {"obs_mean_pair_corr": obs, "null_mean": float(null.mean()), "null_sd": sd,
                     "excess": float(hx), "z": float(hx / sd) if sd > 1e-12 else float("inf"),
                     "p": float((np.sum(null >= obs) + 1) / (n_perm + 1))},
        "in_sample_eval_own_pairs": {"obs_mean_pair_corr": obs_e, "null_mean": float(null_e.mean()),
                                     "null_sd": sd_e, "excess": float(ix),
                                     "z": float(ix / sd_e) if sd_e > 1e-12 else float("inf"),
                                     "p": float((np.sum(null_e >= obs_e) + 1) / (n_perm + 1))},
        "ratio_heldout_over_insample": float(hx / ix) if ix else None,
    }


# ----------------------------- GPU: pooled SAE feature extraction -----------------------------
def _collect_calibration_states(mdl, tok, dev, hs_idx, calib_rows, max_states=64):
    """Collect individual per-token hidden-state vectors (pre-SAE) from a few prompts,
    for matched-L0 calibration of the random-SAE control. calibrate_l0() calls the
    *real* SAE's own .encode() per state, one token at a time -- matching how the main
    loop below applies the SAE (per-token, then mean-pooled), so the random shim's k
    equals the real SAE's mean per-token L0."""
    import torch
    states = []
    for text in calib_rows:
        if len(states) >= max_states:
            break
        inp = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True)
            h = out.hidden_states[hs_idx][0]   # [T, d_model]
        for t in range(h.shape[0]):
            states.append(h[t])
            if len(states) >= max_states:
                break
    return states


def resolve_hs_idx(cfg, sae_id, sae_release, gemma_hook_offset=0):
    """Map an SAE's hook to the HF hidden_states index it is applied to.

    Returns (hook_layer, hs_idx, metadata_hook_name). The rule below is the one every
    published receipt was produced with (commit 6543990). It is WRONG for Gemma Scope:
    sae-lens 6.x keeps the hook name in cfg.metadata.hook_name ("blocks.N.hook_resid_post"),
    cfg.hook_name does not exist, and the Gemma sae_id "layer_N/width_*/canonical" contains no
    "resid_post", so Gemma dictionaries are applied to hidden_states[N] (the input to block N)
    although they were trained on resid_post N = hidden_states[N+1]
    (<internal-notes>).
    gemma_hook_offset=1 (sensitivity analysis, notes/PREREG_gemma_hook_sensitivity_20260924.md)
    reads Gemma Scope dictionaries at hidden_states[N+1]; the default 0 keeps the published
    behaviour byte-identical. Non-Gemma suites are never touched by the offset."""
    import re
    hook_layer = getattr(cfg, "hook_layer", None)
    if hook_layer is None:
        # parse from cfg.hook_name ("blocks.6.hook_resid_pre") or sae_id
        # ("layer_12/width_16k/canonical", "blocks.6.hook_resid_pre", "l15r_8x")
        for cand in (getattr(cfg, "hook_name", None), str(sae_id)):
            if not cand:
                continue
            nums = re.findall(r"\d+", str(cand))
            if nums:
                hook_layer = int(nums[0]); break
    assert hook_layer is not None, f"could not determine hook layer for {sae_id}"
    # HF hidden_states[i]: 0=embeddings, i=output of block i-1 (= resid_pre of block i).
    # TransformerLens "resid_pre of block N" == hidden_states[N]; "resid_post of block N"
    # == hidden_states[N+1]. (The comment that stood here until 2026-09-24 claimed gemma-scope
    # "layer_N" matches hidden_states[N]; that is false -- see the docstring.)
    hook_name = str(getattr(cfg, "hook_name", "")) + " " + str(sae_id)
    hs_idx = hook_layer + 1 if "resid_post" in hook_name else hook_layer
    meta = getattr(cfg, "metadata", None)
    meta_hook = getattr(meta, "hook_name", None) if meta is not None else None
    if gemma_hook_offset:
        if str(sae_release).startswith("gemma-scope"):
            assert meta_hook == f"blocks.{hook_layer}.hook_resid_post", \
                f"gemma-scope metadata hook {meta_hook!r} does not match layer {hook_layer}"
            assert hs_idx == hook_layer, "offset would double-count an already-shifted index"
            hs_idx = hook_layer + int(gemma_hook_offset)
    return hook_layer, hs_idx, meta_hook


def compute_pooled_features(model_name, sae_release, sae_id, prompts, lang=None,
                            max_prompts=None, device="cuda", dtype=None,
                            random_sae="none", random_sae_seed=0, gemma_hook_offset=0):
    """Run prompts through `model_name`, encode the layer's residual via the SAE,
    mean-pool feature activations over each prompt's tokens -> Phi [N x n_features].
    Reuses the proven HF forward pattern (output_hidden_states) from
    probe_attribution_runner.py. SAE loaded via sae-lens.

    random_sae: "none" (default, byte-identical to the original behaviour) or
    "random-dir"/"rotate" -- wraps the loaded SAE in a matched random-SAE control
    (random_sae_baseline.RandomSAEShim) calibrated to the real SAE's mean per-token L0,
    per arXiv:2602.14111's random-baseline gate. See module docstring.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    if dtype is None:
        dtype = torch.bfloat16
    print(f"[load] model {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True).eval()
    dev = next(mdl.parameters()).device

    print(f"[load] SAE {sae_release} :: {sae_id}", flush=True)
    if sae_release.startswith("qwenpt:"):   # manual loader for Qwen's released SAEs
        from qwen_sae_adapter import load_qwen_sae
        sae = load_qwen_sae(sae_release[len("qwenpt:"):], sae_id, dev)
    else:
        sae = SAE.from_pretrained(sae_release, sae_id, device=str(dev))
        if isinstance(sae, tuple):       # some sae-lens versions return (sae, cfg, sparsity)
            sae = sae[0]
        sae = sae.to(dev)
    cfg = getattr(sae, "cfg", sae)
    hook_layer, hs_idx, meta_hook = resolve_hs_idx(cfg, sae_id, sae_release, gemma_hook_offset)
    if gemma_hook_offset:
        print(f"[sae] gemma_hook_offset={gemma_hook_offset} metadata.hook_name={meta_hook}", flush=True)
    print(f"[sae] hook_layer={hook_layer} hs_idx={hs_idx} d_sae={getattr(sae.cfg,'d_sae','?')}", flush=True)
    sae_dtype = sae.W_enc.dtype   # captured pre-wrap: RandomSAEShim has no .W_enc, only .W

    # filter by lang and sort by pair_id so that row i corresponds to the SAME concept
    # in both models even when the two sides use DIFFERENT languages (--lang_a/--lang_b).
    sel = [p for p in prompts if lang is None or p.get("lang") == lang]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    rows = [p["prompt"] for p in sel]
    if max_prompts:
        rows = rows[:max_prompts]

    if random_sae != "none":
        from random_sae_baseline import RandomSAEShim, calibrate_l0
        calib_states = _collect_calibration_states(mdl, tok, dev, hs_idx, rows[:8], max_states=64)
        k = calibrate_l0(sae, calib_states, max_states=64)
        sae = RandomSAEShim(sae, random_sae, random_sae_seed, k, device=dev, dtype=sae_dtype)
        print(f"[random-sae] {model_name}: mode={random_sae} seed={random_sae_seed} k={k}", flush=True)

    feats = []
    for i, text in enumerate(rows):
        inp = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True)
            h = out.hidden_states[hs_idx]              # [1, T, d_model]
            acts = sae.encode(h.to(sae_dtype))          # [1, T, d_sae]
            pooled = acts[0].mean(0)                    # mean over tokens -> [d_sae]
        feats.append(pooled.float().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)
    Phi = np.vstack(feats)                              # [N x d_sae]
    print(f"[features] {model_name}: Phi {Phi.shape}", flush=True)
    return Phi


def topk_features(Phi, k):
    """Keep the k highest-variance features (columns)."""
    if k is None or k >= Phi.shape[1]:
        return Phi, np.arange(Phi.shape[1])
    var = Phi.var(0)
    idx = np.argsort(var)[::-1][:k]
    idx.sort()
    return Phi[:, idx], idx


# ----------------------------- self-test (no GPU) -----------------------------
def selftest():
    """Plant S genuinely-shared features (correlated across models) in a sea of
    independent-noise features; verify match_and_null recovers them above the
    permutation null, and that pure-noise dictionaries do NOT beat the null."""
    rng = np.random.default_rng(0)
    N, K, S = 200, 400, 30
    shared = rng.standard_normal((N, S))           # S shared latent signals
    # model A: shared signals + noise features
    Pa = np.hstack([shared + 0.3 * rng.standard_normal((N, S)),
                    rng.standard_normal((N, K - S))])
    # model B: same shared signals (permuted columns) + different noise
    perm = rng.permutation(S)
    Pb = np.hstack([shared[:, perm] + 0.3 * rng.standard_normal((N, S)),
                    rng.standard_normal((N, K - S))])
    res = match_and_null(Pa, Pb, n_perm=200, seed=1, n_perm_hungarian=20)
    print("[selftest:planted]   greedy_obs=%.3f null=%.3f excess=%.3f p=%.4f z=%.1f  mutualNN=%d (null %.1f) p=%.4f"
          % (res["observed"]["greedy_max_mean"], res["null_greedy_max_mean"]["mean"],
             res["greedy_excess_over_null"], res["greedy_p"], res["greedy_z"],
             res["observed"]["n_mutual_nn"], res["null_mutual_nn"]["mean"], res["mutual_p"]))
    ok_planted = (res["greedy_p"] < 0.01 and res["greedy_excess_over_null"] > 0.05
                  and res["observed"]["n_mutual_nn"] >= S * 0.5)

    # negative control: two independent noise dictionaries (NO shared structure)
    Pa2 = rng.standard_normal((N, K)); Pb2 = rng.standard_normal((N, K))
    res2 = match_and_null(Pa2, Pb2, n_perm=200, seed=2)
    print("[selftest:noise-only] greedy_obs=%.3f null=%.3f excess=%.3f p=%.4f  <-- excess~0, p NOT significant expected"
          % (res2["observed"]["greedy_max_mean"], res2["null_greedy_max_mean"]["mean"],
             res2["greedy_excess_over_null"], res2["greedy_p"]))
    ok_noise = (res2["greedy_p"] > 0.05 and abs(res2["greedy_excess_over_null"]) < 0.03)

    # KEY construct-validity demonstration: the RAW greedy_max_mean is LARGE even
    # for pure noise (max over K columns of N-sample noise corr). Only the
    # excess-over-null is meaningful -- exactly this paper's thesis.
    print("[selftest:thesis] pure-noise RAW greedy_max_mean=%.3f (looks 'high' but excess=%.3f, p=%.3f) "
          "-> permutation null is mandatory" % (res2["observed"]["greedy_max_mean"],
                                                res2["greedy_excess_over_null"], res2["greedy_p"]))
    # held-out frozen-pair evaluator: planted survives out of sample, noise does not
    h = heldout_frozen_pairs(Pa[:N//2], Pb[:N//2], Pa[N//2:], Pb[N//2:], n_perm=200, seed=3)
    h2 = heldout_frozen_pairs(Pa2[:N//2], Pb2[:N//2], Pa2[N//2:], Pb2[N//2:], n_perm=200, seed=4)
    # The design guarantees: planted -> held-out excess at the p floor and positive; noise ->
    # held-out at null while in-sample re-selection is optimistic. It does NOT guarantee a
    # particular ratio (a selection half of 100 docs vs 400 features is heavily optimised
    # in-sample), so the ratio is asserted > 0, not against an arbitrary bar.
    ok_held = (h["held_out"]["p"] < 0.01 and h["ratio_heldout_over_insample"] > 0.0
               and h2["held_out"]["p"] > 0.05
               and h2["in_sample_eval_own_pairs"]["excess"] > h2["held_out"]["excess"])
    print("[selftest:heldout] planted held p=%.4f ratio=%.2f | noise held p=%.3f held excess=%+.4f insample excess=%+.4f -> %s"
          % (h["held_out"]["p"], h["ratio_heldout_over_insample"], h2["held_out"]["p"],
             h2["held_out"]["excess"], h2["in_sample_eval_own_pairs"]["excess"], ok_held))
    passed = ok_planted and ok_noise and ok_held
    print("SELFTEST", "PASS" if passed else "FAIL",
          "(planted recovered above null:%s ; noise stays at null:%s)" % (ok_planted, ok_noise))
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model_a"); ap.add_argument("--sae_a_release"); ap.add_argument("--sae_a_id")
    ap.add_argument("--model_b"); ap.add_argument("--sae_b_release"); ap.add_argument("--sae_b_id")
    ap.add_argument("--prompts"); ap.add_argument("--lang", default="en")
    ap.add_argument("--lang_a", default=None,
                    help="language for model A's prompts (default: --lang); with "
                         "--lang_b enables cross-lingual cells, incl. model_a==model_b")
    ap.add_argument("--lang_b", default=None, help="language for model B's prompts")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random_sae", choices=["none", "random-dir", "rotate"], default="none",
                    help="arXiv:2602.14111 random-baseline gate: wrap each loaded SAE in a "
                         "matched-dimension, matched-L0 random control (random_sae_baseline.py) "
                         "before matching. 'none' (default) is byte-identical to no flag.")
    ap.add_argument("--random_sae_seed", type=int, default=0,
                    help="seed for the random-SAE control; model B side always uses "
                         "seed+10000 so the two sides are independent")
    ap.add_argument("--select_prompts", default=None,
                    help="HELD-OUT MODE: select reciprocal pairs on THIS corpus, score the frozen "
                         "pairs on --prompts (disjoint docs) with an eval-only null. Output is a "
                         "heldout_* receipt, not a match_and_null receipt.")
    ap.add_argument("--gemma-hook-offset", dest="gemma_hook_offset", type=int, choices=[0, 1],
                    default=0,
                    help="SENSITIVITY (notes/PREREG_gemma_hook_sensitivity_20260924.md): 1 applies "
                         "Gemma Scope dictionaries to hidden_states[N+1] (their training hook, "
                         "resid_post N). 0 (default) is byte-identical to the published pipeline, "
                         "which reads hidden_states[N]. Output filename is auto-prefixed 'ghoff1_'.")
    ap.add_argument("--output")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    for req in ("model_a", "sae_a_release", "sae_a_id", "model_b", "sae_b_release",
                "sae_b_id", "prompts", "output"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required unless --selftest")

    if args.gemma_hook_offset:
        # tag + auto-prefix so a read-point sensitivity receipt can never overwrite or be
        # mistaken for a published receipt.
        d, b = os.path.split(args.output)
        if not b.startswith("ghoff1_"):
            args.output = os.path.join(d, "ghoff1_" + b)
    prompts = [json.loads(l) for l in open(args.prompts)]
    lang_a = args.lang_a or args.lang
    lang_b = args.lang_b or args.lang
    # BOTH sides get independent seeds when the random-SAE control is enabled (default
    # "none" passes these through unused, so behaviour stays byte-identical).
    Pa = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id,
                                 prompts, lang=lang_a, max_prompts=args.max_prompts,
                                 random_sae=args.random_sae, random_sae_seed=args.random_sae_seed,
                                 gemma_hook_offset=args.gemma_hook_offset)
    Pb = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id,
                                 prompts, lang=lang_b, max_prompts=args.max_prompts,
                                 random_sae=args.random_sae, random_sae_seed=args.random_sae_seed + 10_000,
                                 gemma_hook_offset=args.gemma_hook_offset)
    n = min(Pa.shape[0], Pb.shape[0])
    Pa, Pb = Pa[:n], Pb[:n]
    if args.select_prompts:
        # HELD-OUT MODE. Universe (top-k by variance) and pairs come from the selection
        # corpus only; --prompts is the evaluation corpus. Both corpora must be disjoint --
        # asserted on the prompt text, not trusted from the file names.
        sel_prompts = [json.loads(l) for l in open(args.select_prompts)]
        sel_text = {p["prompt"] for p in sel_prompts if p.get("lang") == lang_a}
        ev_text = {p["prompt"] for p in prompts if p.get("lang") == lang_a}
        assert not (sel_text & ev_text), "selection and evaluation corpora overlap"
        Pa_s = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id,
                                       sel_prompts, lang=lang_a, max_prompts=args.max_prompts,
                                       gemma_hook_offset=args.gemma_hook_offset)
        Pb_s = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id,
                                       sel_prompts, lang=lang_b, max_prompts=args.max_prompts,
                                       gemma_hook_offset=args.gemma_hook_offset)
        ns = min(Pa_s.shape[0], Pb_s.shape[0]); Pa_s, Pb_s = Pa_s[:ns], Pb_s[:ns]
        Pa_s, ia = topk_features(Pa_s, args.topk); Pb_s, ib = topk_features(Pb_s, args.topk)
        res = heldout_frozen_pairs(Pa_s, Pb_s, Pa[:, ia], Pb[:, ib], n_perm=args.n_perm, seed=args.seed)
        res.update({"mode": "heldout_frozen_pairs", "select_prompts": args.select_prompts,
                    "eval_prompts": args.prompts, "corpora_disjoint_asserted": True,
                    "model_a": args.model_a, "sae_a": f"{args.sae_a_release}::{args.sae_a_id}",
                    "model_b": args.model_b, "sae_b": f"{args.sae_b_release}::{args.sae_b_id}",
                    "lang": args.lang, "topk": args.topk, "feat_idx_a": ia.tolist(), "feat_idx_b": ib.tolist()})
        if args.gemma_hook_offset:
            res["gemma_hook_offset"] = args.gemma_hook_offset
        Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.output, "w"), indent=2)
        h = res["held_out"]; e = res["in_sample_eval_own_pairs"]
        print(f"[done:heldout] pairs={res['n_pairs_selected']} held excess={h['excess']:+.4f} "
              f"p={h['p']:.4f} z={h['z']:.1f} | insample(eval,own) excess={e['excess']:+.4f} "
              f"| ratio={res['ratio_heldout_over_insample']:.3f} -> {args.output}")
        return
    Pa, ia = topk_features(Pa, args.topk)
    Pb, ib = topk_features(Pb, args.topk)
    res = match_and_null(Pa, Pb, n_perm=args.n_perm, seed=args.seed, n_perm_hungarian=20)
    res.update({"model_a": args.model_a, "sae_a": f"{args.sae_a_release}::{args.sae_a_id}",
                "model_b": args.model_b, "sae_b": f"{args.sae_b_release}::{args.sae_b_id}",
                "lang": args.lang, "lang_a": lang_a, "lang_b": lang_b, "topk": args.topk,
                "feat_idx_a": ia.tolist(), "feat_idx_b": ib.tolist()})

    output = args.output
    if args.gemma_hook_offset:
        res["gemma_hook_offset"] = args.gemma_hook_offset
    if args.random_sae != "none":
        # tag + auto-prefix so a random-control receipt can never be mistaken for a
        # real-SAE receipt (arXiv:2602.14111 random-baseline gate).
        res["random_sae"] = {"mode": args.random_sae, "seed_a": args.random_sae_seed,
                             "seed_b": args.random_sae_seed + 10_000}
        d, b = os.path.split(output)
        if not b.startswith("rand_"):
            output = os.path.join(d, "rand_" + b)

    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(output, "w"), indent=2)
    print(f"[done] greedy excess={res['greedy_excess_over_null']:+.4f} p={res['greedy_p']:.4f} "
          f"z={res['greedy_z']:.1f} | mutualNN={res['observed']['n_mutual_nn']} "
          f"(null {res['null_mutual_nn']['mean']:.1f}) p={res['mutual_p']:.4f} -> {output}")


if __name__ == "__main__":
    main()
