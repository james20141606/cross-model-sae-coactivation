#!/usr/bin/env python3
"""What do the shared features DO? Interpret top mutual-NN matched feature pairs.

For a model pair with validated cross-family sharing (e.g. gemma-2-9b <-> mistral-7b on
C4), recompute prompt-pooled SAE activations, take the top-K mutual-NN matched feature
pairs by correlation, and for each side dump the top-N max-activating documents. If a
matched pair's two features fire on the same documents/topics in two independently
trained models, universality becomes tangible and auditable.

Outputs JSON: per matched pair -> corr, per-model top docs (text snippets + activation),
and the doc-overlap count (how many of the top-N docs coincide across models).

Usage (GPU):
  python interpret_matched_features.py \
    --model_a google/gemma-2-9b --sae_a_release gemma-scope-9b-pt-res-canonical --sae_a_id layer_20/width_16k/canonical \
    --model_b mistralai/Mistral-7B-v0.1 --sae_b_release mistral-7b-res-wg --sae_b_id blocks.16.hook_resid_pre \
    --prompts data/c4_slice.jsonl --max_prompts 2000 --topk 1500 --k_pairs 20 --n_top_docs 5 \
    --output results/cross_model_sae/interp_gemma2-9b__mistral7b.json
"""
import argparse, json, os
from pathlib import Path
import numpy as np

from cross_model_sae_match import compute_pooled_features, _zscore_cols, topk_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True); ap.add_argument("--sae_a_release", required=True); ap.add_argument("--sae_a_id", required=True)
    ap.add_argument("--model_b", required=True); ap.add_argument("--sae_b_release", required=True); ap.add_argument("--sae_b_id", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--lang", default="en")
    ap.add_argument("--max_prompts", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--k_pairs", type=int, default=20)
    ap.add_argument("--n_top_docs", type=int, default=5)
    ap.add_argument("--audit_prompts", default=None,
                    help="optional HELD-OUT prompts jsonl: matches are fitted on --prompts, "
                         "but top documents (and a held-out pooled corr per matched pair) "
                         "are computed on these documents instead (kills the in-sample "
                         "top-doc-audit objection)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts)]
    sel = [p for p in prompts if p.get("lang") == args.lang]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    sel = sel[:args.max_prompts]
    texts = [p["prompt"] for p in sel]

    # With --audit_prompts, run ONE forward pass per model over fit+audit docs
    # (re-loading each model twice OOMs a 46GB L40S), then split the rows.
    audit_texts, n_fit = None, len(sel)
    feed = sel
    if args.audit_prompts:
        asel = [json.loads(l) for l in open(args.audit_prompts)]
        asel = [p for p in asel if p.get("lang") == args.lang]
        asel.sort(key=lambda p: p.get("pair_id", 0))
        asel = asel[:args.max_prompts]
        audit_texts = [p["prompt"] for p in asel]
        feed = [dict(p, pair_id=i) for i, p in enumerate(sel + asel)]

    Pa_all = compute_pooled_features(args.model_a, args.sae_a_release, args.sae_a_id,
                                     feed, lang=args.lang, max_prompts=None)
    Pb_all = compute_pooled_features(args.model_b, args.sae_b_release, args.sae_b_id,
                                     feed, lang=args.lang, max_prompts=None)
    Pa, Pb = Pa_all[:n_fit], Pb_all[:n_fit]
    n = min(Pa.shape[0], Pb.shape[0], len(texts))
    Pa, Pb, texts = Pa[:n], Pb[:n], texts[:n]

    PaK, ia = topk_features(Pa, args.topk)
    PbK, ib = topk_features(Pb, args.topk)
    Za, ka = _zscore_cols(PaK); Zb, kb = _zscore_cols(PbK)
    ia, ib = ia[ka], ib[kb]
    C = (Za.T @ Zb) / Za.shape[0]
    a2b = C.argmax(1); b2a = C.argmax(0)
    recip = np.where(b2a[a2b] == np.arange(C.shape[0]))[0]
    order = recip[np.argsort(C[recip, a2b[recip]])[::-1]][:args.k_pairs]

    # doc-dump source: fit corpus by default; held-out corpus with --audit_prompts
    Pa_doc, Pb_doc, texts_doc, audit_file = Pa, Pb, texts, None
    if args.audit_prompts:
        audit_file = args.audit_prompts
        Pa_doc, Pb_doc = Pa_all[n_fit:], Pb_all[n_fit:]
        texts_doc = audit_texts
        na = min(Pa_doc.shape[0], Pb_doc.shape[0], len(texts_doc))
        Pa_doc, Pb_doc, texts_doc = Pa_doc[:na], Pb_doc[:na], texts_doc[:na]
        print(f"[audit] top docs + held-out corr from {na} held-out docs ({audit_file})", flush=True)

    out = {"model_a": args.model_a, "model_b": args.model_b, "n_docs": int(n),
           "audit_prompts": audit_file, "n_audit_docs": int(Pa_doc.shape[0]),
           "k_pairs": int(len(order)), "pairs": []}
    for r in order:
        fa = int(ia[r]); fb = int(ib[a2b[r]]); corr = float(C[r, a2b[r]])
        acts_a = Pa_doc[:, fa]; acts_b = Pb_doc[:, fb]
        top_a = np.argsort(acts_a)[::-1][:args.n_top_docs]
        top_b = np.argsort(acts_b)[::-1][:args.n_top_docs]
        overlap = len(set(top_a.tolist()) & set(top_b.tolist()))
        # pooled corr of this matched pair on the doc-dump corpus (== fit corpus unless
        # --audit_prompts, in which case this is the HELD-OUT calibration number)
        sa, sb = acts_a.std(), acts_b.std()
        doc_corr = float(np.corrcoef(acts_a, acts_b)[0, 1]) if sa > 1e-9 and sb > 1e-9 else 0.0
        out["pairs"].append({
            "feature_a": fa, "feature_b": fb, "pooled_corr": corr,
            "doc_corpus_corr": doc_corr,
            "top_doc_overlap": overlap,
            "top_docs_a": [{"doc": int(i), "act": float(acts_a[i]), "text": texts_doc[i][:160]} for i in top_a],
            "top_docs_b": [{"doc": int(i), "act": float(acts_b[i]), "text": texts_doc[i][:160]} for i in top_b],
        })
        print(f"[pair] A:{fa} B:{fb} fit_corr={corr:.3f} doc_corr={doc_corr:.3f} "
              f"overlap={overlap}/{args.n_top_docs} | A-top: {texts_doc[top_a[0]][:70]!r}", flush=True)

    ov = [p["top_doc_overlap"] for p in out["pairs"]]
    out["mean_top_doc_overlap"] = float(np.mean(ov)) if ov else 0.0
    out["mean_doc_corpus_corr"] = float(np.mean([p["doc_corpus_corr"] for p in out["pairs"]])) if out["pairs"] else 0.0
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(f"[done] {len(order)} matched pairs, mean top-{args.n_top_docs} doc overlap "
          f"{out['mean_top_doc_overlap']:.2f} -> {args.output}")


if __name__ == "__main__":
    main()
