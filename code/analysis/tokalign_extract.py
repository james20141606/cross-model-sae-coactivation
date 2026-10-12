#!/usr/bin/env python3
"""Per-token SAE activation extraction for the token-aligned sharing test.

Design registered in PREREGISTRATION_token_aligned.md (commit df330c6) BEFORE this file
existed. One (model, SAE) side per invocation; 5 unique sides cover the 3 registered
cells (Mistral blocks.16 shared by two cells, extracted once).

FORWARD PATH IS BYTE-IDENTICAL to cross_model_sae_match.compute_pooled_features (the
computation that produced every committed lextok receipt): same AutoTokenizer /
AutoModelForCausalLM / SAE loading, same bf16 dtype, same hs_idx rule, same
`sae.encode(h)` and `acts[0].mean(0)` pooling. The only additions:
  * `return_offsets_mapping=True` at tokenizer-call time (does not change input_ids;
    the offset_mapping tensor is popped before the model forward);
  * the per-token activations are SAVED (sparse: value > 0) instead of discarded;
  * the bf16-pooled vector is saved per doc (fp16) -- this is gate G0's anchor.

Shards are written atomically (tmp + os.replace) and SKIPPED IF PRESENT, so an evicted
job resumes where it stopped. Nothing here reads any cached pooled matrix or any frozen
pair set: extraction is verdict-blind.

Shard schema (npz, one per --shard_size docs):
  doc_idx   int32 [n_tok]   row -> document index (absolute, into the sorted corpus)
  tok_pos   int32 [n_tok]   row -> token position within its doc (0-based, incl. specials)
  off_start int32 [n_tok]   char span start (offset_mapping)
  off_end   int32 [n_tok]   char span end; start==end marks specials/empty spans
  input_id  int32 [n_tok]   token id (QA/debug only)
  indptr    int64 [n_tok+1] CSR pointers into feat/val
  feat      int32 [nnz]     SAE feature index (FULL dictionary)
  val       f16   [nnz]     activation value (>0)
  pooled    f16   [n_docs_shard, d_sae]  the bf16 mean over ALL token rows, per doc
  doc_start, doc_end int64  absolute doc range [start, end)
  meta      str             json: model, sae, hs_idx, d_sae, interpreter, versions

Usage (GPU):
  python tokalign_extract.py --model google/gemma-2-2b \
    --sae_release gemma-scope-2b-pt-res-canonical --sae_id layer_12/width_16k/canonical \
    --prompts data/c4_slice_20k.jsonl --out_dir cache/tokalign/gemma2-2b_l12
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DENSITY_GUARD_L0 = 500.0   # mean per-token L0 above this -> fail loudly (storage model wrong)


def load_docs(prompts_path, lang="en", max_prompts=None):
    """EXACTLY lexical_token_test.py's selection: filter lang, sort by pair_id."""
    prompts = [json.loads(l) for l in open(prompts_path)]
    sel = [p for p in prompts if lang is None or p.get("lang") == lang]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    if max_prompts:
        sel = sel[:max_prompts]
    return [p["prompt"] for p in sel]


def load_model_and_sae(model_name, sae_release, sae_id):
    """Verbatim port of the loading block of compute_pooled_features."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    dtype = torch.bfloat16
    print(f"[load] model {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True).eval()
    dev = next(mdl.parameters()).device

    print(f"[load] SAE {sae_release} :: {sae_id}", flush=True)
    if sae_release.startswith("qwenpt:"):
        from qwen_sae_adapter import load_qwen_sae
        sae = load_qwen_sae(sae_release[len("qwenpt:"):], sae_id, dev)
    else:
        sae = SAE.from_pretrained(sae_release, sae_id, device=str(dev))
        if isinstance(sae, tuple):
            sae = sae[0]
        sae = sae.to(dev)
    import re
    cfg = getattr(sae, "cfg", sae)
    hook_layer = getattr(cfg, "hook_layer", None)
    if hook_layer is None:
        for cand in (getattr(cfg, "hook_name", None), str(sae_id)):
            if not cand:
                continue
            nums = re.findall(r"\d+", str(cand))
            if nums:
                hook_layer = int(nums[0]); break
    assert hook_layer is not None, f"could not determine hook layer for {sae_id}"
    hook_name = str(getattr(cfg, "hook_name", "")) + " " + str(sae_id)
    hs_idx = hook_layer + 1 if "resid_post" in hook_name else hook_layer
    d_sae = int(getattr(sae.cfg, "d_sae"))
    print(f"[sae] hook_layer={hook_layer} hs_idx={hs_idx} d_sae={d_sae}", flush=True)
    sae_dtype = sae.W_enc.dtype
    return tok, mdl, dev, sae, sae_dtype, hs_idx, d_sae


def extract_shard(tok, mdl, dev, sae, sae_dtype, hs_idx, d_sae, texts, doc_start, meta):
    """Extract one shard; returns dict of arrays for np.savez."""
    import torch
    doc_idx, tok_pos, off_s, off_e, input_id = [], [], [], [], []
    indptr = [0]
    feats, vals = [], []
    pooled_rows = []
    for di, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        if "offset_mapping" not in enc:
            raise RuntimeError("tokenizer returned no offset_mapping (not a fast "
                               "tokenizer?) -- hard error per registration")
        offsets = enc.pop("offset_mapping")[0].tolist()   # [T][2]
        inp = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True)
            h = out.hidden_states[hs_idx]                 # [1, T, d_model]
            acts = sae.encode(h.to(sae_dtype))            # [1, T, d_sae]
            pooled = acts[0].mean(0)                      # bf16 mean over ALL rows (G0)
            A = acts[0].float()                           # [T, d_sae]
            nz = (A > 0)
            nz_idx = nz.nonzero()                         # [nnz, 2] (t, f)
            nz_val = A[nz]
        pooled_rows.append(pooled.float().cpu().numpy().astype(np.float16))
        T = A.shape[0]
        ids = inp["input_ids"][0].tolist()
        if len(offsets) != T:
            raise RuntimeError(f"offset/token length mismatch doc {doc_start + di}: "
                               f"{len(offsets)} vs {T}")
        nz_t = nz_idx[:, 0].cpu().numpy()
        nz_f = nz_idx[:, 1].to(torch.int32).cpu().numpy()
        nz_v = nz_val.cpu().numpy().astype(np.float16)
        # per-token CSR within this doc, appended in token order
        counts = np.bincount(nz_t, minlength=T)
        order = np.argsort(nz_t, kind="stable")
        feats.append(nz_f[order]); vals.append(nz_v[order])
        for t in range(T):
            doc_idx.append(doc_start + di); tok_pos.append(t)
            off_s.append(offsets[t][0]); off_e.append(offsets[t][1])
            input_id.append(ids[t])
            indptr.append(indptr[-1] + int(counts[t]))
    return {
        "doc_idx": np.asarray(doc_idx, np.int32),
        "tok_pos": np.asarray(tok_pos, np.int32),
        "off_start": np.asarray(off_s, np.int32),
        "off_end": np.asarray(off_e, np.int32),
        "input_id": np.asarray(input_id, np.int32),
        "indptr": np.asarray(indptr, np.int64),
        "feat": (np.concatenate(feats) if feats else np.zeros(0, np.int32)),
        "val": (np.concatenate(vals) if vals else np.zeros(0, np.float16)),
        "pooled": np.stack(pooled_rows),
        "doc_start": np.int64(doc_start),
        "doc_end": np.int64(doc_start + len(texts)),
        "meta": np.str_(json.dumps(meta)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sae_release", required=True)
    ap.add_argument("--sae_id", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--shard_size", type=int, default=1000)
    args = ap.parse_args()

    texts = load_docs(args.prompts, args.lang, args.max_prompts)
    n = len(texts)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = [(s, min(s + args.shard_size, n)) for s in range(0, n, args.shard_size)]
    todo = [(s, e) for s, e in shards
            if not (out / f"shard_{s:05d}_{e:05d}.npz").exists()]
    print(f"[plan] {n} docs, {len(shards)} shards, {len(todo)} to do "
          f"(skip-if-exists resume)", flush=True)

    receipt_path = out / "extract_receipt.json"
    if not todo and receipt_path.exists():
        print("[done] all shards + receipt already present; nothing to do", flush=True)
        return 0

    tok, mdl, dev, sae, sae_dtype, hs_idx, d_sae = load_model_and_sae(
        args.model, args.sae_release, args.sae_id)
    import transformers, sae_lens, torch
    meta = {"model": args.model, "sae": f"{args.sae_release}::{args.sae_id}",
            "hs_idx": hs_idx, "d_sae": d_sae, "prompts": args.prompts,
            "n_docs": n, "shard_size": args.shard_size,
            "interpreter": sys.executable,
            "transformers": transformers.__version__, "sae_lens": sae_lens.__version__,
            "torch": torch.__version__,
            "prereg_commit": "df330c6"}

    t0 = time.time()
    total_tok, total_nnz = 0, 0
    for s, e in shards:
        p = out / f"shard_{s:05d}_{e:05d}.npz"
        if p.exists():
            print(f"[skip] {p.name}", flush=True)
            continue
        ts = time.time()
        arrs = extract_shard(tok, mdl, dev, sae, sae_dtype, hs_idx, d_sae,
                             texts[s:e], s, meta)
        total_tok += len(arrs["doc_idx"]); total_nnz += len(arrs["feat"])
        tmp = out / f".tmp_shard_{s:05d}_{e:05d}.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, p)
        l0 = len(arrs["feat"]) / max(1, len(arrs["doc_idx"]))
        print(f"[shard {s:05d}-{e:05d}] {len(arrs['doc_idx'])} tokens, "
              f"nnz={len(arrs['feat'])} (L0={l0:.1f}), {time.time()-ts:.1f}s -> {p.name}",
              flush=True)
        if l0 > DENSITY_GUARD_L0:
            raise RuntimeError(f"mean L0 {l0:.1f} > {DENSITY_GUARD_L0}: sparse storage "
                               "model wrong for this SAE -- fail loudly per registration")

    # side receipt: written LAST, only when every shard exists
    missing = [f"shard_{s:05d}_{e:05d}.npz" for s, e in shards
               if not (out / f"shard_{s:05d}_{e:05d}.npz").exists()]
    if missing:
        print(f"[FAIL] {len(missing)} shards still missing after pass", file=sys.stderr)
        return 1
    import socket
    receipt = dict(meta)
    receipt.update({"n_shards": len(shards), "host": socket.gethostname(),
                    "gpu": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "none"),
                    "elapsed_s_this_pass": round(time.time() - t0, 1),
                    "tokens_this_pass": total_tok, "nnz_this_pass": total_nnz})
    tmpr = out / ".tmp_receipt.json"
    json.dump(receipt, open(tmpr, "w"), indent=2)
    os.replace(tmpr, receipt_path)
    print(f"[done] {len(shards)} shards complete -> {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
