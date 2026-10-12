#!/usr/bin/env python3
"""BOS probe for the SAE-matching arm: which tokenizers prepend BOS, and how long are the C4 docs?

App. threats (v) says the pooled feature (mean over every token of a model's own
tokenization, `cross_model_sae_match.py`: `tok(text)` with default special tokens, then
`acts[0].mean(0)`) carries a BOS term c_f/T_d when the tokenizer prepends BOS, and that at
the pinned revisions Gemma-2, Mistral-7B and Llama-3.2 prepend BOS while Qwen3-Base does not.
This script is the receipt for that sentence. It loads tokenizers only (no weights), at the
revisions pinned in tab:model-revisions, from the offline HF cache, tokenizes exactly as the
pipeline does, and records per model:
  - the tokenizer's bos_token / bos_token_id / add_bos_token attribute,
  - the fraction of C4 slice documents whose first input id is the BOS id,
  - T_d (token count incl. special tokens) over the 2,000-document C4 slice,
plus the character-length distribution of the C4 files the paper uses (build_c4_slice.py
truncates each streamed document to its first --max_chars=300 characters).

Run on the cluster (tokenizers only; read-only on the cache and data):
  HF_HUB_OFFLINE=1 HF_HOME=<HF_CACHE> \
  <ENV_ANALYSIS>/bin/python code/analysis/bos_probe_ct.py \
      --data_dir <REPO_ROOT>/data \
      --out results/cross_model_sae/bos_probe_ct.json
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

# hf id -> revision pinned in paper/sections/tab_hyperparams.tex (tab:model-revisions)
PINNED = {
    "google/gemma-2-2b": "c5ebcd40d208330abc697524c919956e692655cf",
    "google/gemma-2-9b": "33c193028431c2fde6c6e51f29e6f17b60cbfac6",
    "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
}
C4_FILES = ["c4_slice.jsonl", "c4_fit.jsonl", "c4_heldout.jsonl", "c4_slice_20k.jsonl"]


def summ(xs):
    return dict(n=len(xs), min=min(xs), max=max(xs), mean=round(statistics.mean(xs), 2),
                median=statistics.median(xs))


def read_prompts(path):
    return [json.loads(l)["prompt"] for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer

    out = dict(script="code/analysis/bos_probe_ct.py",
               transformers=transformers.__version__,
               hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"),
               tokenization="tok(text) with default add_special_tokens, as cross_model_sae_match.py",
               c4_files={}, tokenizers={})

    for name in C4_FILES:
        p = os.path.join(args.data_dir, name)
        raw = open(p, "rb").read()
        lens = [len(t) for t in read_prompts(p)]
        out["c4_files"][name] = dict(sha256=hashlib.sha256(raw).hexdigest(), chars=summ(lens),
                                     n_at_300=sum(x == 300 for x in lens))

    docs = read_prompts(os.path.join(args.data_dir, "c4_slice.jsonl"))  # docs 1-2000
    for hf_id, rev in PINNED.items():
        tok = AutoTokenizer.from_pretrained(hf_id, revision=rev, trust_remote_code=True)
        ids = [tok(t)["input_ids"] for t in docs]
        bos = tok.bos_token_id
        first = tok.convert_ids_to_tokens(ids[0][:3])
        n_bos_first = sum(1 for x in ids if bos is not None and x[0] == bos)
        out["tokenizers"][hf_id] = dict(
            revision=rev, cls=type(tok).__name__,
            bos_token=tok.bos_token, bos_token_id=bos,
            add_bos_token=getattr(tok, "add_bos_token", None),
            first_three_tokens_doc1=first,
            frac_docs_first_id_is_bos=n_bos_first / len(ids),
            prepends_bos=(n_bos_first == len(ids)),
            T_d_c4_slice=summ([len(x) for x in ids]),
        )
        print(hf_id, rev[:8], "bos=%r" % tok.bos_token, "prepends=%s" % (n_bos_first == len(ids)),
              "first=%s" % first, "T_d=%s" % out["tokenizers"][hf_id]["T_d_c4_slice"], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
