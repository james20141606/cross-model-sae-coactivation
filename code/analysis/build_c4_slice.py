#!/usr/bin/env python3
"""Build a natural-text C4 slice for the SAE-matching template-confound control.

The 206-prompt set spans 9 templated concept families; a skeptic can ask whether the
mutual-NN feature matches are category/template-tuned rather than general. Control: rerun
the headline pairs on a few thousand C4 documents (natural text, no templates). If the
mutual-NN sharing (z over permutation null) survives on C4, the result is not a
template artifact.

Streams allenai/c4 (en), takes the first N docs, truncates each to ~max_chars, writes
data/c4_slice.jsonl with the same schema as prompts_v3_multi.jsonl (pair_id, lang, prompt).

Run on the cluster (needs `datasets` + network):
  <ENV_TRAIN>/bin/python code/analysis/build_c4_slice.py --n 2000
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max_chars", type=int, default=300)
    ap.add_argument("--out", default="data/c4_slice.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= args.n:
            break
        text = " ".join(ex["text"].split())[:args.max_chars]
        if len(text) < 50:
            continue
        rows.append({"pair_id": len(rows), "lang": "en", "family": "c4",
                     "prompt": text, "target": ""})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(rows)} docs (<= {args.max_chars} chars each)")


if __name__ == "__main__":
    main()
