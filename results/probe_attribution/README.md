# Probe-Attribution Result Artifacts

These JSON files are the canonical per-model outputs of the v1 probe-attribution
pipeline reported in `paper/main.tex` (Section: Results, Table referencing
"Probe-attribution scaling on Qwen3"). One file per model size:

| Model | File |
|-------|------|
| Qwen3-0.6B | `probe_attribution_Qwen_Qwen3-0.6B.json` |
| Qwen3-1.7B | `probe_attribution_Qwen_Qwen3-1.7B.json` |
| Qwen3-4B   | `probe_attribution_Qwen_Qwen3-4B.json` |
| Qwen3-8B   | `probe_attribution_Qwen_Qwen3-8B.json` |
| Qwen3-14B  | `probe_attribution_Qwen_Qwen3-14B.json` (n=3/pair, high-variance, reported with explicit small-sample caveat) |

## Reproduction

```bash
# Generate (requires HF cache + GPU)
python experiments/06_probe_attribution_run.py \
    --model Qwen/Qwen3-0.6B \
    --prompts data/prompts_en_zh_ja_ko.jsonl \
    --output results/probe_attribution/probe_attribution_Qwen_Qwen3-0.6B.json

# Repeat for {1.7B, 4B, 8B, 14B}.
```

Headline metric (per file, key `mean_pairwise_r`):
- Pearson correlation of mid-layer logit-lens probe-attribution vectors
  between aligned prompt pairs across en/zh/ja/ko.
- 6 language pairs per model; mean over pairs reported in paper.

## Schema (top-level keys)

- `model_id`: e.g. `Qwen/Qwen3-0.6B`
- `n_prompts_per_lang`: {en: int, zh: int, ja: int, ko: int}
- `layers`: list of layer indices probed
- `mid_layer_band`: layers used for cross-lingual aggregation
- `pairwise_correlations`: dict of {`en-zh`: float, `en-ja`: float, ...}
- `mean_pairwise_r`: float (the headline number per model)
- `per_layer_correlations`: per-layer Pearson r values
