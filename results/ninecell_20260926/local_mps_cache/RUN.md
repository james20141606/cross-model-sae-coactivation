# ninecell_20260926 / local_mps_cache: run record

Prereg: `notes/PREREG_ninecell_ladder_20260926.md`. It was committed at 3a0b358 (03:01). Deviation 1 is b196641 (07:05); it covers the local CPU fallback, the 5 cached cells, and the dropped R bootstrap.

- **Platform.** macOS, local CPU, numpy count engine, `nice -n 10`. There is no SLURM job, so there is no JID.
- **Script.** `code/analysis/ninecell_strengthen_20260926.py` at commit 6bd9078 (sha256 prefix fb5370c52f35). Every A_ and B_ receipt carries this hash.
- **Pooled features.** From `~/gemmahook_cache/pooled/`, extracted 2026-09-24 on Apple MPS with unsloth Gemma weights. Each sha256 matches its `.meta.json`:
  - gemma2-2b 16k hs12: 4f4bfb96…
  - gemma2-9b 16k hs20: 66f54959…
  - mistral7b wg hs16: 8faf49d1…
  - qwen3-1.7b w32k hs15: bfa54de1…
- **Corpus.** `~/lanfilt_cache/c4_slice.jsonl`, md5 30fe9a250f82eb2bcc94ff2282e1407f (same as `data/c4_slice.jsonl`).
- **sklearn.** Used only for the TF-IDF/KMeans strata. Throwaway install at `/tmp/ct_sk` (sklearn 1.9.1); not a shared env.

## Commands
```bash
# symlinks /tmp/ct_local/pooled/{gemma2-2b,gemma2-9b,mistral7b,qwen3-1.7b}.npy -> the four cache files above
bash code/scripts/run_ninecell_local_fallback_20260926.sh      # = /tmp/ct_local/run_local.sh, started 07:06
#   B (analyze_b, OMP 2) ran sequentially for the 5 cells, 07:06-07:19, 138-157 s each.
#   A (analyze_a --device numpy --n_boot 0) first cell ran sequentially. At 07:21 the parent loop
#   was killed and the remaining 4 A cells were started in parallel, 1 thread each, nice 10:
for C in gemma2-2b__mistral7b qwen3-1.7b__mistral7b qwen3-1.7b__gemma2-9b gemma2-9b__mistral7b; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 python ninecell_strengthen_20260926.py analyze_a \
    --cell $C --pooled /tmp/ct_local/pooled --out results/ninecell_20260926/local_mps_cache/A_$C.json \
    --device numpy --n_boot 0 --platform local_mps_cache & done
python ninecell_strengthen_20260926.py summarize --dir results/ninecell_20260926/local_mps_cache
```
The progress log is missing timestamps because macOS `date` has no `-Is`; the times inside each receipt (`date`) are correct. `summarize` was re-run after table-format-only changes to the code (labels, header, relative path). The receipts were not recomputed.

## Receipts
- `A_<cell>.json`: G0a, plus the Q-A point estimates. Arms are base, len, top10, and top200/top500 on the ladder cells. The calibration reps are included.
- `B_<cell>.json`: G0b, the consecutive split, the 10 random splits, and the Δ_E bootstrap draws.
- `NINECELL_SUMMARY.json`: decision branches plus per-cell rows.
- `tab_ninecell.tex`: generated table.
- `logs/`: logs for each run.
