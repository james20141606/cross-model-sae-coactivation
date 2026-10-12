# PREREG — nine-cell length/topic nulls, nine-cell random split, document-bootstrap intervals

Written 2026-09-26 03:00–03:30 +08:00, **before any output of these analyses exists**, except the
items listed in §2 as already seen. The commit time of this file is the evidence.
Brief: `<internal-notes>`. Review inputs:
`<internal-notes>` (W4, W5, W15, W19) and `<internal-notes>`
("needs new experiments").
Code: `code/analysis/ninecell_strengthen_20260926.py` (written after this file, committed separately).
Receipts: `results/ninecell_20260926/`.

## 1. Questions

- **Q-A (length/topic coverage).** The length-decile null and the coarse topic null (k = 10) were
  run on one cell (Gemma-2-2B↔Mistral-7B, App `app:lencov`). The title says "Length". Do the
  nine cross-family C4 cells of the held-out table still beat a permutation null that preserves
  document length (word-count deciles) and one that preserves coarse topic (TF-IDF/KMeans,
  k = 10), once each is calibrated against partitions with the same block sizes but random
  membership?
- **Q-B (random split).** The nine-cell held-out test uses consecutive halves of the 2,000-document
  slice (rows 0–999 select, 1000–1999 evaluate). Prop. 2 assumes exchangeable halves. Do all nine
  cells stay significant when the halves are drawn at random (10 seeded splits), and how far does
  the held-out excess move relative to the consecutive split?
- **Q-C (intervals).** R and Δ_E are reported as point estimates. What are their document-bootstrap
  95% intervals?

## 2. Already seen before this registration (non-blind items)

- G2–M (Gemma-2-2B↔Mistral-7B): length and k = 10 topic arms in the paper
  (`covctl_c4_gemma2-2b__mistral7b.json`, n_perm 200, excess +14.6% → +14.3% / +14.6%; no
  size-matched calibration for the length arm).
- The three ladder cells' `covctl_k200/k500_c4_*.json` receipts carry a `length_matched` arm
  (n_perm 2,000) that the paper never reports. I read the G9–M k200 one while locating inputs
  (base excess 267.5, length-matched 265.7). I have **not** read the Q1.7–G2 or L1–M length arms.
- Topic k = 200/500 point R for the three ladder cells (Table `tab:covctl`); random 50/50 split on
  the 20,000-document slice for the three ladder cells (App Q(iv), ratios 0.97–0.99); consecutive
  held-out Δ_E, r for all nine cells (Table `tab:ladder`, `heldout_*.json`).
- Blind: length and topic-10 arms for Q1.7–G9, Q1.7–M, Q8–G2, Q8–G9, Q8–M, plus Q1.7–G2 and L1–M
  with calibration; every random-split result on the 2,000-document slice; every interval.

## 3. Sample and pipeline (held fixed)

- Corpus: `data/c4_slice.jsonl` (md5 30fe9a250f82eb2bcc94ff2282e1407f), rows `lang=en` sorted by
  `pair_id`, N = 2,000; texts = first 300 characters of C4 records (App E).
- Pooled features: `cross_model_sae_match.compute_pooled_features`, unchanged (published read points,
  Gemma offset 0; mean-pool over all tokens incl. BOS; bf16 forward), on cluster GPU, original
  `google/*`, `meta-llama/*`, `Qwen/*`, `mistralai/*` weights. Six models: gemma2-2b (16k),
  gemma2-9b (16k), mistral7b (wg), qwen3-1.7b (w32k), qwen3-8b (w64k), llama3.2-1b (chanind).
- Cells (A↔B as published): qwen3-1.7b↔gemma2-2b, qwen3-1.7b↔gemma2-9b, qwen3-1.7b↔mistral7b,
  qwen3-8b↔gemma2-2b, qwen3-8b↔gemma2-9b, qwen3-8b↔mistral7b, gemma2-2b↔mistral7b,
  gemma2-9b↔mistral7b, llama3.2-1b↔mistral7b.
- Universe: top-1,500 columns by variance (`topk_features`), z-scored with `_zscore_cols`.
  Statistic: reciprocal-argmax count M. Strata builders imported verbatim from
  `c4_covariate_null.py` (`build_strata`, `strata_perm`, `size_matched_random_strata`).
- **Reproduction gate G0** (before reading any Q-A/B/C number of a cell):
  G0a: observed M on all 2,000 docs within ±3 of `c4n2k_<cell>.json` `observed.n_mutual_nn`;
  G0b: consecutive-split held-out Δ_E within ±0.005 of `heldout_<cell>.json` `held_out.excess`.
  A cell that fails G0 is still reported, flagged "extraction did not reproduce", and its verdict
  is "uncertain" (branch 0). No cell is dropped silently.
- Permutation arithmetic may run on GPU in float64 (batched); `selftest` checks it returns counts
  identical to the numpy path on synthetic data. Permutations are drawn with numpy
  `default_rng` exactly as in `c4_covariate_null.null_under`.

## 4. Estimands

**Q-A** (per cell; n_perm 2,000 for base/length/topic, seeds 0/1/2):
Δ_base, Δ_len, Δ_top10 = M − mean null M under a global, length-decile, and topic-10 permutation of
B's rows. Calibrations: Δ_cal,len and Δ_cal,top = mean Δ over 20 size-matched random partitions
(block sizes of the length / topic strata, random membership; n_perm 200 each).
R_len = Δ_len / Δ_cal,len; R_top10 = Δ_top10 / Δ_cal,top (the paper's eq. `eq:ratio`).
Also reported, not verdict-bearing: raw Δ_len/Δ_base, Δ_top10/Δ_base, p and z of each arm.

**Q-B** (per cell; 10 splits, split seed 20260926+s, s = 0..9): universe and pairs selected on the
selection half, frozen pairs scored on the other half with `heldout_frozen_pairs` (n_perm 2,000,
seed 0) — the published held-out path, only the split changes. Outputs per split: Δ_E, p, r,
pair count. Summaries: min Δ_E, max p, median and range of r, and
ρ = median over splits of Δ_E(random) / Δ_E(consecutive, recomputed).

**Q-C** (intervals, percentile 95%):
- R_len, R_top10 and Δ_base/1,500 for nine cells: B = 200 document resamples with replacement
  (seed 777+b); top-1,500 re-screened on each resample; strata labels carried with the document;
  per resample n_perm 100 for base/len/top10 and 5 size-matched partitions × 40 permutations for
  each calibration.
- R_k200, R_k500 for the three ladder cells (Q1.7–G2, G9–M, L1–M): same resamples, same scheme,
  strata from `build_strata(texts, 10, k, seed 0)` (as the covctl_k receipts).
- Δ_E for nine cells: consecutive split, frozen published-path pairs; B = 1,000 resamples of the
  evaluation documents (seed 5000+b), columns re-z-scored on the resample, Δ_E_b = frozen-pair mean
  correlation minus the mean of 20 evaluation-only permutations. Interval is conditional on the
  selected pair set.
- Caveat fixed now: with-replacement resampling duplicates documents, which shifts M and its null
  together; the interval is a document-resampling interval, not a guaranteed-coverage CI. The
  reduced per-resample n_perm adds Monte-Carlo noise, which widens intervals.
- r gets no bootstrap interval (re-selection on a resample with duplicates is biased); its spread
  over the 10 random splits of Q-B is reported instead.

## 5. Decision rules (fixed now)

Significance level for Q-A: Bonferroni over 9 cells × 2 arms, α = 0.05/18 ≈ 0.0028
(permutation floor 1/2001 ≈ 0.0005). For Q-B: α = 0.05/9 per cell, applied to every split.

**Q-A.** A cell *survives* an arm iff p ≤ 0.05/18 **and** R ≥ 0.50.
- A1 support: all 9 cells survive both arms (length and topic-10).
- A2 partial: 1–2 cells fail at least one arm, and no cell fails the length arm with R_len < 0.50.
- A3 weakened: ≥3 cells fail at least one arm, **or** any cell fails the length arm with
  R_len < 0.50.
- A0 uncertain: the cell failed G0 (reported, not counted either way).

**Q-B.**
- B1 support: every cell, every split: Δ_E > 0 and p ≤ 0.05/9.
- B2 partial: at least one split of at least one cell fails; the cell(s) and split count named.
- Optimism diagnostic (descriptive, reported in all branches): ρ < 0.90 in ≥3 cells → "the
  consecutive split may overstate held-out excess"; ρ > 1.10 in ≥3 cells → "the consecutive split
  is conservative"; otherwise "within ±10%".

**Q-C.**
- C1: every R interval (R_len, R_top10 for nine cells; R_k200, R_k500 for three) has lower bound
  ≥ 0.50, and every Δ_E interval has lower bound > 0 → intervals are reported as tightening the
  existing claims.
- C2: some R lower bound < 0.50 → that R is reported as imprecise; any sentence resting on it
  says so.
- Independent of branch: the abstract's "R ≥ 0.88" also rests on lexical and token rungs that get
  **no** interval here; text must say intervals cover the topic/length rungs only.

## 6. Paper wording per branch (templates; numbers filled only from receipts)

- A1: "Length-decile and coarse-topic ($k{=}10$) nulls, each calibrated against size-matched
  random strata, leave $R_{\text{len}}\in[a,b]$ and $R_{\text{top}}\in[c,d]$ in all nine cells
  (all $p=1/2001$; App X)." Figure 1/abstract "(one mid-range cell)" qualifiers can be replaced by
  "(all nine cells)".
- A2: "... in seven [eight] of nine cells; in <cells> the length [topic] null leaves R = <v>
  (App X)." Qualifier "(one mid-range cell)" becomes "(<k> of nine cells)".
- A3: "Length [topic] explains a substantial part of the excess in <cells> (R = <v>); the
  coactivation claim holds for <k> of nine cells." The writer must flag to the author that the
  title word "Length" no longer covers the nine-cell set.
- B1: "Ten seeded random splits of the 2,000 documents confirm all nine cells (every $p=1/2001$;
  held-out excess <ρ-range> of the consecutive split; App X)."
- B2: "... all but <cell> (<n>/10 splits significant)."
- C1/C2: "Document-bootstrap 95% intervals: $R_{\text{len}}$ <..>, $R_{\text{top}}$ <..>,
  $\Delta_E$ <..> (App X); the lexical and token rungs have point estimates only."

Whatever the branch, results are reported. Thresholds above are not changed after reading results.

## 7. Deviations policy

Any deviation (missing model weights, a cell that cannot run, a changed n_perm because of time)
is written into the receipt and the report as a deviation with its reason, not folded in silently.
If the cluster is unreachable before 09:00, the fallback is: report the pre-existing `length_matched`
arms of the three `covctl_k200_c4_*` receipts as "existing, not pre-registered, uncalibrated"
coverage, and run nothing new locally except with explicit approval.

## 8. Deviation 1 (appended 2026-09-26 07:08 +08:00, before any Q-A/Q-B/Q-C output exists)

cluster was unreachable from 03:00 to 07:05: the university VPN was not connected, the route went through
en0, and ssh timed out. The coordinator approved a local fallback at 07:05. It changes the
platform and the cell coverage. It does not change the analyses or the decision rules.

- **Platform.** Local CPU (macOS, numpy; count engine `numpy` = the published loop), run under
  `nice -n 10`. Receipts are tagged `platform=local_mps_cache`.
- **Features.** These are the existing pooled caches in `~/gemmahook_cache/pooled/`, extracted
  2026-09-24 on Apple MPS by `gemma_hook_sensitivity.py extract`. That run used the same
  2,000-document slice (md5 30fe9a…), the published read points
  (gemma2-2b `hs12`, gemma2-9b `hs20`, mistral7b `hs16`, qwen3-1.7b `hs15`), and mean-pooling
  including BOS. The differences from the published setting:
  - Gemma weights are the ungated `unsloth/gemma-2-*` re-uploads, not `google/*`, and were not
    checked bit for bit.
  - The forward pass ran on MPS, not CUDA.
- **Reproduction check already on record** (`gemma_hook_sensitivity/GEMMA_HOOK_SENSITIVITY.json`,
  gate G0, 2026-09-24). Local screen M vs the published receipts:
  - dM = +1 / +3 / +1 / 0 for Q1.7–G2 / Q1.7–G9 / G9–M / G2–M.
  - Held-out Δ_E differs by ≤ 0.0019.
  - Q1.7–M was not in that gate; it is checked here by G0a/G0b.
  - This agent's G0 tolerance is ±3, so Q1.7–G9 (dM = +3) passes here although it failed the
    stricter ±2 gate of 09-24.
- **Coverage.** 5/9 cells: Q1.7–G2, Q1.7–G9, Q1.7–M, G2–M, G9–M. Q8–G2, Q8–G9, Q8–M and L1–M
  have no local cache and are not run. The decision rules in §5 are applied to these 5 cells
  only. The receipt and the report say "5 of 9 cells". They do not count as the nine-cell
  branch A1/B1.
- **Q-C scope.** The R bootstrap is not run: it takes about 1 h per cell locally. Δ_E bootstrap
  is run (B = 1,000). R therefore stays at point estimates in this fallback.
- **Code change needed for this.** When `n_boot = 0`, `decide()` counted the missing R intervals
  as "lower bound < 0.5". It now skips them and lists them as not computed. This is a reporting
  fix only; no statistic changed.
- **If the cluster connects before the local run ends,** the full nine-cell job is submitted at once.
  The cluster results then govern, and the local results serve as a consistency check only.
