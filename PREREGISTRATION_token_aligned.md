# PREREGISTRATION: token-aligned, order-sensitive compositional sharing (S1/S2)

Committed BEFORE any per-token activation exists anywhere (local or cluster; verified in
PLAN_token_aligned_sharing.md §2: every existing dump is post-`mean(0)`). Git history is
the timestamp. This freezes the design of PLAN_token_aligned_sharing.md §3–§5 into
binding rules; where the plan marked a number as a *proposal*, the value below is the
registered value. Nothing in this file may be retuned after the first receipt exists;
amendments go into `experiments_detail.md` as new entries and are only legitimate if they
are committed before the receipt they affect.

**NO INCENTIVE DECLARATION** (the 6848d4c discipline). All three registered outcome
branches move the paper up — branch 1 lifts the scope limit, branch 2 is the sharpest
negative the paper could own (it closes the shared-unigram hypothesis *against* the
published positive, instantiating the paper's own thesis once more), branch 3 refutes the
token-detector deflation and sharpens the published claim's boundary. I have no reason to
prefer any branch and will report whichever the receipts give. If I find myself wanting
one, that is a bug.

---

## 0. Frozen inputs (checksums pinned at registration time)

- Corpus: `data/c4_slice_20k.jsonl` (cluster), 20,000 docs, md5
  `64e994008170f6c5738bdb5dc19f285e`. Document order and selection are exactly
  `lexical_token_test.py`'s: filter `lang == "en"`, sort by `pair_id`. Texts are already
  single-space-normalized by `build_c4_slice.py` (`" ".join(text.split())[:300]`).
- Committed pooled caches (the G0 anchors), cluster `cache/`:
  - `lextok20k_qwen3-1.7b__gemma2-2b.npz`  md5 `74651e10dc0f402f139f2e94d604ee95`
  - `lextok20k_gemma2-9b__mistral7b.npz`   md5 `be768bbd5cb2530c16a2fa0554cfece8`
  - `lextok20k_llama3.2-1b__mistral7b.npz` md5 `88b3cf06172d758832f9a59dabad522c`
- Frozen pair sets: `frozen_pairs.feat_a/feat_b` (mutual-NN, prereg 6848d4c) from the
  committed receipts (local md5s pinned):
  - `results/cross_model_sae/lextok20k_qwen3-1.7b__gemma2-2b.json` n=822, md5 `5512e05136307e87921dc1a2a9141f5f`
  - `results/cross_model_sae/lextok20k_gemma2-9b__mistral7b.json`  n=806, md5 `8466990af60fa6eea751634bff66b3be`
  - `results/cross_model_sae/lextok20k_llama3.2-1b__mistral7b.json` n=520, md5 `a2f9ffdb45a0da948f5b2a66ed767498`
  These indices are **columns of the cached top-1500 matrices**, not dictionary IDs;
  gate G0 recovers the dictionary anchoring.
- Cells (identical to the lextok cells; nothing added, nothing dropped):
  | cell | model_a / sae_a | model_b / sae_b |
  |---|---|---|
  | strong | Qwen/Qwen3-1.7B-Base, qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50::layer14 | google/gemma-2-2b, gemma-scope-2b-pt-res-canonical::layer_12/width_16k/canonical |
  | median | google/gemma-2-9b, gemma-scope-9b-pt-res-canonical::layer_20/width_16k/canonical | mistralai/Mistral-7B-v0.1, mistral-7b-res-wg::blocks.16.hook_resid_pre |
  | weak | meta-llama/Llama-3.2-1B, chanind/sae-llama-3.2-1b-res::blocks.8.hook_resid_post | mistralai/Mistral-7B-v0.1, mistral-7b-res-wg::blocks.16.hook_resid_pre |
  5 unique (model, SAE) sides; Mistral blocks.16 shared by two cells (extracted once).
- Extraction interpreter: `<ENV_TRAIN>/bin/python` — the interpreter
  that produced every committed lextok receipt, so G0 is apples-to-apples. Analysis:
  `<ENV_ANALYSIS>`. Seed 0 everywhere. n_perm = 2000 for every
  verdict-bearing null (permutation p floor 1/2001 = 0.0004997...).
- Model forward and SAE encode are **byte-identical** to `compute_pooled_features`
  (bf16, `output_hidden_states`, same `hs_idx` rule, `sae.encode`); the only additions
  are `return_offsets_mapping=True` at tokenizer-call time (which does not change
  `input_ids`) and saving the per-token activations instead of discarding them.

## 1. Alignment unit (registered; PLAN §3.1)

1. Tokenize each document with each side's tokenizer exactly as the original extraction
   does, additionally requesting `return_offsets_mapping=True` (fast tokenizers only; a
   non-fast tokenizer is a hard error, not a silent fallback).
2. Words = maximal runs of non-whitespace characters of the source text (`\S+` spans).
3. Each model token is assigned to the word whose span overlaps its char span most;
   ties → leftmost word; BOS/special tokens and tokens with empty spans are dropped from
   alignment (they still enter G0's pooled reconstruction, which mirrors the original
   `acts[0].mean(0)` over ALL token rows).
4. Word-occurrence activation (per side, per feature) = **mean** of the word's assigned
   tokens' SAE activations (registered primary). **Max**-pooling is a registered
   sensitivity arm: reported, never verdict-bearing.
5. Occurrence (doc d, word w) enters the aligned series iff: the word has ≥1 assigned
   token on BOTH sides; w ≠ 0 (BOS bleed; PLAN §3.2 drop-first-word rule); w ≠ last word
   of the doc (300-char truncation can split the final word; registered exclusion,
   applied to every doc uniformly); and doc d passed QA on both sides (G1 doc rule).
6. Word type key (for S2/N3): lowercased word, stripped of leading/trailing ASCII
   punctuation (`string.punctuation`); occurrences whose type is empty after stripping
   are excluded from S2 only.

## 2. Statistics and nulls (registered; PLAN §3.2–3.3)

All series are float32; correlations per frozen pair (i,j) over the shared occurrence
axis; the cell statistic is the unweighted mean over G2-passing pairs. One permutation
draw = one re-ordering of side B's occurrence rows (all features move together, as in
the pooled-level null); side A is never permuted.

- **S1 (token-aligned sharing).** Within-document centering on both sides (subtract each
  feature's per-doc mean over included occurrences), then Pearson r per pair; mean over
  pairs. Document-pooled statistics are exactly invariant under N1, so any excess is
  strictly beyond the published document-level result.
- **N1 (primary null for S1).** Permute side B's occurrence rows independently within
  each document; 2,000 draws; p = (#{null ≥ obs}+1)/2001.
- **N2 (secondary position control).** Permute side B's occurrence rows across documents
  within relative-position bins (deciles of w / n_words(d)); 2,000 draws. Reported to
  separate position-profile alignment from content alignment; direction must agree with
  N1 before rung 2 is read (PLAN §7 risk 5); never verdict-bearing on its own.
- **S2 (order-sensitivity via exact unigram conditioning).** Restrict to word types with
  ≥20 included occurrences corpus-wide; demean both sides within word type (global fixed
  effect); Pearson r per pair over retained occurrences; mean over G2(S2)-passing pairs.
- **N3 (null for S2).** Permute side B's retained occurrence rows within word type,
  derangement (fixed points re-permuted until none remain); 2,000 draws.
- **R_tok (headline ratio).** To keep numerator and denominator on the same footing,
  the registered headline is support-matched:
  `R_tok = excess(S2 over N3) / excess(S1|S2support over N1|S2support)`,
  where S1|S2support is S1 (within-doc centering) computed on exactly the S2-retained
  occurrences and S2-passing pairs. The naive ratio excess(S2)/excess(S1) over their own
  supports is also reported. This is a registered operationalization of PLAN §3.3's
  R_tok, chosen (before any data) because the ≥20-occurrence type restriction shifts the
  occurrence composition toward frequent words and would otherwise confound the ratio.
- **Bigram narrowing arm (registered, reported, non-verdict).** Demean within
  (prev-word-type, word-type) cells with ≥10 occurrences; same null structure (permute
  within bigram cell). Beyond bigram, the appendix's "narrowed, not closed" language
  applies verbatim.
- **S3 (scramble) is NOT run now** (PLAN §3.4/§8): deferred unless rung 2 lands PARTIAL,
  and registered as directional supporting evidence only, never headline.

## 3. Provenance gates (read before any verdict; PLAN §4)

- **G0 — frozen-pair recovery via pooled reconstruction.** Per side: recompute the
  pooled matrix exactly as the published pipeline (bf16 mean over all token rows,
  including specials) on the same 20k docs in the same order; for each of the 1500
  cached columns, find the full-dictionary feature maximizing Pearson r over the 20,000
  shared docs (float64).
  - Side PASS: ≥1490/1500 columns match at r ≥ 0.999 (correlation-matching, not
    rank-matching, because bf16 forward nondeterminism across GPU types can flip
    rank boundaries).
  - A frozen pair is *recovered* iff both its columns match at r ≥ 0.999 and the
    column→dictionary map is injective on the matched set.
  - Cell PASS: both sides PASS and ≥95% of the cell's frozen pairs recovered.
    Unrecovered pairs are dropped and counted in the receipt.
  - Cell FAIL → **stop; nothing downstream in that cell is interpretable.** The only
    registered fallback (PLAN §7 risk 3) is a clearly-labelled *rematched arm*:
    re-freeze mutual-NN pairs from the reconstructed pooled matrices and report it as
    the weaker arm; it can never be promoted to the primary claim.
  - Dump-integrity assertion (plumbing, not a scientific gate): the sparse fp16
    per-token dump must reproduce this run's pooled matrix on the frozen-pair columns
    (per-column r ≥ 0.999); failure is a code bug and blocks everything.
- **G1 — alignment QA.** Fixed sample = docs 0–49 of the slice (registered). Per side:
  offsets in-bounds, non-overlapping, monotone; ≥99.5% of non-space chars covered by
  token spans; ≥95% of words receive ≥1 token on that side; cross-word tokens (span
  overlapping non-space chars of ≥2 words) ≤0.5% of tokens, every instance listed in
  the receipt for inspection. Doc rule (full corpus): a doc fails QA on a side if any
  token span is invalid or its char coverage <99%; failed docs are excluded on both
  sides; >1% of docs failing → G1 FAIL (stop and fix; plumbing gate, no verdict).
- **G2 — support adequacy.** A frozen (recovered) pair enters S1 only with ≥50 included
  occurrences where at least one side's word-level activation is nonzero; enters S2 only
  with ≥50 such occurrences among S2-retained ones. Per cell and per rung: <100
  surviving pairs → that rung in that cell is **NOT-TESTABLE — a gate outcome, not a
  verdict, and not a negative** (the lextok adequacy-gate discipline applied to support).
  Retained pair counts are always reported.

## 4. Verdict rules (registered; PLAN §5 bars adopted as-is)

Cells that are NOT-TESTABLE on a rung drop out of that rung's 3-cell tally; if ≥2 cells
are NOT-TESTABLE the rung is NOT-TESTABLE overall and no verdict language enters the
paper for it.

- **Rung 1.** Per cell: PASS iff S1 beats N1 at the permutation floor (p = 1/2001) AND
  mean per-pair centered r ≥ 0.05. TOKEN-ALIGNED iff ≥2 of 3 cells PASS. NOT ALIGNED iff
  S1 is within the N1 central 95% band in ≥2 of 3 cells. Else MIXED (report per cell).
- **Rung 2.** A cell enters the rung-2 tally only if it PASSed rung 1 (otherwise its
  R_tok denominator is noise); if <2 cells enter, rung 2 is NOT-EVALUABLE and the
  overall outcome is governed by rung 1 alone. Per entering cell: SURVIVES iff
  R_tok ≥ 0.30 AND S2 beats N3 at the floor; COLLAPSES iff R_tok < 0.10 OR S2 within
  the N3 central 95% band. ORDER-SENSITIVE iff ≥2 cells SURVIVE; LEXICALLY EXHAUSTED
  iff ≥2 cells COLLAPSE; else PARTIAL. (Deliberately lower bars than R_lex's 0.70/0.30:
  exact conditioning removes strictly more than the TF-IDF proxy; the asymmetric floor
  keeps "collapse" a strong claim.)

## 5. Outcome branches — all first-class, with the registered paper consequence

- **B1 = TOKEN-ALIGNED + ORDER-SENSITIVE.** The paper's main scope limit is lifted one
  level: results section reports S1/S2/R_tok; limitations rescoped from "lexical-scale"
  to "at least context-sensitive at word granularity", explicitly bounded below
  composition by the bigram caveat; abstract may gain one clause stating the lifted
  scope (licensed by this branch only).
- **B2 = TOKEN-ALIGNED + LEXICALLY EXHAUSTED.** The sharpest negative this paper could
  own: the surviving universality is fully explained by shared token-identity detectors.
  Results section reports the closure; the appendix's "narrowed, not closed" becomes
  "closed (negatively)"; the document-level R_lex survival is retro-diagnosed as an
  artifact of the nuisance design's weakness (median OOF R² ≤ 0.14) — the paper's own
  thesis (a control that removes nothing licenses nothing) applied against its own
  positive. Abstract updated to state the negative closure (licensed).
- **B3 = NOT ALIGNED (rung 1 fails).** The published claim survives unchanged (it never
  asserted token alignment) but gains a sharper boundary: sharing is document-scale
  only. The pre-registered irony is stated explicitly: pure token detectors WOULD have
  been token-aligned, so B3 also refutes the deflationary token-detector reading.
  Abstract at most gains a scope-tightening clause.
- **MIXED / PARTIAL.** Per-cell reporting in results + limitations use "partial"
  language; abstract untouched. S3 becomes eligible (PLAN §8) as a directional
  tiebreaker only.
- **NOT-TESTABLE (G2) / G0-G1 failure.** Gate outcome reported (appendix); abstract
  untouched; no verdict language anywhere.

In every branch: receipts are committed verbatim, the reader's printed branch is quoted
verbatim in the ELN, numbers in the paper are emitted by gate scripts (plant-verify
discipline), body stays ≤9pp, and no existing evidence is deleted.

## 6. Committed reader

`code/analysis/tokalign_summary.py` is committed BEFORE any extraction job is submitted.
It refuses partial input (all 3 cell receipts or an explicit gate-failure receipt per
cell), applies §4 verbatim, and prints exactly one branch line. Its `--selftest` must
fire EVERY branch from synthetic receipts: B1, B2, B3, MIXED/PARTIAL, NOT-TESTABLE, and
the refusal path. The analysis runner's `--selftest` must show the pipeline can return
either rung-2 verdict on planted data: planted unigram features → S1 passes, S2
collapses; planted context features → S1 passes, S2 survives; planted doc-topic
features → S1 at null (within-doc centering kills document-scale signal by
construction).

## 7. Execution plan (registered so deviations are visible)

Smoke first: strong-cell sides on 500 docs — schema + G0 recovery on the 500-row
prefix of the cached matrices; plumbing only, never a result. Then 5 extraction sides
(sbatch array, per-shard skip-if-exists receipts so evictions resume), then 3 CPU/GPU
analysis jobs, then the reader. Anchored cost (PLAN §6): ≈4–6 GPU-h primary battery,
worst case ≤20 GPU-h.

## AMENDMENT A3 (2026-08-16, post-receipt, NON-BLIND — disclosed)

External review of the completed battery found that the S2 guardrail does not earn
branch B1's full license. Two artifact routes are open that the registered design does
not exclude:

1. **Selection leakage vs N3.** The frozen pairs were selected by mutual argmax of the
   document-pooled correlation on the same 20,000 documents
   (`code/analysis/lexical_token_test.py`, `mutual_nn_pairs`). N3's within-type
   derangement moves occurrences ACROSS documents, destroying exactly the document
   alignment that selection optimized, while the type-centered observed statistic still
   contains document-level variation — so S2 beating N3 can re-detect the published
   document-level result rather than demonstrate survival of word-identity conditioning.
2. **Non-commuting S2c centering.** Within-document centering followed by type demeaning
   is not a joint document+type fixed-effect residualization (the two projections do not
   commute). Counterexample: docs [A,A,B] and [A,B,B] with a pure unigram signal
   f(A)=1, f(B)=0 retain identical nonzero residuals on both sides after the sequential
   centering and can attain r=1 — so the S2c arm does not close route 1.

Additionally (mechanical): `stat_block` drops zero-residual-variance pairs per
transformation, so R_tok's numerator and denominator are not asserted to score an
identical pair set, and no common scored-pair mask is written to the receipts.

**Consequence.** B1's S2 rung is demoted to **UNRESOLVED** pending a selection-clean
rerun with: (a) pair selection and evaluation on disjoint documents; (b) genuine joint
document+type fixed-effect residualization; (c) the identical residualization applied
under every null draw; (d) an asserted common scored-pair mask written to the receipts.
B1's S1 rung stands as registered: token-aligned within-document sharing not reducible
to document means. This demotion is non-blind (all receipts had been seen when the
review occurred) and is disclosed as such here and in the paper.

## AMENDMENT A3-EXEC (2026-08-17, pre-clean-run — committed before any clean job)

Executes A3's selection-clean rerun. NON-BLIND with respect to the ORIGINAL battery
(those receipts were seen); BLIND with respect to the clean statistics (no selection-clean
number exists anywhere when this is committed; git history is the timestamp). Every choice
below is fixed here and may not be retuned after the first clean receipt exists.

### E1. Disjoint-document split (A3 requirement a)

- `rng = numpy.random.default_rng(42)`; `perm = rng.permutation(20000)`;
  `split_A = sort(perm[:10000])`, `split_B = sort(perm[10000:])`. Seed 42, registered
  here before running. The sha256 of each split's int64 bytes is written to every receipt.
- **Selection (split A only):** restrict the md5-pinned pooled caches (§0
  `cache/lextok20k_*.npz`, top-1500 columns) to split-A rows; z-score per column exactly
  as `lexical_token_test._zcols` (float64; sd<1e-12 → 1); pair by the verbatim
  `lexical_token_test.mutual_nn_pairs` reciprocal-argmax rule on C=(Za.T@Zb)/n_A.
  The candidate pool (top-1500 columns/side) is unchanged from the published pipeline.
- **Evaluation (split B only):** every S1/S2 statistic and every null draw uses
  occurrences from split-B documents exclusively (registered inclusion rules of §1
  unchanged; doc-QA rule unchanged). S2's ≥20-occurrence type retention is counted
  within split-B occurrences only.
- **G0 stays full-corpus:** G0's column→dictionary matching is per-side identity
  anchoring (no cross-model pairing enters it), so it cannot leak selection; it runs
  exactly as registered on all 20k docs. A selected pair enters evaluation iff both its
  columns are G0-recovered (r ≥ 0.999, injective); the ≥95%-recovered cell rule applies
  to the split-A-selected pairs. G1/G2 rules unchanged (G2 support counted on split-B
  occurrences).

### E2. Genuine joint document+type fixed effects (A3 requirement b)

- S2-clean residualization = alternating projections in **float64**: sweep =
  (subtract per-doc mean, then per-type mean) over split-B retained occurrences;
  iterate sweeps until max|per-doc residual mean| < 1e-8 AND max|per-type residual
  mean| < 1e-8 on BOTH sides (absolute, activation scale O(1)). Convergence tolerance
  1e-8, max 500 sweeps; non-convergence is a hard error (no receipt, job fails).
  The converged sweep count and final deviations are written to the receipt.
- The A3 counterexample (docs [A,A,B]/[A,B,B], f(A)=1, f(B)=0) must residualize to
  exactly 0 (≤ tol) in the selftest — this is the visible proof that the sequential
  S2c defect is closed.

### E3. Identical residualization under every null draw (A3 requirement c)

- N3-clean: one draw = within-type cyclic derangement (registered scheme, unchanged)
  applied to side B's RAW retained rows, then the FULL E2 alternating-projection
  residualization re-run on the permuted matrix (same tolerance, same convergence
  requirement) before the statistic. n_perm = 2000; p floor 1/2001. A
  `null_reresidualized: true` marker plus the per-draw convergence guarantee go into
  the receipt.
- N1 (S1's null) permutes within documents; within-doc permutation commutes EXACTLY
  with per-doc centering, so re-centering per draw is mathematically the identity —
  stated here so the asymmetry with N3 is visibly reasoned, not an oversight.
- N2 stays as registered (direction check only, never verdict-bearing), unchanged.

### E4. Asserted common scored-pair mask (A3 requirement d)

- ONE mask over selected pairs: G2(S2) support (≥50 nonzero-occurrence support among
  split-B retained occurrences) AND residual sd > 1e-8 on both sides under the joint-FE
  transform AND under the doc-centering transform. Fixed before any null draw.
- `R_tok_clean = excess(S2_joint over N3_clean) / excess(S1|S2support over N1)`, both
  numerator and denominator computed on EXACTLY this mask (code asserts mask identity);
  `n_pairs_scored` and the mask's sha256 are written to the receipt. The naive ratio is
  reported, never verdict-bearing.

### E5. Bars (unchanged from §4) and branches

- S1-clean cell PASS iff p_N1 at the 1/2001 floor AND mean per-pair centered r ≥ 0.05.
- S2-clean cell SURVIVES iff R_tok_clean ≥ 0.30 AND p_N3 at the floor; COLLAPSES iff
  R_tok_clean < 0.10 OR S2 obs within N3's central 95% band. G2 NOT-TESTABLE rules
  unchanged (<100 surviving pairs on a rung).
- **B1-clean** iff ≥2 of 3 cells have (S1 PASS AND S2 SURVIVES): the word-identity
  claim is EARNED under the leakage-clean design; the paper's "unresolved" clauses
  (abstract, results, limitations, appendix, conclusion) are replaced by the earned
  claim, citing the clean design.
- **B2-clean** iff ≥2 of 3 cells have (S1 PASS AND S2 COLLAPSES): the sharing IS
  reducible to word identity + document means; reported as the sharpest negative, and
  the paper says so.
- **B3-clean** otherwise (intermediate or NOT-TESTABLE per the original G2 rules):
  the "unresolved" language STAYS; the clean-rerun outcome is reported per cell in the
  appendix. No verdict language beyond what the receipts license.

### E6. Scope, artifacts, execution

- Omitted from the clean rerun (registered): bigram, max-pool, and S2c arms — S2c is
  superseded by E2's joint FE; the original battery's reported arms stand as published.
- Receipts: `results/cross_model_sae/tokaclean20k_{tag}.json` (smoke: `smoke_` prefix,
  never read by the reader). Analysis: `code/analysis/tokalign_clean_analysis.py`
  (reuses the committed alignment/gate/statistic machinery of `tokalign_analysis.py`).
  Reader: `code/analysis/tokalign_clean_summary.py`, committed before submission; it
  REFUSES (exit 4) any receipt lacking the four A3 requirement markers (split hashes,
  FE convergence report, `null_reresidualized`, mask hash), and its selftest must fire
  B1-clean, B2-clean, B3-clean and the refusal path.
- Execution: same envs, seed 0, n_perm 2000 (smoke: reduced perms, plumbing only,
  strong cell on the 500-doc smoke dumps); per-token dumps are REUSED from the
  completed battery (JID 15816850; 20 shards + receipt per side verified intact
  2026-08-17) — no re-extraction.

## AMENDMENT A4 (2026-08-17, non-blind: A3 receipts and the round-14 review both seen)

External review of the A3 implementation found two residual leakage routes and one reader
weakness; consequences bound before the A4 battery runs.

**Defects being closed.**
(a) *Split-B token gating*: `word_matrices` received the full-corpus token dumps; a
split-A document at least two words longer than the longest split-B document aliases, via
the radix encoding, into a split-B (doc, word) slot. Whether this bit on the real A3 data
is an empirical question; the A4 receipt reports the audit either way
(`aliasing_audit: {radix, max_a_widx, n_aliasing_a_docs}`).
(b) *Candidate-universe screening*: the top-1500 candidate columns were screened by pooled
variance over all 20k documents. The dumps carry the FULL dictionary, so A4 re-screens the
universe by pooled variance computed on split A only (Welford over A documents), then runs
mutual-NN pair selection on split-A pooled correlations within that universe. G0 is
accordingly redefined: dump-integrity reconstruction stays a hard gate; overlap with the
frozen full-corpus pairs is REPORTED (informational), not gated -- the A4 pairs are
A-derived by construction.
(c) *Reader hardening*: the clean reader accepted boolean flags and any hex-shaped hash.
It now refuses: smoke receipts; n_perm != 2000; split hashes differing from the registered
pair (c1aa4ff9... / a85cbc07...); duplicate mask hashes across cells; missing
aliasing-audit or screening-provenance fields.

**Procedure.** Identical to A3-EXEC otherwise: same seed-42 halves, same converged joint
doc+type FE re-applied under every N3 draw, same asserted common mask, same bars
(S1 r>=0.05 at floor; R_tok_clean >=0.3 survive / <0.1 collapse; G2 >=100 pairs).

**Consequences (bound now).**
- B1-clean SURVIVES A4: the earned claim stands with A4 as its carrying receipt; the paper
  cites the A4 battery and the aliasing audit outcome.
- The verdict CHANGES under A4: the A3-based claim is withdrawn to the A4 branch's wording
  (B2-clean or NOT-TESTABLE as registered), disclosed as "the leakage the review named was
  live"; no averaging of the two batteries.
- Wording refinements land regardless (licensed by round 14): "current-word identity" ->
  "normalized current-word type (lowercased, punctuation-stripped)"; the bigram caveat is a
  DISCLAIMER, not evidence from a clean bigram arm; the maximally exact formulation is
  "shared occurrence-level variation survives joint document and normalized-current-word-type
  fixed effects; this does not establish compositional or syntactic sharing."
