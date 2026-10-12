# PLAN: token-aligned, order-sensitive compositional sharing test

Status: **design document only — nothing here has been run, registered, or committed.**
Written 2026-08-12 as a feasibility study in response to the external review point:
"Test token-aligned, order-sensitive compositional sharing; the current positive is
deliberately document-level and lexical-scale." Before any receipt exists, the thresholds
in §5 must be frozen into `experiments_detail.md` (the same pre-registration discipline as
commit 6848d4c for the lextok run). Until then every number in §5 is a *proposal*.

---

## 1. What the current positive is, precisely

The headline positive (`paper/sections/results.tex`, §"Cross-model SAE feature matching")
is a **document-level** statistic:

- **Representation.** For each model, run the shared corpus through the model; encode one
  mid-layer residual stream through that model's released SAE; **mean-pool the per-token
  feature activations over each document's tokens** (`cross_model_sae_match.py:
  compute_pooled_features`, line `pooled = acts[0].mean(0)`) → one scalar per
  (document, feature). Keep the top-1,500 features by variance across documents; z-score
  columns. The compared object is `Phi ∈ R^{N_docs × 1500}` per model.
- **Corpus.** 2,000 (headline) / 20,000 (lextok control) C4 documents, each truncated to
  ≤300 characters (`build_c4_slice.py --max_chars 300`) ≈ 55–80 tokens / ~45–55
  whitespace words per document.
- **Pairing rules.** Correlate A-features with B-features **across documents**
  (`C = Za^T Zb / N`); match by (i) reciprocal mutual-NN (primary), (ii) Lan-style greedy
  argmax, (iii) Hungarian. Null: permute the **document order** of side B, 200×/2,000×.
- **R_lex survival.** `lexical_token_test.py` freezes the raw mutual-NN pairs (822/806/520
  in the strong/median/weak cells), residualizes each pooled feature against a
  unigram-TF-IDF→SVD(k) nuisance design with a shuffled-design calibration arm, and asks
  whether the frozen pairs still co-activate: R_lex = 0.933/0.942/0.918 at k=4096
  (`results/cross_model_sae/lextok20k_*.json`).

**Why it is document-level.** Mean-pooling was chosen deliberately to sidestep the
tokenizer-grid mismatch — different tokenizers split the same text differently, so there
is no shared token axis to correlate on. The cost, conceded in `limitations.tex` and
`appendix.tex` (covctl paragraph): the statistic is **order-insensitive** (word-order
scrambling does not attenuate it, +16.1% excess under scrambling), so the claim is scoped
to shared *lexical-scale* structure, and the TF-IDF nuisance model (median out-of-fold
R² 0.028–0.140 per feature) *narrows* but cannot *close* the shared-unigram-detector
hypothesis.

## 2. Data inventory: what exists, what doesn't

Checked 2026-08-12, local repo + `cluster:<REPO_ROOT>`.

| Artifact | Where | Granularity |
|---|---|---|
| `results/cross_model_sae/*.json` (local + cluster) | receipts | summary stats; `c4_*` receipts carry `feat_idx_a/b` (top-1500 → dictionary IDs, 2k corpus); `lextok20k_*` receipts carry `frozen_pairs.feat_a/b` (**column indices into the top-1500 matrices**, dictionary mapping *not* stored) |
| `cache/lextok20k_{3 cells}.npz` (cluster, 40–69 MB each) | pooled features | `Pa,Pb ∈ float32[20000 × 1500]` — **document-level, post-topk** |
| `results/cross_model_sae/cache/pooled_lexres_*.npz` (cluster) | pooled features | 2k-doc equivalents |
| `data/c4_slice.jsonl`, `data/c4_slice_20k.jsonl` (cluster; 20k is a strict superset of 2k, md5 recorded in ELN) | corpus | text — per-token alignment can always be recomputed from this |
| `results/02_reproduce_anthropic/**/graphs/*.pkl` (cluster, 1.4 GB) | v0 attribution graphs | qwen3-0.6b only, templated prompts, abandoned pipeline — **not usable** for this test |

**Conclusion: no per-token SAE activation data is preserved anywhere** — local or cluster.
Every dump is post-`mean(0)`. The per-token tensors existed only transiently inside the
extraction loop. **New GPU inference is required** (§6). The good news: the frozen corpus,
frozen pair sets, and pooled caches make the new run *anchorable* — the per-token dump can
be validated by reconstructing the pooled matrices and matching them against the committed
caches (§4, gate G0).

## 3. The test: two rungs, separated confounds

The review's phrase bundles two claims that need **different nulls**:

- **Token-aligned** (rung 1): matched features co-activate at the *same positions within*
  a document, not merely on the same documents.
- **Order-sensitive / compositional** (rung 2): the aligned co-activation is not exhausted
  by shared token-identity (unigram) detectors — it depends on context.

Conflating them is the design's main trap (§7): two Gemma/Mistral features that both fire
on the token "cricket" are perfectly token-aligned yet purely lexical.

### 3.1 Alignment unit: whitespace words via character spans (recommended)

An exact cross-tokenizer token grid **does not exist** — Gemma-2 (SentencePiece),
Mistral-7B-v0.1 (SentencePiece/Metaspace), Qwen3 and Llama-3.2 (byte-BPE) split the same
text differently and disagree on leading-space attachment. The finest *well-defined*
shared grid is character spans of the identical source text. Concretely:

1. Tokenize each document with both models' fast tokenizers, `return_offsets_mapping=True`.
2. Split the document into whitespace-delimited **words**; each word owns a char span.
3. Assign each model token to the word with which its char span overlaps most
   (ties → leftmost); drop BOS/special tokens and tokens with empty spans.
4. Per model, per word occurrence: **word-level activation = mean of the word's token
   activations** (registered primary; max-pool as a sensitivity arm — SAE activations are
   non-negative, and features often fire on a word's final piece, so max is the natural
   alternative).
5. Exclusions (registered): words assigned zero tokens on either side; documents where
   round-trip offsets fail QA. Report coverage; expect >95% of words 1-1 aligned on
   English C4. Manual QA on a fixed 50-doc sample before the real run.

Both models now emit one activation vector per **word occurrence** `(doc_id, word_idx)`
— a shared, order-preserving axis. N ≈ 20,000 docs × ~50 words ≈ **10⁶ aligned
occurrences** (2k-doc arm: ~10⁵). Character-level or token-intersection alignments are
finer but inherit every leading-space/byte-fallback edge case; word-level alignment is
deterministic, auditable, and sufficient for both rungs. (Scope note for the paper: this
is *word*-aligned, the finest grid that is well-defined across tokenizers; "token-aligned"
in the review's sense.)

### 3.2 Rung 1 — token-aligned sharing (S1)

For each **frozen** pair (i,j) from the lextok receipts (the same pairs whose document-level
survival is the published positive):

- Series: word-level activations `a_i(d,w)`, `b_j(d,w)` over aligned occurrences.
- **Within-document centering**: subtract each feature's per-document mean. This removes
  the entire document-level (topical) component — by construction, everything the current
  paper already measures — leaving only position-level structure.
- Statistic S1: mean over frozen pairs of the Pearson correlation of the centered series
  (support-gated, §5). Also report per-pair distribution.
- **Null N1 (the order-shuffled null, primary):** independently permute side B's word
  order *within each document*; recompute; n_perm = 2,000 → p floor 1/2001.
  Document-pooled statistics are **exactly invariant** under N1 (a within-doc permutation
  does not move the doc mean), so any excess over N1 is strictly information the published
  document-level result cannot contain. This is the property that makes the test an
  upgrade rather than a rehash.
- **Null N2 (position control, secondary):** permute side B *across* documents within
  relative-position bins (deciles of word index / doc length). Preserves absolute/relative
  position profiles, destroys content alignment. Separates "both models have
  beginning-of-document features" (survives N1, dies vs N2-adjusted excess) from genuine
  content alignment. Drop word position 0 in all arms (BOS bleed).

### 3.3 Rung 2 — order-sensitivity via exact unigram conditioning (S2)

The token-level analogue of R_lex, with the nuisance removed *exactly* instead of via a
TF-IDF proxy:

- Restrict to word **types** (lowercased, stripped of punctuation) with ≥20 aligned
  occurrences in the corpus; report the retained type/occurrence counts.
- **Word-type fixed effects:** demean both sides' occurrence series within word type.
  What survives cannot be a unigram detector — the unigram is literally conditioned out.
  This does at occurrence granularity what the TF-IDF/SVD design (median R² ≤ 0.14) could
  only approximate at document granularity, which is exactly the gap the appendix concedes
  ("narrowed, not closed").
- Statistic S2: same mean frozen-pair correlation on the residuals.
- **Null N3:** permute side B's occurrences *within word type* (derangement across
  documents; fixed points excluded). Preserves each feature's word-type activation profile
  exactly; destroys context alignment.
- Headline ratio: **R_tok = excess(S2 over N3) / excess(S1 over N1)** — the fraction of
  the token-aligned sharing that survives exact lexical conditioning; the direct
  token-level counterpart of R_lex.
- Scope note to register up front: after word-identity conditioning both models still read
  the *same* neighboring words, so surviving signal is "context-sensitive," bounded away
  from unigram — not proof of syntax. A bigram-conditioned variant (demean within
  (prev-word, word) cells with ≥10 occurrences) is the registered narrowing arm; beyond
  bigram the same "narrowed, not closed" language as the current appendix applies.

### 3.4 Optional rung 2b — direct order manipulation (S3, flagged)

Re-run a 2k-doc subset with word order scrambled *within documents*, re-extract, and
compare aligned same-word-type co-activation on scrambled vs. natural text. This is the
literal "does order matter" manipulation, but scrambled text is off-distribution for both
the models and the SAEs, so attenuation confounds order destruction with distribution
shift. Register it as *directional supporting evidence only*, never as the headline; S2's
natural-context-variation design is the primary order-sensitivity instrument precisely
because it uses only in-distribution text.

## 4. Provenance gates (before any verdict is read)

- **G0 — pooled reconstruction.** Mean-pool the new per-token dump over each document's
  tokens and match the resulting `[20000 × d_sae]` matrix's top-1500-variance columns
  against the committed `cache/lextok20k_*.npz` `Pa/Pb` (column-matching by correlation
  ≈ 1.0, tolerance registered, e.g. per-column r > 0.999 for ≥ 1490/1500 columns). This
  (a) proves the new extraction is the same computation that produced the published
  receipts, and (b) recovers the top-1500 → dictionary-ID mapping that the lextok receipts
  do not store, anchoring `frozen_pairs.feat_a/b` to actual SAE features. bf16 forward
  nondeterminism across GPU types makes *rank-boundary* flips possible — hence
  correlation-matching, not exact-rank matching. If G0 fails, stop: nothing downstream is
  interpretable.
- **G1 — alignment QA.** On the fixed 50-doc sample: word coverage ≥95% per side, zero
  cross-word token assignments on manual inspection, offset round-trip exact.
- **G2 — support adequacy.** A frozen pair enters S1/S2 only with ≥50 aligned occurrences
  where at least one side is non-zero (per-token SAE activity is sparse, L0 ≈ 50–100 over
  16k–65k dims; many pooled-level pairs may be thin at token level). Report retained pair
  counts per cell; if <100 pairs survive the gate in a cell, that cell's verdict is
  NOT-TESTABLE (the lextok adequacy-gate discipline, applied to support instead of R²).

## 5. Registered outcome branches (both first-class)

Bars mirror the lextok pre-registration style (to be frozen in `experiments_detail.md`
before any receipt exists; the numbers below are the proposal):

- **Rung 1 verdict.** TOKEN-ALIGNED if S1 beats N1 at the permutation floor (p = 1/2001)
  with mean per-pair centered correlation ≥ 0.05 in ≥2 of 3 cells; NOT ALIGNED if S1 is
  within the N1 95% band in ≥2 of 3 cells; else MIXED (report per-cell).
- **Rung 2 verdict.** ORDER-SENSITIVE (survives) if R_tok ≥ 0.30 *and* S2 beats N3 at the
  floor, in ≥2 of 3 cells; LEXICALLY EXHAUSTED (collapses) if R_tok < 0.10 or S2 within
  null band, in ≥2 of 3 cells; else PARTIAL. (Deliberately lower bars than R_lex's
  0.70/0.30: exact conditioning removes strictly more than the TF-IDF proxy, so even
  partial survival is informative; the asymmetric floor keeps "collapse" a strong claim.)

**What the paper says in each branch — all three are publishable within this paper's
audit framing:**

1. **S1 passes, S2 survives.** The main scope limit of the paper is lifted one level: the
   cross-model sharing is token-aligned and not exhausted by unigram identity — shared
   *context-sensitive* structure, bounded below composition by the bigram caveat.
   Limitations rewritten accordingly; the "lexical-scale" scoping becomes "at least
   context-sensitive at word granularity."
2. **S1 passes, S2 collapses.** The sharpest result the paper could own: the surviving
   universality is **fully explained by shared token-identity detectors**. This *closes*
   (negatively) the shared-unigram hypothesis the appendix currently marks "narrowed, not
   closed," and retro-diagnoses the document-level R_lex survival as an artifact of the
   nuisance design's weakness (median out-of-fold R² ≤ 0.14) rather than evidence of
   supra-lexical sharing — the paper's own thesis (the control decides, and a control that
   removes nothing licenses nothing) instantiated once more, against its own positive.
3. **S1 fails.** The sharing is *document-scale only*: matched features track slow topical
   variables, with no position-level alignment at all. The published claim survives
   unchanged (it never asserted token alignment) but gains a sharper boundary; and the
   token-detector deflation is itself refuted — pure token detectors *would* have been
   token-aligned, so failure here rules out the deflationary reading too. State this
   pre-registered irony explicitly.

## 6. Cost

**Can it be computed from existing dumps? No.** Only document-pooled matrices exist (§2).
One new extraction pass per unique (model, SAE) side is required, saving per-token
activations this time.

- **Unique sides:** the 3 registered cells (strong/median/weak = qwen3-1.7b↔gemma2-2b,
  gemma2-9b↔mistral7b, llama3.2-1b↔mistral7b) comprise **5 unique sides** (Mistral-7B
  blocks.16 is shared by two cells). Models + SAEs are already in
  `<HF_CACHE>/hub` (revisions recorded in the appendix).
- **GPU-hours (anchored on JID 14915906):** the full 20k lextok cell — extraction of
  *both* sides at batch=1 *plus* the CPU test — took 1h35–2h27 on one H100/H200. So
  ≈ 30–70 min per side; a per-token dump adds only I/O. **Primary battery (5 sides, 20k
  docs): ≈ 4–6 GPU-h.** Smoke (+1), optional scrambled arm S3 (5 sides × 2k docs, ≈ 1–2),
  contingency for cancelled-job resubmits: **realistic total 8–12 GPU-h, worst case
  ≤ 20 GPU-h.** Trivial against the project budget; same
  `<GPU_PARTITION>,<GPU_PARTITION>,<GPU_PARTITION>,<GPU_PARTITION>` partitions, 192G nodes (the
  9B↔7B pairing OOMs on l40s — but note extraction is now per-*side*, one model resident
  at a time, so even that constraint relaxes).
- **Storage:** dump sparse per-token `(doc_id, token_idx, feat_idx, value)` over the
  *full* dictionary (one pass; lets G0 recover the topk mapping): L0 ≈ 50–100 → ~1–2 GB
  per side fp16-sparse, **≤ 10 GB total** on /scratch. Dense-top-1500 fallback:
  ≈ 4.5 GB/side.
- **Envs:** extraction in `<ENV_TRAIN>` (the interpreter that produced
  every committed lextok receipt — keeps G0 an apples-to-apples check); alignment +
  permutation analysis is GPU-free (numpy over ≤10⁶ × 1500 float32, chunked; a CPU node
  or the tail of the same job) and can run in `<ENV_ANALYSIS>`.
- **Calendar:** ~2–3 sessions: (1) alignment module + selftest with planted
  synthetic features (pure-lexical must collapse under N3, planted contextual must
  survive — the `lexical_token_test.py` CASE discipline), smoke on 200 docs; (2) freeze
  pre-registration, dispatch extraction; (3) analysis + receipts.

## 7. Risks

1. **Biggest: the inferential gap between "token-aligned" and "order-sensitive."**
   Alignment alone cannot distinguish shared token detectors from shared composition
   (rung 1 passes either way), and after word-identity conditioning the *residual context
   is still the same text on both sides* — so a rung-2 survival is "context-sensitive
   beyond unigram," not "compositional/syntactic"; while the one direct order manipulation
   available (scrambling, S3) confounds order destruction with distribution shift, since
   both models and both SAEs were trained on natural text. Mitigation: the two-null
   architecture (N1/N3) plus the bigram narrowing arm, and honest scoping language
   registered *before* the outcome is known. This is a claims-drafting risk as much as a
   statistical one: the paper must commit now to what each branch may be called.
2. **Per-pair support sparsity (G2).** Token-level SAE activity is sparse; frozen pairs
   that co-activate at document scale may have thin aligned nonzero support, especially in
   the weak Llama cell (520 pairs already the noisiest). If G2 guts a cell, the verdict is
   NOT-TESTABLE, not negative — the gate must be registered as such.
3. **Frozen-pair anchoring.** `lextok20k_*` receipts store pairs as top-1500 column
   indices without the dictionary mapping; recovery depends on G0's column-matching
   surviving bf16/GPU-type nondeterminism. Fallback if G0 partially fails: re-freeze pairs
   from the reconstructed pooled matrices and report both (a rematch is second-best — the
   lextok design chose frozen pairs precisely to avoid reconstituted-total artifacts, so
   any rematch must be labeled as the weaker arm).
4. **Alignment edge cases.** Leading-space conventions, byte-fallback tokens, CJK-free
   English C4 keeps this mild, but the 300-char truncation can split a final word
   mid-character-span → drop final partial words on both sides (registered exclusion).
5. **Position-0 / BOS bleed and doc-length effects** — handled by N2 and the drop-first-word
   rule; verify N1 and N2 agree in direction before reading rung 2.

## 8. Decision recommendation

Run rungs 1+2 (S1, S2, N1–N3, G0–G2) on the three registered cells at 20k docs: ~5 GPU-h,
every branch strengthens the paper, and branch 2 in particular would convert the review's
"deliberately document-level and lexical-scale" from a conceded limitation into a
measured, closed question. Defer S3 (scramble) unless rung 2 lands PARTIAL, where a
directional tiebreaker is worth its caveats.
