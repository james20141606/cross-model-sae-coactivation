# Lab-notebook excerpts: registration entries

Verbatim excerpts of the append-only lab notebook, apart from identity scrubbing (user names,
cluster name, account, partitions and paths are replaced by placeholders such as USER,
CLUSTER_ACCOUNT, <DATA_ROOT>, <GPU_PARTITION>; pointers to internal review notes by
<internal-notes>). Each entry was written, and committed, before the result it governs;
`REGISTRATIONS.md` lists the commit that fixes each. The full notebook is not shipped.

| # | timestamp | what the entry fixes |
|---|---|---|
| 1 | 2026-07-07T04:40:00-04:00 | probe-transfer hardening: expected pattern stated before the run |
| 2 | 2026-07-16T09:05:00-04:00 | lexical-residualized null: design fixed before running |
| 3 | 2026-07-20T18:39:33-04:00 | matched random-dictionary control: added and launched |
| 4 | 2026-07-27T19:05:00-04:00 | fine-grained topic (TF-IDF strata) control: decision rule |
| 5 | 2026-07-27T19:40:00-04:00 | amendment to the fine-grained control, before any k=200/500 receipt |
| 6 | 2026-07-27T22:20:00-04:00 | lexical residualization at 20k documents: registration (adequacy gate, k rule, branches) |
| 7 | 2026-07-31T21:05:00-0700 | Lan et al. pairing rule inside the lexical pipeline: registration |
| 8 | 2026-08-04T06:30:00-04:00 | Procrustes pairing baseline: registered, selftested, dispatched |
| 9 | 2026-08-15T23:58:00-06:00 | token-aligned / order-sensitive test: registration |
| 10 | 2026-08-16T01:05:00-06:00 | token-aligned test: amendment A1 before data |
| 11 | 2026-08-16T01:45:00-06:00 | token-aligned test: amendment A2 (smoke plumbing) |
| 12 | 2026-08-17T00:04:00-06:00 | token-aligned test: amendment A3 (rescope) |
| 13 | 2026-08-17T02:45:00-06:00 | token-aligned test: amendment A3-EXEC (leakage-clean design frozen) |
| 14 | 2026-09-24T11:54:29+08:00 | Lan et al. filters: pre-registration entry |
| 15 | 2026-09-24T14:16:34+08:00 | Gemma Scope read point: pre-registration entry |
| 16 | 2026-09-26T03:10:00+08:00 | nine-cell length/topic nulls, random split, bootstrap intervals: pre-registration and code |

---

## 2026-07-07T04:40:00-04:00   R13 adversarial pre-review: 3 attacks assessed, text armor patched, hardening experiment submitted
**Why:** simulate the 3 harshest realistic reviewers vs the Finding-5-folded paper; patch thin spots; identify the last real lever.
**Attack (a) universality skeptic ("probe transfer = preserved linear separability, not shared features"):**
```
REAL SURFACE FOUND: cross_model_probe_transfer.py fits mutual-NN on the SAME 206 concept
prompts used for eval (label-free, but same corpus). Given co-activation, same-corpus
transfer ~1.0 is expected; discriminating evidence = non-matched control + null pair.
PATCHES: (1) app:threats caveat (iii) "probe-transfer corpus scope" — full honest disclosure
+ why the controls still discriminate; (2) related (v) now distinguishes our MAP-FREE
transfer (identity over matched pairs) from learned-map stitching (removes the self-own
where we criticized functional transfer while headlining one); (3) limitations pointer
updated. HARDENING EXPERIMENT (submitted, results-gated): --match_prompts option — matching
fit on the INDEPENDENT 2000-doc C4 slice, eval on concept prompts. selftest PASS; smoke
JID 12812168 (null pair gpt2<->mistral) running; 3 capable pairs to follow on smoke-pass.
Pre-registered expectations: capable pairs transfer >> control (ideally ~1.0); null pair
transfer ~= control. A capable-pair FAILURE would confirm circularity and force weakening
Finding 5 (will be logged + folded honestly either way).
```
**Attack (b) SAE skeptic ("everything inherits SAE seed instability"):**
```
PATCH: app:threats caveat (i) rewritten with 3-part bound: (1) direction-of-bias — seed noise
DEGRADES reciprocal matching and spurious matches are absorbed by the permutation null (can
shrink, cannot manufacture the positive); (2) claim is existential (a transferable matched
set exists), not per-feature identity; (3) positive already spans 3 independently trained
SAE suites from 3 groups (GemmaScope / community Mistral / QwenScope) = de facto
recipe-robustness. No new experiment needed.
```
**Attack (c) audit skeptic ("two papers stapled"):**
```
FOUND: intro said "contribution is NOT a new universality positive" while abstract headlines
one — quotable contradiction. PATCH: reframed as "one instrument applied twice: a battery
that falsifies three artifacts and, with the same nulls, certifies the one survivor — down
to a zero-shot-usable interface". Existing armor retained: audit-order results, fig1 arc,
null-pair tie-in, Lan audit showing the battery matters beyond our own claims.
```
**Page budget:** all patches appendix-heavy; related (v) growth compensated by trims in the
same section (Lan sentence, son2025 clause). GATE: pdflatex exit 0, main text still ends
p10, 24pp, 0 undefined refs.

---

## 2026-07-16T09:05:00-04:00   PRE-REGISTRATION: lexical-residualized cross-model null (design fixed BEFORE running)
**Why:** The 07-15 correction left one named gap: the length/topic-matched nulls rule out doc-length and a coarse k=10 topic partition, but NOT lexical covariance — and the scramble non-collapse means the surviving overlap is compatible with shared unigram detectors. This entry fixes the design BEFORE any number is seen, so the result cannot be tuned toward a preferred verdict.
**Command(s):** ```bash
# design only — no run yet. Script: code/analysis/lexical_residual_null.py (to be written)
```
**Output digest (PRE-COMMITTED CHOICES — no results seen yet):**
- Cell: Gemma2-2B(gemma-scope layer_12/width_16k/canonical) <-> Mistral-7B(mistral-7b-res-wg blocks.16.hook_resid_pre), data/c4_slice.jsonl, N=2000, topk=1500, n_perm=200 — the exact covctl receipt cell.
- Extraction: REUSE compute_pooled_features/topk_features/_zscore_cols/null_under from cross_model_sae_match.py + c4_covariate_null.py. No reinvention.
- Lexical rep L: TfidfVectorizer(max_features=4000, stop_words="english") — EXACTLY the config build_strata() already uses — then TruncatedSVD(n_components=k).
- PRIMARY k = 128. Justification fixed in advance: 5-fold leaves 1600 training docs vs 128 predictors (12.5:1), so out-of-fold OLS is well-conditioned and cannot degenerate to interpolation; 128 comps of a 4000-term TF-IDF on 2000 short docs (median 50 words) spans the dominant lexical covariance.
- Robustness sweep: k in {64, 128, 256}. ALL reported, not the best.
- Folds: KFold(n_splits=5, shuffle=True, random_state=0). SAME folds for BOTH models.
- Regressor: OLS LinearRegression (with intercept), multi-output, fit on 4 folds -> predict held-out fold -> residual = actual - predicted. Cross-fitting is mandatory: in-sample OLS would over-remove and bias toward killing the effect.
- CALIBRATION CONTROL (pre-committed): `lexical_shuffled` — row-permute L, refit the same cross-fitted OLS. Same k, same spectrum, destroys doc<->lexical correspondence. Without it, a drop in excess is uninterpretable: projecting out ANY k-dim subspace removes some signal. Real lexical explanation requires real-L excess << shuffled-L excess.
- Statistic: unchanged mutual-NN + 200-perm global-random null (null_under(..., strata=None)).
**Interpretation contract (fixed in advance):** excess survives w/ large z -> sharing is NOT merely shared unigram detectors, supports stronger universality. Excess collapses toward 0 (and shuffled-L control does NOT collapse) -> MAJOR NEGATIVE: cross-model universality substantially explained by shared lexical statistics, central claim must scope down to lexical-level sharing. BOTH outcomes reported as they land.
**Observation:** Baseline to beat (un-residualized, from covctl receipt): obs=689, null 469.8+-9.8, excess +219.2 (14.6% of 1500), z=22.3. Next: write script, smoke at N=96/topk=256/k=32, then full N=2000.

---

## 2026-07-20T18:39:33-04:00   Matched random-SAE control added + launched (arXiv:2602.14111 random-baseline gate)

**Why:** arXiv:2602.14111 ("Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?") means any reviewer can ask, of the cross-model SAE-matching headline result: would a random encoder of matched dimension and matched sparsity produce the same z? Ported the matched random-SAE control already built for <SIBLING_PROJECT> (`random_sae_baseline.py`, RandomSAEShim: random-dir / rotate modes, calibrated per-token L0, independent seeds per side) into circuit_tracing_iclr and wired it into `cross_model_sae_match.py`. Interpretation rule fixed BEFORE seeing results: if reciprocal-matching z stays high under random SAEs, the "cross-family sharing" claim deflates to activation-statistics correlation; if z collapses to ~0, the claim is SAE-specific and hardened.

**Command(s):**
```bash
git status --short                          # clean before starting (no concurrent edits to cross_model_sae_match.py)
cp a sibling analysis's random_sae_baseline.py circuit_tracing_iclr/code/analysis/random_sae_baseline.py   # + orientation fix (below)
# edited code/analysis/cross_model_sae_match.py: +--random_sae {none,random-dir,rotate} +--random_sae_seed,
#   calibration+wrap in compute_pooled_features() (per-token L0 calibrated via a small batch of that
#   model's own real-SAE activations), independent seeds (seed, seed+10000) A/B, output auto-prefixed
#   rand_ + "random_sae" JSON key when enabled; --random_sae none path untouched (byte-identical)
/usr/bin/python3 -m py_compile code/analysis/cross_model_sae_match.py code/analysis/random_sae_baseline.py
/usr/bin/python3 code/analysis/cross_model_sae_match.py --selftest
/usr/bin/python3 <scratch>/check_random_sae_shim.py     # CPU-only unit check, both W_enc orientations
ssh -o BatchMode=yes -o ConnectTimeout=6 cluster 'echo ok'
scp code/analysis/{cross_model_sae_match.py,random_sae_baseline.py} cluster:.../code/analysis/
scp scripts/slurm/run_cross_model_sae_randcontrol.sbatch cluster:.../scripts/slurm/
ssh cluster ... py_compile + --selftest + bash -n   # re-verified on cluster mirror (<ENV_TRAIN> env, sae_lens present)
ssh cluster 'cd <REPO_ROOT> && sbatch scripts/slurm/run_cross_model_sae_randcontrol.sbatch'
```

**Output digest:**
- ⚠️ CAVEAT CONFIRMED + FIXED: `qwen_sae_adapter.py`'s `QwenSAE.W_enc` is `[d_sae, d_model]` — TRANSPOSED vs sae_lens's `[d_in, d_sae]` convention (the adapter's real matmul uses a separately-cached `_W_encT`; `.W_enc` is kept transposed only as a dtype/shape probe, qwen_sae_adapter.py:26-28). Unmodified, RandomSAEShim would read d_in/d_sae from the wrong axes for every `qwenpt:` SAE. Fixed in `RandomSAEShim.__init__` by detecting `W_enc.shape[0] == cfg.d_sae` (transposed) vs `W_enc.shape[1] == cfg.d_sae` (already sae_lens-orientation) and transposing before building `d_in x d_sae`.
- selftest (local `/usr/bin/python3`, numpy 2.0.2/scipy 1.13.1, no torch needed for this path): `SELFTEST PASS` — identical numbers pre- and post-edit (`greedy_obs=0.261 null=0.209 excess=0.053 p=0.0050 z=40.3`; noise-only `excess=-0.003 p=0.995`); confirmed via `git stash` that the pre-existing numpy RuntimeWarnings (divide-by-zero in the synthetic "thesis" test) are unrelated to this change.
- CPU-only both-orientation unit check (fake sae_lens-style `[d_in,d_sae]` SAE and fake qwenpt-style `[d_sae,d_in]` SAE built from the SAME underlying weights): calibrated k recovered exactly (12/12) for both; `shim.W.shape == (d_in, d_sae)` for BOTH orientations (proves the transpose-detection actually fires for qwenpt-style); matched-L0 exact (all rows == k) for random-dir and rotate; determinism under fixed seed (`torch.equal` on repeated construction + encode); different seed -> different W; cross-orientation equivalence — sae_lens-style and qwenpt-style real SAEs with the same weights (just stored transposed) produce `allclose` shim.W. **ALL CHECKS PASSED.**
- Re-verified identically on the cluster mirror (`<REPO_ROOT>`, <ENV_TRAIN> env): py_compile OK, `--selftest` PASS (same numbers), `bash -n` on the new sbatch OK, `data/c4_slice.jsonl` present (734433 bytes, matches the 2000-doc C4 slice used for the real c4_* receipts).
- New `scripts/slurm/run_cross_model_sae_randcontrol.sbatch`: 3 sequential cells with `--random_sae random-dir`, model/SAE/prompt args copied verbatim from `results/cross_model_sae/c4_{qwen3-1.7b__gemma2-2b,gemma2-9b__mistral7b,llama3.2-1b__mistral7b}.json` — highest-z (Qwen3-1.7B<->Gemma2-2B, mutual_z=42.7), mid (Gemma2-9B<->Mistral-7B, mutual_z=24.0), low (Llama-3.2-1B<->Mistral-7B, mutual_z=12.3). PROMPTS=data/c4_slice.jsonl, topk=1500, n_perm=200, max_prompts=2000, time=04:00:00, skip-if-exists on `rand_*.json`.
- Submitted: **JID 14426362** (`ct-xmodel-sae-randctl`), `squeue` shows `PENDING (QOSGrpGRES)` immediately after submit — normal cap-saturation queueing, not a failure; will run when a GPU frees up. Results not yet landed — no receipt to fold into the paper yet.

**Observation:** the qwenpt W_enc-transpose bug would have silently corrupted the random control for exactly the headline cell (Qwen3-1.7B<->Gemma2-2B) if not caught — verified fixed by an explicit cross-orientation-equivalence unit check, not just "it ran without erroring." `--random_sae none` (the default, used by every existing receipt) is unmodified/byte-identical: selftest numbers match pre-edit exactly and the only new code path is gated behind the flag. Next: poll `squeue -u USER` / check `logs/xmodel_sae/ct-xmodel-sae-randctl-14426362.{out,err}` for completion, scp back `results/cross_model_sae/rand_c4_*.json`, fold into paper per the interpretation rule above (pre-registered before results seen).

---

## 2026-07-27T19:05:00-04:00   PRE-REGISTRATION: fine-grained lexical/BoW control (written BEFORE results)

**Why:** the internal re-review (<internal-notes>) names one decision-relevant
weakness: the surviving cross-model sharing is still compatible with shared lexical/unigram
detectors. The published covariate control used a COARSE k=10 TF-IDF partition on ONE cell.
This entry fixes the decision rule *before* the receipts exist, so the read cannot be chosen
after the fact. Jobs are queued (smoke JID 14852351); no covctl_k200/k500 receipt exists yet.

**Design:** code/analysis/c4_covariate_null.py on three representative C4-robust cells
spanning the effect range -- strong Qwen3-1.7B<->Gemma2-2B (z=42.7), median
Gemma2-9B<->Mistral-7B (z=24.0), weak Llama-3.2-1B<->Mistral-7B (z=12.3); the same three
cells as the random-dictionary control -- at n_bow_clusters in {200,500}, n_perm=2000.

**Primary statistic:** mutual-NN EXCESS = obs - null_mean, NOT z. As clusters get finer the
within-stratum permutation has less freedom, so the null mean rises and the null SD shrinks;
z can inflate for a purely mechanical reason, the excess cannot. Retention :=
bow_matched_excess / baseline_random_excess, computed per cell.

**Pre-registered decision rule (fixed now):**
  SURVIVES  retention >= 0.70 at k=200 in >=2 of 3 cells AND p at the permutation floor.
            -> the k=200 lexical partition does not explain the sharing; report as the
               control it is, keep the positive framing.
  PARTIAL   retention 0.30-0.70.
            -> report as partial attenuation, quantified; the positive is retained but
               explicitly bounded as "not fully separable from lexical structure".
  FAILS     retention < 0.30 in >=2 of 3 cells.
            -> the sharing is substantially a bag-of-words/topic covariate. Do NOT bury:
               rewrite title/abstract as a negative measurement audit (the internal reviewer states this
               would be the stronger paper), demote Finding 1 to a controlled negative.
  k=500 is reported as a robustness ladder alongside k=10 (published) and k=200; if k=200 and
  k=500 disagree, the FINER (k=500, more conservative) result governs the verdict.

**Known limitation of the control, stated in advance:** TF-IDF/KMeans strata are themselves a
lexical model. Surviving them bounds *how much* of the sharing coarse-to-medium lexical
structure explains; it cannot exclude lexical structure that KMeans does not represent. This
is the same caveat already carried by the under-powered lexical-residualization test.

**Command(s):**
```bash
scp code/scripts/slurm/run_c4_covariate_null_fine.sbatch cluster:$ROOT/scripts/slurm/
ssh cluster 'bash -lc "cd $ROOT && SMOKE=1 sbatch --array=0-2 --time=00:40:00 scripts/slurm/run_c4_covariate_null_fine.sbatch"'   # JID 14852351
# full array (0-5 = 3 cells x {k=200,k=500}) submitted after smoke passes; JID recorded in the next entry.
```

**Observation:** rule fixed, results pending. Whatever comes back gets reported against this
rule verbatim.

---

## 2026-07-27T19:40:00-04:00   PRE-REGISTRATION AMENDMENT (still before any k=200/500 receipt)

**Why:** while the first full array (JID 14852432) was running I found a flaw in my own
design and cancelled it. The rule below replaces the primary statistic in the 19:05 entry.
No covctl_k200/k500 receipt existed at the time of this amendment (14852432 was cancelled
with zero outputs written; `ls covctl_k*.json` -> 0 files).

**The flaw:** permuting *within* strata approaches the identity as blocks shrink. At k=200
over N=2000 the mean block is 10 docs; at k=500 it is 4. So the null mean rises toward the
observed count and the excess decays **mechanically** as the partition gets finer --- at
block size 1 the excess is exactly 0 by construction, whatever the effect is. A low
retention at k=200/500 would therefore have been **uninterpretable**: "the sharing is
lexical" and "the null absorbed the signal because the blocks are small" predict the same
number. My 19:05 rule ("on disagreement the finer k governs") was, if anything, backwards:
finer is more conservative for a partly mechanical reason.

**The fix (new arm `bow_size_matched_random`):** permute within strata that have the SAME
block-size multiset but RANDOM membership, so they carry no lexical information. This
isolates the mechanical component. Unit-tested locally: block-size multiset preserved,
200/200 blocks, label agreement with the real clusters 0.0045 (chance).
This is the same calibration the paper already applies to the lexical-residualization test,
where a random subspace of equal rank removed as much as the real lexical one.

**Amended primary statistic:**
  lexical-specific retention  R := bow_matched_excess / bow_size_matched_random_excess
  (was: bow_matched_excess / baseline_random_excess, which conflates the two effects)

**Amended decision rule (fixed now, before results):**
  SURVIVES  R >= 0.70 at BOTH k=200 and k=500 in >=2 of 3 cells, bow_matched p at the
            permutation floor -> the lexical partition does not explain the sharing.
  PARTIAL   0.30 <= R < 0.70 -> quantified partial attenuation; positive retained but
            explicitly bounded.
  FAILS     R < 0.30 in >=2 of 3 cells -> sharing is substantially bag-of-words; do NOT
            bury; rewrite title/abstract as a negative measurement audit per the internal reviewer.
  The raw bow/baseline retention is still reported, but as a descriptive number only, since
  it is contaminated by the block-size effect quantified by the new arm.

**Command(s):**
```bash
ssh cluster 'bash -lc "scancel 14852432"'          # cancelled mid-flight, no outputs kept
scp code/analysis/c4_covariate_null.py cluster:$ROOT/code/analysis/
ssh cluster 'bash -lc "cd $ROOT && sbatch scripts/slurm/run_c4_covariate_null_fine.sbatch"'   # JID 14852578
```

**Observation:** the amendment makes the experiment able to return an interpretable
negative, which the original design could not. Results pending.

---

## 2026-07-27T22:20:00-04:00   PRE-REGISTRATION: the token/unigram-detector test (written before the run exists)

**Why:** the post-covariate internal re-review is still borderline/reject, and it moved to the
alternative *we* flagged: the surviving positive does not separate a broadly interesting
shared representation from shared **token/unigram** detectors. The fine-topic control
(SURVIVES, R=0.93-1.06) removed the TOPIC explanation for three cells; it did not touch the
unigram one, because the statistic mean-pools and is order-insensitive by construction.
SSH to the cluster is down (auth expired), so this entry fixes the rule while no result is even
obtainable. Nothing below can have been influenced by data.

**NO INCENTIVE DECLARATION.** Both branches move the paper up, and I am recording that so
the analysis has no stake in the outcome:
  * survival  -> the last live alternative is excluded; weak accept.
  * collapse  -> reframe as a coherent negative measurement audit (title/abstract already
                 read "Universality, Audited ... and Its Artifacts"); also weak accept.
                 a sibling project reached weak accept this session by exactly this move -- deleting
                 its own headline instrument after that instrument failed calibration.
I therefore have no reason to prefer either branch, and will report whichever the receipts
give. If I find myself wanting one, that is a bug.

**Design.**
  corpus : 20,000 C4 documents (the internal review asked 10k-50k), same builder and max_chars=300 as the
           published 2,000-doc slice so the statistic stays comparable; seed 0.
           The old lexical run failed for lack of data: 1,600 docs/fold could not fit a
           bag-of-words map, median OOF R^2 ~0.037 (Gemma) / ~0.013 (Mistral). 20k docs
           gives 16k/fold, a 10x increase.
  cells  : strong  Qwen3-1.7B <-> Gemma2-2B    (C4 z=42.7)
           median  Gemma2-9B  <-> Mistral-7B   (C4 z=24.0)
           weak    Llama-3.2-1B <-> Mistral-7B (C4 z=12.3)
           -- the same three cells as the random-dictionary and fine-topic controls.
  lexical design : TF-IDF(max_features=50000, stop_words=english) -> TruncatedSVD,
           k in {256, 1024, 4096}, fitted INSIDE each training fold (vocabulary, idf and
           SVD basis never see held-out docs). k up to 4096 because the previous k<=256
           truncation discarded exactly the rare-word detail a unigram detector fires on.
  calibration : every arm is repeated with a ROW-SHUFFLED design of identical k and
           spectrum. Projecting out ANY k-dim subspace removes some signal; only the
           excess over the shuffled arm is lexical-specific. Same calibrated-ratio logic
           that changed the covariate verdict.
  n_perm : 2000 everywhere (floor 1/2001 = 0.0005, clears Bonferroni over the 13- and
           17-cell rosters).

**(a) FIRST-STAGE ADEQUACY GATE -- hard, checked before anything downstream is read.**
A first stage that cannot fit real lexical features better than shuffled ones explains
nothing, and every downstream number is then uninterpretable rather than reassuring. Gate,
per cell, per k, on BOTH models:
    A1  paired per-feature: frac( R2_real > R2_shuffled ) >= 0.75
    A2  median OOF R2_real - median OOF R2_shuffled >= 0.02
Both must hold on model A AND model B. The runner EXITS NON-ZERO if the gate fails, and the
arm is recorded as `adequacy_failed`. A gate-failing arm may NOT be reported as survival --
it is reported as "the lexical hypothesis is not testable at this corpus size with this
proxy", which is a limitation, not a result. (This is precisely the error the previous
lexres run avoided only by refusing to claim anything.)

**(b) SURVIVAL OF THE ORIGINAL MATCHED PAIRS -- not a rematch of totals.**
The old run rematched on residuals and compared totals, so a similar total recomputed from
DIFFERENT pairs would have looked like survival. New primary statistic:
    1. from the RAW pooled features, run the standard top-1500 + mutual-NN matching and
       FREEZE that pair set M (recorded in the receipt).
    2. residualize both models (cross-fitted, within-fold).
    3. c_res = mean over M of corr(residual_a[i], residual_b[j]) for the frozen (i,j).
    4. null: within-fold permutation of model B's document order, recompute the mean over
       the SAME M, 2000 times.  (Within-fold, because cross-fitting stamps fold structure
       into both models; a global permutation would let fold alignment masquerade as
       sharing -- CASE3 of the existing self-test.)
    5. retention_real = (c_res - null_mean_res) / (c_raw - null_mean_raw)
       retention_shuf = same with the shuffled design
       R_lex = retention_real / retention_shuf          <- PRIMARY
R_lex, not retention_real, because the shuffled arm absorbs the generic cost of projecting
out a k-dim subspace.

**(c)/(d)** n_perm = 2000; and the size-matched partition calibration is repeated over
**20 independently sampled partitions** (the covariate arm used one, which the internal reviewer correctly
called the weak point), reporting mean and spread and using the mean as the calibrator.

**DECISION RULE (fixed now).** Over the cells whose adequacy gate PASSES:
    SURVIVES   R_lex >= 0.70 in >= 2 of 3 cells, pair-survival p at the 1/2001 floor
               -> shared-unigram/token detectors do not explain the sharing at this
                  corpus size and lexical capacity. Report as the control it is; the
                  positive framing stands, scope-bounded.
    PARTIAL    0.30 <= R_lex < 0.70
               -> report quantified partial attenuation; the positive is retained but
                  explicitly bounded as not fully separable from lexical structure.
    COLLAPSES  R_lex < 0.30 in >= 2 of 3 cells
               -> the surviving positive is substantially shared lexical statistics.
                  ACTION, pre-committed: rewrite title and abstract as a negative
                  measurement audit; demote Finding 1 from "the defensible positive" to a
                  fourth controlled negative; promote the audit half (three artifacts
                  killed + the Lan-protocol re-analysis) to the contribution. Do NOT bury,
                  do NOT re-cut the corpus, do NOT search for a surviving subset.
    If the ADEQUACY GATE FAILS on >= 2 of 3 cells at every k, the verdict is
    NOT-TESTABLE: report that bag-of-words proxies cannot represent these activations even
    at 20k documents, keep the token-detector alternative explicitly live in the
    limitations, and claim nothing from the run.
Ties/edge cases: if k values disagree, the LARGEST adequacy-passing k governs (it is the
most capable lexical proxy, so it is the strongest test of the lexical hypothesis).

**Deliverables to be committed with the result, regardless of branch:** every per-cell JSON;
the frozen pair set M inside each receipt; the adequacy table; and a read-back-verified
emitted LaTeX table (the emit+re-read pattern that caught defects in two other papers).

**Observation:** rule fixed while the cluster is unreachable. Code to follow; it will be
written to this spec and self-tested offline, and the run submitted the moment SSH returns.

---

## 2026-07-31T21:05:00-0700   PRE-REGISTRATION: apply R_lex to a PUBLISHED pairing rule (written before the code exists)

**Why:** `tab:lextok` currently establishes "our matches survive our control" (R_lex
0.933/0.942/0.918). That is a statement about this paper. The field's cross-model
universality results are established at the correlation level, and a primary-source pass
found no lexical-residualization control in any of them. This arm turns the control into a
claim about the field: transplant a *published* pairing rule into the same R_lex machinery
and report how much of ITS excess survives.

**LITERATURE VERIFICATION FIRST (three claims checked, two corrected).** Full text of
arXiv:2410.06981v4 (44pp) pulled and read; `grep -i` over it returns **zero** hits for
lexical / n-gram / bag-of-words / unigram / token overlap / confound. Their only baselines
are random-pairing and shuffled-pairing nulls (their §3 "Baseline Tests Types") plus
randomized-weight SAEs (their App. D). So the "no lexical-residualization control" claim
holds for this paper, on the primary source.
  * CORRECTION 1: arXiv:2410.06981 and OpenReview `rbHOLX8OWh` are **the same work**, not
    two results. v3 title "Sparse Autoencoders Reveal Universal Feature Spaces Across Large
    Language Models"; v4 (2025-05-21) retitled "Quantifying Feature Space Universality
    Across Large Language Models via Sparse Autoencoders". Counting them separately would
    have inflated the roster.
  * CORRECTION 2: arXiv:2502.03714 (Universal SAEs, Thasarathan et al.) trains ONE shared
    SAE ingesting activations from any model. It has **no cross-model feature-pairing step**
    to transplant, so it is not a candidate for this arm at all.
  * CORRECTION 3: arXiv:2603.26236 is "A Universal Vibe? Finding and Controlling
    Language-Agnostic Informal Register with SAEs". It is cross-LINGUAL within ONE model
    (Gemma-2-9B-IT), not cross-model, and the phrase is "trivial lexical sensitivity", which
    they address by *dataset design* (every target term polysemous), not by residualization.
    It is evidence the concern is in the air; it is not a cross-model pairing pipeline.
  * VENUE: Lan et al. is an **unpublished arXiv preprint**. `references.bib` already has this
    right (`booktitle = {arXiv preprint arXiv:2410.06981}`). Two code docstrings say
    "ICLR 2025"/"ICLR'25" and are wrong; fixed in this commit, since this arm cites the work.

**Which pipeline, and the exact reimplementation scope.** Lan et al. (arXiv:2410.06981),
**pairing rule only**. Verbatim from their §3 "Highest Activation Correlation":
    "for each feature in SAE A, we find its highest correlated feature in SAE B (as this
     allows for many-to-1 mappings, this is termed 1A-to-ManyB)"
That is the whole transplant: `ia = arange(K_A)`, `jb = C.argmax(1)`. NOT reimplemented
here: their SVCCA/RSA downstream (already reimplemented separately in
`lan_protocol_audit.py` and reported in App. `tab:lan`; not needed for R_lex), their
semantic-subspace selection, and their two filters -- see G0.

**G0 -- declared scope of the transplant (a declaration, not a test).** Lan's non-concept
keyword list IS published (their App. K: `\\n`, `\n`, empty string, space, `.` `,` `!` `?`
`-`, tokenizer padding token), but applying it needs each feature's top-5 *dataset examples*,
which the cached pooled features do not carry; their 1-1 filter is applied downstream of the
argmax. We therefore transplant the **unfiltered** 1A-to-ManyB argmax rule -- for which Lan
also report results and explicitly state (App. J) that "the scores still show high similarity
even with the inclusion of many-to-1 pairings". Declared now, not discovered later.

**REIMPLEMENTATION VALIDITY GATE -- mandatory, read BEFORE any R_lex number.**
  G1  the mapping must be many-to-1: n_distinct_targets < K_A.        (their §3, App. J)
  G2  the 1-1 fraction must lie in Lan's published interval. Their Tables 5+6
      (Pythia-70m x Pythia-160m, 55 layer pairs, 32768-feature SAEs) give
      1-1 fraction min 0.1593, max 0.6324, median 0.4131. Gate: our frac_one_to_one in
      [0.1593, 0.6324] on >= 2 of 3 cells.
  G3  nesting invariant: mutual-NN pair set must be a STRICT subset of the argmax pair set.
      (Correctness of both implementations; a failure means one of the two rules is miscoded.)
  Fail G1 or G3, or fail G2 on >= 2 of 3 cells  ->  BRANCH 5. Arm stops. No R_lex is read.

**DECLARED WEAKNESS OF G2, stated now rather than after seeing it.** For an arbitrary argmax
assignment over K_B targets the 1-1 fraction tends to (1-1/K_B)^(K_A-1) -> 1/e; at our
K=1500 that reference value is **0.3680**, which sits inside Lan's published interval. G2 is
therefore NECESSARY, NOT SUFFICIENT. It rules out gross miscoding (an accidentally reciprocal
rule gives 1.000; a Hungarian/one-to-one rule gives 1.000) and confirms the rule is of the
right family. It does NOT certify numerical fidelity to their tables, and no achievable gate
would: their Tables 5/6 are computed at TOKEN granularity on SAME-TOKENIZER Pythia pairs
after a filter needing per-feature dataset examples we do not have. The real anchor is that
the rule is fully specified in the two prose sentences quoted above and is two lines of code;
the receipt stores those sentences verbatim so a reader can check the transplant themselves.
Anyone who finds G2 too weak should read the result as a claim about *the argmax pairing rule
as specified*, not as a certified replication of Lan's numbers. It is stated that way in the
paper.

**Metric: the existing R_lex, unchanged.** Same `code/analysis/lexical_token_test.py`, same
20,000 C4 documents from the same cached pooled features, same topk=1500, same
k in {256,1024,4096} TF-IDF(50000, stop_words=english) -> TruncatedSVD fitted inside each
training fold, same 5-fold cross-fitting, same row-shuffled calibration design, same
within-fold permutation null, n_perm=2000, seed 0. The ONLY thing that changes is the frozen
pair set M. Implemented as a `--pairing {mutual,lan_argmax}` flag; `mutual` (the default)
stays byte-identical, verified by re-running the existing selftest and by a
pairing-equivalence assertion.

**Pre-registered adequacy gate -- unchanged from 6848d4c.** A1 frac(R2_real > R2_shuffled)
>= 0.75 AND A2 median dR2 >= 0.02, on BOTH models. The nuisance model must demonstrably
bite or the arm is void. Runner exits non-zero if no k passes.

**Cells:** the same three, from the same caches:
  strong  Qwen3-1.7B <-> Gemma2-2B      median  Gemma2-9B <-> Mistral-7B
  weak    Llama-3.2-1B <-> Mistral-7B
Ours for comparison (committed, `tab:lextok`): R_lex = 0.933 / 0.942 / 0.918, mutual-NN pair
counts 822 / 806 / 520.

**INTERPRETATION BRANCHES -- all fixed now, with evidential bases.** Scored over cells whose
adequacy gate passes, at the largest adequacy-passing k (same tie rule as 6848d4c).
  B1 THEIR-EXCESS-COLLAPSES    R_lex^lan < 0.30 in >= 2 of 3 gate-passing cells.
     Basis: argmax-paired excess is substantially shared lexical structure where
     reciprocal-paired excess is not. Licenses: "the field's pairing rule needs this control,
     and here is what it costs" -- with the number.
  B2 THEIR-EXCESS-SURVIVES-LIKE-OURS   R_lex^lan >= 0.70 in >= 2 of 3, pair-survival p at the
     1/2001 floor. Basis: the control is passable by the field's rule too. Licenses ONLY
     "the control has now been run on the field's own pairing rule and it passes" -- a
     service to the field, not a correction of it. Does NOT license any claim that our
     pairing rule is better.
  B3 PARTIAL                   0.30 <= R_lex^lan < 0.70 in >= 2 of 3. Report the quantified
     attenuation and the gap to ours. No collapse claim.
  B4 NUISANCE MODEL DOES NOT BITE   adequacy gate fails at every k on >= 2 of 3 cells. VOID:
     "not testable under the argmax pairing at this corpus size"; claim nothing. Note: the
     first stage does not depend on the pairing, and this identical design already passed the
     gate on these exact caches, so a failure here would indicate a BUG and will be reported
     as one rather than as a limitation.
  B5 REIMPLEMENTATION DOES NOT REPRODUCE   G1/G2/G3 as above. NO comparison is licensed; the
     arm stops and is reported as stopped. This branch is why the gate is read first.

**NO INCENTIVE DECLARATION.** B1 makes the strongest methods claim, so B1 is the branch I
have an incentive to find. Recording it. B2 is fully reportable -- "we ran the control the
field has not run, on the field's own pairing rule, and it passes" is true and useful. If I
find myself wanting B1, that is a bug.

**PRE-COMMITTED FRAMING, required in the paper on EVERY branch.** The claim is about **the
control**, not about their competence. Lan et al.'s result may be entirely correct at the
correlation level and still need this control before it can bear a representational-*sharing*
interpretation. Their scope is narrower than the regime we transplant into: they pair at
token granularity on same-tokenizer pairs and make rotation-invariant *space*-level claims,
while we pair on document-mean-pooled activations across families. A collapse here is
evidence about the argmax pairing rule in the cross-family document-pooled regime, NOT a
refutation of their within-scope finding. This sentence goes in whatever the receipts say.

**Explicit output paths (two sibling repos hit silent default-path collisions this week;
nothing here shares a stem with `lextok20k_*`):**
  receipts  results/cross_model_sae/lanpair_lextok20k_{stem}.json
  table     paper/sections/tab_lanpair.tex
  reader    code/analysis/lanpair_summary.py   (REFUSES partial input)
  sbatch    code/scripts/slurm/run_lanpair_lextok.sbatch
  logs      logs/lanpair/

**Persisted per cell (every registered readout, or the reader refuses to score):**
pairing_rule, lan_rule_quote, K_A, n_pairs_argmax, n_pairs_mutual, n_distinct_targets,
frac_one_to_one, one_over_e_reference, lan_table56_interval, nesting_ok, G1/G2/G3 verdicts,
full frozen pair index lists for BOTH rules, raw pair null, and per k: adequacy (both
models), pair_survival_real, pair_survival_shuffled, retention_real, retention_shuffled,
R_lex, interpretable.

**Command(s):** none yet -- this entry is committed before the code exists.

**Observation:** design fixed. Nothing below this line can have been influenced by an
R_lex^lan number, because none exists. Next: implement the flag, re-verify the `mutual` path
is byte-identical, run the three cells, read the validity gate, then score.

---

## 2026-08-04T06:30:00-04:00   Procrustes-aligned pairing baseline: registered, selftested, dispatched (JID 15260604)
**Why:** codex round 5 item 3 + DR sweep: arXiv:2605.24577 argues the standard universality
metric is blind to a rotation failure; 2607.08499 predicts Procrustes pre-alignment finds more
universal features than post-hoc pairing. Our two-pairing-rule answer becomes a three-rule
answer, with the third being the one the rotation account predicts should win.
**Command(s):** ```bash
# PREREGISTRATION_procrustes_baseline.md (3be9b7d) committed BEFORE the pairing existed
# lexical_token_test.py: --pairing procrustes; alignment over the SHARED DOCUMENT axis
#   (Q = UV^T from svd(Ub^T Ua)), rank 64, split seed 0, fit on train half, ALL downstream
#   quantities restricted to the held-out half
# selftest CASE7: collapses on pure-lexical (want <0.30), survives on shared latent (>=0.70)
# procrustes_summary.py: reader with a LEAKAGE CHECK (N_docs must equal n_eval_docs;
#   rank/split_seed must match the registration) read before any R_lex
sbatch code/scripts/slurm/run_proclex_lextok.sbatch   # -> 15260604
```
**Output digest:** selftest 10/10 PASS locally and on the cluster; reader REFUSES with all
three receipts missing, as designed.
**Observation:** Anticipated failure mode is P4: halving the corpus halves the first stage's
power, and the registration forbids enlarging the corpus to rescue it (that would be an
adaptive stopping rule on the same statistic).

---

## 2026-08-15T23:58:00-06:00   PRE-REGISTRATION: token-aligned / order-sensitive sharing (S1/S2), written before any per-token data exists

**Why:** execute the externally-reviewed 8->9 path ("test token-aligned, order-sensitive
compositional sharing; the current positive is deliberately document-level and
lexical-scale"). PLAN_token_aligned_sharing.md established feasibility on 2026-08-12 and
verified (§2) that NO per-token SAE activation dump exists anywhere -- every committed
artifact is post-mean(0) -- so nothing below can have been influenced by data.

**Command(s):**
```bash
ssh cluster 'md5sum $ROOT/data/c4_slice_20k.jsonl $ROOT/cache/lextok20k_*.npz'
md5 -q results/cross_model_sae/lextok20k_{qwen3-1.7b__gemma2-2b,gemma2-9b__mistral7b,llama3.2-1b__mistral7b}.json
# wrote PREREGISTRATION_token_aligned.md from PLAN sections 3-5; committed before any code runs
```

**Output digest:**
```
corpus  c4_slice_20k.jsonl              md5 64e994008170f6c5738bdb5dc19f285e  (20,000 docs)
caches  lextok20k_qwen3-1.7b__gemma2-2b 74651e10... / gemma2-9b__mistral7b be768bbd... /
        llama3.2-1b__mistral7b 88b3cf06...
frozen pairs: 822 / 806 / 520 (mutual-NN, prereg 6848d4c), pinned by receipt md5s
Registered: S1 = within-doc-centered frozen-pair corr vs N1 (within-doc word-order shuffle,
  n_perm=2000; pooled stats exactly invariant under N1); S2 = word-type fixed-effect
  residuals vs N3 (within-type derangement); headline R_tok support-matched; bars
  S1: p at 1/2001 floor AND mean r>=0.05 in >=2/3 cells; R_tok >=0.30 survive / <0.10
  collapse; gates G0 (pooled reconstruction r>=0.999, >=1490/1500 cols, >=95% pairs
  recovered), G1 (50-doc QA), G2 (>=50 nonzero-support occurrences/pair, <100 pairs ->
  NOT-TESTABLE, a gate outcome not a verdict). All three branches first-class with
  registered paper consequences; B2 (S2-collapse) registered as the sharpest negative.
  No-incentive declaration recorded. S3 (scramble) deferred unless PARTIAL.
```

**Observation:** thresholds frozen while no receipt is obtainable; next step is the
committed reader (tokalign_summary.py) whose selftest must fire every branch, committed
before any extraction job is submitted.

---

## 2026-08-16T01:05:00-06:00   Extraction + analysis code with planted-data selftest; AMENDMENT A1 (S2c arm) registered pre-data

**Why:** implement the registered pipeline. tokalign_extract.py keeps the forward path
byte-identical to compute_pooled_features (same loaders, bf16, hs_idx rule, sae.encode,
mean(0) pooling saved per doc as the G0 anchor) and only ADDS offset_mapping capture and
the sparse per-token dump; shards are atomic + skip-if-exists so evictions resume.
tokalign_analysis.py implements G0/G1/G2, the word alignment, S1/N1/N2, S2/N3, the
support-matched R_tok, and the bigram/max-pool arms.

**AMENDMENT A1 (registered NOW, before any receipt or dump exists anywhere).** The
registered S2 (word-type FE only, per PLAN sec 3.3) leaves doc-scale co-variation in the
residuals, and N3 (cross-document within-type permutation) destroys doc identity -- so an
S2 survival could in principle be carried by document-scale sharing beyond unigram rather
than position-level structure. To keep branch language honest, a reported (never
verdict-bearing) arm S2c is added: within-doc centering THEN word-type demeaning, same
N3 nulls. Claims-drafting guardrail registered with it: if B1 fires but S2c excess <= 0 in
>=2 entering cells, the branch-1 paper language must be scoped to "beyond-unigram sharing
at document-and-word scale" instead of "context-sensitive at word granularity". Verdict
bars are UNCHANGED. Also fixed at implementation time (both pre-data, both
definition-preserving): N3's derangement is realized as a within-group random CYCLIC
derangement (an iterative rejection scheme stalls when many types each hold one fixed
point; the cyclic construction has zero fixed points by definition), and R_tok's
denominator guard (excess<=0 -> NaN -> reader falls through to the band clause).

**Command(s):**
```bash
/usr/bin/python3 code/analysis/tokalign_analysis.py --selftest
```

**Output digest:**
```
CASE A unigram   S1 obs=0.998 p=0.0050 | S2 excess=+0.0085 R_tok=0.008 (collapses, as planted)
CASE B context   S1 obs=0.998 p=0.0050 | S2 excess=+1.0065 R_tok=1.009 p=0.0050 (survives)
CASE C doctopic  S1 obs=-0.0032 band=[-0.0064,0.0053] (at null: doc-scale signal killed)
G0 synthetic     recovered 40/40 columns, min r64=0.999999 | dump integrity min r=1.0
G1 clean passed=True; corrupted-offsets doc fails doc-QA while clean doc passes
SELFTEST PASS (pipeline can return every registered rung outcome)
First selftest iteration FAILED (R_tok~1 on planted unigram): both synthetic sides drew
IDENTICAL noise because all-length-4 vocab meant side B never split words and the rng
streams stayed aligned -- fixed by explicit struct_seed/noise_seed separation + mixed-
length vocab. Logged because a same-noise artifact is exactly what S2 must not reward.
```

**Observation:** the pipeline discriminates planted unigram from planted context sharing
and kills planted doc-topic signal by construction; next: sync to cluster, smoke on the
strong cell at 500 docs (schema + G0 recovery), then the 5-side battery.

---

## 2026-08-16T01:45:00-06:00   Smoke: extraction + G0 PASS; G1 tripped by CJK byte-fallback -> plumbing amendment A2

**Why:** registered execution plan (smoke on the strong cell, 500 docs) before the battery.

**Command(s):**
```bash
ssh cluster 'bash -lc "cd $ROOT && sbatch --export=ALL,SMOKE=1 --array=0,1 code/scripts/slurm/run_tokalign_extract.sbatch"'  # JID 15816506
ssh cluster 'bash -lc "cd $ROOT && sbatch --export=ALL,SMOKE=1 --array=0 code/scripts/slurm/run_tokalign_analysis.sbatch"'   # JID 15816612 (FAILED: lextok receipt not on cluster), 15816675
scp results/cross_model_sae/lextok20k_{...3 cells...}.json cluster:$ROOT/results/cross_model_sae/   # md5-verified vs registration
```

**Output digest:**
```
JID 15816506 (extract, 2 sides x 500 docs): COMPLETED in 88s/120s.
  qwen3-1.7b_l14: L0=50.0 exactly (TopK SAE), 32k dict. gemma2-2b_l12: L0~155 (JumpReLU), 16k dict.
JID 15816675 (analysis): G0 PASS -- side A 1498/1500 cols matched at r>=0.999 (min r 0.999998,
  median margin 0.287), side B 1500/1500; 821/822 frozen pairs recovered ON 500 DOCS ONLY.
  Dump-integrity PASS. G1 FAIL: Qwen side, doc 10, char 147 ("破" inside a truncated title):
  byte-BPE splits the multi-byte char, two tokens share char span (147,148) -> my QA counted
  any double-covered char as invalid. Receipt written with gates, rc=3, per registration.
```

**AMENDMENT A2 (plumbing, pre-verdict, licensed by the registered G1 "stop and fix" rule):**
single-char overlap runs = benign byte-fallback (counted, reported); overlap runs >=2 chars
remain invalid. This is exactly PLAN sec 7 risk 4 (byte-fallback edge case) materializing.
Selftest extended: single-char overlap must pass, 2-char overlap must fail -> 11+2 PASS.
No verdict-bearing statistic existed when this was amended (the only receipt is the smoke
gate-failure receipt above, which contains no S1/S2 numbers).

**Observation:** G0's column-matching against the committed caches is far above threshold
even at 500 docs, so frozen-pair anchoring is solid; next re-run smoke analysis with A2 and
dispatch the 5-side full extraction in parallel (extraction code untouched by A2).

---

## 2026-08-17T00:04:00-06:00   RESCOPE APPLIED: token-aligned claim retreats to S1; S2 word-identity attribution demoted to UNRESOLVED (amendment A3)

**Why:** act on the verified external review (see 2026-08-16T23:50 entry). The retreat,
exactly this far and no further: S1 stands as the finding (token-aligned within-document
sharing not reducible to document means, r=0.524/0.408/0.342, p at the 1/2001 floor in all
three cells); R_tok=1.064/1.301/1.307 is still reported but no longer read as word-identity
survival -- two artifact routes named (selection leakage vs N3; non-commuting S2c
centering) plus the R_tok support-mismatch caveat; the conditioning key is now described
accurately (lowercased, punctuation-stripped word types, >=20 occurrences).

**Command(s):**
```bash
# paper: abstract.tex, results.tex (sec:results:tokalign), limitations.tex,
#        conclusion.tex, appendix.tex (app:tokalign) rescoped
# registration: PREREGISTRATION_token_aligned.md += AMENDMENT A3 (append-only;
#        B1's S2 rung -> UNRESOLVED pending selection-clean rerun: disjoint-doc pair
#        selection/eval, joint doc+type FE residualization, identical residualization
#        per null draw, asserted common scored-pair mask in receipts; non-blind, disclosed)
# gates: check_tokalign_claims.py rewritten (the "r{=}0.34" substring shortcut removed;
#        each of 0.524/0.408/0.342 must appear verbatim; scope widened to appendix.tex +
#        conclusion.tex; selftest now plants BOTH an S1 value and an R_tok value)
#        emit_tokalign_table.py caption demoted to match (re-emitted tab_tokalign.tex)
git add -f results/cross_model_sae/tokalign20k_{qwen3-1.7b__gemma2-2b,gemma2-9b__mistral7b,llama3.2-1b__mistral7b}.json
cd paper && pdflatex x3 + bibtex
for c in code/analysis/check_*.py; do /usr/bin/python3 $c; done
# live plant: 0.524 -> 9.524 across all 5 scanned tex files -> gate exit 1; restored -> exit 0
```

**Output digest:**
```
0 undefined references; body 9pp (References tops p10)  [required ~15 lines of prose
  compression across results/limitations/discussion/conclusion to absorb the longer
  caveat language; every pinned number preserved verbatim]
check_body_pages / check_c4_table / check_lexical_claims / check_lextok_claims /
check_plants / check_receipts_tracked / check_rule_coverage / check_tokalign_claims
  -> ALL exit 0
check_tokalign_claims --selftest: clean-pass=True s1-plant-caught=True rtok-plant-caught=True
emit_tokalign_table --selftest: clean emit verifies True; planted 0.111 caught True
live S1 plant across 5 tex files: gate exit 1 (caught); restore: exit 0
receipts: 3x tokalign20k_*.json force-added past .gitignore results/* (now tracked)
```

**Observation:** the paper now claims exactly what the receipts support -- S1's advance is
kept, the S2 reading is demoted with the artifact routes named in body, limitations,
appendix and registration; the selection-clean rerun is the registered follow-up. Next:
commit + push (push explicitly requested with the rescope instruction).

---

## 2026-08-17T02:45:00-06:00   AMENDMENT A3-EXEC registered: selection-clean rerun design frozen before any clean job

**Why:** execute A3's registered follow-up (the selection-clean rerun that B1's demoted S2
rung is pending on) with every design choice fixed and committed BEFORE any clean statistic
exists: split seed, convergence tolerance, bars, branch rules, receipt schema markers.

**Command(s):**
```bash
ssh cluster 'echo SOCKET_ALIVE'                      # alive
ssh cluster 'for d in .../cache/tokalign/*/; do ...' # all 5 sides: 20 shards + receipt intact
# PREREGISTRATION_token_aligned.md += AMENDMENT A3-EXEC (append-only)
```

**Output digest:**
```
Cluster state verified: cache/tokalign 3.1G, 5 sides x 20 shards + extract_receipt.json
  (JID 15816850 dumps intact -> REUSED, no re-extraction); lextok20k_{3 cells}.json and
  cache/lextok20k_*.npz present on cluster.
A3-EXEC freezes: split seed 42 (50/50 doc split, selection on A via verbatim
  mutual_nn_pairs on split-A pooled caches, evaluation on B only); joint doc+type FE by
  float64 alternating projections to max residual group-mean < 1e-8 (max 500 sweeps,
  non-convergence = hard error); N3-clean re-residualized per draw with the same
  procedure; ONE common scored-pair mask (joint-FE AND doc-centering residual sd > 1e-8,
  both sides, + G2 support) asserted for R_tok_clean numerator AND denominator, sha256 in
  receipt. Bars UNCHANGED (S1: floor + r>=0.05; S2: R_tok>=0.30 survive / <0.10 collapse).
Branches: B1-clean (word-identity claim EARNED, unresolved clauses replaced),
  B2-clean (sharpest negative, reported as such), B3-clean (intermediate/NOT-TESTABLE,
  unresolved stays). Non-blind wrt the original battery, blind wrt clean statistics.
```

**Observation:** the design is now binding; next step is the analysis + reader code
(committed before submission), then smoke (strong cell, reduced perms), then the 3-cell
battery at n_perm=2000 on the reused dumps.

---

## 2026-09-24T11:54:29+08:00   PREREG: Lan et al. filters added to the Lan-protocol audit (no result exists yet)
**Why:** final review B P1-2 — Table 19 omits Lan et al.'s filters (non-concept, shared keyword, 1-1, |r|>0.1); brief item 2 asks to add them and rerun.
**Command(s):** ```bash
curl -sL https://arxiv.org/pdf/2410.06981v4 | pdftotext   # §3 filters, App. A (|r|<0.1 dropped), App. K (non-concept list)
curl raw.githubusercontent.com/wlg1/univ_feat_geom/main/run_pipeline/{run_expm_fns,interpret_fns,correlation_fns}.py   # code order F1 keyword -> F2 1-1 (keeps first) -> F3 r>0.1
ssh -o ConnectTimeout=8 cluster 'echo ok'   # Operation timed out (VPN down) -> local MPS re-extraction
# C4 slice rebuilt from allenai/c4 en shard 00000 with build_c4_slice.py logic: md5 30fe9a250f82eb2bcc94ff2282e1407f (= ELN md5 of the cluster slice)
```
**Output digest:**
- notes/PREREG_lan_filters_20260924.md: filter definitions (code followed where text and code differ), cross-tokenizer strip normalisation, reproduction gate G0 (|Δ|≤0.02 vs lan{T,C}_* receipts), estimands obs/rand/perm/exc/share on the filtered set, decision rules D1–D3, sentences for every outcome.
- Gemma-2-9B: google/gemma-2-9b is gated and no local token → unsloth/gemma-2-9b re-upload; G0 is the check.
**Observation:** prereg committed before any activation is extracted. Next: implement code/analysis/lan_filter_audit.py, extract locally, read G0 first.

---

## 2026-09-24T14:16:34+08:00   PREREG: Gemma Scope read-point sensitivity (hidden_states[N] vs [N+1])
**Why:** the hook investigation (<internal-notes>) found every Gemma Scope cell reads hidden_states[N] while the dictionaries were trained on resid_post N = hidden_states[N+1]; the main session chose disclosure + a pre-registered sensitivity analysis with main-text numbers unchanged.
**Command(s):**
```bash
# written by hand; no tier-1 output exists yet
git add notes/PREREG_gemma_hook_sensitivity_20260924.md experiments_detail.md
```
**Output digest:**
- Tier 1 = 6 held-out Gemma cells + 8 C4-screen cells (incl. G2-G9 and w65k); only change = Gemma layer_N -> hidden_states[N+1].
- Gates: G0 (offset 0 reproduces receipts, |dM|<=2, |dDelta_E|<=0.005); H (FVU lower and L0 closer to nominal 82/72/68 at N+1, BOS excluded).
- Decision: retained iff held-out p = 1/2001, Delta_E > 0, screen p <= 1/201; band |dDelta_E|, |dr| <= 0.05; outcomes A/B/C with paper sentences fixed in the file.
- Non-blind: G9-M at idx 21 and G9 FVU/L0 were seen in the investigation; the verdict statement is evaluated on the 5 blind cells.
- Environment: local MPS, unsloth Gemma re-uploads (cannot be bit-checked vs gated google repos; G0 is the check). cluster SSH socket down at 14:12 (no re-auth run).
- c4_fit.jsonl (md5 bbafbe4a) verified row-identical to rows 0-999 of the sorted slice.
**Observation:** registration committed before any run; next, a default-off --gemma-hook-offset switch in cross_model_sae_match.py, committed separately.

---

## 2026-09-26T03:10:00+08:00   STRENGTHEN: prereg + code for nine-cell length/topic nulls, nine-cell random split, bootstrap intervals (cluster unreachable)
**Why:** `<internal-notes>`; the most frequent (c)-class review items for #60363 are controls on 3 cells / length null on 1 cell (title says "Length"), untested i.i.d. halves for the nine-cell held-out, and no intervals for R / Delta_E (`<internal-notes>` W4, W5, W15).
**Command(s):**
```bash
git commit notes/PREREG_ninecell_ladder_20260926.md          # 3a0b358, 03:01, before any code/result
PYTHONPATH=/tmp/ct_sk ~/.venvs/lanfilt/bin/python code/analysis/ninecell_strengthen_20260926.py selftest
# local smoke (pipe only, numbers not read): 1000 docs of qwen3-1.7b__gemma2-2b from ~/gemmahook_cache, n_perm 20, n_boot 2
git commit code/analysis/ninecell_strengthen_20260926.py code/scripts/slurm/run_ninecell_strengthen_20260926.sbatch   # ba74310
route -n get <cluster-ip>   # interface en0 -> the university VPN not connected; main told to ask the author
```
**Output digest:**
- selftest PASS: numpy==torch(fp64) reciprocal counts identical; planted length-driven sharing base excess 25.8, R_len 0.144 (control can remove what it names); planted length-free sharing R_len 0.995, p 0.010; held-out planted p 0.005 vs noise p 0.284; 10 decision-branch cases.
- smoke: analyze_a / analyze_b / summarize exit 0, all receipt keys present.
- sklearn for the local smoke only: `uv pip install --target /tmp/ct_sk scikit-learn joblib threadpoolctl` (throwaway, not a shared env).
**Observation:** design = one GPU job (published compute_pooled_features for six models into /dev/shm, fp64 batched permutation counts for Q-A/Q-C on GPU, numpy Q-B in background). Waiting for the author to connect the university VPN.

---

