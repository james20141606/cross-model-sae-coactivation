# PREREG — Lan et al. filtering steps added to the Lan-protocol audit (App. `app:lan`, Table 19)

Written 2026-09-24 ~12:00 +08:00, **before any analysis output of this experiment exists**
(no activations extracted, no statistic computed). Commit time of this file is the evidence.
Brief: `<internal-notes>` item 2; reviewer
finding B P1-2 (`<internal-notes>`).

## 1. Question

Table 19 (`tab:lan`) transplants the Lan et al. (arXiv:2410.06981v4) statistic — greedy
activation-correlation pairing, then SVCCA/RSA of the paired decoder rows against a
random-pairing baseline — but omits the filters their pipeline applies before scoring. Does
adding those filters change any of the three audit findings of Table 19?

- (i) the permuted-pairing null sits far above the random-pairing baseline, so part of the
  signal above random comes from the pairing step itself (intro: "from 9% on C4 to 94% on
  templated prompts");
- (ii) on the templated corpus the statistic cannot separate the strongest surviving pair
  (Gemma-2-9B↔Mistral) from the corpus-artifact pair (Pythia-70M↔Mistral);
- (iii) on C4 the statistic separates only the strongest pair; the weak surviving pair
  (Qwen3-1.7B↔Mistral) is indistinguishable from the artifact pair.

## 2. The filters, as defined by the source (checked 2026-09-24)

Source text: arXiv 2410.06981v4 §3 "Filter by Top Tokens", "Filter Many-to-1 Mappings",
App. A ("all experiments filter feature pairs by non-concept keywords, 1-1, and low
correlation (less than 0.1)"), App. K (non-concept list). Source code:
`github.com/wlg1/univ_feat_geom/run_pipeline/run_expm_fns.py::run_expm` and
`interpret_fns.py` (fetched 2026-09-24). Where text and code differ, **the code is followed**
and the difference is stated.

- **F1 keyword filter** (code: one loop does both parts). For each paired feature, take its
  top-5 activating token *positions* over the corpus (`highest_activating_tokens`, k=5,
  flattened over all tokens) and decode each token id on its own (`tokenizer.decode`).
  Drop the pair if either feature's top-5 contains a non-concept token
  (`['.', '\\n', '\n', '', ' ', '-', ',', '!', '?', '<|endoftext|>', '<bos>', '|bos|', '<pad>']`);
  otherwise keep it only if the two top-5 lists share at least one token string.
- **F2 one-to-one** (code, `oneToOne_bool`): targets that receive exactly one source feature
  in the *unfiltered* pairing keep that pair; for a target that receives several, keep only
  the first surviving pair in iteration order. (The paper text says "removing many-to-1
  pairings"; the code keeps one representative. Code followed; text reading = secondary S2.)
- **F3 low correlation** (code): keep pairs with max correlation > 0.1.
- Order F1 → F2 → F3 (code order). Their random baseline on the filtered set is
  `shuffle_rand`: permute the B rows among the kept pairs.

## 3. Our adaptation (fixed now; each departure is forced by our setting and disclosed)

Everything in Table 19 is held fixed: same eight cells (4 pairs × {templates N=206 en,
C4 N=2,000}), same models, released SAEs and hooks as the receipts `lan{T,C}_*__mistral7b.json`
(A = the non-Mistral model, B = Mistral-7B-v0.1 `mistral-7b-res-wg` resid_pre 16), same
mean-pooled per-document SAE features, same top-512-by-variance feature pool per model, same
z-scored correlation, same A→B argmax pairing (1A-to-ManyB: many A may map to one B), same
`svcca` (top-20 PCs) and `rsa` functions imported unchanged from
`code/analysis/lan_protocol_audit.py`.

- **Top-5 tokens.** Token-level SAE activations are recorded in the same forward pass that
  produces the pooled features (identical tokenization, `add_special_tokens` default, same
  hidden-state index). Top-5 = the 5 highest-activation token positions of that feature
  over all tokens of the corpus, restricted to activation > 0 (if fewer than 5 positive
  positions exist, the positive ones only; Lan's `topk` would otherwise return arbitrary
  zero-activation positions).
- **Token-string normalisation (forced by cross-tokenizer pairs).** Lan compared
  same-tokenizer pairs, so raw decoded strings were comparable. Our pairs mix SentencePiece
  (Mistral, Gemma: decode drops the leading-space marker) with byte-level BPE (GPT-2, Pythia,
  Qwen: decode keeps a leading space). Primary: compare `s = decode(id).strip()`.
  Non-concept iff `s ∈ {'', '.', ',', '!', '?', '-', '\\n'}` or the id is one of the
  tokenizer's special ids (covers `<s>`, `<bos>`, `<|endoftext|>`, `<pad>` for every model —
  their list names "the respective padding token of the tokenizer"). Shared keyword iff the
  two sets of `s` intersect (case-sensitive, as in their code).
- **Instance unit.** Lan correlate over tokens; Table 19 correlates pooled documents. The
  pairing is not changed here (changing it would change the unfiltered statistic too and is
  outside this control); stated as a remaining difference.
- **Estimability.** SVCCA uses 20 PCs, so a scored set needs K ≥ 21 pairs. Below that the
  cell (or the permutation draw) is "not estimable" and no number is reported for it.

## 4. Estimands (per cell)

Primary variant P (F1 strip-normalised + F2 code + F3):
- `K_f` = number of pairs kept; retention `K_f / 512`.
- `obs_f` = RSA (and SVCCA) of the kept pairs' decoder rows.
- `rand_f` = mean over 200 `shuffle_rand` draws (B rows permuted within the kept pairs),
  seed 0. `sd_rand_f` = SD of those draws.
- `perm_f` = our permuted-pairing null pushed through the **whole** pipeline: permute Zb rows,
  argmax pairing, F1 (same per-feature top-5 tokens; the shared-keyword test is re-evaluated
  on the new pairs), F2 (counts on the permuted pairing), F3 (permuted correlation), score.
  200 draws, seed 1. A draw is estimable iff it keeps ≥ 21 pairs. `perm_f` = mean over
  estimable draws; reported only if ≥ 100/200 draws are estimable, else "not estimable";
  the estimable fraction and the median kept size are always reported.
- `exc_f = obs_f − perm_f`; `share_f = (perm_f − rand_f)/(obs_f − rand_f)` (the estimand of
  the intro's 9–94%).
- Unfiltered comparator: the same local activations run through the unchanged
  `lan_statistics` (seed 0, n_base=20, topk=512) = gate G0 below, plus the same with
  n_base=200 for like-for-like baselines.

Secondary (reported in the appendix table note, no decision attached):
- S1 literal strings: F1 with raw `decode` strings, no strip (Lan's code verbatim + special ids).
- S2 paper-text 1-1: drop every pair whose target receives >1 source after F1.
- Per-filter retention counts (after F1, after F2, after F3).

RSA is the primary statistic (Table 19's "exc." column is RSA); SVCCA is reported beside it.

## 5. Reproduction gate G0 (read first, before any filtered number)

Local re-extraction on this Mac (cluster unreachable: `ssh cluster` times out 2026-09-24 11:30,
the university VPN down). Apple M5 Pro, MPS, bf16 model, float32 SAE; transformers 4.56.2, sae-lens
6.51.1, torch 2.14. C4 slice rebuilt from `allenai/c4` en train shard 00000 with
`build_c4_slice.py`'s logic: md5 `30fe9a250f82eb2bcc94ff2282e1407f` = the ELN md5 of the
slice used on the cluster. Gemma-2-9B: `google/gemma-2-9b` is gated and this machine has no
licensed token, so the ungated re-upload `unsloth/gemma-2-9b` (bf16) is used; G0 is the
check that it behaves as the original.

G0 per cell: |local obs − receipt obs| ≤ 0.02 for both RSA and SVCCA.
- If G0 passes in all 8 cells: the filtered results are presented as filters applied to the
  Table 19 pipeline.
- If G0 fails in some cells: those cells are still reported, flagged, and compared only with
  their **local** unfiltered values (same activations, so the filtered-vs-unfiltered contrast
  is internal); the appendix states the size of the mismatch.
- If G0 fails in ≥ 3 of 8 cells: the appendix says the local re-extraction does not reproduce
  Table 19 closely enough to call it the same pipeline; the filtered-vs-local-unfiltered
  contrast is still reported, and the intro/main text are **not** changed on its basis.

## 6. Decision rules (primary variant P, RSA; the three non-null pairs)

Noise yardstick for a difference between two cells: `2·sqrt(sd_rand_f1² + sd_rand_f2²)`.
"Separated" means difference ≥ max(0.05, yardstick); "indistinguishable" means difference
< max(0.05, yardstick).

- **D1 (finding i).**
  - *persists*: in ≥ 4 of the 6 non-null cells, perm_f is estimable, `perm_f − rand_f ≥ 0.02`
    and `share_f ≥ 0.10`.
  - *removed*: in ≥ 4 of the 6 non-null cells, perm_f is not estimable (the filters discard
    the permuted pairings) or `perm_f − rand_f < 0.02`.
  - otherwise *mixed* (report per-corpus pattern).
- **D2 (finding ii).** Compare Gemma-9B vs Pythia on templates, `exc_f` (if either perm_f is
  not estimable, use `obs_f − rand_f` for both and say so).
  - *persists*: indistinguishable.
  - *removed*: Gemma separated above Pythia.
  - *reversed*: Pythia separated above Gemma (reported as persists-in-kind).
- **D3 (finding iii).** On C4, Qwen vs Pythia `exc_f` (same fallback).
  - *persists*: Qwen and Pythia indistinguishable (or Pythia separated above Qwen).
  - *removed*: Qwen separated above Pythia.
  Also reported: whether Gemma C4 remains separated above both.
- Any non-null cell with `K_f < 21` makes the finding that uses it *not testable* for that
  cell; if D2 or D3 lacks a cell, that decision is "not testable".

## 7. Sentences written in advance

Appendix `app:lan`, replacing "We reproduce the pairing and similarity statistic; their
additional filters (...) are not applied, so our transplant tests the statistic, not their
full pipeline.":

- Base sentence (all outcomes): "We reproduce the pairing and similarity statistic, and
  separately add their filters (\Cref{app:lan_filters})."
- App. `app:lan_filters` (new subsection) carries the table, the prereg path, the receipts,
  and one outcome paragraph per decision:
  - D1 persists: "With their filters, the permuted-pairing null still sits above their
    random baseline in N of 6 non-null cells (share X–Y%): the filters do not remove the
    pairing step's contribution."
  - D1 removed: "With their filters, the permuted pairings are mostly discarded (by the
    shared-keyword and $|r|{>}0.1$ steps) or no longer exceed their random baseline, so the
    pairing-step inflation we report is a property of the unfiltered statistic; their full
    pipeline guards against it in part."
  - D1 mixed: "With their filters, the pairing-step inflation remains on [corpus] but not on
    [corpus]."
  - D2 persists / removed; D3 persists / removed: "With their filters the templated corpus
    still does not / now does separate the strongest pair from the artifact"; "On C4 the
    filtered statistic still does not / now does separate the weak surviving pair from the
    artifact."
  - Retention always stated: "The filters keep A–B% of the 512 pairs (their reported range
    for pretrained SAEs is 20–30%)."

Main text (intro l.26–29, net +0 lines):
- D1 persists: intro sentence unchanged except "re-implemented" → "re-implemented, with or
  without their filters," only if it fits in the same lines; otherwise unchanged.
- D1 removed or mixed: the intro claim is scoped to the unfiltered statistic, e.g. "re-
  implemented without their token filters on three of our pairs ..." and the "9–94%" clause
  kept (it is still the unfiltered measurement), with the filtered caveat in the appendix.
- G0 fails in ≥ 3 cells: main text unchanged.
- Abstract and conclusion do not mention the Lan audit; they are not changed under any
  outcome.

## 8. What this does not test

No new claim beyond these eight cells. Lan's token-level instance unit, Spearman/Euclidean RSA
variant, and same-tokenizer model pairs are not reproduced; the filtered result is "their
filters applied to our transplant", not a replication of their numbers.

## 9. Deviations

None yet. Any code defect found after results exist is recorded here as a dated
"deviation" entry, never by editing §§1–8.

### Deviation note — 2026-09-24 13:55 +08:00 (appended after results; §§1–8 unchanged)
Found by the independent verifier (`<internal-notes>` §2),
not by us before the run. §2 says "where text and code differ, the code is followed". At the public
HEAD of `wlg1/univ_feat_geom` (commit 9951f22b, 2025-03-23), the `else:` after the keyword loop in
`run_expm` is indented at the level of the `for`, so it is a Python for-else. The loop has no
`break`, so the `else` always runs and resets `filt_corr_ind_A/B` to the unfiltered set: F1 has no
effect in that version. In earlier commits (558a46b4 and before, including cdf80003 and 781e13a1)
there is no such `else`, and F1 filters. We implemented F1 as the paper text states and as those
earlier commits do. The paper's reported retention of 20–30% implies F1 was active in their runs.
"Code followed" therefore holds for the ≤558a46b4 code and for F2/F3, but not for F1 at the current
HEAD. No result, decision or paper sentence changes because of this note.
