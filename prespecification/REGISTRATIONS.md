# Registrations: what was fixed before which result, and where

Every registration below was committed to the project's (private) development repository before
the receipt it governs existed. The commit hashes are those of that history, which is not part of
this public release; they are given so the order of events can be checked against it on request.
Stand-alone registration files ship verbatim apart from path and host scrubbing; registrations that were written into the lab
notebook ship as the excerpts in `notebook_excerpts.md`.

| Registration | Where it is in this package | Commit (date, UTC offset as recorded) | Paper |
|---|---|---|---|
| Probe-transfer hardening: capable pairs transfer >> control, null pair ~ control | `notebook_excerpts.md` #1 | `83aa3b0` (2026-07-07 -0700) | App. threats (in-sample scoping of the probe-transfer result) |
| Lexical-residualized null (first, 2,000-document attempt) | `notebook_excerpts.md` #2 | `d58fb66` (2026-07-16 -0700) | App. lexres (reported as under-powered) |
| Matched random-dictionary control and its two readings | `notebook_excerpts.md` #3 | `7c923fe` (2026-07-20 -0700) | Table 1 row 2; random-dictionary paragraph |
| Fine-grained topic (TF-IDF strata) control: decision rule | `notebook_excerpts.md` #4 | `85b7328` (2026-07-27 -0700) | Table 1 rows 5-6; `tab_covctl` |
| Amendment: block-size calibration before any k=200/500 receipt | `notebook_excerpts.md` #5 | `65a43c1` (2026-07-27 -0700) | same |
| Lexical residualization at 20,000 documents (adequacy gate, largest-passing-k rule, branches) | `notebook_excerpts.md` #6 | `6848d4c` (2026-07-27 -0700) | Table 1 row 7; `tab_lextok` |
| Lan et al. pairing rule inside the lexical pipeline | `notebook_excerpts.md` #7; `code/analysis/_prechange_3237ecb_lexical_token_test.py` is the pre-change code | `3237ecb` (2026-07-31 -0700) | Table 1 row 8; `tab_lanpair` |
| Leakage-controlled Procrustes pairing baseline (r=64, 50/50 split, branches P1-P4) | `PREREGISTRATION_procrustes_baseline.md`; `notebook_excerpts.md` #8 | `3be9b7d` (2026-08-04 -0700) | Table 1 row 9; `tab_proclex` |
| Token-aligned / order-sensitive test (S1/S2, N1-N3, G0-G2, branches B1-B3) | `PREREGISTRATION_token_aligned.md` (design notes in `PLAN_token_aligned_sharing.md`, written before registration and not binding); `notebook_excerpts.md` #9 | `df330c6` (2026-08-15 -0600) | Table 1 rows 10-12; `tab_tokalign` |
| Amendment A1 (S2c arm), before data | `notebook_excerpts.md` #10 | `c766a48` (2026-08-16 -0600) | same |
| Amendment A2 (CJK byte-fallback in the G1 check), from the smoke | `notebook_excerpts.md` #11 | `af89c96` (2026-08-16 -0600) | same |
| Amendment A3 (S2 word-identity attribution demoted; S1 stands) | `notebook_excerpts.md` #12 | `f1ef9df` (2026-08-17 -0600) | same |
| Amendment A3-EXEC (leakage-clean rerun frozen: split seed 42, joint FE, unchanged bars) | `notebook_excerpts.md` #13 | `b0b07cd` (2026-08-17 -0600) | Table 1 rows 10 and 12 (verdict-bearing) |
| Amendment A4 (B-only token gating, A-only universe screening, hardened reader) | `PREREGISTRATION_token_aligned.md` (amendment section) | `fb55a98` (2026-08-17 -0700); implementation `798e08d`, split-hash fix `b544e55` | same |
| Lan et al. filtering steps added to the Lan-protocol audit | `notes/PREREG_lan_filters_20260924.md`; `notebook_excerpts.md` #14 | `4cbd289` (2026-09-24 +0800); deviation note `baf68d8` | App. lan_filters; `tab_lanfilt` |
| Gemma Scope read point: `hidden_states[N]` vs `[N+1]` | `notes/PREREG_gemma_hook_sensitivity_20260924.md`; `notebook_excerpts.md` #15 | `f2cbf42` (2026-09-24 +0800) | App. gemmahook; `tab_gemmahook` |
| Nine-cell length/topic nulls, nine-cell random split, document-bootstrap intervals | `notes/PREREG_ninecell_ladder_20260926.md`; `notebook_excerpts.md` #16 | `3a0b358` (2026-09-26 +0800) | results (5 of 9 cells, local fallback under deviation 1, commit b196641) in `results/ninecell_20260926/local_mps_cache/`; App. ninecell, `tab_ninecell`; nine-cell cluster run (after submission) in `results/ninecell_20260926/cluster_18577359/` |

Not registered, and labelled so in the paper: the held-out frozen-pair follow-up (Table 1 rows 3-4,
`heldout_*.json`). The shared-core control is registered and unrun.

The readers that apply these rules print the registration they implement (for example
`code/analysis/tokalign_summary.py` prints the branch line fixed by `df330c6`), and the receipts
carry the registration commit where one existed at run time (`prereg_commit`, `a3_exec_commit`).
