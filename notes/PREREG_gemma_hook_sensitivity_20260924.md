# PREREG — Gemma Scope read-point sensitivity: `hidden_states[N]` vs `hidden_states[N+1]`

Written 2026-09-24 14:15–14:40 +08:00, **before any tier-1 output of this experiment exists**
except the results listed in §2 as already seen. Commit time of this file is the evidence.
Brief: `<internal-notes>` (discipline);
investigation: `<internal-notes>` (§1–§4).

## 1. Question

Every Gemma Scope cell in the paper encodes `hidden_states[N]` of the HF forward pass
(= the input to block N, resid_pre N) with the `layer_N` dictionary, while the dictionary was
trained on `blocks.N.hook_resid_post` = `hidden_states[N+1]`. Cause: the convention written in
commit 6543990 (2026-06-28): `hs_idx = hook_layer + 1 if "resid_post" in hook_name`, where
`hook_name = cfg.hook_name + sae_id`; under sae-lens 6.x `cfg.hook_name` does not exist (the
name lives in `cfg.metadata.hook_name`) and the Gemma sae_id `layer_N/width_*/canonical`
contains no "resid_post", so the +1 branch never fired. Cluster logs (`hook_layer=N hs_idx=N`)
and a local fingerprint (§2) confirm the cluster read index N. The other suites (Mistral,
Qwen-Scope, chanind Llama, Pythia, GPT-2) read their own training point.

**Question.** If the only change is that every Gemma Scope dictionary reads
`hidden_states[N+1]` (its training hook), does any Gemma cell of the held-out confirmation
(Table `tab:c4_verdicts`) or of the C4 screen (Table `tab:c4_counts`) change verdict, and by how
much do the numbers move?

**What is decided in advance, independent of the result.** The main-text numbers stay on the
original read point (index N): every rung of the control ladder was computed at that point, so
the paper is internally consistent there, and mixing read points inside one table before the
deadline is worse than a disclosed sensitivity analysis. The training-hook numbers are this
pre-registered sensitivity. Only the decision rule in §6 can change a main-text count or sentence.

## 2. Results already seen before this registration (non-blind items)

The hook investigation (2026-09-24 ~13:00–14:00, `/tmp/hookcheck/`) computed, before this file:
- **G9–M (Gemma-2-9B ↔ Mistral-7B) at idx 21**, local MPS, unsloth/gemma-2-9b:
  screen M = 841 (null μ 550.6), mutual z 27.5 (idx 20 local: 820, z 24.7; receipt 819, z 24.0);
  held-out pairs 798, Δ_E 0.617, retained r 0.816 (idx 20 local: 0.601 / 0.792; receipt
  0.602 / 0.791). The screen and held-out used `rows[:1000]`/`rows[1000:]` of the slice.
- **Gemma-2-9B L20 16k FVU / L0** on 200 C4 docs at idx 19–22 (idx 20: FVU 0.190, L0 47.9
  no-BOS; idx 21: FVU 0.156, L0 64.8; nominal 68). Pooled top-1500 per-feature correlation
  between idx 20 and idx 21: median 0.993.
- FVU scans of the non-Gemma suites at neighbouring indices (their read points are correct).

Consequences: **G9–M is non-blind.** Its tier-1 numbers are recomputed with the committed code
(§4) and reported, but the "verdict survives" statement of §6 is evaluated separately on the
**five blind held-out cells** (Q1.7–G2, Q8–G2, Q1.7–G9, Q8–G9, G2–M) and reported with G9–M
flagged "seen before registration". Nothing about Gemma-2-2B at idx 13, nor any Qwen3-8B cell,
nor any 65k result has been computed.

## 3. The only change, and everything held fixed

- **Change:** Gemma Scope `layer_N` → `hidden_states[N+1]` (flag `--gemma-hook-offset 1`; the
  default `0` leaves the pipeline byte-identical to the published one). Applies to every Gemma
  dictionary: 2B `layer_12/width_16k/canonical` (path `average_l0_82`), 2B
  `layer_12/width_65k/canonical` (`average_l0_72`), 9B `layer_20/width_16k/canonical`
  (`average_l0_68`).
- **Fixed:** corpus `data/c4_slice.jsonl` (md5 30fe9a250f82eb2bcc94ff2282e1407f), rows = `lang=en`
  sorted by `pair_id`, 2,000 docs; selection half = rows 0–999 (= `c4_fit.jsonl`, md5
  bbafbe4a34537b9d73f72bcf3e94d5d0, row-identity checked locally 2026-09-24), evaluation half =
  rows 1000–1999 (= `c4_heldout.jsonl`, md5 dd016e9893d9f29f298f05b1ce58a63e on the cluster);
  tokenization with `add_special_tokens` default (BOS kept); mean-pool over all tokens incl. BOS;
  model in bf16, SAE input cast to the SAE dtype; top-1,500 features by variance per side;
  `match_and_null` (n_perm 200, seed 0, n_perm_hungarian 20) for the screen;
  `heldout_frozen_pairs` (top-k chosen on the selection half, n_perm 2000, seed 0) for held-out;
  cell orientation (model_a, model_b) exactly as in the receipts (the null permutes model_b rows);
  non-Gemma sides at their published indices (Mistral-7B `blocks.16.hook_resid_pre` → 16,
  Qwen3-1.7B-Base `layer14` → 15, Qwen3-8B-Base `layer18` → 19).
- Statistics code is imported unchanged from `code/analysis/cross_model_sae_match.py`.

## 4. Cells, environment, weights

**Tier 1 (decision tier).**
- Held-out (6 cells, the Gemma rows of `tab:c4_verdicts`, receipts `heldout_*.json`):
  Q1.7–G2, Q8–G2, Q1.7–G9, Q8–G9, G9–M, G2–M.
- Screen (8 cells, receipts `c4_*.json`): the same six + G2–G9 (within family, both sides
  shift) + the 65k recipe check `w65k_gemma2-2b__mistral7b`.

**Primary environment: local Apple-silicon MPS** (cluster GPU is held at the group QOS limit;
the cluster SSH socket is also down as of 14:12). torch 2.14.0, transformers 4.56.2,
sae-lens 6.51.1 (cluster: CUDA, sae-lens 6.43.0). Weights:
- Gemma: **unsloth re-uploads** `unsloth/gemma-2-2b` (rev 25319945f7fd…) and
  `unsloth/gemma-2-9b` (rev 9145841c83ad…) — the `google/gemma-2-*` repos are gated and no HF
  token is available locally, so the weights **cannot be compared bit-for-bit** with the cluster's;
  gate G0 is the equivalence check and this is disclosed.
- Mistral `mistralai/Mistral-7B-v0.1` (rev 27d67f1b…), Qwen `Qwen/Qwen3-1.7B-Base`
  (rev ea980cb0…), `Qwen/Qwen3-8B-Base` (rev 49e3418f…); SAEs from the same releases as the
  cluster (gemma-scope canonical, `mistral-7b-res-wg`, Qwen-Scope via `qwen_sae_adapter.py`).
- If a cluster replication with the original weights completes, it is reported next to the local
  numbers; the decision uses the local run.

Every receipt records: script sha256, input corpus md5, the pooled-feature file sha256 of each
side, model repo + revision, SAE release/id/path, hs_idx used, device, library versions, seeds.

## 5. Gates (checked before any N+1 number is interpreted)

- **G0 (switch off reproduces the published receipts).** With offset 0, the local pipeline must
  reproduce every tier-1 receipt: screen |ΔM| ≤ 2 reciprocal pairs; held-out |ΔΔ_E| ≤ 0.005.
  A cell failing G0 has its N+1 result reported as "G0 failed — not interpreted" and takes no
  part in the decision.
- **H (the N+1 index is the training point).** For each of the three Gemma dictionaries, on the
  first 200 C4 docs, BOS excluded: FVU(N+1) < FVU(N) **and** |L0(N+1) − nominal| <
  |L0(N) − nominal| (nominal 82 / 72 / 68 from the release path). If H fails for a dictionary,
  stop: its cells are not interpreted and the failure is reported. FVU = Σ‖x−x̂‖² / Σ‖x−x̄‖²
  over tokens; L0 = mean count of active latents per token; also reported with BOS included.

## 6. Primary outcome and decision rule

Per held-out cell at N+1: **retained** iff held-out p equals the floor 1/2001 **and**
Δ_E > 0 **and** the N+1 screen mutual p ≤ 1/201.
**Numerical-equivalence band:** |Δ_E(N+1) − Δ_E(N)| ≤ 0.05 **and** |r(N+1) − r(N)| ≤ 0.05,
where Δ_E(N), r(N) are the published receipt values.

Outcomes (evaluated on the five blind cells; G9–M reported with its non-blind flag and, if it
disagrees with the five, the disagreement is stated):
- **A — all retained, all in band.** Main text unchanged apart from the disclosure qualifier.
  Appendix sentence: "Reading every Gemma Scope dictionary at its training hook
  (`hidden_states[N+1]`) moves the held-out excess of the six Gemma cells by [x, y] and the
  retained fraction by [u, v]; every verdict is unchanged (Table X)."
- **B — all retained, ≥1 out of band.** Appendix table shows both columns; the appendix sentence
  names the out-of-band cells and their N+1 values; the main-text qualifier reads "(at the training
  hook the held-out excess moves by up to Z; App. X)" in place of the plain pointer.
- **C — any cell not retained.** That cell is removed from "all nine cross-family cells" in the
  abstract, intro and results ("k of nine"), with the reason; the main session is told so the
  FINAL_ABSTRACTS copy is synchronised.
- If G0 or H blocks a cell, the outcome is evaluated on the remaining cells and the blocked
  cells are named as untested.

## 7. Secondary (descriptive, no decision)

- Screen M, null μ/σ, mutual z, and excess % at N and N+1 for all 8 screen cells.
- 65k recipe check: z(16k) − z(65k) at N+1 vs the published 22.1 → 14.1 (−8.0).
- G2–G9 within-family cell at N+1.
- FVU / L0 (with and without BOS) at N and N+1 for the three Gemma dictionaries, used to correct
  the roster table.

## 8. Tiers 2–3 (lexical / token-aligned arms, 20k docs)

Not part of this decision. They run only if tier 1 is finished and compute is available; each
follows its own originally registered rule, enters the appendix only if complete by
2026-09-25 08:00 +08:00, and otherwise is recorded in the ELN only. They cannot change a
main-text sentence.

## 9. Disclosure edits made regardless of outcome

- Roster table: the hook column states the actual read point (`hidden_states[N]` = input to
  block N) and the training hook (resid_post N); the L0 column is corrected from the receipts
  (Mistral 88.2, etc.) with BOS inclusion stated.
- Threats/limitations: the read point, the FVU/L0 at both indices, and this analysis.
- Main text: one existing phrase replaced by a qualified one (net 0 lines), e.g. "each model's
  released SAE (Gemma dictionaries read one block before their training hook; App. X)".
- Feature labels (obituary, cricket, …) worded "at our read point"; the steering interpretation
  qualified as "injecting the released feature direction at our read point".

## 10. Deviations

Any departure from this file after results exist is written as "deviation from pre-registration"
with the reason, in the appendix and the ELN. A code defect found later is disclosed, not
silently fixed.
