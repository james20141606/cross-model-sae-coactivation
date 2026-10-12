# Release inventory

What ships, what each piece needs, and which script produced which receipt. Receipt names are the
ones the paper cites; `*` stands for a cell tag such as `qwen3-1.7b__gemma2-2b`.

## Layout

| path | contents | needs |
|---|---|---|
| `results/` | every receipt the paper reads from (JSON), plus `results/probe_attribution/README.md` | nothing |
| `code/analysis/` | extraction, matching, controls, readers, table emitters, `check_*.py` gates | see below |
| `code/scripts/slurm/`, `scripts/slurm/`, `scripts/*.slurm` | the SLURM launch configurations as run (identity strings replaced) | a GPU cluster |
| `experiments/` | the v0/v1 probe-attribution and attribution-graph experiments (`probe_attribution_v2.py`, `utils.py` adapter over `circuit-tracer`) | GPU |
| `code/figures/`, `paper/figures_src/`, `code/figstyle/` | figure scripts (data panels are drawn from receipts) | matplotlib |
| `paper/main.tex`, `paper/sections/*.tex`, `paper/main.pdf` | the arXiv version's LaTeX (read by the gates), style files, figure PDFs, and its compiled PDF | nothing |
| `PREREGISTRATION_*.md`, `notes/PREREG_*.md`, `prespecification/` | registrations and notebook excerpts; `prespecification/REGISTRATIONS.md` is the index | nothing |
| `data/` | the multilingual prompt sets (the C4 slices are rebuilt, README section 2) | nothing |
| `tools/` | `verify_package.py`, `reproduce_main_table.py`, `run_offline_checks.sh` | Python 3.9+ |
| `setup.sh` | environment for the attribution-graph arm (PyTorch 2.6.0 / CUDA 12.4, `circuit-tracer` 4bb8c0e) | conda |

## Receipt families: producer, reader, paper location

GPU = needs a GPU to regenerate; CPU = regenerates from other receipts or from tokenizers only.

| receipts (`results/cross_model_sae/` unless stated) | produced by | launch config | read by | paper |
|---|---|---|---|---|
| `c4_*.json` (2,000-doc screen, n_perm 200), `c4n2k_*.json` (n_perm 2,000) | `cross_model_sae_match.py` (GPU) | `run_sae_match_*.sbatch`, `run_c4_roster_nperm2000.sbatch` | `check_ladder_table.py` (`c4_*`), paper appendix (`c4n2k_*`) | Table 1 row 1; C4 count and verdict tables |
| `heldout_*.json` (frozen pairs, 1,000 + 1,000 docs) | `cross_model_sae_match.py --select_prompts` (GPU) | `run_c4_heldout_frozen.sbatch` (`submit_heldout_grid.sh`) | `check_ladder_table.py`, `check_c4_table.py` | Table 1 rows 3-4; held-out table (not pre-registered) |
| `c4h_*.json` | `cross_model_sae_match.py` (GPU) | `run_sae_match_llama32_heldout.sbatch` and siblings | paper appendix | App. C4 held-out screen |
| `rand_c4_*.json` | `cross_model_sae_match.py --random_sae random-dir` + `random_sae_baseline.py` (GPU) | `scripts/slurm/run_cross_model_sae_randcontrol.sbatch` | `check_ladder_table.py` | Table 1 row 2 |
| `covctl_k{200,500}_c4_*.json`, `covctl_c4_*.json` | `c4_covariate_null.py` (GPU) | `run_c4_covariate_null*.sbatch` | `covctl_summary.py`, `emit_covctl_table.py` | Table 1 rows 5-6; `tab_covctl`; their length-decile arm in App. ninecell |
| `ninecell_20260926/local_mps_cache/{A,B}_*.json`, `NINECELL_SUMMARY.json`, `tab_ninecell.tex` | `ninecell_strengthen_20260926.py` (`analyze_a`, `analyze_b`; local CPU on pooled features extracted for App. gemmahook) | `code/scripts/run_ninecell_local_fallback_20260926.sh`; run record `RUN.md` (the nine-cell GPU job `run_ninecell_strengthen_20260926.sbatch` was not run) | `ninecell_strengthen_20260926.py summarize` | App. ninecell; `tab_ninecell`; Fig. 2 caption |
| `lextok20k_*.json` (+ `_k256/_k1024/_k4096`) | `lexical_token_test.py` (GPU) | `run_lexical_token_test.sbatch` | `lextok_summary.py`, `check_lextok_claims.py` | Table 1 row 7; `tab_lextok` |
| `lanpair_lextok20k_*.json` | `lexical_token_test.py --pairing` (Lan rule; GPU) | `run_lanpair_lextok.sbatch` | `lanpair_summary.py` | Table 1 row 8; `tab_lanpair` |
| `proclex_lextok20k_*.json` | `lexical_token_test.py --pairing` (Procrustes; GPU) | `run_proclex_lextok.sbatch` | `procrustes_summary.py` | Table 1 row 9; `tab_proclex` |
| `splithalf20k_*.json` | `pooled_splithalf.py` (pooled features of the lexical runs) | as in its docstring | `check_tokalign_claims.py`, gates (`GATE_INPUTS`) | App. threats (iv) |
| `tokalign20k_*.json`, `tokaclean20k_*.json`, `tokalign_extract/*/extract_receipt.json` | `tokalign_extract.py` (GPU), then `tokalign_analysis.py` / `tokalign_clean_analysis.py` (CPU, large memory) | `run_tokalign_extract.sbatch`, `run_tokalign_analysis.sbatch`, `run_tokalign_clean.sbatch` | `tokalign_summary.py`, `tokalign_clean_summary.py`, `emit_tokalign_table.py`, `check_tokalign_claims.py` | Table 1 rows 10-12; `tab_tokalign` |
| `lanT_*`, `lanC_*` (Lan protocol on templates / C4), `lanfiltT_*`, `lanfiltC_*`, `LANFILT_DECISIONS.json` | `lan_protocol_audit.py`, `lan_filter_audit.py` (GPU or MPS) | local runs (device recorded in the receipts) | `lanfilt_summary.py`, `check_lanfilt_claims.py` | App. lan, lan_filters; `tab_lanfilt` |
| `gemma_hook_sensitivity/` (`GEMMA_HOOK_SENSITIVITY.json`, `cluster/`, `cluster_q8g2/`, `off{0,1}_*.json`, `GEMMA_HOOK_COMBINED.json`, `interp_explore_*.json`) | `gemma_hook_sensitivity.py` (GPU / MPS), `gemmahook_summary.py` (CPU) | `run_gemma_hook_sensitivity.sbatch` | `check_gemmahook_claims.py` | App. gemmahook; `tab_gemmahook` |
| `bos_probe_ct.json` | `bos_probe_ct.py` (tokenizers only) | command in its docstring | `check_c4_table.py` | App. threats (v), C4-slice note |
| `interp*_*.json`, `INTERP_OVERLAP_NULL.json` | `interpret_matched_features.py` (GPU), `interp_overlap_null.py` (CPU) | `run_interp_heldout.sbatch` | paper appendix; regenerated by `run_offline_checks.sh` | App. interpretation |
| `probe_transfer_*.json` | `cross_model_probe_transfer.py` (GPU) | `run_probe_transfer_c4match.sbatch` | paper appendix | App. threats (probe transfer) |
| `steer*_*.json` | `cross_model_steer_transfer.py` (GPU) | arguments in the script's `argparse` block | paper appendix | steering-transfer null |
| `xling_*`, `zh_*`, `hard2000_*`, layer-band cells (`gemma2-2bL6__...`), template cells (`gemma2-2b__mistral7b.json`, ...), `SUMMARY.json` | `cross_model_sae_match.py` (GPU), `cross_model_sae_summary.py` (CPU) | `scripts/slurm/run_cross_model_sae.sbatch` | paper appendix | SAE-matching tables on templated prompts |
| `results/cross_model_causal*/` | `cross_model_causal.py`, `cross_model_causal_group.py` (GPU), `cross_model_causal_summary.py` | `scripts/slurm/run_cross_model_causal*.sbatch` | paper appendix | causal-transfer null |
| `results/probe_attribution_v2/`, `results/probe_attribution/` | `experiments/probe_attribution_v2.py`, `experiments/06_probe_attribution_run.py` (GPU) | `scripts/slurm/*` | paper | v1 probe-attribution sweep |
| `results/probe_attribution_controls{,_n200}/` (+ `STATS.json`, `LANGPAIR_SPLIT.json`) | `probe_attribution_runner.py` (GPU); `probe_residual_stats.py`, `probe_residual_by_langpair.py`, `probe_controls_summary.py` (CPU) | `run_probe_controls{,_n200}.sbatch` | paper | construct-validity controls; depth-corrected residual |
| `results/rosetta_check/` | `rosetta_sublinear_check.py`, `rosetta_slope_bootstrap.py` (CPU) | none | regenerated by `run_offline_checks.sh` | 70M-to-7B match appendix |

`smoke_*.json` are pipeline smokes that were committed alongside the runs; no number in the paper
is read from a smoke.

## Known limits of this package

- The C4 slices, model weights, SAE weights and pooled activations are not shipped; everything
  upstream of a receipt needs them (README section 2-3).
- `code/analysis/check_receipts_tracked.py` checks git tracking and needs the repository;
  `tools/verify_package.py` runs its path lists against this package instead.
- `code/analysis/check_plants.py` and `check_rule_coverage.py` recompile the paper and need the
  the full LaTeX toolchain and the development repository's build scripts; they are included as code only.
- `code/analysis/emit_tokalign_table.py` without flags rewrites `paper/sections/tab_tokalign.tex`; its
  caption template writes `9.4\times10^{5}` where the submitted caption reads `940,000` (numbers
  identical). Use `--selftest` and `check_tokalign_claims.py`, as `run_offline_checks.sh` does.
  `lanfilt_summary.py` and `gemmahook_summary.py` likewise rewrite their tables (to identical bytes);
  their `check_*` counterparts verify without writing.
- `code/analysis/_prechange_3237ecb_lexical_token_test.py` is the lexical pipeline as it stood at the
  Lan-pairing registration; `run_lanpair_lextok.sbatch` and `run_proclex_lextok.sbatch` pin it by SHA-256 and re-run it per cell to
  show the default (mutual-NN) path is unchanged by the pairing option.
