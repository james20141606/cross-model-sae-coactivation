# Cross-Model SAE Coactivation Survives Length, Topic, and Lexical Controls

Xupeng Chen (New York University), Runhao Li (University of Southern California),
Jiayi Cheng (New York University), Michael Yue (Columbia University),
Chenqian Le (New York University)

Code, receipts and registrations for the paper. The paper asks whether cross-model matches of
sparse-autoencoder (SAE) features, such as the "obituary" features of Gemma-2-9B and Mistral-7B,
reflect shared feature content or only cheap coincidence: two sets of 1,500 features reciprocally
match about half of each other under idealized chance, and any document property both models encode
(length, topic, vocabulary) can make features coactivate. We run the pairing rule inside a
document-permutation null, score frozen pairs on documents the rule never saw, and add controls that
each fix one named nuisance. All nine cross-family cells (Gemma-2, Mistral, Qwen3, Llama-3) hold on
disjoint C4 text (p = 1/2001 each, Bonferroni over the 17-cell roster; a follow-up that was not
pre-registered); on three representative cells every control leaves at least 0.88 of the excess its
reference arm leaves; regressing out a unigram TF-IDF lexical model leaves 92-94% of the excess; and
a registered token-aligned test keeps 0.925-0.986 of the token-level excess after conditioning on
the document and the current word. The claim is coactivation, not feature identity, and does not
extend to compositional sharing, a capability trend, or shared computation. Three decisive-looking
secondary results were rejected by the same controls and are reported.

This repository holds the analysis code, the run configurations, the receipts (the JSON outputs that
every number in the paper is read from), the registrations, and the checkers that pin the paper's
numbers to the receipts. It holds no model weights, no activation caches and no C4 text beyond the
short document excerpts that the feature-interpretation receipts quote; the C4 slices are rebuilt by
a script (below).

## Citation

```bibtex
@article{chen2026crossmodelsae,
  title   = {Cross-Model {SAE} Coactivation Survives Length, Topic, and Lexical Controls},
  author  = {Chen, Xupeng and Li, Runhao and Cheng, Jiayi and Yue, Michael and Le, Chenqian},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## 1. Re-compute the paper's numbers without a GPU

Everything in this section reads only the bundled receipts. Tested with the system Python 3.9.6 of
macOS (`/usr/bin/python3`, numpy 2.0.2) and with a virtualenv (Python 3.9.6, numpy 2.0.2,
scipy 1.13.1, matplotlib). The two `tools/` Python scripts need only the standard library; some
readers import numpy. Run from the repository root:

```bash
python3 tools/verify_package.py            # manifest SHA-256s; every receipt the paper cites is here
python3 tools/reproduce_main_table.py      # Table 1 (the control ladder): each row's value triple recomputed
                                           # from receipts and searched for in the text of paper/main.pdf
bash tools/run_offline_checks.sh           # every receipt-only check, one line per check; exit 0 = all pass
```

`tools/reproduce_main_table.py` compares the strong/median/weak value triple of each Table 1 row (it
prints, but does not compare, the p column) and looks for that string anywhere in the PDF text, not
at its row. It reuses the derivation in `code/analysis/check_ladder_table.py`, the code of our own
table gate, so it is a reproduction from the receipts, not an independent re-derivation. The PDF
comparison needs `pdftotext` (poppler) and is skipped with a note if it is missing. Expected last
line: `PDF comparison: 12/12 rows found verbatim in paper/main.pdf`.

`tools/run_offline_checks.sh` runs and reports the exit code of:

| command | what it establishes |
|---|---|
| `code/analysis/check_ladder_table.py` (+ `--selftest`) | every cell of Table 1 and of the C4 count table equals its receipt; planted errors are caught |
| `code/analysis/check_main_numbers.py` (+ `--selftest`) | located pins for numbers in the abstract, body and Appendix L / Table 14, each derived from a receipt; withdrawn text must stay absent |
| `code/analysis/emit_ninecell_table.py --check` | Table 14 equals its rendering from the nine-cell cluster receipts |
| `code/analysis/check_c4_table.py` | the C4 verdict table and the held-out ratio range |
| `code/analysis/emit_covctl_table.py --check` | the topic-strata table (`tab_covctl`) against `covctl_k*_c4_*.json` |
| `code/analysis/lextok_summary.py --check`, `lanpair_summary.py --check`, `procrustes_summary.py --check` | the three lexical-residualization tables and their registered verdicts |
| `code/analysis/tokalign_summary.py`, `tokalign_clean_summary.py`, `check_tokalign_claims.py`, `emit_tokalign_table.py --selftest` | the token-aligned arm: registered branch lines, table cells, prose pins |
| `code/analysis/covctl_summary.py` | the length/topic-matched null verdicts |
| `code/analysis/check_lanfilt_claims.py` | the Lan et al. filter follow-up (`tab_lanfilt`) |
| `code/analysis/check_gemmahook_claims.py` | the Gemma Scope read-point sensitivity (`tab_gemmahook`), with SHA-256 pins on its receipts |
| `code/analysis/check_lextok_claims.py`, `check_lexical_claims.py`, `check_body_pages.py` | claim sites in the TeX and the compiled PDF (need `pdftotext`) |
| `code/analysis/rosetta_sublinear_check.py`, `rosetta_slope_bootstrap.py`, `interp_overlap_null.py` | re-derive three appendix receipts and confirm the regenerated JSON equals the shipped one |

Optional and slow (about 12 CPU-minutes): `RUN_SLOW=1 bash tools/run_offline_checks.sh` also runs
`code/analysis/probe_residual_stats.py`, which re-derives
`results/probe_attribution_controls_n200/STATS.json` from the per-model logit-lens matrices.

Figures 1 and 2 are drawn from receipts by `code/figures/fig1_control_ladder_20261010.py` and
`code/figures/main_results_20261010.py` (need matplotlib; `compose.export` in `code/figstyle/` also
needs a headless Chrome for the PDF). Each writes `paper/figures/<dir>/*.json` listing the receipt
files it read with their SHA-256; the shipped JSONs were regenerated from the receipts in this
repository.

## 2. Data and models

All models, SAE suites and the corpus are public; the HuggingFace IDs and pinned snapshot SHAs are in
the paper (Table 3, "Pinned snapshot revisions").

- Models: Gemma-2-2B/9B, Mistral-7B-v0.1, Qwen3-1.7B/8B-Base, Llama-3.2-1B, Pythia-70M, GPT-2 (SAE
  arm); Qwen3 0.6B-14B, Gemma-3 1B-27B-it and the 17-model control set (probe-attribution arm).
- SAEs: Gemma Scope (`gemma-scope-*-pt-res-canonical`), `mistral-7b-res-wg`, `gpt2-small-res-jb`,
  `pythia-70m-deduped-res-sm`, `chanind/sae-llama-3.2-1b-res` via `sae-lens`; Qwen-Scope
  (`Qwen/SAE-Res-Qwen3-*`) via `code/analysis/qwen_sae_adapter.py`.
- Prompts: `data/prompts_v3_multi.jsonl` (the 206-pair multilingual set; `code/analysis/build_prompts_v3.py`)
  and `data/prompts_expanded_v2.jsonl`.
- C4 is not redistributed here. Rebuild the two slices (needs `datasets` and network access):

  ```bash
  python code/analysis/build_c4_slice.py --n 2000                                   # data/c4_slice.jsonl
  python code/analysis/build_c4_slice.py --n 20000 --out data/c4_slice_20k.jsonl    # lexical / token arms
  ```

  The script streams the first documents of `allenai/c4` `en/train` and truncates each to 300
  characters; the first 2,000 lines of the 20k slice equal the 2,000 slice. Pass `revision=` to
  `load_dataset` to pin the revision listed in the paper's Table 3.

## 3. What needs a GPU

Every step that runs a language model: feature extraction and matching
(`code/analysis/cross_model_sae_match.py`), the covariate and random-dictionary controls, the lexical
arms (`lexical_token_test.py` and its pairing variants), token-level extraction (`tokalign_extract.py`),
the Gemma read-point re-run (`gemma_hook_sensitivity.py`), the Lan et al. audits, probe attribution
(`experiments/probe_attribution_v2.py`, `code/analysis/probe_attribution_runner.py`), probe transfer,
steering and the causal tests. The reported runs used single H100-80GB / H200-141GB GPUs (a few cells
ran on an Apple-silicon laptop via MPS; the receipts record the device). `RELEASE.md` maps each
receipt family to the script and launch configuration that produced it.

The SLURM scripts in `code/scripts/slurm/` and `scripts/slurm/` are shipped as run, with
site-specific strings replaced by placeholders: `CLUSTER_ACCOUNT`, `<GPU_PARTITION>`, `<DATA_ROOT>`
(scratch root), `<REPO_ROOT>` (checkout), `<HF_CACHE>`, `<ENV_ANALYSIS>` / `<ENV_TRAIN>` (an analysis
environment with `sae-lens`, and the environment the extraction jobs used), `<HOME>`, `USER`. In
executable lines they appear as environment variables (`${REPO_ROOT}`, ...); in `#SBATCH` directives
they must be edited by hand. Receipt `host` fields read `<node>` or `local-machine`. `setup.sh` pins
PyTorch 2.6.0 (CUDA 12.4) and `circuit-tracer` at commit `4bb8c0e`.

## Directory map

| path | contents |
|---|---|
| `results/` | every receipt the paper reads from (JSON), with run records (`RUN.md`) and job logs for the nine-cell runs |
| `code/analysis/` | extraction, matching, controls, readers, table emitters, `check_*.py` gates, `qwen_sae_adapter.py` |
| `code/figures/`, `code/fig1_rc.py`, `code/figstyle/`, `paper/figures_src/` | figure scripts (every data mark is read from a receipt) and the shared figure style |
| `code/scripts/slurm/`, `scripts/slurm/`, `scripts/*.slurm` | SLURM launch configurations as run |
| `experiments/` | v0/v1 probe-attribution and attribution-graph experiments (`probe_attribution_v2.py`, `utils.py` adapter over `circuit-tracer`) |
| `PREREGISTRATION_*.md`, `PLAN_token_aligned_sharing.md`, `notes/PREREG_*.md`, `prespecification/` | registrations and lab-notebook excerpts; `prespecification/REGISTRATIONS.md` is the index |
| `paper/` | the arXiv version's LaTeX (`main.tex`, `sections/*.tex`, read by the gates), figure PDFs and the compiled `main.pdf` |
| `data/` | the multilingual prompt sets |
| `tools/` | `verify_package.py`, `reproduce_main_table.py`, `run_offline_checks.sh` |
| `MANIFEST.sha256` | SHA-256 of every file |
| `RELEASE.md` | receipt family -> producing script -> launch config -> reader -> paper location; known limits |

Path and host scrubbing: absolute paths, the cluster's name, account, partitions and node names are
replaced by the placeholders above, and pointers to internal notes read `<internal-notes>`. The full
lab notebook is not released; `prespecification/notebook_excerpts.md` holds the entries that fix a
design or rule before its result. The commit hashes cited in the registrations refer to the private
development history.

## License

Code, receipts and documentation in this repository: MIT (see `LICENSE`). Third-party material keeps
its own license: the C4 document excerpts quoted in the feature-interpretation receipts come from
`allenai/c4` (ODC-BY 1.0; Common Crawl terms of use apply), and `paper/*.sty` are the ICLR 2027
style files and the `fancyhdr`/`natbib` packages under their own terms. Models and SAE suites are
not redistributed.

## Contact

Xupeng Chen, xc1490@nyu.edu
