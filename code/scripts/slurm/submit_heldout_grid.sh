#!/bin/bash
# P2-B full grid: held-out C4 replication.
# (B1) 8 C4-robust cells + pythia null control, matching refit on data/c4_heldout.jsonl
# (B2) held-out top-doc audit for g9m (fit on c4_fit, audit on c4_heldout)
# (B3) probe transfer g9m with matching fitted on the held-out half
# Run from <REPO_ROOT> on the cluster.
set -uo pipefail
cd ${REPO_ROOT}

sub_cell () {  # sub_cell TAG  A RA IA  B RB IB
  local tag=$1 a=$2 ra=$3 ia=$4 b=$5 rb=$6 ib=$7
  MODEL_A=$a SAE_A_RELEASE=$ra SAE_A_ID=$ia \
  MODEL_B=$b SAE_B_RELEASE=$rb SAE_B_ID=$ib \
  PROMPTS=data/c4_heldout.jsonl PROMPT_LANG=en TOPK=1500 N_PERM=200 TAG=$tag \
  sbatch --job-name="ct-$tag" scripts/slurm/run_cross_model_sae.sbatch
}

GA="google/gemma-2-2b gemma-scope-2b-pt-res-canonical layer_12/width_16k/canonical"
GB="google/gemma-2-9b gemma-scope-9b-pt-res-canonical layer_20/width_16k/canonical"
MI="mistralai/Mistral-7B-v0.1 mistral-7b-res-wg blocks.16.hook_resid_pre"
Q1="Qwen/Qwen3-1.7B-Base qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 layer14"
Q8="Qwen/Qwen3-8B-Base qwenpt:Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 layer18"
PY="EleutherAI/pythia-70m-deduped pythia-70m-deduped-res-sm blocks.3.hook_resid_post"

# 8 robust cells (cross-family)
sub_cell c4h_gemma2-2b__mistral7b   $GA $MI
sub_cell c4h_gemma2-9b__mistral7b   $GB $MI
sub_cell c4h_qwen3-1.7b__mistral7b  $Q1 $MI
sub_cell c4h_qwen3-8b__mistral7b    $Q8 $MI
sub_cell c4h_qwen3-1.7b__gemma2-2b  $Q1 $GA
sub_cell c4h_qwen3-1.7b__gemma2-9b  $Q1 $GB
sub_cell c4h_qwen3-8b__gemma2-2b    $Q8 $GA
sub_cell c4h_qwen3-8b__gemma2-9b    $Q8 $GB
# null control (must stay at null on held-out too)
sub_cell c4h_pythia70m__mistral7b   $PY $MI
# within-family anchors (cheap; replicate the two anchor cells as well)
sub_cell c4h_gemma2-2b__gemma2-9b   $GA $GB
sub_cell c4h_qwen3-8b__qwen3-1.7b   $Q8 $Q1

# (B2) held-out top-doc audit, headline pair
TAG=interp_heldout_gemma2-9b__mistral7b sbatch --job-name=ct-interp-heldout scripts/slurm/run_interp_heldout.sbatch

# (B3) probe transfer with held-out matching corpus
PAIR=g9m MATCH_PROMPTS=../../data/c4_heldout.jsonl OUT_SUFFIX=_heldout \
  sbatch --job-name=ct-probe-xferC4h scripts/slurm/run_probe_transfer_c4match.sbatch
