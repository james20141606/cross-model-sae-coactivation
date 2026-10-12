#!/bin/bash
#SBATCH --job-name=ct-match-w65k
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# R13 recipe-robustness cell: same capable pair (gemma2-2b <-> mistral7b, C4), but the
# Gemma SAE swapped 16k -> 65k released width. If mutual-NN z stays significant, the
# "different seeds/recipes would give different matches" attack is answered with data.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token 2>/dev/null || true)
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT/code/analysis"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
"$ENV_PY" -u cross_model_sae_match.py \
  --model_a google/gemma-2-2b --sae_a_release gemma-scope-2b-pt-res-canonical --sae_a_id layer_12/width_65k/canonical \
  --model_b mistralai/Mistral-7B-v0.1 --sae_b_release mistral-7b-res-wg --sae_b_id blocks.16.hook_resid_pre \
  --prompts ../../data/c4_slice.jsonl --lang en \
  --output ../../results/cross_model_sae/c4_w65k_gemma2-2b__mistral7b.json
