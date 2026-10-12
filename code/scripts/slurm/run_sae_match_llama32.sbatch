#!/bin/bash
#SBATCH --job-name=ct-match-llama32
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# R17 Deferred M1: intermediate-scale MODERN checkpoint in the SAE arm to break the
# capability-vs-recipe confound. Llama-3.2-1B (modern recipe, 1B, NEW family) <-> the fixed
# Mistral-7B reference on the 2000-doc C4 natural-text slice, identical config to the
# c4_*__mistral7b.json headline roster (topk=1500, n_perm=200). SAE: chanind/sae-llama-3.2-1b-res
# (sae-lens-native residual SAE, layer 8/16 ~= 0.5 depth). Loads via the existing from_pretrained
# path -- no matching-code change.
#
# SMOKE=1 -> tiny de-risk run (60 docs, n_perm=20) written to smoke_*.json.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token 2>/dev/null || true)
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT/code/analysis"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} SMOKE=${SMOKE:-0}"

if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--max_prompts 60 --n_perm 20"
  OUT="$ROOT/results/cross_model_sae/smoke_c4_llama3.2-1b__mistral7b.json"
else
  EXTRA="--n_perm 200"
  OUT="$ROOT/results/cross_model_sae/c4_llama3.2-1b__mistral7b.json"
fi

"$ENV_PY" -u cross_model_sae_match.py \
  --model_a meta-llama/Llama-3.2-1B --sae_a_release chanind/sae-llama-3.2-1b-res --sae_a_id blocks.8.hook_resid_post \
  --model_b mistralai/Mistral-7B-v0.1 --sae_b_release mistral-7b-res-wg --sae_b_id blocks.16.hook_resid_pre \
  --prompts ../../data/c4_slice.jsonl --lang en --topk 1500 $EXTRA \
  --output "$OUT"
echo "[DONE] $(date -Is) -> $OUT"
