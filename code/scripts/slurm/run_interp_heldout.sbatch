#!/bin/bash
#SBATCH --job-name=ct-interp-heldout
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:20:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# HELD-OUT top-document audit (kill-argument unresolved-major fix): fit mutual-NN
# matches on the FIT half of the C4 slice (docs 0-999), dump top documents + per-pair
# held-out corr on the HELD-OUT half (docs 1000-1999). Headline pair = g9m.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python      # has sae_lens
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

FIT="${FIT:-data/c4_fit.jsonl}"
AUDIT="${AUDIT:-data/c4_heldout.jsonl}"
MAX_PROMPTS="${MAX_PROMPTS:-1000}"
K_PAIRS="${K_PAIRS:-20}"
TAG="${TAG:-interp_heldout_gemma2-9b__mistral7b}"

"$ENV_PY" -u code/analysis/interpret_matched_features.py \
  --model_a "${MODEL_A:-google/gemma-2-9b}" \
  --sae_a_release "${SAE_A_RELEASE:-gemma-scope-9b-pt-res-canonical}" \
  --sae_a_id "${SAE_A_ID:-layer_20/width_16k/canonical}" \
  --model_b "${MODEL_B:-mistralai/Mistral-7B-v0.1}" \
  --sae_b_release "${SAE_B_RELEASE:-mistral-7b-res-wg}" \
  --sae_b_id "${SAE_B_ID:-blocks.16.hook_resid_pre}" \
  --prompts "$FIT" --audit_prompts "$AUDIT" \
  --max_prompts "$MAX_PROMPTS" --topk 1500 --k_pairs "$K_PAIRS" --n_top_docs 5 \
  --output "results/cross_model_sae/${TAG}.json" || echo "FAILED $TAG"
echo DONE
