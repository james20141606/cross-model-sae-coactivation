#!/bin/bash
#SBATCH --job-name=ct-c4-covctl
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT_B
#SBATCH --partition=<GPU_PARTITION>
# C4 cross-model SAE-sharing covariate-controlled null + token-scramble.
# Reuses <ENV_TRAIN> (sae_lens + sklearn). Smoke: MAX_PROMPTS=96 TOPK=256 N_PERM=100.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
OUT=results/cross_model_sae
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/$OUT"
cd "$ROOT"
MODEL_A="${MODEL_A:-google/gemma-2-2b}"; SAE_A_RELEASE="${SAE_A_RELEASE:-gemma-scope-2b-pt-res-canonical}"; SAE_A_ID="${SAE_A_ID:-layer_12/width_16k/canonical}"
MODEL_B="${MODEL_B:-mistralai/Mistral-7B-v0.1}"; SAE_B_RELEASE="${SAE_B_RELEASE:-mistral-7b-res-wg}"; SAE_B_ID="${SAE_B_ID:-blocks.16.hook_resid_pre}"
PROMPTS="${PROMPTS:-data/c4_slice.jsonl}"; PROMPT_LANG="${PROMPT_LANG:-en}"
TOPK="${TOPK:-1500}"; N_PERM="${N_PERM:-200}"; MAX_PROMPTS="${MAX_PROMPTS:-}"
TAG="${TAG:-covctl_c4_gemma2-2b__mistral7b}"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
echo "[INFO] A=$MODEL_A B=$MODEL_B topk=$TOPK n_perm=$N_PERM max_prompts=${MAX_PROMPTS:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
MP_ARG=""; [ -n "$MAX_PROMPTS" ] && MP_ARG="--max_prompts $MAX_PROMPTS"
"$ENV_PY" -u code/analysis/c4_covariate_null.py \
  --model_a "$MODEL_A" --sae_a_release "$SAE_A_RELEASE" --sae_a_id "$SAE_A_ID" \
  --model_b "$MODEL_B" --sae_b_release "$SAE_B_RELEASE" --sae_b_id "$SAE_B_ID" \
  --prompts "$PROMPTS" --lang "$PROMPT_LANG" --topk "$TOPK" --n_perm "$N_PERM" $MP_ARG \
  --output "$OUT/${TAG}.json" || echo "FAILED $TAG"
[ -s "$OUT/${TAG}.json" ] && echo "[OK] wrote $OUT/${TAG}.json" || echo "[EMPTY] $OUT/${TAG}.json"
echo DONE
