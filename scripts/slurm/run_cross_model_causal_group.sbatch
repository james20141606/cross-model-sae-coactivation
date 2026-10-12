#!/bin/bash
#SBATCH --job-name=ct-causal-group
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:20:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
OUT=results/cross_model_causal_group
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/$OUT"; cd "$ROOT"
MODEL_A="${MODEL_A:?}"; SAE_A_RELEASE="${SAE_A_RELEASE:?}"; SAE_A_ID="${SAE_A_ID:?}"
MODEL_B="${MODEL_B:?}"; SAE_B_RELEASE="${SAE_B_RELEASE:?}"; SAE_B_ID="${SAE_B_ID:?}"
PROMPTS="${PROMPTS:-data/prompts_v3_multi.jsonl}"; PROMPT_LANG="${PROMPT_LANG:-en}"
MAX_PROMPTS="${MAX_PROMPTS:-120}"; TOPK="${TOPK:-1500}"; KGROUP="${KGROUP:-30}"; NPERM="${NPERM:-200}"; TAG="${TAG:?}"
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} A=$MODEL_A B=$MODEL_B max_prompts=$MAX_PROMPTS k_group=$KGROUP n_perm=$NPERM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$ENV_PY" -u code/analysis/cross_model_causal_group.py \
  --model_a "$MODEL_A" --sae_a_release "$SAE_A_RELEASE" --sae_a_id "$SAE_A_ID" \
  --model_b "$MODEL_B" --sae_b_release "$SAE_B_RELEASE" --sae_b_id "$SAE_B_ID" \
  --prompts "$PROMPTS" --lang "$PROMPT_LANG" --max_prompts "$MAX_PROMPTS" \
  --topk "$TOPK" --k_group "$KGROUP" --n_perm "$NPERM" --output "$OUT/${TAG}.json" || echo "FAILED $TAG"
[ -s "$OUT/${TAG}.json" ] && echo "[OK] $OUT/${TAG}.json" || echo "[EMPTY] $OUT/${TAG}.json"
echo DONE
