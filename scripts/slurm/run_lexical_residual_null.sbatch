#!/bin/bash
#SBATCH --job-name=ct-lexres
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT_B
#SBATCH --partition=<GPU_PARTITION>
# NOTE: rtx6000_courant / rtx6000_economics are rejected by sbatch for account
# CLUSTER_ACCOUNT_B ("partition is not valid for this job") -- they are
# group-restricted. <GPU_PARTITION>,<GPU_PARTITION> is the pair proven by run_c4_covariate_null.
# C4 cross-model SAE-sharing: LEXICAL-RESIDUALIZED null (cross-fitted TF-IDF-SVD residuals).
# Reuses <ENV_TRAIN> (sae_lens + sklearn), same extraction path as run_c4_covariate_null.sbatch.
# Smoke: MAX_PROMPTS=96 TOPK=256 N_PERM=100 SVD_K=32 TAG=smoke_lexres
# FAILURE MUST BE VISIBLE: `set -e` + no `||` swallowing, so a crashed python exits
# non-zero and SLURM reports FAILED. The inherited run_c4_covariate_null.sbatch
# pattern (`set -uo pipefail` + `|| echo FAILED` + unconditional `echo DONE`) makes
# a crashed job exit 0 and report COMPLETED -- i.e. job state is NOT a receipt.
set -euo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
OUT=results/cross_model_sae
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/$OUT/cache"
cd "$ROOT"
MODEL_A="${MODEL_A:-google/gemma-2-2b}"; SAE_A_RELEASE="${SAE_A_RELEASE:-gemma-scope-2b-pt-res-canonical}"; SAE_A_ID="${SAE_A_ID:-layer_12/width_16k/canonical}"
MODEL_B="${MODEL_B:-mistralai/Mistral-7B-v0.1}"; SAE_B_RELEASE="${SAE_B_RELEASE:-mistral-7b-res-wg}"; SAE_B_ID="${SAE_B_ID:-blocks.16.hook_resid_pre}"
PROMPTS="${PROMPTS:-data/c4_slice.jsonl}"; PROMPT_LANG="${PROMPT_LANG:-en}"
TOPK="${TOPK:-1500}"; N_PERM="${N_PERM:-200}"; MAX_PROMPTS="${MAX_PROMPTS:-}"
SVD_K="${SVD_K:-64,128,256}"; N_FOLDS="${N_FOLDS:-5}"
RIDGE_ALPHAS="${RIDGE_ALPHAS:-}"; RIDGE_MAX_FEATURES="${RIDGE_MAX_FEATURES:-20000}"
TAG="${TAG:-lexres_c4_gemma2-2b__mistral7b}"
CACHE="${CACHE:-$OUT/cache/pooled_${TAG}.npz}"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
echo "[INFO] A=$MODEL_A B=$MODEL_B topk=$TOPK n_perm=$N_PERM svd_k=$SVD_K folds=$N_FOLDS max_prompts=${MAX_PROMPTS:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
MP_ARG=""; [ -n "$MAX_PROMPTS" ] && MP_ARG="--max_prompts $MAX_PROMPTS" || true
RIDGE_ARG=""; [ -n "$RIDGE_ALPHAS" ] && RIDGE_ARG="--ridge_alphas $RIDGE_ALPHAS --ridge_max_features $RIDGE_MAX_FEATURES" || true
# record the exact code the job ran with, so the receipt is attributable
echo "[CODE] sha256(lexical_residual_null.py)=$(sha256sum code/analysis/lexical_residual_null.py | cut -d' ' -f1)"
echo "[CODE] sha256(this sbatch)=$(sha256sum scripts/slurm/run_lexical_residual_null.sbatch | cut -d' ' -f1)"
echo "[CODE] git=$(git rev-parse --short HEAD 2>/dev/null || echo NA) dirty=$(git status --porcelain 2>/dev/null | wc -l)"
# selftest gates the real run: if the method cannot discriminate, do not report numbers
"$ENV_PY" -u code/analysis/lexical_residual_null.py --selftest
"$ENV_PY" -u code/analysis/lexical_residual_null.py \
  --model_a "$MODEL_A" --sae_a_release "$SAE_A_RELEASE" --sae_a_id "$SAE_A_ID" \
  --model_b "$MODEL_B" --sae_b_release "$SAE_B_RELEASE" --sae_b_id "$SAE_B_ID" \
  --prompts "$PROMPTS" --lang "$PROMPT_LANG" --topk "$TOPK" --n_perm "$N_PERM" \
  --svd_k "$SVD_K" --n_folds "$N_FOLDS" --cache "$CACHE" $MP_ARG $RIDGE_ARG \
  --output "$OUT/${TAG}.json"
test -s "$OUT/${TAG}.json"
echo "[OK] wrote $OUT/${TAG}.json"
echo DONE
