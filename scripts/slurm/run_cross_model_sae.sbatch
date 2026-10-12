#!/bin/bash
#SBATCH --job-name=ct-xmodel-sae
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:20:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#
# Cross-model SAE feature matching with permutation null.
# DISCIPLINE: smoke this on a 2h debug-gpu placeholder FIRST (set MAX_PROMPTS=20,
# N_PERM=20) to confirm the sae-lens release/id strings load and Phi is non-empty,
# THEN batch the full run. The <ENV_TRAIN> env has sae_lens 6.43.0 (<ENV_ANALYSIS> does NOT).
#
# Default pair = within-family Gemma validation (Lan et al.'s validated same-family
# setup): if greedy_excess>0 with small p here, the pipeline is sound and we proceed
# to the cross-family arms (Gemma<->Llama via gemma-scope/llama_scope; Qwen SAEs need
# manual weight loading -- not a registered sae-lens release).
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python      # has sae_lens
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
OUT=results/cross_model_sae
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/$OUT"
cd "$ROOT"

# --- configurable pair (defaults = within-family Gemma-2 validation) ---
MODEL_A="${MODEL_A:-google/gemma-2-2b}"
SAE_A_RELEASE="${SAE_A_RELEASE:-gemma-scope-2b-pt-res-canonical}"
SAE_A_ID="${SAE_A_ID:-layer_12/width_16k/canonical}"
MODEL_B="${MODEL_B:-google/gemma-2-9b}"
SAE_B_RELEASE="${SAE_B_RELEASE:-gemma-scope-9b-pt-res-canonical}"
SAE_B_ID="${SAE_B_ID:-layer_20/width_16k/canonical}"
PROMPTS="${PROMPTS:-data/prompts_v3_multi.jsonl}"
PROMPT_LANG="${PROMPT_LANG:-en}"      # NB: NOT $LANG -- collides with the shell locale (C.UTF-8)
LANG_A="${LANG_A:-}"                  # optional: per-side language (cross-lingual cells)
LANG_B="${LANG_B:-}"
TOPK="${TOPK:-1500}"
N_PERM="${N_PERM:-200}"
MAX_PROMPTS="${MAX_PROMPTS:-}"     # set to e.g. 20 for smoke
TAG="${TAG:-$(echo "${MODEL_A}__${MODEL_B}" | tr /: _)}"

echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
echo "[INFO] A=$MODEL_A ($SAE_A_RELEASE :: $SAE_A_ID)"
echo "[INFO] B=$MODEL_B ($SAE_B_RELEASE :: $SAE_B_ID)  lang=$PROMPT_LANG lang_a=${LANG_A:--} lang_b=${LANG_B:--} topk=$TOPK n_perm=$N_PERM max_prompts=${MAX_PROMPTS:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

MP_ARG=""; [ -n "$MAX_PROMPTS" ] && MP_ARG="--max_prompts $MAX_PROMPTS"
LA_ARG=""; [ -n "$LANG_A" ] && LA_ARG="--lang_a $LANG_A"
LB_ARG=""; [ -n "$LANG_B" ] && LB_ARG="--lang_b $LANG_B"
"$ENV_PY" -u code/analysis/cross_model_sae_match.py \
  --model_a "$MODEL_A" --sae_a_release "$SAE_A_RELEASE" --sae_a_id "$SAE_A_ID" \
  --model_b "$MODEL_B" --sae_b_release "$SAE_B_RELEASE" --sae_b_id "$SAE_B_ID" \
  --prompts "$PROMPTS" --lang "$PROMPT_LANG" $LA_ARG $LB_ARG --topk "$TOPK" --n_perm "$N_PERM" $MP_ARG \
  --output "$OUT/${TAG}.json" || echo "FAILED $TAG"

if [ -s "$OUT/${TAG}.json" ]; then echo "[OK] wrote $OUT/${TAG}.json ($(wc -c < "$OUT/${TAG}.json") bytes)";
else echo "[EMPTY] $OUT/${TAG}.json"; fi
echo DONE
