#!/bin/bash
#SBATCH --job-name=ct-covctl-fine
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1 --time=03:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --array=0-5
# NOTE on partitions: l40s (44 GiB usable) is NOT enough for the gemma2-9b <-> mistral7b
# cell. c4_covariate_null.py extracts real activations from BOTH models and then re-extracts
# from both on scrambled text; the 9B+7B pair OOMs on the second pass. 80 GiB+ only.
# internal re-review, single most decision-relevant weakness: the surviving cross-model sharing
# is still compatible with shared LEXICAL / unigram detectors. The published covariate
# control (covctl_c4_gemma2-2b__mistral7b.json) used a COARSE k=10 TF-IDF partition on ONE
# cell -- too coarse to exclude bag-of-words sharing, and n=1 cell.
#
# This runs the FINE-grained version on three representative C4-robust cells spanning the
# effect range (strong / median / weak -- the same three cells as the random-dictionary
# control), at k=200 and k=500 TF-IDF/KMeans clusters, n_perm=2000.
#
# Read the EXCESS (obs - null_mean), not z: as clusters get finer the within-stratum
# permutation has less freedom, so the null mean rises AND the null SD shrinks. z can
# inflate for a purely mechanical reason; the excess cannot.
#
# SMOKE=1 -> tiny de-risk run (200 docs, topk 256, n_perm 100, k=20) to smoke_*.json.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python   # same env as the published covctl receipt
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

T=${SLURM_ARRAY_TASK_ID:-0}
CELL=$((T % 3)); KSEL=$((T / 3))
case "$CELL" in
  0) # strongest surviving cross-family cell (C4 z=42.7, excess +29.3%)
     A=Qwen/Qwen3-1.7B-Base; RA=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; IA=layer14
     B=google/gemma-2-2b; RB=gemma-scope-2b-pt-res-canonical; IB=layer_12/width_16k/canonical
     TAG=qwen3-1.7b__gemma2-2b ;;
  1) # median cell (C4 z=24.0, excess +17.7%)
     A=google/gemma-2-9b; RA=gemma-scope-9b-pt-res-canonical; IA=layer_20/width_16k/canonical
     B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre
     TAG=gemma2-9b__mistral7b ;;
  2) # weakest surviving cell (C4 z=12.3, excess +7.7%)
     A=meta-llama/Llama-3.2-1B; RA=chanind/sae-llama-3.2-1b-res; IA=blocks.8.hook_resid_post
     B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre
     TAG=llama3.2-1b__mistral7b ;;
esac
case "$KSEL" in
  0) NBOW=200 ;;
  1) NBOW=500 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--max_prompts 200 --topk 256 --n_perm 100 --n_bow_clusters 20"
  OUT="$ROOT/results/cross_model_sae/smoke_covctlfine_${TAG}.json"
else
  EXTRA="--topk 1500 --n_perm 2000 --n_bow_clusters $NBOW"
  OUT="$ROOT/results/cross_model_sae/covctl_k${NBOW}_c4_${TAG}.json"
fi

"$ENV_PY" -u code/analysis/c4_covariate_null.py \
  --model_a "$A" --sae_a_release "$RA" --sae_a_id "$IA" \
  --model_b "$B" --sae_b_release "$RB" --sae_b_id "$IB" \
  --prompts data/c4_slice.jsonl --lang en $EXTRA \
  --output "$OUT"
RC=$?
# Fail LOUDLY. Without this the script exits 0 on a crashed python (set -u -o pipefail has
# no -e), the final echo lies, and sacct reports COMPLETED for a run that wrote nothing --
# which is exactly what happened to the two gemma2-9b tasks of JID 14852578 (CUDA OOM on
# the scramble re-extraction, reported COMPLETED, produced no receipt).
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[FAIL] rc=$RC output=$OUT exists=$([ -s "$OUT" ] && echo yes || echo no)" >&2
  exit 1
fi
echo "[DONE] $(date -Is) -> $OUT"
