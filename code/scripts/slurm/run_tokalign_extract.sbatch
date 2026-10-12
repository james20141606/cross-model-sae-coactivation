#!/bin/bash
#SBATCH --job-name=ct-tokx
#SBATCH --output=<REPO_ROOT>/logs/tokalign/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/tokalign/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=06:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --array=0-4
# Per-token SAE extraction for the token-aligned sharing test.
# Registered at PREREGISTRATION_token_aligned.md (df330c6) BEFORE this file existed.
# One (model, SAE) side per array task; 5 sides cover the 3 registered cells.
# Shards are skip-if-exists, so a cancelled/evicted task is simply resubmitted.
# SMOKE=1 -> 500 docs into cache/tokalign_smoke/ (plumbing only, NOT a result).
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python   # receipt-identical interpreter (G0)
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
mkdir -p "$ROOT/logs/tokalign"
cd "$ROOT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

SLICE=data/c4_slice_20k.jsonl
if [ ! -s "$SLICE" ]; then echo "[FAIL] $SLICE missing" >&2; exit 1; fi
MD5=$(md5sum "$SLICE" | cut -d' ' -f1)
if [ "$MD5" != "64e994008170f6c5738bdb5dc19f285e" ]; then
  echo "[FAIL] $SLICE md5 $MD5 != registered 64e994008170f6c5738bdb5dc19f285e" >&2; exit 1
fi

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) M=Qwen/Qwen3-1.7B-Base; R=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; I=layer14
     SIDE=qwen3-1.7b_l14 ;;
  1) M=google/gemma-2-2b; R=gemma-scope-2b-pt-res-canonical; I=layer_12/width_16k/canonical
     SIDE=gemma2-2b_l12 ;;
  2) M=google/gemma-2-9b; R=gemma-scope-9b-pt-res-canonical; I=layer_20/width_16k/canonical
     SIDE=gemma2-9b_l20 ;;
  3) M=mistralai/Mistral-7B-v0.1; R=mistral-7b-res-wg; I=blocks.16.hook_resid_pre
     SIDE=mistral7b_b16 ;;
  4) M=meta-llama/Llama-3.2-1B; R=chanind/sae-llama-3.2-1b-res; I=blocks.8.hook_resid_post
     SIDE=llama3.2-1b_b8 ;;
  *) echo "unknown task"; exit 2 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  OUT_DIR="$ROOT/cache/tokalign_smoke/$SIDE"
  EXTRA="--max_prompts 500 --shard_size 250"
else
  OUT_DIR="$ROOT/cache/tokalign/$SIDE"
  EXTRA="--shard_size 1000"
fi

"$ENV_PY" -u code/analysis/tokalign_extract.py \
  --model "$M" --sae_release "$R" --sae_id "$I" \
  --prompts "$SLICE" --out_dir "$OUT_DIR" $EXTRA
RC=$?
if [ $RC -ne 0 ]; then echo "[FAIL] rc=$RC" >&2; exit 1; fi
if [ ! -s "$OUT_DIR/extract_receipt.json" ]; then
  echo "[FAIL] rc=0 but no receipt at $OUT_DIR/extract_receipt.json" >&2; exit 1
fi
echo "[DONE] $(date -Is) -> $OUT_DIR"
