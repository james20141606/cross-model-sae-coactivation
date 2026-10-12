#!/bin/bash
#SBATCH --job-name=ct-probe-n200
#SBATCH --output=<REPO_ROOT>/logs/probe_controls/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/probe_controls/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:20:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
OUT=results/probe_attribution_controls_n200
mkdir -p "$ROOT/logs/probe_controls" "$ROOT/$OUT"
cd "$ROOT"

MODELS="${MODELS:-EleutherAI/pythia-410m-deduped}"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
echo "[INFO] models=$MODELS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
for M in $MODELS; do
  TAG=$(echo "$M" | tr / _)
  echo "==== [RUN] $M ===="
  "$ENV_PY" -u code/analysis/probe_attribution_runner.py \
    --model "$M" --prompts data/prompts_v3_multi.jsonl \
    --output "$OUT/${TAG}.json" || echo "FAILED $M"
  # sanity: non-empty output with raw matrix
  if [ -s "$OUT/${TAG}.json" ]; then
    echo "[OK] wrote $OUT/${TAG}.json ($(wc -c < "$OUT/${TAG}.json") bytes)"
  else
    echo "[EMPTY] $OUT/${TAG}.json"
  fi
done
echo DONE
