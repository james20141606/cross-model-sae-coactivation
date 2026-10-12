#!/bin/bash
#SBATCH --job-name=ct-probe-controls
#SBATCH --output=<REPO_ROOT>/logs/probe_controls/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/probe_controls/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
mkdir -p "$ROOT/logs/probe_controls" "$ROOT/results/probe_attribution_controls"
cd "$ROOT"

# MODELS overridable; default = the construct-validity contrast set
MODELS="${MODELS:-EleutherAI/pythia-410m-deduped}"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} models=$MODELS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
for M in $MODELS; do
  TAG=$(echo "$M" | tr / _)
  echo "==== [RUN] $M ===="
  "$ENV_PY" -u code/analysis/probe_attribution_runner.py \
    --model "$M" --prompts data/prompts_expanded_v2.jsonl \
    --output "results/probe_attribution_controls/${TAG}.json" || echo "FAILED $M"
done
echo DONE
