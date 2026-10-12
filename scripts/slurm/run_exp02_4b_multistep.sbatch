#!/bin/bash
# Circuit Tracing — exp02 multistep on Qwen3-4B (vs the 0.6B run that gave 0/4).
# Hypothesis: 4B's larger capacity (36 layers vs 28) resolves 2-hop reasoning
# circuits at the attribution-graph threshold (0.01), giving us at least 1-2
# graphs from the 4 multistep prompts.
#
# 4B is ~4× the model + ~4× the transcoder; we bump mem to 128G and time to 1h.
#
#SBATCH --job-name=ct-exp02-4b-multistep
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>

set -euo pipefail

ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
LOG_DIR=$ROOT_DIR/logs/exp02
mkdir -p "$LOG_DIR"

export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub
# attribution gradient computation needs ~40GB chunks; expandable_segments
# avoids fragmentation that pushed the H100 80GB run OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}"

"$ENV_PYTHON" experiments/02_reproduce_anthropic/run.py \
    --model qwen3-4b --experiment multistep

echo
echo "[DONE] exp02 4b multistep at $(date -Is)"
