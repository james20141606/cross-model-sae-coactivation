#!/bin/bash
# Circuit Tracing Exp 02 — multistep reasoning prompts on Qwen3-0.6B.
# Following exp02-antonyms (3/8 OK) and exp03 (2/14 OK on en multistep), now
# focus exclusively on the multistep set (4 prompts), which has the highest
# convergence rate so far. Likely yield: 3-4/4.
#
#SBATCH --job-name=ct-exp02-multistep
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>

set -euo pipefail

ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
LOG_DIR=$ROOT_DIR/logs/exp02
mkdir -p "$LOG_DIR"

export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub

echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}"

"$ENV_PYTHON" experiments/02_reproduce_anthropic/run.py \
    --model qwen3-0.6b --experiment multistep

echo
echo "[DONE] exp02 multistep at $(date -Is)"
