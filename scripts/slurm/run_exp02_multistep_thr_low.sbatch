#!/bin/bash
# Circuit Tracing — exp02 multistep on Qwen3-0.6B at LOW threshold (0.001).
# All previous multistep runs at threshold=0.01 gave 0/4 across 0.6B and 4B.
# This tests: is 2-hop deficit budget-bound (more iterations needed → lower
# threshold helps) or structural (model just doesn't form attribution-graph
# circuits for 2-hop)?
#
#SBATCH --job-name=ct-exp02-multistep-thr001
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}"

"$ENV_PYTHON" experiments/02_reproduce_anthropic/run.py \
    --model qwen3-0.6b --experiment multistep --threshold 0.001

echo
echo "[DONE] exp02 multistep thr=0.001 at $(date -Is)"
