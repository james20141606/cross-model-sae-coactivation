#!/bin/bash
# Circuit Tracing Exp 03 — cross-lingual feature sharing on Qwen3-0.6B.
# Probes the same antonym/multistep semantics across en/zh/es/ja/ko prompts.
# Goal: feature-overlap heatmap → does Qwen share circuits across languages or
# learn separate per-language circuits?
#
#SBATCH --job-name=ct-exp03-crosslang
#SBATCH --output=<REPO_ROOT>/logs/exp03/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp03/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>

set -euo pipefail

ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
LOG_DIR=$ROOT_DIR/logs/exp03
mkdir -p "$LOG_DIR"

export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub

echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}"

"$ENV_PYTHON" experiments/03_cross_lingual/run.py \
    --model qwen3-0.6b --device cuda:0 \
    --threshold 0.01

echo
echo "[DONE] exp03 cross-lingual at $(date -Is)"
