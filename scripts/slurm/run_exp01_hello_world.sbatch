#!/bin/bash
# Circuit Tracing Exp 01 — hello-world attribution graph on Qwen3-0.6B.
# Adapted from scripts/01_hello_world.slurm (which used a generic `gpu` partition
# and a separate `circuit_tracing` conda env). the cluster uses partitions+account,
# and we reuse `<ENV_TRAIN>` env which already has transformer-lens 3.1.0 +
# sae-lens 6.43 + circuit_tracer + numpy 1.26.4.
#
#SBATCH --job-name=ct-hello
#SBATCH --output=<REPO_ROOT>/logs/exp01/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp01/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>

set -euo pipefail

ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
LOG_DIR=$ROOT_DIR/logs/exp01
mkdir -p "$LOG_DIR"

export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub

echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}"

# config.py now defaults to mwhanna/qwen3-0.6b-transcoders-lowl0 directly
# (the bare -transcoders repo lacks config.yaml; lowl0 has it). Run normally.
"$ENV_PYTHON" experiments/01_hello_world/run.py \
    --model qwen3-0.6b --device cuda:0 \
    --prompt "The opposite of small is"

echo
echo "[DONE] exp01 hello-world at $(date -Is)"
