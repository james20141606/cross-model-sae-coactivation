#!/bin/bash
# CT exp02 multistep on Qwen3-8B (largest mwhanna-supported scale).
# 0.6B 0/4, 4B 0/4 (with H100 80GB OOM, H200 OOM partial). 8B test on H200 141G.
#
#SBATCH --job-name=ct-exp02-multistep-8b
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
set -euo pipefail
ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
mkdir -p $ROOT_DIR/logs/exp02
export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
cd $ROOT_DIR
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}
"$ENV_PYTHON" experiments/02_reproduce_anthropic/run.py \
    --model qwen3-8b --experiment multistep
echo "[DONE] $(date -Is)"
