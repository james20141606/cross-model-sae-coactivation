#!/bin/bash
# CT exp02 antonyms on Qwen3-1.7B (intermediate scale: 0.6B 3/8 ✓, 4B OOM).
# 1.7B + mwhanna-qwen3-1.7b-transcoders may resolve 2-3 of the 5 failures or
# fail uniformly — confirms whether antonym convergence is structural at this
# attribution threshold.
#
#SBATCH --job-name=ct-exp02-antonyms-1p7b
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
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
    --model qwen3-1.7b --experiment antonyms
echo "[DONE] $(date -Is)"
