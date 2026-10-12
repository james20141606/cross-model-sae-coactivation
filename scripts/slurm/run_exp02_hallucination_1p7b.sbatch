#!/bin/bash
# CT exp02 hallucination on Qwen3-1.7B.
#
#SBATCH --job-name=ct-exp02-halluc-1p7b
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=00:45:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
set -euo pipefail
ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
mkdir -p $ROOT_DIR/logs/exp02
export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
cd $ROOT_DIR
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}
"$ENV_PYTHON" experiments/02_reproduce_anthropic/run.py \
    --model qwen3-1.7b --experiment hallucination
echo "[DONE] $(date -Is)"
