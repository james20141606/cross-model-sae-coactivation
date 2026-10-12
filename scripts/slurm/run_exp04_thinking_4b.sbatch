#!/bin/bash
# CT exp04 thinking-mode on Qwen3-4B.
# Hypothesis: 2-hop reasoning failed in exp02 because chain happens latently;
# thinking mode forces explicit intermediate tokens → attribution graph
# should resolve the chain at each step.
#
#SBATCH --job-name=ct-exp04-thinking-4b
#SBATCH --output=<REPO_ROOT>/logs/exp04/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp04/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
set -euo pipefail
ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
mkdir -p $ROOT_DIR/logs/exp04
export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
cd $ROOT_DIR
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}
"$ENV_PYTHON" experiments/04_thinking_mode/run.py --model qwen3-4b --device cuda:0
echo "[DONE] $(date -Is)"
