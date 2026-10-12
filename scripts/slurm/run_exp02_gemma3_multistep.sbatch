#!/bin/bash
# CT exp02 multistep on Gemma3-4B (gemma-scope-2 transcoders).
# Test if 2-hop deficit is Qwen3-family-specific or universal across architectures.
#
#SBATCH --job-name=ct-exp02-gemma3-4b-multistep
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128G --gres=gpu:1 --time=01:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>

set -euo pipefail
ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
mkdir -p $ROOT_DIR/logs/exp02
export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(${ENV_ANALYSIS}/bin/python -c "from huggingface_hub import get_token; print(get_token() or '')" 2>/dev/null)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
cd $ROOT_DIR
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}
"$ENV_PYTHON" experiments/02_reproduce_anthropic/run.py \
    --model gemma3-4b --experiment multistep
echo "[DONE] $(date -Is)"
