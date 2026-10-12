#!/bin/bash
# CT exp02 antonyms at threshold=0.001 (10× tighter than default 0.01).
# Earlier antonyms@0.01 → 3/8 graphs converged. Test if lower threshold
# unlocks more circuits or if 5/8 fails are structural like multistep was.
#
#SBATCH --job-name=ct-exp02-antonyms-thr001
#SBATCH --output=<REPO_ROOT>/logs/exp02/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/exp02/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/circuit-tracer:${PYTHONPATH:-}"
"$ENV_PYTHON" experiments/02_reproduce_anthropic/run.py \
    --model qwen3-0.6b --experiment antonyms --threshold 0.001
echo "[DONE] $(date -Is)"
