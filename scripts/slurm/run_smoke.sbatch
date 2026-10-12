#!/bin/bash
# Circuit Tracing M1 smoke — config import + experiment-script syntax check.
# Real runs need transcoder/SAE checkpoints + GPUs; smoke just confirms the
# Python skeleton is sound.
#
#SBATCH --job-name=circuit-smoke
#SBATCH --output=<REPO_ROOT>/logs/smoke/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/smoke/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>

set -euo pipefail

ROOT_DIR="${REPO_ROOT}"
ENV_PYTHON="${ENV_ANALYSIS}/bin/python"
LOG_DIR="${ROOT_DIR}/logs/smoke"
mkdir -p "$LOG_DIR"

echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "=== 1) config + utils import ==="
"${ENV_PYTHON}" -c "
from experiments.config import QWEN3_0_6B, QWEN3_4B, GEMMA3_4B, RESULTS_DIR
print(f'QWEN3_0_6B.name={QWEN3_0_6B.name} layer count cfg present')
print(f'GEMMA3_4B.name={GEMMA3_4B.name}')
print(f'results dir={RESULTS_DIR}')"

echo
echo "=== 2) syntax check on all 5 experiment run.py ==="
for n in 01_hello_world 02_reproduce_anthropic 03_cross_lingual 04_thinking_mode 05_cross_model; do
  if [ -f "experiments/${n}/run.py" ]; then
    printf "  %-26s " "${n}"
    "${ENV_PYTHON}" -m py_compile "experiments/${n}/run.py" && echo "compile OK"
  else
    printf "  %-26s skip (no run.py)\n" "${n}"
  fi
done

echo
echo "[DONE] circuit-tracing smoke passed at $(date -Is)"
