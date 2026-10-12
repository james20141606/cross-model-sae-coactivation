#!/bin/bash
# Pre-download Qwen3-0.6B + mwhanna-qwen3-0.6b-transcoders so exp01 hello-world
# starts cold-cache-fast next time. Uses <ENV_TRAIN> env (it has transformer-lens
# 3.1.0 + sae-lens 6.43.0 + circuit_tracer + numpy 1.26.4).
#
# CPU job, ~3-5 min wall (network-bound).
#
#SBATCH --job-name=ct-predownload
#SBATCH --output=<REPO_ROOT>/logs/predownload/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/predownload/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>

set -euo pipefail

ROOT_DIR=${REPO_ROOT}
ENV_PYTHON=${ENV_TRAIN}/bin/python
LOG_DIR=$ROOT_DIR/logs/predownload
mkdir -p "$LOG_DIR"

export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p $HF_HOME

echo "[INFO] host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID"
echo "[INFO] HF_HOME=$HF_HOME"

cd "$ROOT_DIR"
"$ENV_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

# 1) Qwen3-0.6B (base + tokenizer)
qwen_path = snapshot_download("Qwen/Qwen3-0.6B")
print(f"[OK] Qwen3-0.6B → {qwen_path}")
print(f"     dir size = {sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(qwen_path) for f in fs) / 1e9:.2f} GB")

# 2) mwhanna's qwen3-0.6b transcoders (HF repo for layer_X.safetensors)
tx_path = snapshot_download("mwhanna/qwen3-0.6b-transcoders-lowl0")
print(f"[OK] qwen3-0.6b-transcoders-lowl0 → {tx_path}")
print(f"     dir size = {sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(tx_path) for f in fs) / 1e9:.2f} GB")
PY

echo
echo "=== HF_HOME usage ==="
du -sh $HF_HOME 2>&1 | head

echo
echo "[DONE] predownload finished at $(date -Is)"
