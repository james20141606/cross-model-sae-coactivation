#!/bin/bash
#SBATCH --job-name=ct-toka
#SBATCH --output=<REPO_ROOT>/logs/tokalign/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/tokalign/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=160G --gres=gpu:1 --time=06:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --array=0-2
# Token-aligned sharing analysis (G0-G2 gates + S1/S2 + nulls), one registered cell
# per array task. Registered at PREREGISTRATION_token_aligned.md (df330c6).
# The GPU only accelerates the permutation loops; it changes no number's definition.
# SMOKE=1 -> reads cache/tokalign_smoke dumps, n_perm=200, receipt auto-prefixed smoke_.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python   # analysis env per registration
mkdir -p "$ROOT/logs/tokalign" "$ROOT/results/cross_model_sae"
cd "$ROOT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# the offline planted-data self-test must pass on this host before any real receipt
"$ENV_PY" -u code/analysis/tokalign_analysis.py --selftest || { echo "[FAIL] selftest" >&2; exit 1; }

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TAG=qwen3-1.7b__gemma2-2b;  DA=qwen3-1.7b_l14; DB=gemma2-2b_l12 ;;
  1) TAG=gemma2-9b__mistral7b;   DA=gemma2-9b_l20;  DB=mistral7b_b16 ;;
  2) TAG=llama3.2-1b__mistral7b; DA=llama3.2-1b_b8; DB=mistral7b_b16 ;;
  *) echo "unknown task"; exit 2 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  CACHE_DIR="$ROOT/cache/tokalign_smoke"; EXTRA="--smoke --n_perm 200 --n_perm_sens 100"
  OUT="$ROOT/results/cross_model_sae/smoke_tokalign20k_${TAG}.json"
else
  CACHE_DIR="$ROOT/cache/tokalign"; EXTRA="--n_perm 2000 --n_perm_sens 500"
  OUT="$ROOT/results/cross_model_sae/tokalign20k_${TAG}.json"
fi

"$ENV_PY" -u code/analysis/tokalign_analysis.py \
  --cell "$TAG" \
  --dir_a "$CACHE_DIR/$DA" --dir_b "$CACHE_DIR/$DB" \
  --pairs_receipt "$ROOT/results/cross_model_sae/lextok20k_${TAG}.json" \
  --cache_npz "$ROOT/cache/lextok20k_${TAG}.npz" \
  --corpus data/c4_slice_20k.jsonl \
  --seed 0 $EXTRA \
  --output "$OUT"
RC=$?
# rc=2: G2 NOT-TESTABLE (receipt kept, registered gate outcome). rc=3: G0/G1 gate
# failure (receipt kept, nothing downstream interpretable). Anything else nonzero, or
# a missing/empty receipt, is a hard failure -- do NOT let sacct show COMPLETED for a
# job that wrote nothing (the 2026-07-27 lesson).
if [ ! -s "$OUT" ]; then echo "[FAIL] rc=$RC and no receipt at $OUT" >&2; exit 1; fi
if [ $RC -eq 2 ]; then echo "[GATE] G2 NOT-TESTABLE -- receipt kept" >&2; exit 2; fi
if [ $RC -eq 3 ]; then echo "[GATE] G0/G1 FAILED -- receipt kept" >&2; exit 3; fi
if [ $RC -ne 0 ]; then echo "[FAIL] rc=$RC" >&2; exit 1; fi
echo "[DONE] $(date -Is) -> $OUT"
