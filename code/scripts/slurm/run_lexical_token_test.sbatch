#!/bin/bash
#SBATCH --job-name=ct-lextok
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G --gres=gpu:1 --time=08:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --array=0-2
# The token/unigram-detector test. Design pre-registered in experiments_detail.md at
# commit 6848d4c BEFORE code/analysis/lexical_token_test.py was written; do not retune.
#
# 80 GiB+ partitions only: the 9B<->7B cell keeps both models resident (this is the same
# OOM that silently killed two covctl tasks of JID 14852578 on an l40s).
#
# The AC's spec, all four parts:
#   (a) hard first-stage adequacy gate vs a shuffled design -> runner exits non-zero
#   (b) survival of the FROZEN original mutual-NN pairs, not a rematch of totals
#   (c) n_perm=2000
#   (d) 20 independently sampled size-matched partitions (in the covariate arm)
# plus the larger corpus the internal review asked for: 20k C4 docs, vs 2k for the headline receipts.
#
# SMOKE=1 -> 2000 docs, topk 256, n_perm 100, k=64 (plumbing only, NOT a result).
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae" "$ROOT/cache"
cd "$ROOT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# the offline self-test must pass on this host before any real run is allowed
"$ENV_PY" -u code/analysis/lexical_token_test.py --selftest || { echo "[FAIL] selftest" >&2; exit 1; }

SLICE="${SLICE:-data/c4_slice_20k.jsonl}"
# The slice must ALREADY exist. Do not build it here: every array task would run the same
# check-then-build, and `-s` is true the moment streaming starts, so task 1 would happily
# read a half-written file and silently analyse a truncated corpus. Build it once, up
# front, with build_c4_slice.py, then submit.
if [ ! -s "$SLICE" ]; then
  echo "[FAIL] $SLICE missing. Build it ONCE before submitting:" >&2
  echo "       $ENV_PY code/analysis/build_c4_slice.py --n 20000 --out $SLICE" >&2
  exit 1
fi
NSLICE=$(wc -l < "$SLICE")
if [ "${SMOKE:-0}" != "1" ] && [ "$NSLICE" -lt 20000 ]; then
  echo "[FAIL] $SLICE has $NSLICE docs, expected 20000 (truncated or partial build)" >&2
  exit 1
fi
echo "[slice] $SLICE has $NSLICE docs"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) A=Qwen/Qwen3-1.7B-Base; RA=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; IA=layer14
     B=google/gemma-2-2b; RB=gemma-scope-2b-pt-res-canonical; IB=layer_12/width_16k/canonical
     TAG=qwen3-1.7b__gemma2-2b ;;                       # strong  (C4 z=42.7)
  1) A=google/gemma-2-9b; RA=gemma-scope-9b-pt-res-canonical; IA=layer_20/width_16k/canonical
     B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre
     TAG=gemma2-9b__mistral7b ;;                        # median  (C4 z=24.0)
  2) A=meta-llama/Llama-3.2-1B; RA=chanind/sae-llama-3.2-1b-res; IA=blocks.8.hook_resid_post
     B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre
     TAG=llama3.2-1b__mistral7b ;;                      # weak    (C4 z=12.3)
  *) echo "unknown task"; exit 2 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--max_prompts 2000 --topk 256 --n_perm 100 --svd_k 64"
  OUT="$ROOT/results/cross_model_sae/smoke_lextok_${TAG}.json"
  CACHE=""
else
  # KS lets a single k be run as its own SHORTER job. Cluster-wide admin sweeps cancelled
  # 288 jobs at 20:01 and 38 at 22:31 on 2026-07-28, and cell 0 was hit twice purely for
  # being the longest-running cell. Shorter jobs are less likely to overlap a sweep; the
  # per-k receipts are merged afterwards and the k-independent parts cross-checked.
  KS="${KS:-256,1024,4096}"
  EXTRA="--topk 1500 --n_perm 2000 --svd_k $KS"
  if [ "$KS" = "256,1024,4096" ]; then
    OUT="$ROOT/results/cross_model_sae/lextok20k_${TAG}.json"
  else
    OUT="$ROOT/results/cross_model_sae/lextok20k_${TAG}_k${KS}.json"
  fi
  CACHE="--cache $ROOT/cache/lextok20k_${TAG}.npz"
fi

"$ENV_PY" -u code/analysis/lexical_token_test.py \
  --model_a "$A" --sae_a_release "$RA" --sae_a_id "$IA" \
  --model_b "$B" --sae_b_release "$RB" --sae_b_id "$IB" \
  --prompts "$SLICE" --lang en $EXTRA $CACHE \
  --output "$OUT"
RC=$?
# Fail loudly. rc=2 is the pre-registered adequacy-gate failure: the receipt IS written and
# is a real (NOT-TESTABLE) finding, so keep it and surface the code rather than pretending
# the job succeeded. Any other non-zero rc, or a missing/empty receipt, is a hard failure.
if [ ! -s "$OUT" ]; then
  echo "[FAIL] rc=$RC and no receipt at $OUT" >&2; exit 1
fi
if [ $RC -eq 2 ]; then
  echo "[GATE-FAIL] adequacy gate failed at every k -- receipt kept, verdict NOT-TESTABLE" >&2
  exit 2
fi
if [ $RC -ne 0 ]; then
  echo "[FAIL] rc=$RC" >&2; exit 1
fi
echo "[DONE] $(date -Is) -> $OUT"
