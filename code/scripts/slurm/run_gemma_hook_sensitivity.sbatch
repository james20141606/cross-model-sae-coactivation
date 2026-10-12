#!/bin/bash
#SBATCH --job-name=ct-ghs
#SBATCH --output=<HOME>/ghs_20260924/logs/%x-%j.out
#SBATCH --error=<HOME>/ghs_20260924/logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# Gemma Scope read-point sensitivity, tier 1 (notes/PREREG_gemma_hook_sensitivity_20260924.md),
# cluster run with the ORIGINAL google/gemma-2-* weights. Added 2026-09-24 evening: the local
# Qwen3-8B-Base download stalled (huggingface.co unreachable through the local proxy), so the two
# Q8 cells run here (deviation from prereg sec. 4, disclosed); the other six cells are re-run in
# the same job as a cluster replication with the original weights.
#
# One job, sequential: extract (GPU, batch-1 forwards over the 2,000-doc slice, both Gemma read
# points in one pass) for gemma2-9b, gemma2-2b, qwen3-8b, qwen3-1.7b, mistral7b; then match (numpy,
# ~6-8 min) for all 8 cells at offsets 0 and 1 with G0 against the tracked published receipts.
#
# Storage (2026-09-24: /scratch at 99.94% of 5 TB): nothing is written to /scratch.
#   code + tracked published receipts + outputs: $HOME/ghs_20260924 (receipts are JSON, < 1 MB)
#   pooled features (~3.5 GB) and any downloaded weights: /dev/shm (node-local, removed at exit)
#   HF cache: /dev/shm hub; repos already complete in <HF_CACHE>/hub are
#   symlinked read-only and the job runs HF_HUB_OFFLINE=1 if all are present.
set -uo pipefail
RUN=${GHS_RUN:-$HOME/ghs_20260924}   # a second copy (<GPU_PARTITION>, 2026-09-25 01:50) sets GHS_RUN
# 2026-09-25 02:40 rerun for the Q8-G2 cell (18441889 failed gemma2-2b: the 65k Gemma Scope params
# were not in the scratch cache and the job runs offline). Defaults unchanged; the rerun sets
#   GHS_MODELS="gemma2-2b qwen3-8b"   (only the models the missing cell needs)
#   GHS_HUB_OVERRIDE=$HOME/ghs_hub/hub (gemma-scope-2b repo: 65k params downloaded there on the login
#   node, sha256 checked against the HF LFS pointer; 16k params symlinked read-only to the scratch blob)
ENV_PY=${ENV_ANALYSIS}/bin/python
SCRATCH_HUB=${HF_CACHE}/hub
SHM=/dev/shm/${USER}_ghs_${SLURM_JOB_ID:-local}
export HF_HOME=$SHM/hf HF_HUB_CACHE=$SHM/hf/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
export GHS_CACHE=$SHM/pooled GHS_C4=${REPO_ROOT}/data/c4_slice.jsonl
export GHS_GEMMA_WEIGHTS=google
trap 'rm -rf "$SHM"' EXIT
mkdir -p "$HF_HUB_CACHE" "$GHS_CACHE" "$RUN/logs" "$RUN/pooled_meta"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
df -h /dev/shm | tail -1
echo "[space] home: $(du -sh $RUN 2>/dev/null | cut -f1) used by run dir"
(myquota 2>/dev/null | grep -E "/home|/scratch") || echo "[space] myquota unavailable on this node"

ALL_PRESENT=1
for m in models--google--gemma-2-2b models--google--gemma-2-9b models--Qwen--Qwen3-8B-Base \
         models--Qwen--Qwen3-1.7B-Base models--mistralai--Mistral-7B-v0.1 \
         models--google--gemma-scope-2b-pt-res models--google--gemma-scope-9b-pt-res \
         models--Qwen--SAE-Res-Qwen3-8B-Base-W64K-L0_50 models--Qwen--SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 \
         models--JoshEngels--Mistral-7B-Residual-Stream-SAEs; do
  if [ -n "${GHS_HUB_OVERRIDE:-}" ] && [ -d "$GHS_HUB_OVERRIDE/$m/snapshots" ]; then
    ln -s "$GHS_HUB_OVERRIDE/$m" "$HF_HUB_CACHE/$m"; echo "[hf] $m <- $GHS_HUB_OVERRIDE (override)"
  elif [ -d "$SCRATCH_HUB/$m/snapshots" ] && [ -z "$(ls "$SCRATCH_HUB/$m/blobs" 2>/dev/null | grep incomplete)" ]; then
    ln -s "$SCRATCH_HUB/$m" "$HF_HUB_CACHE/$m"; echo "[hf] $m <- scratch cache (read-only)"
  else
    ALL_PRESENT=0; echo "[hf] $m MISSING in scratch cache -> download into /dev/shm"
  fi
done
[ "$ALL_PRESENT" = "1" ] && export HF_HUB_OFFLINE=1 && echo "[hf] HF_HUB_OFFLINE=1"

cd "$RUN/code/analysis"
sha256sum gemma_hook_sensitivity.py cross_model_sae_match.py qwen_sae_adapter.py
md5sum "$GHS_C4"
for M in ${GHS_MODELS:-gemma2-9b gemma2-2b qwen3-8b qwen3-1.7b mistral7b}; do
  echo "[extract] $M $(date -Is)"
  "$ENV_PY" -u gemma_hook_sensitivity.py extract --model "$M" --device cuda || echo "[FAIL] extract $M rc=$?"
  cp "$GHS_CACHE"/*.meta.json "$RUN/pooled_meta/" 2>/dev/null
done
echo "[match] $(date -Is)"
"$ENV_PY" -u gemma_hook_sensitivity.py match
RC=$?
cp "$GHS_CACHE"/*.meta.json "$RUN/pooled_meta/" 2>/dev/null
echo "[DONE] rc=$RC $(date -Is)"
exit $RC
