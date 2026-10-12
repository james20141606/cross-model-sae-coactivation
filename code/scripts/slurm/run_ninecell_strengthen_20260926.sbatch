#!/bin/bash
#SBATCH --job-name=ct-ninecell
#SBATCH --output=<HOME>/ninecell_20260926/logs/%x-%j.out
#SBATCH --error=<HOME>/ninecell_20260926/logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=200G --gres=gpu:1 --time=03:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# Nine-cell length/topic nulls + random split + document-bootstrap intervals
# (notes/PREREG_ninecell_ladder_20260926.md; code/analysis/ninecell_strengthen_20260926.py).
# fp64 permutation batches need an H100/H200/A100 (no L40s: fp64 throughput ~1/40).
#
# One job, because the pooled features live in node-local /dev/shm (scratch is near quota):
#   1. extract  (GPU, batch-1 forwards, published compute_pooled_features) for six models
#   2. analyze_b for the nine cells in the background on CPU (numpy, OMP 1 thread each)
#   3. analyze_a for the nine cells on the GPU (float64 batched permutation counts)
#   4. summarize -> NINECELL_SUMMARY.json, tab_ninecell.tex
# Storage: code + tracked receipts + outputs in $HOME/ninecell_20260926 (JSON, a few MB);
# pooled (~2.2 GB) and any downloaded weights in /dev/shm, removed at exit. Nothing on /scratch.
set -uo pipefail
RUN=${NC_RUN:-$HOME/ninecell_20260926}
ENV_PY=${ENV_ANALYSIS}/bin/python
SCRATCH_HUB=${HF_CACHE}/hub
SHM=/dev/shm/${USER}_nc_${SLURM_JOB_ID:-local}
export HF_HOME=$SHM/hf HF_HUB_CACHE=$SHM/hf/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
export NC_C4=${REPO_ROOT}/data/c4_slice.jsonl
export NC_RECEIPTS=$RUN/receipts
POOLED=$SHM/pooled; OUT=$RUN/results/ninecell_20260926
trap 'rm -rf "$SHM"' EXIT
mkdir -p "$HF_HUB_CACHE" "$POOLED" "$OUT" "$RUN/logs"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
df -h /dev/shm | tail -1
ALL_PRESENT=1
for m in models--google--gemma-2-2b models--google--gemma-2-9b models--Qwen--Qwen3-8B-Base \
         models--Qwen--Qwen3-1.7B-Base models--mistralai--Mistral-7B-v0.1 models--meta-llama--Llama-3.2-1B \
         models--google--gemma-scope-2b-pt-res models--google--gemma-scope-9b-pt-res \
         models--Qwen--SAE-Res-Qwen3-8B-Base-W64K-L0_50 models--Qwen--SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 \
         models--JoshEngels--Mistral-7B-Residual-Stream-SAEs models--chanind--sae-llama-3.2-1b-res; do
  if [ -d "$SCRATCH_HUB/$m/snapshots" ] && [ -z "$(ls "$SCRATCH_HUB/$m/blobs" 2>/dev/null | grep incomplete)" ]; then
    ln -s "$SCRATCH_HUB/$m" "$HF_HUB_CACHE/$m"; echo "[hf] $m <- scratch cache (read-only)"
  else
    ALL_PRESENT=0; echo "[hf] $m MISSING in scratch cache -> download into /dev/shm"
  fi
done
[ "$ALL_PRESENT" = "1" ] && export HF_HUB_OFFLINE=1 && echo "[hf] HF_HUB_OFFLINE=1"

cd "$RUN/code/analysis"
sha256sum ninecell_strengthen_20260926.py cross_model_sae_match.py c4_covariate_null.py qwen_sae_adapter.py
md5sum "$NC_C4"
"$ENV_PY" -u ninecell_strengthen_20260926.py selftest || { echo "[FAIL] selftest"; exit 2; }

MODELS=${NC_MODELS:-"llama3.2-1b qwen3-1.7b gemma2-2b gemma2-9b mistral7b qwen3-8b"}
for M in $MODELS; do
  echo "[extract] $M $(date -Is)"
  "$ENV_PY" -u ninecell_strengthen_20260926.py extract --model "$M" --pooled "$POOLED" \
    || echo "[FAIL] extract $M rc=$?"
  cp "$POOLED"/*.meta.json "$OUT/" 2>/dev/null
done
ls -la "$POOLED"

CELLS=${NC_CELLS:-"qwen3-1.7b__gemma2-2b qwen3-1.7b__gemma2-9b qwen3-1.7b__mistral7b qwen3-8b__gemma2-2b qwen3-8b__gemma2-9b qwen3-8b__mistral7b gemma2-2b__mistral7b gemma2-9b__mistral7b llama3.2-1b__mistral7b"}
echo "[B] start $(date -Is)"
for C in $CELLS; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$ENV_PY" -u ninecell_strengthen_20260926.py analyze_b --cell "$C" --pooled "$POOLED" \
    --out "$OUT/B_$C.json" > "$RUN/logs/B_${C}_${SLURM_JOB_ID:-local}.log" 2>&1 &
done
echo "[A] start $(date -Is)"
for C in $CELLS; do
  "$ENV_PY" -u ninecell_strengthen_20260926.py analyze_a --cell "$C" --pooled "$POOLED" \
    --out "$OUT/A_$C.json" --device cuda || echo "[FAIL] analyze_a $C rc=$?"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1
done
echo "[A] done $(date -Is); waiting for B"
wait
echo "[B] done $(date -Is)"
"$ENV_PY" -u ninecell_strengthen_20260926.py summarize --dir "$OUT"
NA=$(ls "$OUT"/A_*.json 2>/dev/null | wc -l); NB=$(ls "$OUT"/B_*.json 2>/dev/null | wc -l)
echo "[DONE] A=$NA B=$NB $(date -Is)"
[ "$NA" -eq 9 ] && [ "$NB" -eq 9 ] && [ -s "$OUT/NINECELL_SUMMARY.json" ] || { echo "[FAIL] receipts incomplete" >&2; exit 1; }
