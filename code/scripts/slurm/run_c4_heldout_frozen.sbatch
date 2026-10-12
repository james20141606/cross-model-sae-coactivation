#!/bin/bash
#SBATCH --job-name=ct-c4-heldout
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1 --time=01:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# <GPU_PARTITION> was in this list on the first submission (JID 18176229); task 1
# (Qwen3-8B x Gemma-2-9B) died with a Bus error loading shards on <node> at 6 GB RSS --
# a node-local I/O fault, not memory. --exclude is rejected by site policy, so the fix is
# the partition list: resubmitted as JID 18177194 on the 80 GiB+ partitions the roster
# rerun used (<node> is an L40S node and cannot be scheduled there).
#SBATCH --array=0-8
# app:threats (iv), open item: a pooled-level HELD-OUT evaluation of A-selected pairs for
# the nine headline C4 cells. The headline (c4_*.json) selects and nulls on the same 2,000
# docs (re-selecting under every null draw). Here the reciprocal pairs are selected on
# data/c4_fit.jsonl (docs 0-999) and those FROZEN pairs are scored on data/c4_heldout.jsonl
# (docs 1000-1999) with an eval-only permutation null; the in-sample comparator re-selects
# on the held-out docs themselves so the ratio is on identical documents.
# Both corpora are the two halves of the headline slice (ELN 2026-07-11); the matcher
# asserts disjointness on prompt text. Timing basis: the roster's own n_perm=2000 rerun
# (JID 14915907) took 3-6 min/cell on one GPU; this does two extractions per model, so
# budget ~10 min/cell, 1 h cap. GPU utilisation is >50% during extraction (two 7-9B
# forward passes over 2,000 docs); the numpy null runs after the GPU is released.
#
# SMOKE=1 -> 100 docs each half, n_perm=50, to smoke_heldout_*.json.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT/code/analysis"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

GS2=gemma-scope-2b-pt-res-canonical;  IG2=layer_12/width_16k/canonical
GS9=gemma-scope-9b-pt-res-canonical;  IG9=layer_20/width_16k/canonical
MIS=mistral-7b-res-wg;                IMI=blocks.16.hook_resid_pre
Q17=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; IQ17=layer14
Q8=qwenpt:Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50;    IQ8=layer18

# the nine natural-text-robust cross-family cells (tab:c4_verdicts), same ids as run_c4_roster_nperm2000
case "${SLURM_ARRAY_TASK_ID:-0}" in
   0) A=Qwen/Qwen3-1.7B-Base;RA=$Q17;IA=$IQ17; B=google/gemma-2-2b;RB=$GS2;IB=$IG2; TAG=qwen3-1.7b__gemma2-2b ;;
   1) A=Qwen/Qwen3-8B-Base;RA=$Q8;IA=$IQ8;   B=google/gemma-2-9b;RB=$GS9;IB=$IG9;  TAG=qwen3-8b__gemma2-9b ;;
   2) A=Qwen/Qwen3-8B-Base;RA=$Q8;IA=$IQ8;   B=google/gemma-2-2b;RB=$GS2;IB=$IG2;  TAG=qwen3-8b__gemma2-2b ;;
   3) A=Qwen/Qwen3-1.7B-Base;RA=$Q17;IA=$IQ17; B=google/gemma-2-9b;RB=$GS9;IB=$IG9; TAG=qwen3-1.7b__gemma2-9b ;;
   4) A=google/gemma-2-9b;RA=$GS9;IA=$IG9;   B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=gemma2-9b__mistral7b ;;
   5) A=Qwen/Qwen3-8B-Base;RA=$Q8;IA=$IQ8;   B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=qwen3-8b__mistral7b ;;
   6) A=google/gemma-2-2b;RA=$GS2;IA=$IG2;   B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=gemma2-2b__mistral7b ;;
   7) A=Qwen/Qwen3-1.7B-Base;RA=$Q17;IA=$IQ17; B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=qwen3-1.7b__mistral7b ;;
   8) A=meta-llama/Llama-3.2-1B;RA=chanind/sae-llama-3.2-1b-res;IA=blocks.8.hook_resid_post; B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=llama3.2-1b__mistral7b ;;
   *) echo "unknown task"; exit 2 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--max_prompts 100 --n_perm 50"
  OUT="$ROOT/results/cross_model_sae/smoke_heldout_${TAG}.json"
else
  EXTRA="--n_perm 2000"
  OUT="$ROOT/results/cross_model_sae/heldout_${TAG}.json"
fi

"$ENV_PY" -u cross_model_sae_match.py \
  --model_a "$A" --sae_a_release "$RA" --sae_a_id "$IA" \
  --model_b "$B" --sae_b_release "$RB" --sae_b_id "$IB" \
  --select_prompts ../../data/c4_fit.jsonl \
  --prompts ../../data/c4_heldout.jsonl --lang en --topk 1500 $EXTRA \
  --output "$OUT"
RC=$?
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[FAIL] rc=$RC output=$OUT exists=$([ -s "$OUT" ] && echo yes || echo no)" >&2
  exit 1
fi
echo "[DONE] $(date -Is) -> $OUT"
