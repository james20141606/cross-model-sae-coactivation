#!/bin/bash
#SBATCH --job-name=ct-xmodel-sae-randctl
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=04:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#
# Matched random-SAE control (arXiv:2602.14111 "Sanity Checks for Sparse Autoencoders:
# Do SAEs Beat Random Baselines?" random-baseline gate) for cross-model SAE feature
# matching. Re-runs THREE real cross_model_sae_match.py cells -- highest-z
# (Qwen3-1.7B<->Gemma2-2B, mutual_z=42.7), mid (Gemma2-9B<->Mistral-7B, mutual_z=24.0),
# low (Llama-3.2-1B<->Mistral-7B, mutual_z=12.3) -- with BOTH SAEs replaced by a
# matched-dimension, matched-per-token-L0 random control (random_sae_baseline.py). Each
# side is calibrated+wrapped independently (model B seed = model A seed + 10000).
# Model/SAE/prompt args copied VERBATIM from the real receipts:
#   results/cross_model_sae/c4_qwen3-1.7b__gemma2-2b.json
#   results/cross_model_sae/c4_gemma2-9b__mistral7b.json
#   results/cross_model_sae/c4_llama3.2-1b__mistral7b.json
#
# INTERPRETATION RULE: if reciprocal-matching z stays high under the random SAEs, the
# "cross-family sharing" claim deflates to activation-statistics correlation; if z
# collapses to ~0, the claim is SAE-specific and hardened.
#
# skip-if-exists: each cell is skipped if its rand_*.json receipt already exists and is
# non-empty (safe to resubmit after a partial/killed run).
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python      # has sae_lens
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
OUT=results/cross_model_sae
PROMPTS="${PROMPTS:-data/c4_slice.jsonl}"
TOPK="${TOPK:-1500}"
N_PERM="${N_PERM:-200}"
MAX_PROMPTS="${MAX_PROMPTS:-2000}"
RANDOM_SAE_MODE="${RANDOM_SAE_MODE:-random-dir}"
RANDOM_SAE_SEED="${RANDOM_SAE_SEED:-0}"
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/$OUT"
cd "$ROOT"

echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} mode=$RANDOM_SAE_MODE seed=$RANDOM_SAE_SEED prompts=$PROMPTS topk=$TOPK n_perm=$N_PERM max_prompts=$MAX_PROMPTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

run_cell() {
  local tag="$1" ma="$2" sar="$3" said="$4" mb="$5" sbr="$6" sbid="$7"
  local out="$OUT/rand_${tag}.json"     # cross_model_sae_match.py auto-prefixes rand_ when --random_sae != none
  if [ -s "$out" ]; then
    echo "[SKIP] $out already exists"
    return 0
  fi
  echo "[CELL] $tag : A=$ma ($sar::$said)  B=$mb ($sbr::$sbid)"
  "$ENV_PY" -u code/analysis/cross_model_sae_match.py \
    --model_a "$ma" --sae_a_release "$sar" --sae_a_id "$said" \
    --model_b "$mb" --sae_b_release "$sbr" --sae_b_id "$sbid" \
    --prompts "$PROMPTS" --lang en --topk "$TOPK" --n_perm "$N_PERM" --max_prompts "$MAX_PROMPTS" \
    --random_sae "$RANDOM_SAE_MODE" --random_sae_seed "$RANDOM_SAE_SEED" \
    --output "$OUT/${tag}.json" || echo "FAILED $tag"
  if [ -s "$out" ]; then echo "[OK] wrote $out ($(wc -c < "$out") bytes)";
  else echo "[EMPTY] $out"; fi
}

# highest-z real cell: Qwen3-1.7B <-> Gemma2-2B (mutual_z=42.7) -- qwenpt: manual loader,
# exercises the W_enc-orientation fix in random_sae_baseline.py
run_cell "c4_qwen3-1.7b__gemma2-2b" \
  "Qwen/Qwen3-1.7B-Base" "qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50" "layer14" \
  "google/gemma-2-2b" "gemma-scope-2b-pt-res-canonical" "layer_12/width_16k/canonical"

# mid-z real cell: Gemma2-9B <-> Mistral-7B (mutual_z=24.0)
run_cell "c4_gemma2-9b__mistral7b" \
  "google/gemma-2-9b" "gemma-scope-9b-pt-res-canonical" "layer_20/width_16k/canonical" \
  "mistralai/Mistral-7B-v0.1" "mistral-7b-res-wg" "blocks.16.hook_resid_pre"

# low-z real cell: Llama-3.2-1B <-> Mistral-7B (mutual_z=12.3)
run_cell "c4_llama3.2-1b__mistral7b" \
  "meta-llama/Llama-3.2-1B" "chanind/sae-llama-3.2-1b-res" "blocks.8.hook_resid_post" \
  "mistralai/Mistral-7B-v0.1" "mistral-7b-res-wg" "blocks.16.hook_resid_pre"

echo DONE
