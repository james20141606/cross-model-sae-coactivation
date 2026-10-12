#!/bin/bash
#SBATCH --job-name=ct-slice20k-repro
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1 --time=01:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# End-to-end reproduction check enabled by the superset property.
#
# md5(head -2000 data/c4_slice_20k.jsonl) == md5(data/c4_slice.jsonl), so running the
# matcher on the NEW 20k file restricted to its first 2000 documents is byte-identical
# input to the published run. It must therefore reproduce the published receipt
# c4_gemma2-9b__mistral7b.json (observed.n_mutual_nn = 819, mutual_z = 24.01).
#
# If it does NOT reproduce, the corpus file, the extraction path, or the environment has
# drifted, and nothing computed on the 20k slice is interpretable -- stop and diagnose
# before reading any lextok result. This is the "if it does not, stop and find out why"
# gate, made executable rather than left as an intention.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

# input identity first -- cheap, and if this fails the GPU work is pointless
A=$(head -2000 data/c4_slice_20k.jsonl | md5sum | cut -d' ' -f1)
B=$(md5sum data/c4_slice.jsonl | cut -d' ' -f1)
echo "[md5] head-2000 of 20k slice = $A"
echo "[md5] published 2k slice     = $B"
[ "$A" = "$B" ] || { echo "[FAIL] 20k slice is NOT a superset of the published slice" >&2; exit 1; }

OUT="$ROOT/results/cross_model_sae/repro20k_gemma2-9b__mistral7b.json"
cd "$ROOT/code/analysis"
"$ENV_PY" -u cross_model_sae_match.py \
  --model_a google/gemma-2-9b --sae_a_release gemma-scope-9b-pt-res-canonical --sae_a_id layer_20/width_16k/canonical \
  --model_b mistralai/Mistral-7B-v0.1 --sae_b_release mistral-7b-res-wg --sae_b_id blocks.16.hook_resid_pre \
  --prompts ../../data/c4_slice_20k.jsonl --lang en --topk 1500 --n_perm 200 --max_prompts 2000 \
  --output "$OUT"
RC=$?
[ $RC -eq 0 ] && [ -s "$OUT" ] || { echo "[FAIL] rc=$RC output missing" >&2; exit 1; }

"$ENV_PY" - "$OUT" "$ROOT/results/cross_model_sae/c4_gemma2-9b__mistral7b.json" <<'PY'
import json, sys
new, pub = (json.load(open(p)) for p in sys.argv[1:3])
n_new, n_pub = new["observed"]["n_mutual_nn"], pub["observed"]["n_mutual_nn"]
z_new, z_pub = new["mutual_z"], pub["mutual_z"]
print(f"[repro] mutual_nn new={n_new} published={n_pub} (delta {n_new-n_pub})")
print(f"[repro] mutual_z  new={z_new:.2f} published={z_pub:.2f}")
# bfloat16 GPU nondeterminism is disclosed in the paper, so allow a small tolerance;
# anything larger means real drift.
ok = abs(n_new - n_pub) <= 5 and abs(z_new - z_pub) <= 1.0
print("[repro] REPRODUCES" if ok else "[repro] DOES NOT REPRODUCE -- STOP")
sys.exit(0 if ok else 1)
PY
RC=$?
[ $RC -eq 0 ] || { echo "[FAIL] 20k slice does not reproduce the published cell" >&2; exit 1; }
echo "[DONE] $(date -Is) -> $OUT"
