#!/bin/bash
#SBATCH --job-name=ct-lanpair
#SBATCH --output=<REPO_ROOT>/logs/lanpair/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/lanpair/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 --time=08:00:00
# CLUSTER_ACCOUNT_C, not CLUSTER_ACCOUNT: the advanced account's group GRES
# cap was saturated (QOSGrpGRES / QOSMaxGRESPerUser) and this array sat PD indefinitely
# behind other jobs on that allocation. A probe job on _158_general started immediately.
# This arm needs one GPU it never uses, so it has no reason to contend for the advanced
# allocation. Recorded here because the receipts were produced under THIS account.
# Was CLUSTER_ACCOUNT_C + <GPU_PARTITION>: that QOS (gpu48) carries a GROUP cap of
# gres/gpu=24, and <GPU_PARTITION> is saturated by other groups (one alone holds 24).
# Two runs were cancelled with QOSGrpGRES at 2:05 and 2:08 elapsed against an 8h
# limit -- the group cap evicts running jobs, so multi-hour work cannot finish
# there. The advanced account's h200 QOS has gres/gpu=112 and is what the
# sibling long jobs use successfully.
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --array=0-2
# Apply the SAME lexical-residualization control to Lan et al.'s PUBLISHED pairing rule.
# Design pre-registered in experiments_detail.md at commit 3237ecb BEFORE any of this code
# existed; do not retune. The reimplementation validity gate (G1/G2/G3) is read BEFORE any
# R_lex number and a failure exits 3 (Branch 5) with no R_lex computed.
#
# NO GPU IS NEEDED: every cell reads its pooled-feature cache from cache/lextok20k_*.npz,
# so no model is ever loaded. A GPU is requested only because every partition on this
# cluster is a GPU partition. Small/plentiful partitions are listed first for that reason --
# the 80GiB-only constraint on run_lexical_token_test.sbatch was about keeping two models
# resident and does not apply here.
#
# ENV: <ENV_TRAIN>, the SAME interpreter that produced every committed lextok20k_* receipt
# (numpy 1.26.4 / sklearn 1.8.0). <ENV_ANALYSIS> has numpy 2.x and would make the comparison
# against R_lex = 0.933/0.942/0.918 not apples-to-apples.
#
# Explicit, non-colliding output paths: lanpair_lextok20k_* shares no stem with lextok20k_*.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_TRAIN}/bin/python
mkdir -p "$ROOT/logs/lanpair" "$ROOT/results/cross_model_sae"
cd "$ROOT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0}"

# the offline self-test must pass on this host before any real run is allowed. It now
# covers the Lan pairing two-sidedly (CASE5) and proves the validity gate can FAIL (CASE6).
"$ENV_PY" -u code/analysis/lexical_token_test.py --selftest || { echo "[FAIL] selftest" >&2; exit 1; }

SLICE="${SLICE:-data/c4_slice_20k.jsonl}"
[ -s "$SLICE" ] || { echo "[FAIL] $SLICE missing; build with build_c4_slice.py" >&2; exit 1; }
NSLICE=$(wc -l < "$SLICE")
[ "$NSLICE" -ge 20000 ] || { echo "[FAIL] $SLICE has $NSLICE docs, expected 20000" >&2; exit 1; }

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TAG=qwen3-1.7b__gemma2-2b ;;   # strong  (C4 z=42.7)   ours: R_lex 0.933
  1) TAG=gemma2-9b__mistral7b ;;    # median  (C4 z=24.0)   ours: R_lex 0.942
  2) TAG=llama3.2-1b__mistral7b ;;  # weak    (C4 z=12.3)   ours: R_lex 0.918
  *) echo "unknown task"; exit 2 ;;
esac
CACHE="$ROOT/cache/lextok20k_${TAG}.npz"
[ -s "$CACHE" ] || { echo "[FAIL] cache $CACHE missing -- this arm reuses the committed
       pooled features and must NOT re-extract them, or it would not be the same data" >&2; exit 1; }

# Model/SAE ids are recorded in the receipt for provenance. They are NOT used to extract
# anything: the cache short-circuits compute_pooled_features entirely.
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) A=Qwen/Qwen3-1.7B-Base; RA=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; IA=layer14
     B=google/gemma-2-2b; RB=gemma-scope-2b-pt-res-canonical; IB=layer_12/width_16k/canonical ;;
  1) A=google/gemma-2-9b; RA=gemma-scope-9b-pt-res-canonical; IA=layer_20/width_16k/canonical
     B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre ;;
  2) A=meta-llama/Llama-3.2-1B; RA=chanind/sae-llama-3.2-1b-res; IA=blocks.8.hook_resid_post
     B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre ;;
esac
COMMON=(--model_a "$A" --sae_a_release "$RA" --sae_a_id "$IA"
        --model_b "$B" --sae_b_release "$RB" --sae_b_id "$IB"
        --prompts "$SLICE" --lang en --topk 1500 --cache "$CACHE")

# ---------------------------------------------------------------------------------------
# PRE-FLIGHT: the --pairing flag must not have perturbed the DEFAULT path. Run the
# pre-3237ecb file and the current file with --pairing mutual on THIS cell's real cache
# (reduced k/n_perm for speed) and require every key of the old receipt to be reproduced
# exactly. Per-cell rather than once, because the claim "the default is unchanged" has to
# hold on the actual data the comparison baseline was computed from.
# ---------------------------------------------------------------------------------------
#
# The cluster tree is an rsync mirror, not a git checkout, so the pre-change file cannot be
# recovered here with `git show`. It is provisioned alongside the code as
# _prechange_3237ecb_lexical_token_test.py and pinned by sha256 below, so the comparison
# cannot silently run against the wrong file. Regenerate and verify from any clone with:
#   git show 3237ecb:code/analysis/lexical_token_test.py | shasum -a 256
PRECHANGE="$ROOT/code/analysis/_prechange_3237ecb_lexical_token_test.py"
PRECHANGE_SHA=e995bec54bab56549d71e319213273c116e1a5d5e44872608b4e0c9037f6fac5
[ -s "$PRECHANGE" ] || { echo "[FAIL] $PRECHANGE missing" >&2; exit 1; }
GOT=$(sha256sum "$PRECHANGE" | cut -d' ' -f1)
[ "$GOT" = "$PRECHANGE_SHA" ] || {
  echo "[FAIL] pre-change file sha256 $GOT != $PRECHANGE_SHA" >&2; exit 1; }
EQ="$ROOT/results/cross_model_sae/_equiv_${TAG}"
mkdir -p "$EQ"
cp "$PRECHANGE" "$EQ/old_ltt.py"
cp "$ROOT/code/analysis/cross_model_sae_match.py" "$EQ/"
EQARGS=(--max_prompts 3000 --svd_k 64 --n_perm 50)
"$ENV_PY" -u "$EQ/old_ltt.py" "${COMMON[@]}" "${EQARGS[@]}" --output "$EQ/old.json" >"$EQ/old.log" 2>&1
"$ENV_PY" -u code/analysis/lexical_token_test.py "${COMMON[@]}" "${EQARGS[@]}" \
    --pairing mutual --output "$EQ/new.json" >"$EQ/new.log" 2>&1
"$ENV_PY" - "$EQ/old.json" "$EQ/new.json" <<'PYEOF' || { echo "[FAIL] default path CHANGED" >&2; exit 1; }
import json, sys
o, n = (json.load(open(p)) for p in sys.argv[1:3])
# "cell" is defined as os.path.basename(--output). The two runs MUST write to different
# paths to be compared at all, so this one key differs by construction and comparing it
# is a false positive, not a signal. Excluded by name, loudly, rather than by loosening
# the comparison -- every other key, including every number, still has to match exactly.
EXCLUDE = {"cell"}
def walk(a, b, p=""):
    if isinstance(a, dict):
        return [d for k in a if not (p == "" and k in EXCLUDE)
                for d in (walk(a[k], b[k], p + "/" + k) if k in b
                          else [p + "/" + k + " MISSING"])]
    return [] if a == b else ["%s: old=%r new=%r" % (p, a, b)]
bad = walk(o, n)
print("[equiv] %d old key(s) compared (%s excluded: output-path metadata), %d difference(s); "
      "new-only keys: %s"
      % (len(o) - len(EXCLUDE), "/".join(sorted(EXCLUDE)), len(bad), sorted(set(n) - set(o))))
for b in bad:
    print("   DIFF", b)
sys.exit(1 if bad else 0)
PYEOF
echo "[equiv] default --pairing mutual path is numerically identical to 3237ecb"

# ---------------------------------------------------------------------------------------
# THE ARM: Lan et al.'s 1A-to-ManyB pairing, everything else unchanged.
# ---------------------------------------------------------------------------------------
OUT="$ROOT/results/cross_model_sae/lanpair_lextok20k_${TAG}.json"

# PER-K CHECKPOINTING (2026-08-02). The previous form ran --svd_k 256,1024,4096 in ONE
# call and wrote $OUT only after all three finished. Three runs were then cancelled with
# QOSGrpGRES at 2:01, 2:05 and 2:28 elapsed against an 8h limit -- the group GPU cap
# evicts RUNNING jobs, so no single call needing >2h can ever land here. Two of those
# runs had completed k=256 AND k=1024 and lost both.
#
# Each k now writes its own receipt the moment it finishes, and a k whose receipt already
# exists is skipped, so a resumed run picks up where the eviction hit. The skip-if-exists
# idiom is the one already used in a sibling sweep. The code-hash guard above
# still runs on every submission, so a resumed run cannot mix receipts across code
# versions: if lexical_token_test.py changed, the guard fails before any k is read.
for K in 256 1024 4096; do
  KOUT="${OUT%.json}_k${K}.json"
  if [ -s "$KOUT" ]; then
    echo "[skip] k=$K already has a receipt at $KOUT"
    continue
  fi
  echo "[run ] k=$K -> $KOUT  ($(date -Is))"
  "$ENV_PY" -u code/analysis/lexical_token_test.py "${COMMON[@]}" \
    --n_perm 2000 --svd_k "$K" --pairing lan_argmax --output "$KOUT"
  RC=$?
  [ -s "$KOUT" ] || { echo "[FAIL] k=$K rc=$RC and no receipt at $KOUT" >&2; exit 1; }
  if [ $RC -eq 3 ]; then
    echo "[GATE-FAIL] k=$K reimplementation validity gate FAILED -- Branch 5, arm stops, receipt kept" >&2
    exit 3
  fi
  if [ $RC -eq 2 ]; then
    echo "[GATE-FAIL] k=$K adequacy gate failed -- Branch 4 for this k, receipt kept" >&2
    exit 2
  fi
  [ $RC -eq 0 ] || { echo "[FAIL] k=$K rc=$RC" >&2; exit 1; }
  echo "[ok  ] k=$K done at $(date -Is)"
done

# Merge the per-k receipts into the single artefact the reader expects. Merging is a
# separate, cheap, restartable step: if it fails, every per-k receipt survives.
"$ENV_PY" - "$OUT" 256 1024 4096 <<'PYMERGE' || { echo "[FAIL] merge" >&2; exit 1; }
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); ks = [int(k) for k in sys.argv[2:]]
parts = {}
for k in ks:
    f = out.with_name(out.stem + f"_k{k}.json")
    if not f.exists():
        print(f"[FAIL] merge: missing per-k receipt {f}", file=sys.stderr); sys.exit(1)
    parts[str(k)] = json.loads(f.read_text())
# Carry the first receipt's top level (model pair, pairing rule, provenance), then
# REPLACE its arms with the union of every per-k receipt's arms. The first form kept
# only the first receipt's own arm at top level, so the reader -- which resolves the
# governing k as "largest adequacy-passing k over d['arms']" -- saw a single k and
# silently reported the smallest k as governing. The per-k payloads stay under
# per_k for provenance; nothing is invented at merge time.
base = dict(parts[str(ks[0])])
merged_arms = {}
for k in ks:
    arms = parts[str(k)].get("arms", {})
    if str(k) not in arms:
        print(f"[FAIL] merge: receipt for k={k} does not contain its own arm", file=sys.stderr)
        sys.exit(1)
    merged_arms[str(k)] = arms[str(k)]
base["arms"] = merged_arms
base["per_k"] = parts
base["merged_from"] = [out.with_name(out.stem + f"_k{k}.json").name for k in ks]
out.write_text(json.dumps(base, indent=1))
print(f"[merge] {len(parts)} per-k receipts -> {out} (arms: {sorted(merged_arms)})")
PYMERGE
echo "[DONE] $(date -Is) -> $OUT"
