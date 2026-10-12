#!/bin/bash
# Run every receipt-only check in this package (no GPU, no network). One line per check with its
# exit code; exits 1 if any check fails. PY defaults to python3; RUN_SLOW=1 adds probe_residual_stats.
#
#   bash tools/run_offline_checks.sh
#   PY=/path/to/python RUN_SLOW=1 bash tools/run_offline_checks.sh
cd "$(dirname "$0")/.." || exit 2
PY=${PY:-python3}
LOG=${LOG_DIR:-$(mktemp -d)}; mkdir -p "$LOG"
fail=0
run () {   # run <label> <cmd...>
  local label=$1; shift
  "$@" > "$LOG/$label.log" 2>&1; local rc=$?
  printf 'rc=%d  %-44s %s\n' "$rc" "$label" "$(grep -v '^[[:space:]]*$' "$LOG/$label.log" | tail -1 | cut -c1-110)"
  [ $rc -eq 0 ] || fail=1
}
# Regenerators overwrite a shipped receipt: afterwards the receipt must equal the shipped bytes or,
# failing that, the shipped JSON value (float formatting aside).
regen () {   # regen <label> <receipt> <cmd...>
  local label=$1 rec=$2; shift 2
  cp "$rec" "$LOG/$label.shipped.json"
  run "$label" "$@"
  if cmp -s "$rec" "$LOG/$label.shipped.json"; then
    printf 'rc=0  %-44s %s\n' "$label (compare)" "regenerated $rec is byte-identical to the shipped file"
  elif "$PY" -c "import json,sys; sys.exit(json.load(open(sys.argv[1]))!=json.load(open(sys.argv[2])))" "$rec" "$LOG/$label.shipped.json"; then
    printf 'rc=0  %-44s %s\n' "$label (compare)" "regenerated $rec equals the shipped JSON value"
  else
    printf 'rc=1  %-44s %s\n' "$label (compare)" "regenerated $rec DIFFERS from the shipped file"; fail=1
  fi
  cp "$LOG/$label.shipped.json" "$rec"   # leave the package as shipped
}
A=code/analysis
run verify_package             "$PY" tools/verify_package.py
run reproduce_main_table       "$PY" tools/reproduce_main_table.py
run check_ladder_table         "$PY" $A/check_ladder_table.py
run check_ladder_table_selftest "$PY" $A/check_ladder_table.py --selftest
run check_c4_table             "$PY" $A/check_c4_table.py
run emit_covctl_table_check    "$PY" $A/emit_covctl_table.py --check
run lextok_summary_check       "$PY" $A/lextok_summary.py --check
run lanpair_summary_check      "$PY" $A/lanpair_summary.py --check
run procrustes_summary_check   "$PY" $A/procrustes_summary.py --check
run tokalign_summary           "$PY" $A/tokalign_summary.py
run tokalign_clean_summary     "$PY" $A/tokalign_clean_summary.py
run check_tokalign_claims      "$PY" $A/check_tokalign_claims.py
run emit_tokalign_table_selftest "$PY" $A/emit_tokalign_table.py --selftest
run covctl_summary             "$PY" $A/covctl_summary.py
run check_lanfilt_claims       "$PY" $A/check_lanfilt_claims.py
run check_gemmahook_claims     "$PY" $A/check_gemmahook_claims.py
run check_main_numbers        "$PY" $A/check_main_numbers.py
run check_main_numbers_selftest "$PY" $A/check_main_numbers.py --selftest
run emit_ninecell_table_check  "$PY" $A/emit_ninecell_table.py --check
if command -v pdftotext >/dev/null; then
  run check_lextok_claims      "$PY" $A/check_lextok_claims.py
  run check_lexical_claims     "$PY" $A/check_lexical_claims.py
  run check_body_pages         "$PY" $A/check_body_pages.py
else
  echo "skip  check_lextok_claims / check_lexical_claims / check_body_pages (pdftotext not found)"
fi
regen rosetta_sublinear_check  results/rosetta_check/rosetta_sublinear_v0.json      "$PY" $A/rosetta_sublinear_check.py
regen rosetta_slope_bootstrap  results/rosetta_check/rosetta_slope_bootstrap.json   "$PY" $A/rosetta_slope_bootstrap.py
regen interp_overlap_null      results/cross_model_sae/INTERP_OVERLAP_NULL.json     "$PY" $A/interp_overlap_null.py
if [ "${RUN_SLOW:-0}" = 1 ]; then
  regen probe_residual_stats   results/probe_attribution_controls_n200/STATS.json  "$PY" $A/probe_residual_stats.py
fi
run verify_package_after      "$PY" tools/verify_package.py   # no check altered the package
echo "logs: $LOG"
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
