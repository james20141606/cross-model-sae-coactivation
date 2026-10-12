#!/usr/bin/env python3
"""Summarise the Phase-2 construct-validity controls and decide depth-profile vs semantic.

Reads results/probe_attribution_controls/*.json (produced by probe_attribution_runner.py)
and results/probe_attribution_v2/*.json (original generator) and emits a decomposition
table + SUMMARY.json. Pure CPU/numpy, no GPU.

Decomposition (per model, pooled over the 6 language pairs):
  real            : headline mean_pairwise_r (validation target).
  layer_shuffle   : permute mid-band layer order of lang B within each concept pair.
                    Collapse => signal lives in the shared per-layer DEPTH profile.
  mismatched_pair : derange the concept mapping (control c). Survival => concept
                    identity is irrelevant => DEPTH-PROFILE ARTIFACT.
  semantic_residual = real - mismatched_pair : the part attributable to MATCHING the
                    same concept across languages (the only genuinely cross-lingual part).
"""
import json, glob, os

CTRL = "results/probe_attribution_controls"
ORIG = "results/probe_attribution_v2"

orig = {}
for fp in glob.glob(f"{ORIG}/*.json"):
    d = json.load(open(fp)); orig[d["model"]] = d["mean_pairwise_r"]

EN_ONLY = lambda m: "pythia" in m.lower()  # Pythia = Pile, English-only
rows = []
for fp in sorted(glob.glob(f"{CTRL}/*.json")):
    if "_smoke" in fp or "SUMMARY" in fp:
        continue
    d = json.load(open(fp))
    m = d["model"]
    real = d["mean_pairwise_r"]
    mm = d["mean_mismatched_pair_r"]
    rows.append(dict(
        model=m, multilingual=not EN_ONLY(m),
        real=real, orig=orig.get(m),
        repro_diff=(real - orig[m]) if m in orig else None,
        layer_shuffle=d["mean_layer_shuffle_r"],
        mismatched_pair=mm,
        full_shuffle=d["mean_full_shuffle_r"],
        semantic_residual=real - mm,
        mismatch_survival_frac=(mm / real) if real else None,
    ))

rows.sort(key=lambda r: -r["real"])
print(f"{'model':32s} {'real':>7s} {'repro':>6s}|{'layerS':>7s} {'mismat':>7s} {'floor':>6s}|{'sem_res':>7s} {'surv%':>5s}  type")
for r in rows:
    print(f"{r['model']:32s} {r['real']:+7.3f} {r['repro_diff']:+6.3f}|"
          f"{r['layer_shuffle']:+7.3f} {r['mismatched_pair']:+7.3f} {r['full_shuffle']:+6.3f}|"
          f"{r['semantic_residual']:+7.3f} {100*r['mismatch_survival_frac']:5.0f}  "
          f"{'multiling' if r['multilingual'] else 'EN-only'}")

mono = [r for r in rows if not r["multilingual"]]
multi = [r for r in rows if r["multilingual"]]
verdict = {
    "reproduction_max_abs_diff": max(abs(r["repro_diff"]) for r in rows),
    "raw_metric_ranks_EN_only_above_all_multilingual":
        max(r["real"] for r in mono) > max(r["real"] for r in multi),
    "mean_mismatch_survival_frac": sum(r["mismatch_survival_frac"] for r in rows) / len(rows),
    "mean_layer_shuffle_r": sum(r["layer_shuffle"] for r in rows) / len(rows),
    "semantic_residual_EN_only_mean": sum(r["semantic_residual"] for r in mono) / len(mono),
    "semantic_residual_multilingual_mean": sum(r["semantic_residual"] for r in multi) / len(multi),
}
print("\n=== VERDICT ===")
for k, v in verdict.items():
    print(f"  {k}: {v}")
print("\n  Raw metric: mismatched-pair survives "
      f"{100*verdict['mean_mismatch_survival_frac']:.0f}% on average and layer-shuffle "
      f"collapses to {verdict['mean_layer_shuffle_r']:+.3f} => DEPTH-PROFILE ARTIFACT dominates.")
print(f"  Genuine cross-lingual (semantic_residual): EN-only "
      f"{verdict['semantic_residual_EN_only_mean']:+.3f} vs multilingual "
      f"{verdict['semantic_residual_multilingual_mean']:+.3f} "
      "=> faint real signal only in multilingual models, but it is a small fraction of r.")

out = {"rows": rows, "verdict": verdict}
json.dump(out, open(f"{CTRL}/SUMMARY.json", "w"), indent=2)
print(f"\nwrote {CTRL}/SUMMARY.json")
