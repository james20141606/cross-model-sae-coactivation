#!/usr/bin/env python3
"""Score the Lan-pairing arm against the rule pre-registered at 3237ecb, and emit its table.

Written BEFORE any lanpair receipt exists, so the scoring cannot be tuned to a result.

REFUSES PARTIAL INPUT. Three separate refusals, because a reader that scores whatever it
happens to find is how a two-of-three rule silently becomes a one-of-one rule:
  1. all three cells must have a receipt          -> otherwise REFUSED, exit 2
  2. every registered readout key must be present -> otherwise REFUSED, exit 2
  3. the reimplementation validity gate must PASS on every cell -> otherwise BRANCH 5,
     no R_lex is read or printed, exit 3
Only after all three does it apply the pre-registered branches.

The rule, verbatim from the pre-registration (3237ecb), over cells whose ADEQUACY gate
passes, at the largest adequacy-passing k:
  B1 COLLAPSES  R_lex < 0.30 in >= 2 of 3
  B2 SURVIVES   R_lex >= 0.70 in >= 2 of 3 AND pair-survival p at the 1/(n_perm+1) floor
  B3 PARTIAL    otherwise
  B4 NOT-TESTABLE   adequacy gate fails at every k on >= 2 of 3
  B5 NO-COMPARISON  reimplementation validity gate fails

    python3 code/analysis/lanpair_summary.py            # emit + verify + score
    python3 code/analysis/lanpair_summary.py --check     # verify only
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results/cross_model_sae")
OUT = os.path.join(ROOT, "paper/sections/tab_lanpair.tex")

# (display, stem, band, our committed mutual-NN R_lex from tab:lextok)
CELLS = [("Qwen3-1.7B $\\leftrightarrow$ Gemma2-2B", "qwen3-1.7b__gemma2-2b", "strong", 0.933),
         ("Gemma2-9B $\\leftrightarrow$ Mistral-7B", "gemma2-9b__mistral7b", "median", 0.942),
         ("Llama-3.2-1B $\\leftrightarrow$ Mistral-7B", "llama3.2-1b__mistral7b", "weak", 0.918)]
SURVIVE_R, COLLAPSE_R = 0.70, 0.30

# Registered readouts. Absent any one of them the receipt is incomplete and unscoreable.
TOP_KEYS = ["cell", "N_docs", "topk", "n_perm", "n_folds", "seed", "pairing",
            "pairing_prereg_commit", "reimpl_gate", "frozen_pairs", "pairs_mutual",
            "pairs_lan_argmax", "raw", "arms"]
GATE_KEYS = ["lan_rule_quote", "lan_rule_source", "K_A", "K_B", "n_pairs_argmax",
             "n_pairs_mutual", "n_distinct_targets", "frac_one_to_one",
             "one_over_e_reference", "lan_table56_interval", "G1_many_to_one",
             "G2_frac11_in_lan_interval", "G3_mutual_nested", "passed"]
ARM_KEYS = ["adequacy", "pair_survival_real", "pair_survival_shuffled",
            "retention_real", "retention_shuffled", "R_lex", "interpretable"]


def load_or_refuse():
    cells, missing = [], []
    for disp, stem, band, ours in CELLS:
        p = os.path.join(RES, "lanpair_lextok20k_%s.json" % stem)
        if not os.path.exists(p):
            missing.append(os.path.relpath(p, ROOT))
            continue
        cells.append((disp, stem, band, ours, json.load(open(p))))
    if missing:
        print("REFUSED: %d of %d receipts missing -- a partial roster cannot be scored "
              "against a 2-of-3 rule." % (len(missing), len(CELLS)))
        for m in missing:
            print("   missing", m)
        return None
    bad = []
    for disp, stem, band, ours, d in cells:
        for k in TOP_KEYS:
            if k not in d:
                bad.append("%s: missing top-level %r" % (stem, k))
        if d.get("pairing") != "lan_argmax":
            bad.append("%s: pairing=%r, expected 'lan_argmax'" % (stem, d.get("pairing")))
        for k in GATE_KEYS:
            if k not in d.get("reimpl_gate", {}):
                bad.append("%s: missing reimpl_gate.%r" % (stem, k))
        if not d.get("arms"):
            bad.append("%s: no arms" % stem)
        for kk, arm in d.get("arms", {}).items():
            for k in ARM_KEYS:
                if k not in arm:
                    bad.append("%s k=%s: missing arm.%r" % (stem, kk, k))
    if bad:
        print("REFUSED: %d incomplete readout(s)." % len(bad))
        for b in bad:
            print("  ", b)
        return None
    return cells


def governing_arm(d):
    """Largest adequacy-passing k, per the pre-registration."""
    ok = [(int(k), a) for k, a in d["arms"].items() if a["adequacy"]["gate_passed"]]
    return max(ok, key=lambda t: t[0]) if ok else (None, None)


def score(cells):
    rows = []
    for disp, stem, band, ours, d in cells:
        k, arm = governing_arm(d)
        rows.append({"disp": disp, "stem": stem, "band": band, "ours": ours, "d": d,
                     "k": k, "arm": arm, "gate_passed": arm is not None,
                     "R_lex": arm["R_lex"] if arm else float("nan")})
    passing = [r for r in rows if r["gate_passed"]]
    if len(passing) < 2:
        return rows, "B4 NOT-TESTABLE"
    n_s = sum(1 for r in passing if r["R_lex"] >= SURVIVE_R)
    n_c = sum(1 for r in passing if r["R_lex"] < COLLAPSE_R)
    floors = all(r["arm"]["pair_survival_real"]["p"]
                 <= 1.0 / (r["d"]["n_perm"] + 1) + 1e-12 for r in passing)
    if n_c >= 2:
        return rows, "B1 COLLAPSES"
    if n_s >= 2 and floors:
        return rows, "B2 SURVIVES"
    return rows, "B3 PARTIAL"


def emit(rows):
    g0 = rows[0]["d"]["reimpl_gate"]
    L = ["% AUTO-GENERATED by code/analysis/lanpair_summary.py -- DO NOT HAND-EDIT.",
         "% Numbers formatted from results/cross_model_sae/lanpair_lextok20k_*.json and",
         "% verified by a read-back in that script.",
         "\\begin{table}[h]", "\\centering",
         "\\caption{\\textbf{The same control, applied to a published pairing rule.} "
         "\\citet{lan2024universal} pair cross-model SAE features by taking, for each "
         "feature in $A$, its highest-correlated feature in $B$ (their 1A-to-ManyB rule; "
         "many-to-one by construction). We transplant \\emph{that rule} into the "
         "\\Cref{tab:lextok} pipeline and change nothing else --- same $20{,}000$ C4 "
         "documents, same cached pooled features, same TF-IDF$\\to$SVD$_k$ cross-fitted "
         "design, same row-shuffled calibration, same within-fold null, $n_{\\text{perm}}"
         "{=}2{,}000$. \\emph{reimpl.}: the validity gate read \\emph{before} any "
         "$R_{\\text{lex}}$ --- the mapping must be many-to-one and its $1$--$1$ fraction "
         "must fall inside the $[0.159,0.632]$ interval implied by their Tables~5--6; it is "
         "necessary, not sufficient (an arbitrary argmax assignment gives $\\approx 1/e$). "
         "\\emph{ours}: the mutual-NN $R_{\\text{lex}}$ from \\Cref{tab:lextok} on the same "
         "cell. Receipts: \\texttt{lanpair\\_lextok20k\\_*.json}.}",
         "\\label{tab:lanpair}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Cell & band & $k$ & pairs & $1$--$1$ frac & ret.\\ real & ret.\\ shuf & "
         "$R_{\\text{lex}}$ & ours \\\\", "\\midrule"]
    for r in rows:
        g = r["d"]["reimpl_gate"]
        if not r["gate_passed"]:
            L.append("%s & %s & \\multicolumn{7}{c}{\\emph{adequacy gate failed at every "
                     "$k$ --- uninterpretable}} \\\\" % (r["disp"], r["band"]))
            continue
        a = r["arm"]
        L.append("%s & %s & $%d$ & $%d$ & $%.3f$ & $%.3f$ & $%.3f$ & $%.3f$ & $%.3f$ \\\\"
                 % (r["disp"], r["band"], r["k"], g["n_pairs_argmax"],
                    g["frac_one_to_one"], a["retention_real"], a["retention_shuffled"],
                    a["R_lex"], r["ours"]))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(OUT, "w").write("\n".join(L) + "\n")


def verify(rows):
    if not os.path.exists(OUT):
        print("MISSING %s" % OUT)
        return False
    txt = open(OUT).read()
    bad = []
    for r in rows:
        if not r["gate_passed"]:
            continue
        a, g = r["arm"], r["d"]["reimpl_gate"]
        want = ("%s & %s & $%d$ & $%d$ & $%.3f$ & $%.3f$ & $%.3f$ & $%.3f$ & $%.3f$"
                % (r["disp"], r["band"], r["k"], g["n_pairs_argmax"],
                   g["frac_one_to_one"], a["retention_real"], a["retention_shuffled"],
                   a["R_lex"], r["ours"]))
        if want not in txt:
            bad.append("%s: emitted row does not match receipt" % r["stem"])
    for b in bad:
        print("MISMATCH  " + b)
    print("read-back: %d row(s) checked, %d mismatches" % (len(rows), len(bad)))
    return not bad


def main():
    cells = load_or_refuse()
    if cells is None:
        return 2

    print("==== REIMPLEMENTATION VALIDITY GATE (read before any R_lex) ====")
    gate_ok = True
    for disp, stem, band, ours, d in cells:
        g = d["reimpl_gate"]
        print("  %-24s argmax=%d mutual=%d distinct=%d frac_1-1=%.4f (Lan %.3f-%.3f, 1/e=%.4f)"
              % (stem, g["n_pairs_argmax"], g["n_pairs_mutual"], g["n_distinct_targets"],
                 g["frac_one_to_one"], g["lan_table56_interval"][0],
                 g["lan_table56_interval"][1], g["one_over_e_reference"]))
        print("  %-24s G1=%s G2=%s G3=%s -> %s"
              % ("", g["G1_many_to_one"], g["G2_frac11_in_lan_interval"],
                 g["G3_mutual_nested"], "PASS" if g["passed"] else "FAIL"))
        gate_ok &= bool(g["passed"])
    if not gate_ok:
        print("\nBRANCH 5: the reimplementation does not reproduce Lan et al.'s reported "
              "pairing behaviour. NO comparison is licensed and no R_lex is read.")
        return 3
    print("  quote: %r" % cells[0][4]["reimpl_gate"]["lan_rule_quote"])
    print("  source: %s" % cells[0][4]["reimpl_gate"]["lan_rule_source"])
    print("  NOTE: G2 is necessary, NOT sufficient -- 1/e lies inside the Lan interval.\n")

    rows, verdict = score(cells)
    for disp, stem, band, ours, d in cells:
        print("== %s (%s)  N=%d docs, %d frozen argmax pairs, n_perm=%d"
              % (stem, band, d["N_docs"], d["frozen_pairs"]["n"], d["n_perm"]))
        print("   raw pair excess=%+.5f z=%.1f p=%.4f"
              % (d["raw"]["excess"], d["raw"]["z"], d["raw"]["p"]))
        for k in sorted(d["arms"], key=int):
            a = d["arms"][k]
            ga, gb = a["adequacy"]["model_a"], a["adequacy"]["model_b"]
            print("   k=%-5s gate %s  A frac=%.2f dR2=%+.3f | B frac=%.2f dR2=%+.3f"
                  % (k, "PASS" if a["adequacy"]["gate_passed"] else "FAIL",
                     ga["frac_real_beats_shuffled"], ga["median_delta_r2"],
                     gb["frac_real_beats_shuffled"], gb["median_delta_r2"]))
            print("          ret_real=%.3f ret_shuf=%.3f R_lex=%.3f | real p=%.4f z=%.1f"
                  % (a["retention_real"], a["retention_shuffled"], a["R_lex"],
                     a["pair_survival_real"]["p"], a["pair_survival_real"]["z"]))

    if "--check" not in sys.argv:
        emit(rows)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    ok = verify(rows)

    print("\n---- pre-registered rule (3237ecb) ----")
    for r in rows:
        print("  %-24s gate=%s  governing k=%s  R_lex(lan)=%s   vs ours=%.3f"
              % (r["stem"], "PASS" if r["gate_passed"] else "FAIL", r["k"],
                 ("%.3f" % r["R_lex"]) if r["gate_passed"] else "n/a", r["ours"]))
    print("  VERDICT: %s" % verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
