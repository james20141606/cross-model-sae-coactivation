#!/usr/bin/env python3
"""COMMITTED READER for the token-aligned/order-sensitive sharing test.

Applies PREREGISTRATION_token_aligned.md sec.4 VERBATIM to the three cell receipts
(results/cross_model_sae/tokalign20k_{tag}.json) and prints exactly one branch line.
Committed BEFORE any extraction job is submitted (registration df330c6); the receipts it
will read do not exist yet. Do not retune thresholds here: they are duplicated from the
registration so a reader can diff them against it.

Refusal discipline: partial or malformed input is REFUSED (exit 4), never scored -- a
verdict computed from an incomplete battery would be an unregistered analysis.

Exit codes:
  0  branch printed (B1 / B2 / B3 / MIXED / PARTIAL / rung2-not-testable variants)
  2  overall NOT-TESTABLE (gate outcome, not a verdict -- and not a negative)
  4  REFUSED (missing/malformed receipts)
  1  selftest failure

Selftest fires EVERY branch from synthetic receipts: B1, B2, B3, MIXED, PARTIAL,
NOT-TESTABLE (G2), rung2-NOT-TESTABLE, and the refusal path.
"""
import argparse, json, os, sys

# ---- registered constants (PREREGISTRATION_token_aligned.md, commit df330c6) ----
CELLS = ["qwen3-1.7b__gemma2-2b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
S1_MIN_R = 0.05          # rung 1 cell PASS: mean per-pair centered r >= this ...
N_PERM = 2000            # ... AND p at the permutation floor 1/(N_PERM+1)
RTOK_SURVIVE = 0.30      # rung 2 cell SURVIVES: R_tok >= this AND p_N3 at floor
RTOK_COLLAPSE = 0.10     # rung 2 cell COLLAPSES: R_tok < this OR S2 within N3 95% band
G2_MIN_PAIRS = 100       # < this many surviving pairs -> rung NOT-TESTABLE in that cell

FLOOR = 1.0 / (N_PERM + 1)

CONSEQUENCE = {
    "B1": "registered consequence: scope limit lifted one level -- limitations rescoped "
          "from 'lexical-scale' to 'at least context-sensitive at word granularity', "
          "bounded below composition by the bigram caveat; abstract may gain one clause.",
    "B2": "registered consequence: the sharpest negative -- surviving universality is "
          "fully explained by shared token-identity detectors; 'narrowed, not closed' "
          "becomes 'closed (negatively)'; R_lex survival retro-diagnosed as a "
          "nuisance-weakness artifact; abstract states the negative closure.",
    "B3": "registered consequence: published claim unchanged but sharper boundary -- "
          "sharing is document-scale only; the token-detector deflation is itself "
          "refuted (pure token detectors would have been token-aligned); abstract at "
          "most a scope-tightening clause.",
    "MIXED": "registered consequence: per-cell reporting; 'partial' language in "
             "limitations; abstract untouched.",
    "PARTIAL": "registered consequence: per-cell reporting; 'partial' language in "
               "limitations; abstract untouched; S3 (scramble) becomes eligible as a "
               "directional tiebreaker only.",
    "NOT-TESTABLE": "registered consequence: gate outcome reported (appendix); abstract "
                    "untouched; no verdict language anywhere.",
    "R2NT": "registered consequence: S1 result enters per its rung-1 branch; rung-2 gate "
            "outcome reported (appendix); no order-sensitivity verdict language.",
}


def _req(d, path, cell):
    """Fetch a nested key or refuse: a missing field is malformed input, not a zero."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            print(f"REFUSED: receipt for cell {cell} lacks required field '{path}'",
                  file=sys.stderr)
            sys.exit(4)
        cur = cur[k]
    return cur


def _at_floor(p):
    return p <= FLOOR * (1.0 + 1e-9)


def _within_band(obs, ci95):
    return ci95[0] <= obs <= ci95[1]


def eval_cell_rung1(r, cell):
    """-> (status, reason) with status in PASS / WITHIN_BAND / NEITHER / NOT_TESTABLE."""
    if not _req(r, "gates.G0.passed", cell):
        return "NOT_TESTABLE", "G0 failed (frozen-pair recovery)"
    if not _req(r, "gates.G1.passed", cell):
        return "NOT_TESTABLE", "G1 failed (alignment QA)"
    if not _req(r, "gates.G2.s1.testable", cell):
        n = _req(r, "gates.G2.s1.n_pairs_pass", cell)
        return "NOT_TESTABLE", f"G2(S1): only {n} pairs with adequate support (<{G2_MIN_PAIRS})"
    obs = _req(r, "s1.obs_mean_pair_r", cell)
    p = _req(r, "s1.null_n1.p", cell)
    ci = _req(r, "s1.null_n1.ci95", cell)
    if _at_floor(p) and obs >= S1_MIN_R:
        return "PASS", f"obs r={obs:.4f} >= {S1_MIN_R}, p={p:.6f} at floor"
    if _within_band(obs, ci):
        return "WITHIN_BAND", f"obs r={obs:.4f} inside N1 95% band [{ci[0]:.4f},{ci[1]:.4f}]"
    return "NEITHER", f"obs r={obs:.4f}, p={p:.6f} (floor {FLOOR:.6f})"


def eval_cell_rung2(r, cell, rung1_status):
    """-> (status, reason), status in SURVIVES / COLLAPSES / NEITHER / NOT_TESTABLE /
    NOT_ENTERED. Only rung1-PASS cells with a positive N2-direction check enter."""
    if rung1_status != "PASS":
        return "NOT_ENTERED", f"rung1 status {rung1_status} (denominator not licensed)"
    if not _req(r, "gates.G2.s2.testable", cell):
        n = _req(r, "gates.G2.s2.n_pairs_pass", cell)
        return "NOT_TESTABLE", f"G2(S2): only {n} pairs with adequate support (<{G2_MIN_PAIRS})"
    exc_n2 = _req(r, "s1.excess_n2", cell)
    if exc_n2 <= 0:
        return "NOT_ENTERED", (f"N2 direction check failed (S1 excess over N2 = "
                               f"{exc_n2:+.4f} <= 0): position control disagrees with N1")
    rtok = _req(r, "s2.R_tok", cell)
    obs = _req(r, "s2.obs_mean_pair_r", cell)
    p = _req(r, "s2.null_n3.p", cell)
    ci = _req(r, "s2.null_n3.ci95", cell)
    if rtok >= RTOK_SURVIVE and _at_floor(p):
        return "SURVIVES", f"R_tok={rtok:.3f} >= {RTOK_SURVIVE}, p_N3={p:.6f} at floor"
    if rtok < RTOK_COLLAPSE or _within_band(obs, ci):
        why = (f"R_tok={rtok:.3f} < {RTOK_COLLAPSE}" if rtok < RTOK_COLLAPSE
               else f"S2 obs r={obs:.4f} inside N3 95% band")
        return "COLLAPSES", why
    return "NEITHER", f"R_tok={rtok:.3f}, p_N3={p:.6f}"


def evaluate(receipts):
    """receipts: {tag: dict}. Returns (branch_key, branch_line, per_cell, exit_code)."""
    per_cell = {}
    for tag in CELLS:
        r1, why1 = eval_cell_rung1(receipts[tag], tag)
        r2, why2 = eval_cell_rung2(receipts[tag], tag, r1)
        per_cell[tag] = {"rung1": r1, "rung1_reason": why1,
                         "rung2": r2, "rung2_reason": why2}

    r1s = [c["rung1"] for c in per_cell.values()]
    n_nt1 = r1s.count("NOT_TESTABLE")
    if n_nt1 >= 2:
        rung1 = "NOT-TESTABLE"
    elif r1s.count("PASS") >= 2:
        rung1 = "TOKEN-ALIGNED"
    elif r1s.count("WITHIN_BAND") >= 2:
        rung1 = "NOT ALIGNED"
    else:
        rung1 = "MIXED"

    entered = [t for t in CELLS if per_cell[t]["rung2"] in
               ("SURVIVES", "COLLAPSES", "NEITHER")]
    r2s = [per_cell[t]["rung2"] for t in entered]
    n_nt2 = [per_cell[t]["rung2"] for t in CELLS].count("NOT_TESTABLE")
    if n_nt2 >= 2:
        rung2 = "NOT-TESTABLE"
    elif len(entered) < 2:
        rung2 = "NOT-EVALUABLE"
    elif r2s.count("SURVIVES") >= 2:
        rung2 = "ORDER-SENSITIVE"
    elif r2s.count("COLLAPSES") >= 2:
        rung2 = "LEXICALLY EXHAUSTED"
    else:
        rung2 = "PARTIAL"

    if rung1 == "NOT-TESTABLE":
        return ("NOT-TESTABLE",
                "BRANCH: NOT-TESTABLE -- >=2 cells fail a provenance/support gate on "
                "rung 1. " + CONSEQUENCE["NOT-TESTABLE"], per_cell, 2)
    if rung1 == "NOT ALIGNED":
        return ("B3", "BRANCH: B3 NOT ALIGNED -- S1 within the N1 95% band in >=2 of 3 "
                "cells; the sharing is document-scale only. " + CONSEQUENCE["B3"],
                per_cell, 0)
    if rung1 == "MIXED":
        return ("MIXED", "BRANCH: MIXED on rung 1 -- no 2-of-3 majority in either "
                "direction. " + CONSEQUENCE["MIXED"], per_cell, 0)
    # rung1 == TOKEN-ALIGNED
    if rung2 == "NOT-TESTABLE":
        return ("R2NT", "BRANCH: TOKEN-ALIGNED + RUNG2 NOT-TESTABLE -- >=2 cells lack "
                "S2 support (G2). " + CONSEQUENCE["R2NT"], per_cell, 2)
    if rung2 == "NOT-EVALUABLE":
        return ("R2NT", "BRANCH: TOKEN-ALIGNED + RUNG2 NOT-EVALUABLE -- <2 cells enter "
                "rung 2 (rung-1 PASS + N2 direction check). " + CONSEQUENCE["R2NT"],
                per_cell, 2)
    if rung2 == "ORDER-SENSITIVE":
        return ("B1", "BRANCH: B1 TOKEN-ALIGNED + ORDER-SENSITIVE -- S1 beats the "
                "within-document order-shuffle null and >=30% of the aligned excess "
                "survives exact word-identity conditioning. " + CONSEQUENCE["B1"],
                per_cell, 0)
    if rung2 == "LEXICALLY EXHAUSTED":
        return ("B2", "BRANCH: B2 TOKEN-ALIGNED + LEXICALLY EXHAUSTED -- the aligned "
                "co-activation is token-aligned but does NOT survive exact word-identity "
                "conditioning. " + CONSEQUENCE["B2"], per_cell, 0)
    return ("PARTIAL", "BRANCH: TOKEN-ALIGNED + PARTIAL on rung 2 -- no 2-of-3 majority "
            "between survive and collapse. " + CONSEQUENCE["PARTIAL"], per_cell, 0)


def report(receipts, quiet=False):
    branch_key, line, per_cell, code = evaluate(receipts)
    if not quiet:
        print("=" * 78)
        print("tokalign reader -- PREREGISTRATION_token_aligned.md (df330c6) applied verbatim")
        print(f"bars: S1 r>={S1_MIN_R} & p at 1/{N_PERM + 1} floor | R_tok >= {RTOK_SURVIVE} "
              f"survive / < {RTOK_COLLAPSE} collapse | G2 >= {G2_MIN_PAIRS} pairs")
        print("-" * 78)
        for tag in CELLS:
            c = per_cell[tag]
            r = receipts[tag]
            pc = r.get("prereg_commit", "?")
            print(f"[{tag}]  (receipt prereg_commit={pc})")
            print(f"  rung1 {c['rung1']:<12} {c['rung1_reason']}")
            print(f"  rung2 {c['rung2']:<12} {c['rung2_reason']}")
        print("-" * 78)
    print(line)
    return code


def load_receipts(dirpath):
    receipts = {}
    missing = []
    for tag in CELLS:
        p = os.path.join(dirpath, f"tokalign20k_{tag}.json")
        if not os.path.exists(p):
            missing.append(p)
            continue
        try:
            receipts[tag] = json.load(open(p))
        except Exception as e:
            print(f"REFUSED: receipt {p} unreadable ({e})", file=sys.stderr)
            sys.exit(4)
    if missing:
        print("REFUSED: partial input -- a verdict from an incomplete battery would be "
              "an unregistered analysis. Missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(4)
    return receipts


# --------------------------------- selftest ---------------------------------
def _mk(rung1="pass", g0=True, g1=True, g2s1=True, g2s2=True, n2pos=True,
        rtok=0.6, s2mode="floor"):
    """Synthetic receipt builder. rung1: pass|band|neither. s2mode: floor|band|neither."""
    obs1 = {"pass": 0.20, "band": 0.0001, "neither": 0.20}[rung1]
    p1 = {"pass": FLOOR, "band": 0.45, "neither": 0.30}[rung1]
    obs2, p2 = 0.10, FLOOR
    if s2mode == "band":
        obs2, p2 = 0.0001, 0.5
    elif s2mode == "neither":
        obs2, p2 = 0.05, 0.01
    return {
        "prereg_commit": "df330c6",
        "gates": {"G0": {"passed": g0}, "G1": {"passed": g1},
                  "G2": {"s1": {"testable": g2s1, "n_pairs_pass": 700 if g2s1 else 12},
                         "s2": {"testable": g2s2, "n_pairs_pass": 650 if g2s2 else 8}}},
        "s1": {"obs_mean_pair_r": obs1,
               "null_n1": {"p": p1, "ci95": [-0.001, 0.001], "mean": 0.0, "sd": 0.0005},
               "excess_n2": 0.15 if n2pos else -0.002},
        "s2": {"obs_mean_pair_r": obs2, "R_tok": rtok,
               "null_n3": {"p": p2, "ci95": [-0.001, 0.001], "mean": 0.0, "sd": 0.0005}},
    }


def selftest():
    np_ok = []

    def case(name, receipts, want_key, want_code):
        key, line, _, code = evaluate(receipts)
        good = (key == want_key and code == want_code)
        np_ok.append(good)
        print(f"[{'PASS' if good else 'FAIL'}] {name}: branch={key} exit={code} "
              f"(want {want_key}/{want_code})")
        if not good:
            print(f"       line: {line}")

    T = CELLS
    # B1: all three aligned, R_tok healthy, S2 at floor
    case("B1 order-sensitive", {t: _mk() for t in T}, "B1", 0)
    # B2 via R_tok < 0.10 in all cells
    case("B2 collapse (R_tok)", {t: _mk(rtok=0.05, s2mode="floor") for t in T}, "B2", 0)
    # B2 via S2 inside the null band (R_tok nominally high -> band clause must catch it)
    case("B2 collapse (band)", {t: _mk(rtok=0.5, s2mode="band") for t in T}, "B2", 0)
    # B3: S1 inside the N1 band everywhere
    case("B3 not aligned", {t: _mk(rung1="band") for t in T}, "B3", 0)
    # MIXED rung1: one pass, one band, one neither
    case("MIXED rung1", {T[0]: _mk(), T[1]: _mk(rung1="band"),
                         T[2]: _mk(rung1="neither")}, "MIXED", 0)
    # PARTIAL rung2: survive / collapse / neither across the three cells
    case("PARTIAL rung2", {T[0]: _mk(), T[1]: _mk(rtok=0.05),
                           T[2]: _mk(rtok=0.2, s2mode="neither")}, "PARTIAL", 0)
    # NOT-TESTABLE: two cells fail G2(S1) support
    case("NOT-TESTABLE (G2 S1)", {T[0]: _mk(), T[1]: _mk(g2s1=False),
                                  T[2]: _mk(g2s1=False)}, "NOT-TESTABLE", 2)
    # NOT-TESTABLE via G0 failure in two cells (gate, not verdict)
    case("NOT-TESTABLE (G0)", {T[0]: _mk(), T[1]: _mk(g0=False),
                               T[2]: _mk(g0=False)}, "NOT-TESTABLE", 2)
    # rung2 NOT-TESTABLE: aligned but two cells lack S2 support
    case("R2 NOT-TESTABLE", {T[0]: _mk(), T[1]: _mk(g2s2=False),
                             T[2]: _mk(g2s2=False)}, "R2NT", 2)
    # rung2 NOT-EVALUABLE: aligned cells fail the N2 direction check
    case("R2 NOT-EVALUABLE (N2 dir)", {T[0]: _mk(), T[1]: _mk(n2pos=False),
                                       T[2]: _mk(n2pos=False)}, "R2NT", 2)

    # refusal path 1: missing receipt file
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        json.dump(_mk(), open(os.path.join(td, f"tokalign20k_{T[0]}.json"), "w"))
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--dir", td],
                            capture_output=True, text=True).returncode
        good = rc == 4
        np_ok.append(good)
        print(f"[{'PASS' if good else 'FAIL'}] refusal on partial input: exit={rc} (want 4)")
    # refusal path 2: receipt missing a required field
    with tempfile.TemporaryDirectory() as td:
        for t in T:
            r = _mk()
            del r["s1"]["null_n1"]
            json.dump(r, open(os.path.join(td, f"tokalign20k_{t}.json"), "w"))
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--dir", td],
                            capture_output=True, text=True).returncode
        good = rc == 4
        np_ok.append(good)
        print(f"[{'PASS' if good else 'FAIL'}] refusal on malformed receipt: exit={rc} (want 4)")

    print("SELFTEST", "PASS" if all(np_ok) else "FAIL",
          f"({sum(np_ok)}/{len(np_ok)} branches fired correctly)")
    return 0 if all(np_ok) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default="results/cross_model_sae")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    sys.exit(report(load_receipts(args.dir)))


if __name__ == "__main__":
    main()
