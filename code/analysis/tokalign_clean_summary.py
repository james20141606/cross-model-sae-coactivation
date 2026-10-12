#!/usr/bin/env python3
"""COMMITTED READER for the selection-clean rerun (AMENDMENT A3-EXEC, b0b07cd).

Wraps tokalign_summary's registered bars (unchanged: S1 r>=0.05 at the 1/2001 floor;
R_tok_clean >= 0.30 + floor survive / < 0.10 or in-band collapse; G2 >= 100 pairs) and
applies A3-EXEC E5's branch rules to the three clean receipts
(results/cross_model_sae/tokaclean20k_{tag}.json). Committed BEFORE the battery is
submitted; the receipts it will read do not exist yet.

REFUSAL discipline (exit 4): partial input, malformed receipts, and -- new here -- any
receipt that does not VISIBLY carry the four A3 requirements:
  (a) split block with seed 42 + both split sha256s,
  (b) joint-FE convergence report with max dev < tol on both observed sides,
  (c) null_reresidualized == True,
  (d) common_mask with n_pairs_scored + sha256 and common_mask_asserted marker.
A receipt without these is not a clean receipt and must never be scored.

Exit codes: 0 branch printed (B1-clean / B2-clean / B3-clean intermediate);
2 B3-clean via NOT-TESTABLE gates; 4 REFUSED; 1 selftest failure.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokalign_summary import (CELLS, FLOOR, G2_MIN_PAIRS, N_PERM, RTOK_COLLAPSE,
                              RTOK_SURVIVE, S1_MIN_R, _at_floor, _within_band)

A3EXEC = "b0b07cd"
SPLIT_SEED = 42

CONSEQUENCE = {
    "B1-clean": "registered consequence (A3-EXEC E5): the word-identity claim is EARNED "
                "under the leakage-clean design -- the paper's 'unresolved' clauses are "
                "replaced by the earned claim, citing the clean design.",
    "B2-clean": "registered consequence (A3-EXEC E5): the sharing IS reducible to word "
                "identity + document means -- reported as the sharpest negative, and "
                "the paper says so.",
    "B3-clean": "registered consequence (A3-EXEC E5): intermediate/NOT-TESTABLE -- the "
                "'unresolved' language STAYS; clean-rerun outcome reported per cell in "
                "the appendix; no verdict language beyond what the receipts license.",
}


def _refuse(msg):
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(4)


def _req(d, path, cell):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            _refuse(f"receipt for cell {cell} lacks required field '{path}'")
        cur = cur[k]
    return cur


REGISTERED_SPLIT_A = "4ce0138bdd8888e0640d7429927591bb9bbdfc35ea37aa0a02042f68f781a0ae"
REGISTERED_SPLIT_B = "e381de51b90e5f876ba6778f0e2303b36bdc4c3a0fbacf98bc5fa9f4f3d770ac"


def check_a3_requirements(r, cell):
    """Refuse any receipt that does not visibly enforce the four A3 requirements.
    AMENDMENT A4(c): hardened fail-closed -- smoke receipts, wrong n_perm, split
    hashes differing from the REGISTERED pair, and missing A4 provenance all refuse."""
    if r.get("smoke"):
        _refuse(f"cell {cell}: SMOKE receipt -- the reader scores full batteries only")
    if _req(r, "n_perm", cell) != 2000:
        _refuse(f"cell {cell}: n_perm != 2000 (registered)")
    if _req(r, "split.seed", cell) != SPLIT_SEED:
        _refuse(f"cell {cell}: split.seed != {SPLIT_SEED} (requirement a)")
    if _req(r, "split.sha256_a", cell) != REGISTERED_SPLIT_A or \
            _req(r, "split.sha256_b", cell) != REGISTERED_SPLIT_B:
        _refuse(f"cell {cell}: split hashes differ from the REGISTERED seed-42 halves "
                f"(requirement a; A4(c))")
    for f in ("a4_requirements.token_stream_gated_to_b",
              "a4_requirements.universe_screened_on_a_only",
              "a4_requirements.aliasing_audited"):
        if _req(r, f, cell) is not True:
            _refuse(f"cell {cell}: {f} not enforced (A4)")
    _req(r, "aliasing_audit.radix", cell)
    _req(r, "selection.universe_k", cell)
    if not r.get("smoke") and (_req(r, "split.n_a", cell) != 10000
                               or _req(r, "split.n_b", cell) != 10000):
        _refuse(f"cell {cell}: split is not the registered 50/50 of 20000 "
                f"(requirement a)")
    fe = _req(r, "s2_clean.fe_convergence", cell)
    tol = fe.get("tol")
    if tol is None or tol > 1e-8:
        _refuse(f"cell {cell}: FE tolerance missing or > 1e-8 (requirement b)")
    for k in ("obs_max_dev_a", "obs_max_dev_b"):
        if k not in fe or not (fe[k] < tol):
            _refuse(f"cell {cell}: joint FE not converged ({k}) (requirement b)")
    if _req(r, "s2_clean.null_reresidualized", cell) is not True:
        _refuse(f"cell {cell}: null draws not re-residualized (requirement c)")
    cm = _req(r, "common_mask", cell)
    if not isinstance(cm.get("sha256"), str) or len(cm["sha256"]) != 64 \
            or "n_pairs_scored" not in cm:
        _refuse(f"cell {cell}: common_mask sha256/n_pairs_scored missing "
                f"(requirement d)")
    req = _req(r, "a3_requirements", cell)
    for k in ("disjoint_selection_eval", "joint_fe_converged",
              "null_reresidualized", "common_mask_asserted"):
        if req.get(k) is not True:
            _refuse(f"cell {cell}: a3_requirements.{k} is not True")


def eval_cell_rung1(r, cell):
    if not _req(r, "gates.G0.passed", cell):
        return "NOT_TESTABLE", "G0 failed (selected-pair recovery)"
    if not _req(r, "gates.G1.passed", cell):
        return "NOT_TESTABLE", "G1 failed (alignment QA)"
    if not _req(r, "gates.G2.s1.testable", cell):
        n = _req(r, "gates.G2.s1.n_pairs_pass", cell)
        return "NOT_TESTABLE", (f"G2(S1): only {n} pairs with adequate support "
                                f"(<{G2_MIN_PAIRS})")
    obs = _req(r, "s1.obs_mean_pair_r", cell)
    p = _req(r, "s1.null_n1.p", cell)
    ci = _req(r, "s1.null_n1.ci95", cell)
    if _at_floor(p) and obs >= S1_MIN_R:
        return "PASS", f"obs r={obs:.4f} >= {S1_MIN_R}, p={p:.6f} at floor"
    if _within_band(obs, ci):
        return "WITHIN_BAND", (f"obs r={obs:.4f} inside N1 95% band "
                               f"[{ci[0]:.4f},{ci[1]:.4f}]")
    return "NEITHER", f"obs r={obs:.4f}, p={p:.6f} (floor {FLOOR:.6f})"


def eval_cell_rung2(r, cell, rung1_status):
    if rung1_status != "PASS":
        return "NOT_ENTERED", f"rung1 status {rung1_status}"
    if not _req(r, "gates.G2.s2.testable", cell):
        n = _req(r, "gates.G2.s2.n_pairs_pass", cell)
        return "NOT_TESTABLE", (f"G2(S2): only {n} pairs with adequate support "
                                f"(<{G2_MIN_PAIRS})")
    exc_n2 = _req(r, "s1.excess_n2", cell)
    if exc_n2 <= 0:
        return "NOT_ENTERED", (f"N2 direction check failed (S1 excess over N2 = "
                               f"{exc_n2:+.4f} <= 0)")
    rtok = _req(r, "s2_clean.R_tok_clean", cell)
    obs = _req(r, "s2_clean.obs_mean_pair_r", cell)
    p = _req(r, "s2_clean.null_n3.p", cell)
    ci = _req(r, "s2_clean.null_n3.ci95", cell)
    if rtok != rtok:                                    # nan -> no positive numerator
        return "COLLAPSES", "R_tok_clean undefined (denominator excess <= 0)"
    if rtok >= RTOK_SURVIVE and _at_floor(p):
        return "SURVIVES", f"R_tok_clean={rtok:.3f} >= {RTOK_SURVIVE}, p_N3={p:.6f} at floor"
    if rtok < RTOK_COLLAPSE or _within_band(obs, ci):
        why = (f"R_tok_clean={rtok:.3f} < {RTOK_COLLAPSE}" if rtok < RTOK_COLLAPSE
               else f"S2-clean obs r={obs:.4f} inside N3 95% band")
        return "COLLAPSES", why
    return "NEITHER", f"R_tok_clean={rtok:.3f}, p_N3={p:.6f}"


def evaluate(receipts):
    per_cell = {}
    for tag in CELLS:
        r1, why1 = eval_cell_rung1(receipts[tag], tag)
        r2, why2 = eval_cell_rung2(receipts[tag], tag, r1)
        per_cell[tag] = {"rung1": r1, "rung1_reason": why1,
                         "rung2": r2, "rung2_reason": why2}
    n_earn = sum(1 for c in per_cell.values()
                 if c["rung1"] == "PASS" and c["rung2"] == "SURVIVES")
    n_neg = sum(1 for c in per_cell.values()
                if c["rung1"] == "PASS" and c["rung2"] == "COLLAPSES")
    nt1 = sum(1 for c in per_cell.values() if c["rung1"] == "NOT_TESTABLE")
    nt2 = sum(1 for c in per_cell.values() if c["rung2"] == "NOT_TESTABLE")
    if n_earn >= 2:
        return ("B1-clean", "BRANCH: B1-clean -- S1 passes on split B AND the joint-FE "
                "S2 survives (R_tok_clean >= 0.30 at the N3 floor) in >=2 of 3 cells "
                "under the leakage-clean design. " + CONSEQUENCE["B1-clean"],
                per_cell, 0)
    if n_neg >= 2:
        return ("B2-clean", "BRANCH: B2-clean -- the joint-FE S2 collapses in >=2 of 3 "
                "cells under the leakage-clean design: the sharing IS reducible to word "
                "identity + document means. " + CONSEQUENCE["B2-clean"], per_cell, 0)
    if nt1 >= 2 or nt2 >= 2:
        return ("B3-clean", "BRANCH: B3-clean (NOT-TESTABLE) -- >=2 cells fail a "
                "provenance/support gate (original G2 rules). "
                + CONSEQUENCE["B3-clean"], per_cell, 2)
    return ("B3-clean", "BRANCH: B3-clean (intermediate) -- no 2-of-3 majority for "
            "either survive or collapse. " + CONSEQUENCE["B3-clean"], per_cell, 0)


def report(receipts):
    for tag in CELLS:
        check_a3_requirements(receipts[tag], tag)
    key, line, per_cell, code = evaluate(receipts)
    print("=" * 78)
    print("tokalign CLEAN reader -- AMENDMENT A3-EXEC (b0b07cd) applied verbatim")
    print(f"bars (unchanged): S1 r>={S1_MIN_R} & p at 1/{N_PERM + 1} floor | "
          f"R_tok_clean >= {RTOK_SURVIVE} survive / < {RTOK_COLLAPSE} collapse | "
          f"G2 >= {G2_MIN_PAIRS} pairs")
    print("-" * 78)
    for tag in CELLS:
        c = per_cell[tag]
        r = receipts[tag]
        cm = r.get("common_mask", {})
        print(f"[{tag}]  (a3_exec={r.get('a3_exec_commit', '?')} "
              f"n_scored={cm.get('n_pairs_scored', '?')} "
              f"mask={str(cm.get('sha256', '?'))[:12]})")
        print(f"  rung1 {c['rung1']:<12} {c['rung1_reason']}")
        print(f"  rung2 {c['rung2']:<12} {c['rung2_reason']}")
    print("-" * 78)
    print(line)
    return code

    masks = [c.get("common_mask", {}).get("sha256") for c in receipts.values()]
    if len(masks) != len(set(masks)):
        _refuse("duplicate common_mask.sha256 across cells (A4(c): each cell must "
                "carry its own scored-pair mask)")



def load_receipts(dirpath):
    receipts = {}
    missing = []
    for tag in CELLS:
        p = os.path.join(dirpath, f"tokaclean20k_{tag}.json")
        if not os.path.exists(p):
            missing.append(p)
            continue
        try:
            receipts[tag] = json.load(open(p))
        except Exception as e:
            _refuse(f"receipt {p} unreadable ({e})")
    if missing:
        print("REFUSED: partial input -- a verdict from an incomplete battery would "
              "be an unregistered analysis. Missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(4)
    return receipts


# --------------------------------- selftest ---------------------------------
def _mk(rung1="pass", g0=True, g2s2=True, rtok=0.8, s2mode="floor", smoke=False,
        drop_marker=None):
    obs1 = {"pass": 0.20, "band": 0.0001, "neither": 0.20}[rung1]
    p1 = {"pass": FLOOR, "band": 0.45, "neither": 0.30}[rung1]
    obs2, p2 = 0.10, FLOOR
    if s2mode == "band":
        obs2, p2 = 0.0001, 0.5
    elif s2mode == "neither":
        obs2, p2 = 0.05, 0.01
    r = {"prereg_commit": "df330c6", "a3_exec_commit": A3EXEC, "smoke": smoke,
         "split": {"seed": 42, "n_a": 10000, "n_b": 10000,
                   "sha256_a": "a" * 64, "sha256_b": "b" * 64},
         "gates": {"G0": {"passed": g0}, "G1": {"passed": True},
                   "G2": {"s1": {"testable": True, "n_pairs_pass": 600},
                          "s2": {"testable": g2s2,
                                 "n_pairs_pass": 550 if g2s2 else 9}}},
         "s1": {"obs_mean_pair_r": obs1,
                "null_n1": {"p": p1, "ci95": [-0.001, 0.001], "mean": 0.0,
                            "sd": 0.0005},
                "excess_n2": 0.15},
         "s2_clean": {"obs_mean_pair_r": obs2, "R_tok_clean": rtok,
                      "null_n3": {"p": p2, "ci95": [-0.001, 0.001], "mean": 0.0,
                                  "sd": 0.0005},
                      "null_reresidualized": True,
                      "fe_convergence": {"tol": 1e-8, "max_sweeps": 500,
                                         "obs_max_dev_a": 3e-9,
                                         "obs_max_dev_b": 5e-9}},
         "common_mask": {"n_pairs_scored": 550, "sha256": "c" * 64},
         "a3_requirements": {"disjoint_selection_eval": True,
                             "joint_fe_converged": True,
                             "null_reresidualized": True,
                             "common_mask_asserted": True}}
    if drop_marker:
        cur = r
        keys = drop_marker.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        del cur[keys[-1]]
    return r


def selftest():
    import subprocess, tempfile
    ok = []
    T = CELLS

    def case(name, receipts, want_key, want_code):
        key, line, _, code = evaluate(receipts)
        good = (key == want_key and code == want_code)
        ok.append(good)
        print(f"[{'PASS' if good else 'FAIL'}] {name}: branch={key} exit={code} "
              f"(want {want_key}/{want_code})")

    case("B1-clean earned", {t: _mk() for t in T}, "B1-clean", 0)
    case("B2-clean collapse (R_tok)", {t: _mk(rtok=0.05) for t in T}, "B2-clean", 0)
    case("B2-clean collapse (band)", {t: _mk(rtok=0.5, s2mode="band") for t in T},
         "B2-clean", 0)
    case("B3-clean intermediate", {T[0]: _mk(), T[1]: _mk(rtok=0.05),
                                   T[2]: _mk(rtok=0.2, s2mode="neither")},
         "B3-clean", 0)
    case("B3-clean via S1 band", {t: _mk(rung1="band") for t in T}, "B3-clean", 0)
    case("B3-clean NOT-TESTABLE (G2 S2)", {T[0]: _mk(), T[1]: _mk(g2s2=False),
                                           T[2]: _mk(g2s2=False)}, "B3-clean", 2)
    case("B3-clean NOT-TESTABLE (G0)", {T[0]: _mk(), T[1]: _mk(g0=False),
                                        T[2]: _mk(g0=False)}, "B3-clean", 2)

    def run_dir(builder):
        with tempfile.TemporaryDirectory() as td:
            for i, t in enumerate(T):
                json.dump(builder(i), open(
                    os.path.join(td, f"tokaclean20k_{t}.json"), "w"))
            return subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--dir", td],
                capture_output=True, text=True).returncode

    with tempfile.TemporaryDirectory() as td:
        json.dump(_mk(), open(os.path.join(td, f"tokaclean20k_{T[0]}.json"), "w"))
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--dir", td],
                            capture_output=True, text=True).returncode
        good = rc == 4
        ok.append(good)
        print(f"[{'PASS' if good else 'FAIL'}] refusal on partial input: exit={rc} "
              f"(want 4)")
    for marker, label in (
            ("split.sha256_b", "missing split hash (req a)"),
            ("s2_clean.fe_convergence.obs_max_dev_b", "missing FE convergence (req b)"),
            ("s2_clean.null_reresidualized", "missing re-resid marker (req c)"),
            ("common_mask.sha256", "missing mask hash (req d)"),
            ("a3_requirements.common_mask_asserted", "missing a3 marker")):
        rc = run_dir(lambda i, m=marker: _mk(drop_marker=m))
        good = rc == 4
        ok.append(good)
        print(f"[{'PASS' if good else 'FAIL'}] refusal on {label}: exit={rc} (want 4)")

    print("SELFTEST", "PASS" if all(ok) else "FAIL",
          f"({sum(ok)}/{len(ok)} branches fired correctly)")
    return 0 if all(ok) else 1


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
