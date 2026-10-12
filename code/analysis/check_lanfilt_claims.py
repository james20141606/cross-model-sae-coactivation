#!/usr/bin/env python3
"""Gate: the Lan-filter follow-up (app:lan_filters, tab:lanfilt, intro scope clause) against its
receipts results/cross_model_sae/lanfilt{T,C}_*__mistral7b.json and the Table 19 receipts.

Every expected string is derived here from the receipts; nothing numeric is typed. Checks:
  (1) tab_lanfilt.tex equals what lanfilt_summary.emit() would write from the receipts
      (so a hand edit or a stale table fails);
  (2) each number the app:lan_filters prose quotes, as an exact token, inside that subsection;
  (3) the pre-registered decisions (D1 mixed / D2 removed / D3 persists, recomputed here from
      the receipts) match the outcome sentences in the prose;
  (4) the intro scopes the pairing-inflation claim to the unfiltered statistic, and quotes the
      filtered share on templates (added 2026-09-25, acceptance pass); the Discussion's protocol
      paragraph quotes the unfiltered share range.

    /usr/bin/python3 code/analysis/check_lanfilt_claims.py              # FINDINGS=<n>, exit 1 on any
    /usr/bin/python3 code/analysis/check_lanfilt_claims.py --selftest   # plants must fire
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lanfilt_summary as S  # noqa: E402

ROOT = HERE.parents[1]
APP = ROOT / "paper/sections/appendix.tex"
INTRO = ROOT / "paper/sections/intro.tex"
DISC = ROOT / "paper/sections/discussion.tex"
TAB = ROOT / "paper/sections/tab_lanfilt.tex"


def expected_table():
    cells = S.load()
    real = S.TAB
    try:
        buf = {}

        class _T:
            def write_text(self, t):
                buf["t"] = t
        S.TAB = _T()
        S.emit(cells)
    finally:
        S.TAB = real
    return buf["t"]


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def expected_prose():
    cells = S.load()
    P = {k: v["filtered"]["P"] for k, v in cells.items()}
    nn = [(p, L) for p in S.NONNULL for L in "TC"]
    T = [(p, "T") for p in S.NONNULL]; C = [(p, "C") for p in S.NONNULL]
    rt = [P[k]["K_f"] / P[k]["pool_A"] for k in T]; rc = [P[k]["K_f"] / P[k]["pool_A"] for k in C]
    s1max = max(cells[k]["filtered"]["S1"]["K_f"] for k in cells)
    g0 = max(max(abs(cells[k]["G0"]["delta_rsa"]), abs(cells[k]["G0"]["delta_svcca"])) for k in cells)
    med = [P[k]["perm_kept_sizes_median"] for k in C]
    permT = [P[k]["rsa_perm"] for k in T]
    drawsT = [round(200 * P[k]["perm_estimable_frac"]) for k in T]
    shareT = [P[k]["rsa_share"] for k in T]
    unf = [S.unfiltered_share(p, "T") for p in S.NONNULL]
    dec = S.decide(cells)
    ncB = P[("gemma2-9b", "C")]["nonconcept_B"]
    ex = {k: S.excess(P[k])[0] for k in nn}
    toks = {
        "retention templates": f"${round(100 * min(rt))}$--${round(100 * max(rt))}\\%$",
        "retention C4": f"${pct(min(rc))}$--${pct(max(rc))}\\%$",
        "Mistral non-concept on C4": f"${ncB}$ of\nMistral's $512$",
        "raw-string max kept": f"more than ${s1max}$ pairs",
        "G0 max delta": f"to within ${g0:.3f}$",
        "C4 perm median kept": f"(median ${min(med):.0f}$--${max(med):.0f}$",
        "templates perm RSA": f"${min(permT):.3f}$--${max(permT):.3f}$",
        "templates estimable draws": f"${min(drawsT)}$--${max(drawsT)}$ of $200$ draws",
        "templates share filtered": f"${pct(min(shareT))}$--${pct(max(shareT))}\\%$",
        "templates share unfiltered": f"instead of ${round(100 * min(unf))}$--${round(100 * max(unf))}\\%$",
        "D2 exc": f"(exc.\\ ${ex[('gemma2-9b', 'T')]:.3f}$ vs.\\ ${ex[('pythia70m', 'T')]:.3f}$)",
        "D3 obs-rand": f"${ex[('qwen3-1.7b', 'C')]:.3f}$ vs.\\ ${ex[('pythia70m', 'C')]:.3f}$",
        "D3 kept": f"only ${P[('qwen3-1.7b', 'C')]['K_f']}$ and ${P[('pythia70m', 'C')]['K_f']}$",
        "D3 band": f"noise band (${dec['D3']['band']:.3f}$)",
        "D3 gap": f"gap (${dec['D3']['diff']:.3f}$)",
        "D3 gemma": f"from both (${ex[('gemma2-9b', 'C')]:.3f}$)",
        "SVCCA rows": "keeps the $21$ pairs SVCCA needs",
    }
    outcome = {
        "D1": {"mixed": "the pairing-step inflation remains on templates, much reduced, but on C4\nthe filters leave too few permuted pairs to measure it",
               "persists": None, "removed": None}[dec["D1"]["decision"]],
        "D2": {"removed": "templated corpus does separate the strongest pair from the\nartifact",
               "persists": None, "reversed": None}[dec["D2"]["decision"]],
        "D3": {"persists": "filtered statistic still does not separate the weak surviving pair from the\nartifact",
               "removed": None}[dec["D3"]["decision"]],
    }
    return toks, outcome, dec


def subsection(app):
    i = app.index("\\label{app:lan_filters}")
    return app[i:app.index("\\input{sections/tab_lanfilt}", i)]


def scan(app, intro, tab, disc=None):
    bad = []
    disc = DISC.read_text() if disc is None else disc
    if tab != expected_table():
        bad.append(("tab_lanfilt.tex", "differs from lanfilt_summary.emit() on the receipts"))
    try:
        sub = subsection(app)
    except ValueError:
        return bad + [("app:lan_filters", "subsection or table input not found")]
    toks, outcome, dec = expected_prose()
    for what, t in toks.items():
        if not re.search(r"(?<![\d.])" + re.escape(t) + r"(?![\d])", sub):
            bad.append((what, t))
    for d, sent in outcome.items():
        if sent is None:
            bad.append((d, f"decision {dec[d]['decision']} has no sentence registered in this checker"))
        elif sent not in sub:
            bad.append((d, sent.replace("\n", " ")))
    if dec["G0_n_fail"] != 0:
        bad.append(("G0", f"{dec['G0_n_fail']} cells fail; prose claims every statistic reproduced"))
    # unfiltered RSA share range (Table 19 receipts), quoted in the intro and the tab:lan caption
    allsh = [S.unfiltered_share(p, L) for p in S.NONNULL for L in "TC"]
    lo, hi = round(100 * min(allsh)), round(100 * max(allsh))
    # 2026-09-25 de-numbering: the intro keeps only the upper end (lo lives in the tab:lan
    # caption checked below) plus a words-only direction qualifier.
    if f"(up to ${hi}\\%$ on templated prompts, less on C4)" not in intro:
        bad.append(("intro RSA share upper end", f"up to {hi}% on templated prompts, less on C4"))
    if re.search(r"\$\d+(\.\d+)?\\%\$", intro.replace(f"${hi}\\%$", "")):
        bad.append(("intro RSA share", "a percentage other than the receipt maximum appears in the intro"))
    if f"manufactures ${lo}$--${hi}\\%$ of the RSA" not in app:
        bad.append(("tab:lan caption RSA share range", f"{lo}--{hi}% of the RSA"))
    if "re-implemented without their\nfilters on three of our pairs" not in intro:
        bad.append(("intro scope", "re-implemented without their filters"))
    # filtered templates share (same token the app:lan_filters prose carries), quoted in the intro
    # 2026-09-25 de-numbering: the filtered-share numbers moved out of the intro; they stay
    # pinned in app:lan_filters via toks["templates share filtered"] above. The intro must
    # keep the qualitative verdict, which is only true while the filtered share is > 0.
    _cells = S.load()
    if not all(_cells[(q, "T")]["filtered"]["P"]["rsa_share"] > 0 for q in S.NONNULL):
        bad.append(("intro filtered share", "receipts no longer support 'does not vanish'"))
    if not re.search(r"that share shrinks on templates\s+but does not vanish", intro):
        bad.append(("intro filtered share", "that share shrinks on templates but does not vanish"))
    if f"absorbs\nup to ${hi}\\%$ of the unfiltered" not in disc:
        bad.append(("discussion unfiltered share max", f"absorbs up to {hi}% of the unfiltered"))
    if "\\Cref{tab:lan}\nis the statistic without them" not in app:
        bad.append(("app:lan scope", "Table 19 labelled as the unfiltered statistic"))
    return bad


def main():
    bad = scan(APP.read_text(), INTRO.read_text(), TAB.read_text())
    for what, v in bad:
        print(f"MISMATCH {what}: expected {v}")
    print(f"FINDINGS={len(bad)}")
    print("check_lanfilt_claims:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def selftest():
    a, i, t = APP.read_text(), INTRO.read_text(), TAB.read_text()
    checks = [("clean pass", not scan(a, i, t))]
    checks.append(("table value plant 0.402->0.403", bool(scan(a, i, t.replace("$0.402$", "$0.403$", 1)))))
    checks.append(("table swap plant (kept 86<->149)",
                   bool(scan(a, i, t.replace("& $86$ &", "& $X$ &").replace("& $149$ &", "& $86$ &")
                             .replace("& $X$ &", "& $149$ &")))))
    checks.append(("prose plant 0.370->0.371", bool(scan(a.replace("$0.370$ vs", "$0.371$ vs", 1), i, t))))
    checks.append(("prose substring plant 0.164->0.1642",
                   bool(scan(a.replace("($0.164$)", "($0.1642$)", 1), i, t))))
    checks.append(("prose swap plant (0.324/0.174)",
                   bool(scan(a.replace("$0.324$ vs.\\ $0.174$", "$0.174$ vs.\\ $0.324$", 1), i, t))))
    checks.append(("outcome plant D3 'still does not'->'now does'",
                   bool(scan(a.replace("filtered statistic still does not separate", "filtered statistic now does separate", 1), i, t))))
    checks.append(("intro plant (upper end 94->93)",
                   bool(scan(a, i.replace("(up to $94\\%$ on templated", "(up to $93\\%$ on templated", 1), t))))
    checks.append(("intro plant (stray old number 15%)",
                   bool(scan(a, i.replace("less on C4)", "$15\\%$ on C4)", 1), t))))
    checks.append(("D3 gap plant 0.150->0.15",
                   bool(scan(a.replace("gap ($0.150$)", "gap ($0.15$)", 1), i, t))))
    checks.append(("appendix plant (filtered share token 7.5->7.6)",
                   bool(scan(a.replace("$7.5$--$14.1\\%$", "$7.6$--$14.1\\%$", 1), i, t))))
    checks.append(("intro plant (filtered share verdict dropped)",
                   bool(scan(a, i.replace("but does not vanish", "and vanishes", 1), t))))
    checks.append(("discussion plant (unfiltered 94->93)",
                   bool(scan(a, i, t, disc=DISC.read_text().replace("up to $94\\%$", "up to $93\\%$", 1)))))
    checks.append(("intro plant (scope removed)",
                   bool(scan(a, i.replace("re-implemented without their\nfilters on", "re-implemented on", 1), t))))
    for name, got in checks:
        print(f"  {name:46s} {'OK' if got else 'FAILED'}")
    ok = all(g for _, g in checks)
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
