#!/usr/bin/env python3
"""Gate: the Gemma Scope read-point sensitivity analysis (app:gemmahook, tab:gemmahook, the roster's
read-point/L0 columns, threats (iv), and the main-text qualifier) against its receipts.

Receipts (prereg notes/PREREG_gemma_hook_sensitivity_20260924.md):
  results/cross_model_sae/gemma_hook_sensitivity/GEMMA_HOOK_SENSITIVITY.json        local MPS run
  results/cross_model_sae/gemma_hook_sensitivity/cluster/GEMMA_HOOK_SENSITIVITY.json       cluster 18441889
  results/cross_model_sae/gemma_hook_sensitivity/cluster_q8g2/GEMMA_HOOK_SENSITIVITY.json  cluster 18445512
  results/cross_model_sae/gemma_hook_sensitivity/GEMMA_HOOK_COMBINED.json           gemmahook_summary.py
  results/cross_model_sae/gemma_hook_sensitivity/interp_explore_gemma2-9b__mistral7b.json
  results/cross_model_sae/tokalign_extract/*/extract_receipt.json                   roster L0 (BOS incl.)

Every expected string is derived here from the receipts; nothing numeric is typed. Checks:
  (0) the receipts hash to the pinned sha256 (a re-run or hand edit of a receipt fails loudly);
  (1) tab_gemmahook.tex and GEMMA_HOOK_COMBINED.json equal what gemmahook_summary would write;
  (2) every number the app:gemmahook prose quotes, as an exact token, inside that section;
  (3) the registered outcome and the per-cell status (evaluated / G0-blocked / not run) match the
      sentences; the "Q8 not run" fallback block is present iff the Q8 cells did not run;
  (4) roster: Gemma read point, L0 column from the extraction receipts, no-BOS L0 in the caption;
  (5) threats (iv) and the main-text qualifier point at app:gemmahook.

    /usr/bin/python3 code/analysis/check_gemmahook_claims.py              # FINDINGS=<n>, exit 1 on any
    /usr/bin/python3 code/analysis/check_gemmahook_claims.py --selftest   # plants must fire
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemmahook_summary as S  # noqa: E402

ROOT = HERE.parents[1]
APP = ROOT / "paper/sections/appendix.tex"
RES = ROOT / "paper/sections/results.tex"
ROSTER = ROOT / "paper/sections/tab_sae_roster.tex"
TAB = S.TAB
INTERP = S.DIR / "interp_explore_gemma2-9b__mistral7b.json"
TOKX = ROOT / "results/cross_model_sae/tokalign_extract"

# (0) pins. Update only together with an ELN entry that says why the receipt changed.
PINS = {
    "GEMMA_HOOK_SENSITIVITY.json": "7fe49a53ffd44ec916d982b26c2f9b2a171f4955a3b18f768031f64db29847ba",
    "cluster/GEMMA_HOOK_SENSITIVITY.json": "de4e6a827438c349559e96212f4c6141912b6c86bef2822e4dc2f30ce7422e86",
    "cluster_q8g2/GEMMA_HOOK_SENSITIVITY.json": "01d6aee383eca3f77c82c75aa23baee8499f30c6e57837b81e44f7124d481883",
    "interp_explore_gemma2-9b__mistral7b.json": "04b80353ed4c99a3c0f163e1161781edbdc26385db4d76b4c10c66924a39d6d7",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_table(C):
    real = S.TAB
    buf = {}

    class _T:
        def write_text(self, t):
            buf["t"] = t
    try:
        S.TAB = _T()
        S.emit(C)
    finally:
        S.TAB = real
    return buf["t"]


def section(app):
    i = app.index("\\label{app:gemmahook}")
    return app[i:app.index("\\input{sections/tab_gemmahook}", i)]


def f3(x):
    return f"{x:.3f}"


def sgn3(x):
    return f"{x:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000")


def expected_prose(C):
    toks = {}
    H = {k: v.get("local") or v.get("cluster") for k, v in C["H"].items()}
    for key, name in (("gemma2-2b:16k", "2B-16k"), ("gemma2-2b:65k", "2B-65k"), ("gemma2-9b:16k", "9B-16k")):
        h = H[key]
        toks[f"FVU {name}"] = f"${f3(h['N']['FVU_noBOS'])}\\to{f3(h['N1']['FVU_noBOS'])}$ for {name}"
        toks[f"L0 {name}"] = (f"${h['N']['L0_noBOS']:.1f}\\to{h['N1']['L0_noBOS']:.1f}$ vs.\\ "
                              f"${h['nominal_L0']}$")
    cells = C["cells"]
    ev = C["blind_cells_evaluated"]
    dx = [cells[t]["decision"]["d_excess"] for t in ev]
    dr = [cells[t]["decision"]["d_r"] for t in ev]
    # a range over the evaluated blind cells (min to max of the rounded shifts)
    toks["blind d_excess"] = f"by ${sgn3(min(dx))}$ to ${sgn3(max(dx))}$ and"
    toks["blind d_r"] = f"retained fraction by ${sgn3(min(dr))}$ to ${sgn3(max(dr))}$, inside"
    for t in C["blind_cells_blocked_by_gate"]:
        g = cells[t]["G0"]
        toks[f"G0 dM {t}"] = f"($\\Delta M={g['dM']}$ against a tolerance of $2$"
        toks[f"G0 dexc {t}"] = f"reproduced within ${abs(g['d_excess']):.3f}$"
    nb = cells["gemma2-9b__mistral7b"]["decision"]
    nbr = cells["gemma2-9b__mistral7b"]["replication"]["decision"]
    toks["non-blind"] = (f"(${f3(nb['excess_N'])}\\to{f3(nb['excess_N1'])}$, retained fraction\n"
                         f"${f3(nb['r_N'])}\\to{f3(nb['r_N1'])}$; cluster replication "
                         f"${f3(nbr['excess_N1'])}$ and ${f3(nbr['r_N1'])}$)")
    for t in C["blind_cells_blocked_by_gate"]:
        rp = cells[t].get("replication")
        if rp and not rp["decision"]["blocked_by_gate"] and rp["decision"]["retained"]:
            d, g = rp["decision"], rp["G0"]
            toks[f"replication {t}"] = (f"passes G0 ($\\Delta M={g['dM']}$) and is\nretained "
                                        f"(${f3(d['excess_N'])}\\to{f3(d['excess_N1'])}$, retained fraction "
                                        f"${f3(d['r_N'])}\\to{f3(d['r_N1'])}$)")
    g2m = cells["gemma2-2b__mistral7b"]["decision"]
    q2 = cells["qwen3-1.7b__gemma2-2b"]["decision"]
    toks["screen z G2-M"] = f"${g2m['z_N']:.1f}\\to{g2m['z_N1']:.1f}$"
    toks["screen z Q1.7-G2"] = f"${q2['z_N']:.1f}\\to{q2['z_N1']:.1f}$"
    w = C["screen_only"]["w65k_gemma2-2b__mistral7b"]
    toks["width N+1"] = f"from $z={g2m['z_N1']:.1f}$ to ${w['z_N1']:.1f}$"
    toks["width N"] = f"against ${g2m['z_N']:.1f}$ to ${w['z_N']:.1f}$"
    I = json.load(open(INTERP))
    toks["interp overlap"] = f"at ${I['off1']['mean_top_doc_overlap']:.2f}/5$"
    return toks


def expected_status(C):
    """Sentences that must / must not be present, by registered outcome and per-cell status."""
    must, must_not = [], []
    q8_not_run = [t for t in C["cells_not_run"] if t.startswith("qwen3-8b")]
    if q8_not_run:
        must.append("%GHQ8-BEGIN\nThe two Qwen3-8B cells were not run")
    else:
        must_not.append("%GHQ8-BEGIN")
    n = len(C["blind_cells_evaluated"])
    word = {1: "the one", 2: "both", 3: "all three", 4: "all four", 5: "all five"}[n]
    oc = C["outcome_on_blind_cells"]
    if oc == "A":
        must.append(f"Registered outcome A (all retained, all in band): {word} blind "
                    f"cell{'s' if n > 1 else ''} that ran and passed the gates")
        must.append("inside the registered equivalence band")
        for t in C["blind_cells_evaluated"]:
            must.append(label_prose(t))
    else:
        must.append(f"OUTCOME {oc} HAS NO SENTENCE REGISTERED IN THIS CHECKER")
    for t in C["blind_cells_blocked_by_gate"]:
        must.append("so by the\nregistration it is not interpreted")
        must.append(label_prose(t) + "\nmissed G0 on the local run")
    if any(C["cells"][t]["ran"] and C["cells"][t]["source"] == "cluster" for t in S.CLUSTER_PRIMARY):
        must.append("\\emph{Deviation from the registration:} Qwen3-8B-Base could not be downloaded")
    else:
        must_not.append("\\emph{Deviation from the registration:}")
    return must, must_not


PROSE_NAME = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B", "gemma2-2b": "Gemma-2-2B",
              "gemma2-9b": "Gemma-2-9B", "mistral7b": "Mistral-7B"}


def label_prose(tag):
    a, b = tag.split("__")
    return PROSE_NAME[a] + "$\\leftrightarrow$" + PROSE_NAME[b]


def roster_expected():
    out = {}
    for d, name in (("gemma2-2b_l12", "Gemma-2-2B"), ("gemma2-9b_l20", "Gemma-2-9B"),
                    ("llama3.2-1b_b8", "Llama-3.2-1B"), ("mistral7b_b16", "Mistral-7B-v0.1")):
        r = json.load(open(TOKX / d / "extract_receipt.json"))
        out[name] = round(r["nnz_this_pass"] / r["tokens_this_pass"])
    return out


def scan(app, res, roster, tab, C=None, pins=True):
    bad = []
    C = C if C is not None else S.combine()
    if pins:
        for rel, want in PINS.items():
            p = S.DIR / rel
            if p.exists() and sha(p) != want:
                bad.append((f"pin {rel}", f"sha256 {want[:12]}..., found {sha(p)[:12]}..."))
            if not p.exists():
                bad.append((f"pin {rel}", "pinned but absent"))
        if S.OUT.exists() and json.load(open(S.OUT)) != json.loads(json.dumps(C)):
            bad.append(("GEMMA_HOOK_COMBINED.json", "stale: differs from gemmahook_summary.combine()"))
    if tab != expected_table(C):
        bad.append(("tab_gemmahook.tex", "differs from gemmahook_summary.emit() on the receipts"))
    try:
        sub = section(app)
    except ValueError:
        return bad + [("app:gemmahook", "section or table input not found")]
    ws = lambda x: re.sub(r"\s+", " ", x)   # line breaks in the tex are not content
    subw = ws(sub)
    for what, t in expected_prose(C).items():
        if not re.search(r"(?<![\d.])" + re.escape(ws(t)) + r"(?![\d])", subw):
            bad.append((what, ws(t)))
    must, must_not = expected_status(C)
    for s in must:
        if ws(s) not in subw:
            bad.append(("status sentence", s.replace("\n", " ")))
    for s in must_not:
        if s in sub:
            bad.append(("stale status block", s))
    # (4) roster
    for name, v in roster_expected().items():
        row = next((l for l in roster.splitlines() if l.startswith(name + " &")), "")
        if f"& ${{\\sim}}{v}$ &" not in row:
            bad.append((f"roster L0 {name}", f"~{v} (extract receipt, BOS incl.)"))
        if name.startswith("Gemma") and "\\texttt{resid\\_pre} block" not in row:
            bad.append((f"roster read point {name}", "resid_pre block N (our read point)"))
    H = {k: v.get("local") or v.get("cluster") for k, v in C["H"].items()}
    a, b = H["gemma2-2b:16k"], H["gemma2-9b:16k"]
    cap = (f"their $L_0$ is ${{\\sim}}{round(a['L0_all_docs_N']['noBOS'])}$ (2B) and "
           f"${{\\sim}}{round(b['L0_all_docs_N']['noBOS'])}$ (9B) at our read point\n"
           f"and ${{\\sim}}{round(a['L0_all_docs_N1']['noBOS'])}$ and "
           f"${{\\sim}}{round(b['L0_all_docs_N1']['noBOS'])}$ at the training hook")
    if cap not in roster:
        bad.append(("roster caption no-BOS L0", cap.replace("\n", " ")))
    # (5) threats + main text
    thr = app[app.index("\\label{app:threats}"):]
    if "(iv) every Gemma Scope dictionary was applied one\nblock before its training hook" not in thr:
        bad.append(("threats (iv)", "Gemma read-point disclosure"))
    if "Gemma Scope's is read one block\nbefore its training hook, \\Cref{app:gemmahook}" not in res:
        bad.append(("results.tex qualifier", "Gemma read point pointer"))
    return bad


def read_all():
    return APP.read_text(), RES.read_text(), ROSTER.read_text(), TAB.read_text()


def main():
    bad = scan(*read_all())
    for what, v in bad:
        print(f"MISMATCH {what}: expected {v}")
    print(f"FINDINGS={len(bad)}")
    print("check_gemmahook_claims:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def _pin_plant():
    """A changed pin must fire (simulates a re-run / hand-edited cluster_q8g2 receipt)."""
    real = PINS["cluster_q8g2/GEMMA_HOOK_SENSITIVITY.json"]
    try:
        PINS["cluster_q8g2/GEMMA_HOOK_SENSITIVITY.json"] = "0" * 64
        return bool(scan(*read_all()))
    finally:
        PINS["cluster_q8g2/GEMMA_HOOK_SENSITIVITY.json"] = real


def selftest():
    a, r, ro, t = read_all()
    C = S.combine()
    toks = expected_prose(C)

    def fires(**kw):
        args = dict(app=a, res=r, roster=ro, tab=t, C=C)
        args.update(kw)
        return bool(scan(**args))

    def plant_first_number(s, tok):
        """Change the last digit of the last number inside the prose token as it appears in s."""
        pat = r"\s+".join(re.escape(w) for w in tok.split())   # line breaks in the tex are not content
        mm = re.search(pat, s)
        seg = mm.group(0)
        m = re.search(r"\d(?=\D*$)", seg)
        j = mm.start() + m.start()
        return s[:j] + str((int(s[j]) + 1) % 10) + s[j + 1:]

    checks = [("clean pass", not scan(a, r, ro, t, C))]
    tab_num = re.search(r"\$0\.\d{3}\$", t).group(0)
    checks.append((f"table value plant {tab_num}", fires(tab=t.replace(tab_num, tab_num[:-2] + "9$", 1))))
    checks.append(("table verdict plant retained->not run",
                   fires(tab=t.replace("& retained \\\\", "& not run \\\\", 1))))
    for what in ("FVU 9B-16k", "L0 2B-16k", "blind d_excess", "blind d_r", "non-blind",
                 "screen z G2-M", "width N", "interp overlap", "replication qwen3-1.7b__gemma2-9b"):
        checks.append((f"prose plant {what}", fires(app=plant_first_number(a, toks[what]))))
    checks.append(("prose substring plant 0.617->0.6170",
                   fires(app=a.replace("\\to0.617$", "\\to0.6170$", 1))))
    checks.append(("status plant: drop GHQ8 fallback",
                   fires(app=re.sub(r"%GHQ8-BEGIN.*?%GHQ8-END\n", "", a, flags=re.S))
                   if "%GHQ8-BEGIN" in a else fires(app=a.replace("\\paragraph{Result (", "%GHQ8-BEGIN\n\\paragraph{Result (", 1))))
    checks.append(("status plant: 'not interpreted' removed",
                   fires(app=a.replace("so by the\nregistration it is not interpreted", "so it is kept", 1))))
    checks.append(("range plant: max excess shift +0.006 -> +0.005",
                   fires(app=a.replace("$-0.001$ to $+0.006$", "$-0.001$ to $+0.005$", 1))))
    checks.append(("status plant: outcome word four -> three",
                   fires(app=a.replace("in band): all four", "in band): all three", 1))))
    checks.append(("status plant: Q8-G2 dropped from the retained list",
                   fires(app=a.replace("Qwen3-8B$\\leftrightarrow$Gemma-2-2B,", "", 1))))
    checks.append(("status plant: deviation sentence removed",
                   fires(app=a.replace("\\emph{Deviation from the registration:}", "Note:", 1))))
    checks.append(("pin plant: cluster_q8g2 receipt sha", _pin_plant()))
    checks.append(("roster plant Mistral ~88->~117",
                   fires(roster=ro.replace("& ${\\sim}88$ &", "& ${\\sim}117$ &", 1))))
    checks.append(("roster plant read point 'post layer 20'",
                   fires(roster=ro.replace("\\texttt{resid\\_pre} block 20$^{\\ddagger}$", "post layer 20", 1))))
    checks.append(("threats plant (iv) removed",
                   fires(app=a.replace("(iv) every Gemma Scope dictionary", "(iv) every dictionary", 1))))
    checks.append(("main-text qualifier plant",
                   fires(res=r.replace("\\Cref{app:gemmahook}", "\\Cref{app:threats}", 1))))
    # pin plant: a different receipt content must fail the pin
    C2 = json.loads(json.dumps(C))
    t2 = next(iter(C2["blind_cells_evaluated"]))
    C2["cells"][t2]["decision"]["excess_N1"] += 0.01
    checks.append(("receipt-content plant (combined excess +0.01)", fires(C=C2)))
    for name, got in checks:
        print(f"  {name:52s} {'OK' if got else 'FAILED'}")
    ok = all(g for _, g in checks)
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
