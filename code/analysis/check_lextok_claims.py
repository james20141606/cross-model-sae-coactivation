#!/usr/bin/env python3
"""Gate: the paper must not read the lexical adequacy gate's Delta-R^2 as variance captured.

WHY THIS FILE EXISTS. The paper carried "A lexical model capturing up to 27% of each
feature's variance captures almost none of what the two models share." The +0.27 is
`median_delta_r2` -- a MARGIN over a spectrum-matched SHUFFLED design whose own median
out-of-fold R^2 is about -0.17. The adequacy rule in lexical_token_test.py is

    passed = bool(frac >= GATE_FRAC_BETTER and dmed >= GATE_MEDIAN_DR2)

which requires R^2_real > R^2_shuffled and NEVER requires R^2_real > 0. The nuisance
model's actual median out-of-fold R^2 tops out at 0.140. So "27% of each feature's
variance" overstated the strongest quantity in the receipt by about 2x, and the existing
gates could not see it: check_c4_table.py scores a different table, lextok_summary.py
verifies the emitted table against the receipt (the table was right; the prose was not),
and check_lexical_claims.py enumerates claim SITES without reading any number.

Four rules, every bound DERIVED from the receipts or from the code -- never typed here:

  R1  variance-attribution   any number the prose attaches to "feature variance" /
                             "out-of-fold R^2" / "per-feature R^2" in a sentence about the
                             lexical nuisance model must be <= the largest median
                             R^2_real in the receipts. Signed numbers are exempt (a
                             Delta-R^2 or an R^2_shuffled is always printed with a sign),
                             and so are quantities that match a receipt-derived design
                             constant (the SVD explained-variance ratio and its
                             complement), which are about the DESIGN, not about features.
  R2  max-delta              any "up to / rising to +X" next to Delta-R^2 must equal the
                             largest median_delta_r2 in the receipts at printed precision.
  R3  gate-is-still-relative re-derive from lexical_token_test.py that the adequacy pass
                             condition is purely relative, and from the receipts that the
                             shuffled design's median R^2 is negative. Either changing
                             invalidates the paper's "margin, not variance captured"
                             scoping, so it is a finding that must be read, not ignored.
  R4  design-scope           the compiled PDF must state the nuisance design's actual
                             restrictions (unigram, the TF-IDF term budget, the SVD
                             variance retention), because "removes the lexical signal" is
                             only true of THAT design.

Scopes: paper/sections/*.tex and the compiled PDF (normalised for ligatures, hyphenated
line breaks and the ICLR line-number ruler, the three transformations that make a rule
decorative in the PDF -- see check_lexical_claims.py's history).

    python3 code/analysis/check_lextok_claims.py             # exit 1 on any finding
    python3 code/analysis/check_lextok_claims.py --selftest  # every rule must be able to fire
"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results/cross_model_sae")
PDF = os.path.join(ROOT, "paper/main.pdf")
SRC = os.path.join(ROOT, "code/analysis/lexical_token_test.py")

# The mutual-NN arm is the one the body claims about; the lanpair/proclex arms are
# transplanted pairing rules reported separately.
RECEIPTS = "lextok20k_*.json"


# ------------------------------------------------------------------ receipt facts
def receipts():
    out = []
    for p in sorted(glob.glob(os.path.join(RES, RECEIPTS))):
        if os.path.basename(p).startswith(("lanpair_", "proclex_")):
            continue
        out.append((os.path.basename(p), json.load(open(p))))
    return out


def facts():
    """Every bound this gate enforces, read off the committed receipts."""
    real, delta, shuf, evr, maxfeat, ks = [], [], [], [], [], []
    for _, d in receipts():
        maxfeat.append(d["tfidf_max_features"])
        for k, a in d["arms"].items():
            ks.append(int(k))
            evr.append(a["explained_variance_ratio"])
            for side in ("model_a", "model_b"):
                s = a["adequacy"][side]
                real.append(s["median_r2_real"])
                delta.append(s["median_delta_r2"])
                shuf.append(s["median_r2_shuffled"])
    if not real:
        raise RuntimeError("no %s receipts under %s" % (RECEIPTS, RES))
    kg = max(ks)
    evr_g = max(e for kk, e in zip(ks, evr) if kk == kg)
    return {"max_real": max(real), "max_delta": max(delta), "max_shuf": max(shuf),
            "evr_governing": evr_g, "max_features": max(maxfeat), "k_governing": kg,
            "n_arms": len(real)}


# ------------------------------------------------------------------ text normalising
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
             "\ufb04": "ffl", "\x1c": "fi", "\x1d": "fl", "\x0e": "ff"}


def pdf_text():
    if not os.path.exists(PDF):
        raise RuntimeError("no %s -- compile first" % PDF)
    r = subprocess.run(["pdftotext", "-layout", PDF, "-"],
                       capture_output=True, text=True, timeout=180)
    # never infer "clean" from empty output: a failed pdftotext looks like "no matches"
    if r.returncode != 0:
        raise RuntimeError("pdftotext exit=%d" % r.returncode)
    txt = r.stdout
    for k, v in LIGATURES.items():
        txt = txt.replace(k, v)
    txt = re.sub(r"-\n\s*(?:\d{1,5}\s+)?", "", txt)      # hyphenated line break
    txt = re.sub(r"(?m)^\s*\d{1,5}\s{2,}", " ", txt)     # line-number ruler
    txt = re.sub(r"\n\s*\d{1,5}\s*\n", "\n", txt)
    return txt


DELTA_TOKEN = "DELTAMARGIN"   # deliberately contains no "R2": see flatten()


def flatten(txt, tex=False):
    """Collapse LaTeX and pdftotext renderings onto one plain-text form.

    Two traps, both hit while writing this file:

      * The two R^2 spellings must stay DISTINGUISHABLE -- a Delta-R^2 is a margin and an
        R^2 is a fit, and conflating them in the NORMALISER would blind the very rule this
        gate exists for. The delta token therefore contains no "R2" substring; an earlier
        "DELTAR2" was itself rewritten to "DELTA R2" by the next line, silently disarming
        the max-delta rule everywhere.
      * LaTeX comment stripping must run ONLY on .tex. In pdftotext output a bare "%" is a
        percent sign, and stripping to end-of-line there deletes every number this gate
        reads -- the PDF scope had zero teeth until `tex` was made explicit.
    """
    if tex:
        txt = re.sub(r"(?m)(?<!\\)%.*$", "", txt)
    for cmd in ("emph", "textbf", "textit", "text", "texttt", "mathrm"):
        for _ in range(3):
            txt = re.sub(r"\\%s\{([^{}]*)\}" % cmd, r"\1", txt)
    txt = re.sub(r"\\(?:Cref|cref|ref|label|citep|citet|cite)\{[^{}]*\}", " ", txt)
    txt = re.sub(r"\\Delta\s*R\^?\{?2\}?", " %s " % DELTA_TOKEN, txt)     # tex
    txt = re.sub(r"[\u2206\u0394]\s*R\s*2", " %s " % DELTA_TOKEN, txt)    # pdf glyph
    txt = re.sub(r"R\^?\{?2\}?(?:_\{?[a-z\\{}]*\}?)?", " R2 ", txt)
    txt = txt.replace("{,}", "").replace("\\%", "%").replace("\\,", " ")
    txt = txt.replace("$", " ").replace("~", " ")
    txt = re.sub(r"\\[a-zA-Z]+", " ", txt)
    txt = re.sub(r"[{}]", " ", txt)
    txt = txt.replace("\u2013", "-").replace("\u2014", " - ").replace("\u2212", "-")
    txt = txt.replace("\u2192", " to ").replace("\u2019", "'")
    txt = re.sub(r"(\d),(\d\d\d)\b", r"\1\2", txt)               # 50,000 -> 50000
    # "27 percent" must scan like "27%": the NUM rule keys a percentage on the "%"
    # sign, so a spelled-out "percent" made any percentage invisible to R1 (audit L2).
    txt = re.sub(r"(\d+(?:\.\d+)?)\s*percent\b", r"\1%", txt, flags=re.I)
    return re.sub(r"\s+", " ", txt)


def _split_flat(flat):
    """Split flattened text at [.:;]+space, but NEVER inside parentheses.

    Splitting on ';' inside a parenthetical detached a number from its scope keyword
    ("The lexical nuisance model (...; capturing 0.27 of each feature's variance)")
    and R1 could not see the pairing (audit L1). Depth is clamped at 0 so a stray ')'
    cannot poison the rest of the scope, and an unclosed '(' stops suppressing splits
    after 600 characters so a stray one cannot glue the whole document together.
    """
    out, start, depth, open_pos = [], 0, 0, -1
    for i, ch in enumerate(flat):
        if ch == "(":
            if depth == 0:
                open_pos = i
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        else:
            if depth > 0 and i - open_pos > 600:
                depth = 0            # stray unclosed paren: stop suppressing splits
            if ch in ".:;" and depth == 0 and i + 1 < len(flat) and flat[i + 1] == " ":
                out.append(flat[start:i + 1].strip())
                start = i + 1
    out.append(flat[start:].strip())
    return [s for s in out if s]


def sentences(txt, tex=False):
    return _split_flat(flatten(txt, tex))


def scan_sources(extra=None):
    """(label, sentence) over every scope. `extra` plants a sentence, for --selftest."""
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "paper/sections/*.tex"))):
        for s in sentences(open(p).read(), tex=True):
            out.append((os.path.basename(p), s))
    for s in sentences(pdf_text()):
        out.append(("main.pdf", s))
    if extra:
        out += [("<planted>", s) for s in sentences(extra, tex=True)]
    return out


# ------------------------------------------------------------------ rules
SCOPE = re.compile(r"lexic|nuisance|unigram|tf-?idf|residualiz|first[- ]stage", re.I)
# what the prose is attaching a number TO. Only these count as a claim about how much of a
# FEATURE the nuisance model explains.
ATTRIB = re.compile(r"(?:each |per-)?features?(?:'s|s')? variance|variance (?:of|in|per) (?:each |the )?"
                    r"features?|per-feature R2|out-of-fold R2|of feature variance", re.I)
# unsigned only: a Delta-R^2 or an R^2_shuffled is always printed with an explicit sign,
# so requiring "no sign" is what separates a fit claim from a margin claim.
NUM = re.compile(r"(?<![-+\u2212\d.])(\d+(?:\.\d+)?)\s*(%?)")
UPTO = re.compile(r"(?:up to|rising to|reach(?:es|ing)?|as (?:high|large) as|peaking at)\s*\+?(\d+\.\d+)", re.I)


def _exempt(v, f):
    """Design constants, derived: the SVD explained-variance ratio and its complement.

    These describe the DESIGN ("the components retain 67.7% of that design's variance"),
    not how much of a feature is explained, so a rule about feature variance must not
    flag them -- and must not need a hand-typed exception to avoid it.
    """
    for d in (f["evr_governing"], 1.0 - f["evr_governing"]):
        if abs(v - d) <= 0.005:
            return True
    return False


def rule_variance_attribution(sites, f):
    bad, bound = [], f["max_real"] + 0.005      # slack for 2-dp printing
    for src, s in sites:
        if not (SCOPE.search(s) and ATTRIB.search(s)):
            continue
        for m in NUM.finditer(s):
            raw, pct = float(m.group(1)), m.group(2) == "%"
            v = raw / 100.0 if pct else raw
            if not pct and not (0.0 < raw < 1.0):
                continue                        # k, term counts, years: not an R^2
            # COUNTING/RETENTION percentages are not variance attribution: "N% of
            # features beat ...", "N% of their excess", "N% of (the) pairs" quantify a
            # fraction of ITEMS or of an excess, not a share of a feature's variance.
            tail = s[m.end():m.end() + 40]
            if pct and re.match(r"\s*of\s+(features|their excess|the excess|pairs|the pairs)",
                                tail, re.I):
                continue
            if _exempt(v, f) or v <= bound:
                continue
            bad.append("%s: attributes %s to feature variance, but the largest median "
                       "R^2_real in any receipt is %.3f (that number is the size of a "
                       "Delta-R^2 margin, not variance captured) -- %r"
                       % (src, m.group(0).strip(), f["max_real"], s[:150]))
    return bad


def rule_max_delta(sites, f):
    bad = []
    for src, s in sites:
        if DELTA_TOKEN not in s:
            continue
        for m in UPTO.finditer(s):
            claimed = m.group(1)
            prec = len(claimed.split(".")[1])
            if ("%.*f" % (prec, f["max_delta"])) != claimed:
                bad.append("%s: claims Delta-R^2 %s but the receipts' largest median "
                           "delta is %.3f -- %r" % (src, claimed, f["max_delta"], s[:150]))
    return bad


def rule_gate_is_relative(f):
    """The paper's scoping sentence is only correct while BOTH of these hold."""
    bad = []
    src = open(SRC).read()
    body = src[src.index("def adequacy("):]
    body = body[:body.index("\n# ---")] if "\n# ---" in body else body
    m = re.search(r"^\s*passed\s*=\s*(.+)$", body, re.M)
    if not m:
        bad.append("%s: adequacy() has no `passed =` line -- this gate can no longer "
                   "verify that the adequacy rule is purely relative"
                   % os.path.relpath(SRC, ROOT))
    else:
        cond = m.group(1)
        if not re.fullmatch(r"bool\(frac >= GATE_FRAC_BETTER and dmed >= GATE_MEDIAN_DR2\)",
                            cond.strip()):
            bad.append("%s: adequacy() pass condition changed to %r -- if it now imposes "
                       "an absolute floor on R^2_real, the paper's 'the gate is a margin, "
                       "not an absolute fit' scoping is stale and must be rewritten"
                       % (os.path.relpath(SRC, ROOT), cond.strip()))
    if f["max_shuf"] >= 0:
        bad.append("largest median R^2_shuffled is %+.3f >= 0: the shuffled design no "
                   "longer predicts badly, so Delta-R^2 no longer overstates the fit and "
                   "the paper's scoping paragraph must be re-derived" % f["max_shuf"])
    if not re.search(r'stop_words\s*=\s*"english"', src):
        bad.append("%s: TF-IDF is no longer stopword-stripped, but the paper says it is"
                   % os.path.relpath(SRC, ROOT))
    if re.search(r"ngram_range", src):
        bad.append("%s: TF-IDF now sets ngram_range, so the design is no longer unigram "
                   "and the paper's 'unigram' scoping is wrong"
                   % os.path.relpath(SRC, ROOT))
    return bad


def rule_design_scope(pdf, f):
    """"Removes the lexical signal" is only true of the design actually fitted, so the
    paper has to state that design's limits where a reader will see them."""
    flat = flatten(pdf)   # PDF scope: never strip "%" as a comment (see flatten)
    need = [("unigram", "unigram"),
            (str(f["max_features"]), "the TF-IDF term budget (%d)" % f["max_features"]),
            ("%.1f" % (100.0 * f["evr_governing"]),
             "the SVD variance retention (%.1f%%)" % (100.0 * f["evr_governing"]))]
    return ["compiled PDF never states %s -- the residualization claim is unscoped" % why
            for tok, why in need if tok not in flat]


# ------------------------------------------------------------------ driver
def run(planted_tex=None):
    f = facts()
    pdf = pdf_text()
    sites = scan_sources(planted_tex)
    groups = [("R1 variance-attribution", rule_variance_attribution(sites, f)),
              ("R2 max-delta", rule_max_delta(sites, f)),
              ("R3 gate-is-still-relative", rule_gate_is_relative(f)),
              ("R4 design-scope", rule_design_scope(pdf, f))]
    return f, groups


def selftest():
    """Every rule must be able to fire. Each plant uses the REAL defect where one exists."""
    f = facts()
    checks = []

    def v(txt, tex):
        return rule_variance_attribution(
            [("<plant>", s) for s in sentences(txt, tex=tex)], f)

    # R1: the exact sentence this file was written to catch, in BOTH renderings.
    defect_tex = ("A lexical model capturing up to $27\\%$ of each feature's variance "
                  "captures almost none of what the two models share.")
    defect_pdf = ("A lexical model capturing up to 27% of each feature\u2019s variance "
                  "captures almost none of what the two models share.")
    checks.append(("R1 tex plant", "must fire", bool(v(defect_tex, True))))
    checks.append(("R1 pdf plant", "must fire", bool(v(defect_pdf, False))))
    # ... and must NOT fire on the corrected wording, or it is a rubber stamp that would
    # have blocked the fix as readily as the defect.
    ok_tex = ("A lexical model whose median per-feature $R^2$ tops out at $0.14$ "
              "captures almost none of what the two models share.")
    checks.append(("R1 corrected wording", "must stay silent", not v(ok_tex, True)))
    # ... nor on the design-level retention number, which is about the SVD, not a feature.
    evr_tex = ("SVD-truncated to components retaining $%.1f\\%%$ of even that design's "
               "feature variance." % (100.0 * f["evr_governing"]))
    checks.append(("R1 design constant", "must stay silent", not v(evr_tex, True)))

    # R2: a stale maximum, in both renderings.
    checks.append(("R2 tex plant", "must fire", bool(rule_max_delta(
        [("<p>", s) for s in sentences(
            "The gate passes at every $k$ (median $\\Delta R^2$ up to $+0.27$).",
            tex=True)], f))))
    checks.append(("R2 pdf plant", "must fire", bool(rule_max_delta(
        [("<p>", s) for s in sentences(
            "The gate passes at every k (median \u2206R2 up to +0.27).")], f))))

    # R3: fake a receipt world in which the shuffled design predicts well.
    checks.append(("R3 shuffled-R2>=0", "must fire",
                   bool(rule_gate_is_relative(dict(f, max_shuf=0.01)))))

    # R4: a PDF that never states the design's limits.
    checks.append(("R4 unscoped PDF", "must fire",
                   bool(rule_design_scope("no design detail here at all", f))))

    for name, want, got in checks:
        print("  %-24s %-18s %s" % (name, want, "OK" if got else "FAILED"))
    good = all(got for _, _, got in checks)
    print("\nVERDICT:", "every rule can fail, and none fires on correct text" if good
          else "A RULE IS BLIND OR OVER-FIRES -- this gate is not trustworthy")
    return 0 if good else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    try:
        f, groups = run()
    except Exception as e:                      # a scope that cannot run is a failure
        print("HARNESS ERROR: %s" % e)
        print("FINDINGS=1")
        return 1
    print("receipt bounds (derived, not typed): max median R^2_real=%.4f  "
          "max median dR^2=%.4f  max median R^2_shuf=%+.4f  governing k=%d  "
          "SVD evr=%.4f  tfidf max_features=%d  (%d arm-sides)"
          % (f["max_real"], f["max_delta"], f["max_shuf"], f["k_governing"],
             f["evr_governing"], f["max_features"], f["n_arms"]))
    n = 0
    for name, bad in groups:
        print("\n%s -- %d finding(s)" % (name, len(bad)))
        for b in bad:
            print("   " + b)
        n += len(bad)
    print("\nFINDINGS=%d" % n)
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
