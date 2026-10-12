#!/usr/bin/env python3
"""Per-rule x per-scope firing proof for the lexical-claims detector.

another project reported "91/91 tests all bite" and it was contaminated twice: its 13 scope plants
exercised ONE rule across 13 scopes, so 17 of 18 rules had never been checked anywhere. Its
first honest run was 218 fire / 16 blind, and the blind ones were rules written in LaTeX
syntax ($1{,}197$, \\texttt{v1.0.0}) that cannot match pdftotext output, where those render
as "1,197" and "v1.0.0". Its PDF scopes -- added specifically to catch retracted claims
surviving into the compiled paper -- were decorative for exactly those rules.

So: N rules across M scopes is an N x M obligation, and the PDF scope must be exercised
THROUGH THE REAL PIPELINE (LaTeX -> pdflatex -> pdftotext), not by injecting a string into
the extracted text, because the transformation is where rules die.

Design consequences:
  * tex and figsrc read plain text with no transformation, so injecting a probe is a
    faithful test of the rule there.
  * pdf does transform, so probes are written into a REAL .tex, the paper is REALLY
    recompiled, and the rules are matched against REAL pdftotext output. One compile
    carries all probes.
  * Every probe uses an INFLECTION not literally present in its pattern, so a
    noun/verb blind spot cannot pass (the bug that made a sibling project's first plant miss).

    python3 code/analysis/check_rule_coverage.py     # exit 1 if any rule is blind anywhere
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code/analysis"))
import check_lexical_claims as C  # noqa: E402

# (rule name, its regex, a probe that must trigger it). Probes deliberately use an
# inflection the pattern does not literally contain wherever the language allows.
RULES = [
    ("lexic",        r"lexic",          "the shared lexicon explains it"),
    ("unigram",      r"unigram",        "these unigrams co-fire"),
    ("ngram",        r"n-?gram",        "matched ngrams recur"),
    ("bagofwords",   r"bag.?of.?word",  "a bag of words model"),
    ("tfidf",        r"tf.?idf",        "the TF IDF design"),
    ("residual",     r"residual",       "after residualising the features"),
    ("scrambl",      r"scrambl",        "scrambling removes it"),
    ("wordorder",    r"word.?order",    "word order is destroyed"),
    ("token",        r"token",          "tokenwise detectors persist"),
    ("vocab",        r"vocab",          "the vocabulary is truncated"),
]

MARK = "RULECOVERAGEPROBE"


def fires(pattern, text):
    return re.search(pattern, text, re.I) is not None


def probe_sentence(name, probe):
    # one sentence per rule, tagged so we can find them in the PDF
    return "%s %s %s." % (MARK, name, probe)


def main():
    # ---- scopes without a transformation: inject and check each rule individually ----
    matrix = {}
    for scope in ("tex", "figsrc"):
        scan = C.scan_tex if scope == "tex" else C.scan_figsrc
        for name, pat, probe in RULES:
            s = probe_sentence(name, probe)
            hits = scan(extra_text=s)
            caught = any(MARK.lower() in C.norm(h[2]) and fires(pat, h[2]) for h in hits)
            matrix[(name, scope)] = caught

    # ---- pdf: exercise the REAL pipeline, one compile for all probes ----
    tex_path = os.path.join(ROOT, "paper/sections/appendix.tex")
    orig = open(tex_path).read()
    block = ("\n\n\\paragraph{Rule-coverage probe (temporary).}\n"
             + " ".join(probe_sentence(n, p) for n, _, p in RULES) + "\n")
    anchor = "\\section{AI Use Disclosure}"
    if anchor not in orig:
        print("cannot locate anchor for the PDF probe"); return 1
    try:
        open(tex_path, "w").write(orig.replace(anchor, block + anchor, 1))
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                           cwd=os.path.join(ROOT, "paper"), capture_output=True, timeout=300)
        pdf_txt = C.normalise_pdf(C.pdf_text())
        planted_present = MARK in pdf_txt
        for name, pat, probe in RULES:
            # Anchor on the probe BODY, not on the marker: LaTeX hyphenates the marker
            # across lines ("RULECOV-\nERAGEPROBE") and renders fi as a ligature
            # ("t<fi>df"), so a marker-anchored search reports BLIND for a rule that in
            # fact fires. Collapse whitespace so an injected line number cannot split it.
            flat = re.sub(r"\s+", " ", pdf_txt)
            body = re.sub(r"\s+", " ", probe)
            matrix[(name, "pdf")] = (body in flat) and fires(pat, body)
    finally:
        open(tex_path, "w").write(orig)
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                           cwd=os.path.join(ROOT, "paper"), capture_output=True, timeout=300)

    # ---- report the full N x M matrix ----
    scopes = ("tex", "figsrc", "pdf")
    print("%-12s %8s %8s %8s" % ("rule", *scopes))
    blind = []
    for name, _, _ in RULES:
        row = [matrix.get((name, s), False) for s in scopes]
        print("%-12s %8s %8s %8s" % (name, *["fire" if r else "BLIND" for r in row]))
        blind += [(name, s) for s, r in zip(scopes, row) if not r]
    n = len(RULES) * len(scopes)
    print("\n%d rules x %d scopes = %d obligations; %d fire, %d BLIND"
          % (len(RULES), len(scopes), n, n - len(blind), len(blind)))
    if not planted_present:
        print("WARNING: probe text was not found in the compiled PDF at all -- the PDF "
              "scope proof is not valid")
    for name, s in blind:
        print("  BLIND  rule %-12s in scope %s" % (name, s))
    print("VERDICT:", "every rule fires in every scope" if not blind
          else "SOME RULE IS BLIND IN SOME SCOPE")
    return 0 if (not blind and planted_present) else 1


if __name__ == "__main__":
    sys.exit(main())
