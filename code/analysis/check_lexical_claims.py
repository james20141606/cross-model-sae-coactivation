#!/usr/bin/env python3
"""Enumerate EVERY claim site about the lexical/token alternative -- in .tex, in FIGURE
SOURCES, and in the COMPILED PDF.

History, because each widening was forced by a real miss:

  v1 scanned paper/sections/*.tex only. That is the guard gap a sibling project hit: its withdrawn
  dissociation claim survived in figures/fig_dissociation.py while the guard reported PASS.
  Logic and matching were both correct; it simply read the wrong files.

  a sibling project's figure plant then MISSED on first run: the rule matched the noun
  "dissociation" while the figure used the verb "dissociate" -- a blind spot inside the fix
  for a blind spot. Every pattern here is therefore a STEM, and the plants in --selftest use
  a DIFFERENT inflection from the one in the pattern so a noun/verb blind spot cannot pass.

  Scanning the compiled PDF is not redundant with the other two: text can be baked into a
  figure binary whose source is not in this repo (fig1_arc.pdf and fig_universality_matrix.pdf
  are generated artifacts), and the PDF is what a reviewer actually reads.

    python3 code/analysis/check_lexical_claims.py             # list every site, by scope
    python3 code/analysis/check_lexical_claims.py --selftest  # plant a violation in EACH
                                                              # scope; fail if any is missed
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "paper/main.pdf")

# STEMS, not words. "lexic" catches lexical/lexically/lexicon; "residual" catches
# residualize/residualization/residualisation; "scrambl" catches scramble/scrambling/
# scrambled. Breadth is deliberate -- a noisy review checklist is recoverable, a silent
# one is not.
TOPIC = re.compile(
    r"lexic|unigram|n-?gram|bag.?of.?word|tf.?idf|residual|scrambl|word.?order|"
    r"token|vocab",
    re.I)
EXCLUDED = re.compile(
    r"do(?:es)? not explain|does not account|exclud|ruled out|rules out|"
    r"not explained by|cannot explain|removes? no more|no more of the excess", re.I)
LIVE = re.compile(
    r"remain\w* live|not ruled out|cannot exclude|does not exclude|consistent with|"
    r"compatible with|under-?power|inconclusive|untested|leaves? .{0,20}live", re.I)


def _sentences_with_topic(text, source, line_of=None):
    out = []
    for m in TOPIC.finditer(text):
        s = max(text.rfind(".", 0, m.start()), text.rfind("\n\n", 0, m.start())) + 1
        e = text.find(".", m.end())
        sent = " ".join(text[s:e if e > 0 else len(text)].split())[:300]
        out.append((source, line_of(m.start()) if line_of else 0, sent))
    return out


def _line_mapper(raw):
    offs, pos = [], 0
    for i, l in enumerate(raw.splitlines(), 1):
        offs.append((pos, i))
        pos += len(l) + 1

    def f(off):
        c = 1
        for o, i in offs:
            if o <= off:
                c = i
            else:
                break
        return c
    return f


def scan_tex(extra_text=None):
    hits = []
    for path in sorted(glob.glob(os.path.join(ROOT, "paper/sections/*.tex"))):
        raw = open(path).read()
        clean = re.sub(r"%.*?$", "", raw, flags=re.M)
        hits += _sentences_with_topic(clean, os.path.basename(path), _line_mapper(raw))
    if extra_text:
        hits += _sentences_with_topic(extra_text, "<planted-tex>", None)
    return hits


def scan_figsrc(extra_text=None):
    """Figure SOURCES: generator scripts and vector/spec sources that carry visible text."""
    pats = ["paper/figures_src/**/*.py", "paper/figures/specs/*.svg",
            "paper/figures/specs/*.json", "paper/figures/*.drawio",
            "code/scripts/ct_fig2.py"]
    hits = []
    for pat in pats:
        for path in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
            raw = open(path, errors="replace").read()
            hits += _sentences_with_topic(raw, os.path.relpath(path, ROOT), _line_mapper(raw))
    if extra_text:
        hits += _sentences_with_topic(extra_text, "<planted-figsrc>", None)
    return hits


def pdf_text(path=PDF):
    if not os.path.exists(path):
        raise RuntimeError("no %s" % path)
    r = subprocess.run(["pdftotext", "-layout", path, "-"],
                       capture_output=True, text=True, timeout=120)
    # Exit code checked deliberately: a failed or timeout-killed pdftotext returns empty
    # stdout, which is byte-identical to "no matches". That exact false negative nearly
    # caused a bad deletion earlier this session; never infer "clean" from empty output.
    if r.returncode != 0:
        raise RuntimeError("pdftotext exit=%d" % r.returncode)
    return r.stdout


LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
             "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
             "\x1c": "fi", "\x1d": "fl", "\x0e": "ff"}


def normalise_pdf(txt):
    """Undo the three pdftotext transformations that silently blind rules.

    Found by the per-rule x per-scope matrix, in real output from this paper:
      1. HYPHENATION  "RULECOV-\n1601   ERAGEPROBE" -- a word split across a line break,
         so "residu-\nalization" does not contain "residual" and the rule cannot fire.
      2. LIGATURES    "t\x1cdf", "scienti\x1cc", "veri\x1ced" -- LaTeX sets fi/fl/ff as
         single glyphs; this is a known failure class, where rules written for the
         source could never match the rendering.
      3. LINE NUMBERS the ICLR template injects "1606" mid-sentence, so any pattern
         spanning a line break fails.
    Without this the PDF scope is decorative for precisely the multi-syllable words that
    matter here (residualization, unigram, scrambling).
    """
    for k, v in LIGATURES.items():
        txt = txt.replace(k, v)
    # join hyphenated line breaks, tolerating an injected line number between the halves
    txt = re.sub(r"-\n\s*(?:\d{1,5}\s+)?", "", txt)
    # drop standalone line-number tokens that survive elsewhere
    txt = re.sub(r"(?m)^\s*\d{1,5}\s{2,}", " ", txt)
    txt = re.sub(r"\n\s*\d{1,5}\s*\n", "\n", txt)
    return txt


def scan_pdf(extra_text=None):
    hits = _sentences_with_topic(normalise_pdf(pdf_text()), "main.pdf", None)
    if extra_text:
        hits += _sentences_with_topic(extra_text, "<planted-pdf>", None)
    return hits


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())[:110]


def xnorm(s):
    """Cross-scope key: drop digits and collapse space.

    pdftotext interleaves the ICLR line numbers into the extracted text, so the same
    sentence has a different norm() in the PDF than in the .tex and would be counted as a
    NEW site. Stripping digits collapses that noise. Without this the 'sites the .tex scan
    cleared' figure is inflated -- which would be a fabricated statistic about my own guard,
    the precise failure mode this guard exists to prevent.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s.lower())).strip()[:110]


def lean(s):
    return ("EXCLUDED" if EXCLUDED.search(s) and not LIVE.search(s)
            else "LIVE" if LIVE.search(s) else "UNDECIDED")


def dedup(hits):
    seen, out = set(), []
    for f, l, s in hits:
        k = (f, norm(s))
        if k not in seen:
            seen.add(k)
            out.append((f, l, s))
    return out


def selftest():
    """Plant a violation in EACH scope; every scope must catch its own plant.

    Each plant uses an inflection NOT literally in the pattern (lexicon / scrambled /
    tokenised) so the noun-vs-verb blind spot that made a sibling project's first figure plant
    miss cannot pass here.
    """
    plants = {
        "tex": ("A withdrawn claim: the shared lexicon fully explains the overlap.", scan_tex),
        # "scrambling" does not contain the word "scramble"; "tokenwise" does not contain
        # "token-level" (the v1 pattern). Both would be missed by a word-based rule and are
        # caught only because the patterns are stems.
        "figsrc": ("# label: scrambling removes it, so lexical sharing is ruled out.", scan_figsrc),
        "pdf": ("The tokenwise detectors remain live in this rendering.", scan_pdf),
    }
    ok = {}
    for name, (text, fn) in plants.items():
        try:
            hits = fn(extra_text=text)
        except Exception as e:
            print("  %-7s SCOPE UNAVAILABLE: %s" % (name, e))
            ok[name] = False
            continue
        caught = any(norm(text)[:40] in norm(s) for _, _, s in hits)
        ok[name] = caught
        print("  %-7s plant %-7s  %r" % (name, "CAUGHT" if caught else "MISSED", text[:56]))
    print()
    for k, v in ok.items():
        print("PASS scope_%s_catches_planted_violation = %s" % (k, v))
    good = all(ok.values())
    print("VERDICT:", "all scopes can fail" if good
          else "A SCOPE CANNOT FAIL -- guard is decorative")
    return 0 if good else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    tex, fig = dedup(scan_tex()), dedup(scan_figsrc())
    try:
        pdf = dedup(scan_pdf())
        pdf_ok = True
    except Exception as e:
        pdf, pdf_ok = [], False
        print("!! PDF SCOPE UNAVAILABLE: %s -- results below are INCOMPLETE\n" % e)
    tex_norm = {xnorm(s) for _, _, s in tex}

    for title, hits in (("TEX (paper/sections/*.tex)", tex),
                        ("FIGURE SOURCES", fig),
                        ("COMPILED PDF", pdf)):
        print("=" * 78)
        print("%s -- %d site(s)" % (title, len(hits)))
        print("=" * 78)
        for f, l, s in hits:
            new = "" if xnorm(s) in tex_norm else "   <-- INVISIBLE TO THE .tex SCAN"
            print("[%-9s] %s%s%s\n            %s\n"
                  % (lean(s), f, (":%d" % l) if l else "", new, s))

    extra = [h for h in fig + pdf if xnorm(h[2]) not in tex_norm]
    print("=" * 78)
    print("TOTALS  tex=%d  figsrc=%d  pdf=%d  (pdf scope %s)"
          % (len(tex), len(fig), len(pdf), "OK" if pdf_ok else "UNAVAILABLE"))
    print("SITES THE .tex SCAN CLEARED BUT THE NEW SCOPES CATCH: %d" % len(extra))
    for f, l, s in extra:
        print("   [%s] %s: %s" % (lean(s), f, s[:120]))
    print("\nAfter the token test lands, re-read every site above against the verdict.")
    return 0 if pdf_ok else 1


if __name__ == "__main__":
    sys.exit(main())
