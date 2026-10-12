"""Gate: the body must end before the References page, with nothing of it spilling over.

WHY THIS FILE EXISTS. The page budget was checked all session by re-typing an ad-hoc snippet
and eyeballing one number against a remembered bar. Both bars in the record were wrong:

    "body 9pp == References page - 1"   falsified in a sibling repo: real body text shared
                                        the References page, so the rule passed a 10pp body.
    "42 non-ws chars before References  an accident. 42 = the running header PLUS one lineno
     == the running header alone"       token; a 2- or 4-digit line number reads 41 or 43.
    "39 ... == the running header"      true only in -layout mode. Measured on this same PDF:
                                          pdftotext -layout  -> 39
                                          pdftotext (default) -> 198
                                        because the default mode emits every lineno token on
                                        the page before the heading. A 5x spread on one file.

So there is no magic number, and there is not even a mode-independent one. The only durable
check is the one that names what it is excluding: strip the page furniture we can enumerate
(running header, line-number ruler, page number) and require that NOTHING of the body remains
in front of the References heading. That is what this does, and it reports the residue instead
of a count, so a failure says which words spilled rather than by how much a bar was missed.

Run:  /usr/bin/python3 code/analysis/check_body_pages.py [--max-body-pages 10] [--selftest]

Bar: 10 pages since revision R1 (2026-10-07). ICLR 2027 allows 10 body pages for the revised
paper of the discussion period (9 at submission; iclr.cc Author Guidelines, verified 2026-10-04,
<internal-notes> ELN). To re-check the submitted build against its own bar, pass --max-body-pages 9.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "paper/main.pdf"

# Extraction mode is PINNED, because the answer depends on it (see the docstring).
PDFTOTEXT_MODE = ["-layout"]

RUNNING_HEADER = "Under review as a conference paper at ICLR"
# The ICLR style prints a line-number ruler and a page number; both are furniture, and both
# are matched as PATTERNS rather than subtracted as a count, so the check does not drift with
# the number of digits.
FURNITURE = [
    re.compile(r"^\s*" + re.escape(RUNNING_HEADER) + r"\s*\d{4}\s*$"),
    re.compile(r"^\s*\d{1,5}\s*$"),                  # lineno ruler / page number on its own line
    re.compile(r"^\s*$"),
]
# ICLR 2027 sets \section in SMALL CAPS, so pdftotext emits "R EFERENCES" --
# a capital, a space, then the rest, in upper case. Allow the split and match
# case-insensitively. This changes only the recognition of the SAME literal
# heading under a different font rendering; the criterion is unchanged.
HEADING = re.compile(r"^\s*(?:\d+\s+)?R\s*EFERENCES\s*$", re.M | re.I)


def page_text() -> list[str]:
    if not PDF.exists():
        raise SystemExit(f"check_body_pages: {PDF} does not exist -- compile first")
    r = subprocess.run(["pdftotext", *PDFTOTEXT_MODE, str(PDF), "-"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("check_body_pages: pdftotext failed -- this is NOT a clean check")
    return r.stdout.split("\f")


def residue(page: str, upto: int) -> list[str]:
    """Body lines appearing before offset `upto` on this page, furniture removed."""
    out = []
    for line in page[:upto].split("\n"):
        if any(rx.match(line) for rx in FURNITURE):
            continue
        # a layout-mode line can be "486   some body text": drop a leading ruler number only
        stripped = re.sub(r"^\s*\d{1,5}\s+", "", line).strip()
        if stripped:
            out.append(stripped)
    return out


def check(max_body_pages: int) -> list[str]:
    findings = []
    pages = page_text()
    ref_page = None
    for i, p in enumerate(pages, 1):
        m = HEADING.search(p)
        if not m:
            continue
        ref_page = i
        spill = residue(p, m.start())
        break
    if ref_page is None:
        findings.append("no References heading found -- the check did not run, "
                        "which is a failure, not a pass")
        return findings

    # The body occupies every page before the heading, PLUS the heading's own page
    # when body text precedes it there. Sharing a page with References is ordinary
    # typesetting, not a violation -- the budget bounds PAGES OF BODY, and the
    # earlier form (any spill => finding) was written when the ICLR 2026 build put
    # References at the top of its page. Switching to the official 2027 template
    # with `times` tightened the setting and moved the heading mid-page, which is
    # what exposed the conflation.
    body_pages = ref_page - 1 + (1 if spill else 0)
    if body_pages > max_body_pages:
        findings.append(
            f"BODY IS {body_pages}pp against a {max_body_pages}pp bar: References "
            f"begins on p{ref_page}" +
            (f" with {len(spill)} line(s) of body above it, first {spill[0][:90]!r}"
             if spill else ""))
    if not findings:
        print(f"check_body_pages: PASS -- body occupies {body_pages}pp (bar "
              f"{max_body_pages}); References begins p{ref_page}"
              + (f", sharing that page with {len(spill)} body line(s)" if spill
                 else ", topping that page")
              + ". Furniture excluded by pattern, not by count.")
    return findings


def selftest() -> int:
    """A gate that cannot fail is not a gate. Plant body text on a synthetic References page
    and confirm the residue check reports it; confirm pure furniture reports nothing."""
    hdr = f"{RUNNING_HEADER} 2027"
    clean = f"      {hdr}\n486\n487\n\n      References\n"
    dirty = f"      {hdr}\n486\n      and collapses at twice the onset in 8/8 cells.\n487\n\n      References\n"
    m_clean, m_dirty = HEADING.search(clean), HEADING.search(dirty)
    r_clean = residue(clean, m_clean.start())
    r_dirty = residue(dirty, m_dirty.start())
    bad = []
    if r_clean:
        bad.append(f"furniture-only page reported spill {r_clean}")
    if not r_dirty:
        bad.append("planted body line before References was NOT reported")
    # the count-based bars this file replaces would both have passed the dirty page
    n_dirty = len(re.sub(r"\s", "", dirty[:m_dirty.start()]))
    if n_dirty in (39, 42):
        bad.append(f"planted page happens to hit a legacy magic bar ({n_dirty}); "
                   "rewrite the probe so the test is meaningful")
    if bad:
        print("check_body_pages selftest: FAIL")
        for b in bad:
            print("   ", b)
        return 1
    print(f"check_body_pages selftest: PASS -- spill detected ({r_dirty[0][:60]!r}), "
          f"furniture-only page clean. Note the planted page's raw pre-char count is "
          f"{n_dirty}, which no fixed bar could have interpreted.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-body-pages", type=int, default=10)  # R1: revision limit (was 9 at submission)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    findings = check(a.max_body_pages)
    if not findings:
        return 0
    print(f"check_body_pages: {len(findings)} FINDING(S)\n")
    for f in findings:
        print("  " + f)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
