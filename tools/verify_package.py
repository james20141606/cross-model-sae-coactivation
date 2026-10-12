#!/usr/bin/env python3
"""Integrity check of this package (standard library only).

  1. every file listed in MANIFEST.sha256 exists with that SHA-256, and nothing unlisted ships;
  2. every receipt the paper cites (\\texttt{...json} or results/...json in paper/sections/*.tex)
     resolves to a bundled file;
  3. every receipt the analysis gates read (GATE_INPUTS in code/analysis/check_receipts_tracked.py,
     which in the source repository checks git tracking) is bundled.

    python3 tools/verify_package.py      # exit 1 on any problem
"""
import glob
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code/analysis"))
import check_receipts_tracked as T  # noqa: E402  (only its path lists are used; no git needed)


def main():
    bad = []
    listed = {}
    for line in open(os.path.join(ROOT, "MANIFEST.sha256")):
        h, rel = line.rstrip("\n").split("  ", 1)
        listed[rel] = h
    for rel, h in listed.items():
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            bad.append("MISSING   " + rel)
        elif hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            bad.append("SHA DIFF  " + rel)
    for dp, ds, fs in os.walk(ROOT):
        ds[:] = [d for d in ds if d != ".git"]   # version-control metadata is not package content
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), ROOT)
            if rel != "MANIFEST.sha256" and rel not in listed and "__pycache__" not in rel:
                bad.append("UNLISTED  " + rel)
    cited = T.cited_paths()
    for c in cited:
        if os.path.basename(c) in T.KNOWN_ABSENT:
            continue
        pat = c if c.startswith("results/") else os.path.join("results", "**", c)
        if not glob.glob(os.path.join(ROOT, pat), recursive=True):
            bad.append("CITED BUT ABSENT  " + c)
    for g in T.GATE_INPUTS:
        if not os.path.exists(os.path.join(ROOT, g)):
            bad.append("GATE INPUT ABSENT " + g)
    for b in bad:
        print(b)
    print(f"{len(listed)} manifest entries, {len(cited)} cited receipt path(s), "
          f"{len(T.GATE_INPUTS)} gate input(s): {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
