# figstyle — the one design system for the figures

- `figstyle.py`  palette (validated), matplotlib rc, helpers (`use()`, `bars`, `ci`, `panel_label`, `save`)
- `glyphs.py`    schematic glyphs written straight to SVG (panels, arrows, model stacks, SAE grids, documents, Lucide icons)
- `compose.py`   svgutils composition + rsvg-convert export
- `lucide_icons.json`  2,112 Lucide icons offline (from lucide-static 1.47.0)
- `demo.py`      renders `demo.png` — the reference look

Rule: schematics never carry numbers; data panels are plotted from receipts and composed in.
Copy this directory into `<repo>/code/figstyle/` so a clean clone builds. Run with `a Python with matplotlib`.
