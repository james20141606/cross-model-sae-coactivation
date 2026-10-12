#!/usr/bin/env python3
"""fig:pipeline v2 (S class, PROTOCOL.md §2): the Gemini schematic paper/figures/fig_pipeline_v2/round1_A.png
(gemini-3-pro-image from prompt1.txt; spec.md) wrapped by hero.py into a PDF page of the included width.
Pure schematic: no data slots, no numbers (whitelist: none; only the language codes en/zh/ja/ko and the name C4).
Included at 0.9\textwidth = 4.95 in. Output: paper/figures/fig_pipeline_v2/fig_pipeline_v2.{svg,pdf,png}.
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figstyle"))
import figstyle as fs          # noqa: E402
from hero import Hero          # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

D = ROOT / "paper" / "figures" / "fig_pipeline_v2"
h = Hero(D / "round1_A.png", width_in=0.9 * fs.TEXTWIDTH_IN)
plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})
h.save(D / "fig_pipeline_v2")
