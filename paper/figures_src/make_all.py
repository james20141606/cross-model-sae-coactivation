"""
Build every figure used in main.tex.

Usage:
    cd paper/figures_src && python make_all.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = [
    "make_pipeline", "make_scaling", "make_xlingual",
    "make_thinking", "make_perturbation", "make_layer_dist",
    "make_replacement",
]


def main():
    for name in MODULES:
        print(f"\n[{name}]")
        mod = importlib.import_module(name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            # Most modules have a `plot()` driven by `if __name__ == '__main__'`
            # block; re-run that block by importing fresh.
            import runpy
            runpy.run_module(name, run_name="__main__")


if __name__ == "__main__":
    main()
    print("\nAll figures written to paper/figures/.")
