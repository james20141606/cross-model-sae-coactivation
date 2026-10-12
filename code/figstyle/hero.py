"""hero.py — compose a hero figure: an approved raster schematic (Gemini PNG, no numbers,
empty data slots) as the background of ONE matplotlib figure, with receipt-driven data
panels drawn as inset axes exactly over the slots.

Why matplotlib as the compositor (not svgutils + Chrome): the PNG is embedded losslessly
(Chrome's print path re-encodes images as JPEG), text is Type 42 TrueType, sizes are true
printed pt, and one script owns the whole figure.

    from hero import Hero
    h = Hero("round4.png", width_in=5.5)               # figure = full text width, aspect from the PNG
    ax = h.slot(x, y, w, h, cover=True)                  # slot in PNG pixel coords; returns an Axes
    ... draw with figstyle helpers ...
    h.save("fig1")                                        # fig1.svg (audited) + fig1.pdf (+ fig1.png preview)
"""
import pathlib, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs


class Hero:
    def __init__(self, png, width_in=fs.TEXTWIDTH_IN, dpi=600):
        fs.use()
        self.png = pathlib.Path(png)
        im = Image.open(self.png).convert("RGB")
        self.W, self.H = im.size
        self.width_in = width_in
        self.fig = plt.figure(figsize=(width_in, width_in * self.H / self.W), dpi=dpi)
        bg = self.fig.add_axes([0, 0, 1, 1])
        bg.imshow(im, interpolation="none", aspect="auto")
        bg.set_axis_off()
        self.bg = bg
        self.print_dpi = self.W / width_in

    def slot(self, x, y, w, h, cover=True, pad=0.0):
        """Inset axes over a slot given in PNG pixel coordinates (origin top-left)."""
        if cover:  # hide whatever the raster drew inside the slot
            self.bg.add_patch(matplotlib.patches.Rectangle((x, y), w, h, facecolor="white", edgecolor="none", zorder=2))
        ax = self.fig.add_axes([(x + pad) / self.W, 1 - (y + h - pad) / self.H, (w - 2 * pad) / self.W, (h - 2 * pad) / self.H])
        ax.set_facecolor("none")
        return ax

    def text(self, x, y, s, **kw):
        """Vector text at PNG pixel coords (for labels that must be searchable/auditable)."""
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def save(self, stem, audit=True, min_pt=7.0):
        stem = pathlib.Path(stem)
        svg, pdf, png = stem.with_suffix(".svg"), stem.with_suffix(".pdf"), stem.with_suffix(".png")
        self.fig.savefig(svg, format="svg")
        if audit:
            import importlib.util
            spec = importlib.util.spec_from_file_location("figaudit", HERE / "audit.py")
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            d, out = mod.run(svg, self.width_in, min_pt)
            apng = mod.annotate(svg, d, out)
            errs = [o for o in out if o[0] == "error"]
            if errs:
                lines = "\n".join(f"  {k:10} {m}" for _, k, m, _ in errs[:25])
                raise RuntimeError(f"audit: {len(errs)} layout errors in {svg.name}; see {apng.name}\n{lines}")
        self.fig.savefig(pdf, format="pdf")
        fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
        if "Type 3" in fonts:
            raise RuntimeError(f"Type 3 font in {pdf}:\n{fonts}")
        self.fig.savefig(png, format="png", dpi=200)
        print(f"{pdf.name}: raster {self.W}x{self.H} px = {self.print_dpi:.0f} dpi at {self.width_in} in; fonts ok; audit ok")
        return svg, pdf, png
