"""Shared figure design system for the figures.

One palette, one type scale, one set of mark specs, so the 33 figures across five repos
read as one system. Import from any figure script:

    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N] / "figstyle"))
    import figstyle as fs; fs.use()

or copy this directory into <repo>/code/figstyle/ (preferred: a clean clone must build).

Palette validated 2026-09-22 with the dataviz skill's validate_palette.js (light surface):
lightness band, chroma floor, adjacent-pair CVD separation (worst 10.9 protan), normal-vision
floor (28.4), contrast >= 3:1 -- ALL PASS for the 4-slot set. The fifth slot (VIOLET) is an
extra categorical slot; never use more than five, fold the tail into "other"/facets.

Rules baked in (dataviz + gallery README):
  - colour follows the entity, never its rank; categorical hues in fixed order, never cycled
  - one hue light->dark for magnitude; two poles + neutral midpoint for polarity
  - thin marks, hairline solid grid one step off the surface, generous padding
  - text wears text tokens, never the series colour
  - legend present for >= 2 series; direct-label selectively, never every point
  - schematic panels carry NO axes, ticks or values (those come only from receipts)
"""
import matplotlib

# ---- palette (semantic roles are fixed across all five papers) ----------------------
TEAL   = "#0D9488"   # ours / the contribution / the control that is right / positive result
CORAL  = "#E8573F"   # the baseline that fails / the wrong control / the retracted reading
INDIGO = "#3B4CCA"   # model A / primary entity
AMBER  = "#D97706"   # model B / secondary entity
VIOLET = "#7C3AED"   # fifth categorical slot only
GREY   = "#8A8F98"   # null band, not significant, "not identity", withheld
INK    = "#1F2328"   # primary text
INK2   = "#57606A"   # secondary text (axis labels, captions inside figures)
INK3   = "#8B949E"   # muted text
GRID   = "#E6E8EB"   # hairline gridlines / axes, one step off surface
SURF   = "#FFFFFF"
CATEGORICAL = [TEAL, CORAL, INDIGO, AMBER, VIOLET]

def tint(hex_color, alpha=0.10):
    """Panel-background wash of a palette hue (returns an RGBA tuple for matplotlib)."""
    h = hex_color.lstrip("#"); r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)

def tint_hex(hex_color, alpha=0.10, surface="#FFFFFF"):
    """Same wash pre-blended onto the surface, for SVG fills."""
    h = hex_color.lstrip("#"); s = surface.lstrip("#")
    out = "".join("%02x" % round(int(h[i:i+2], 16) * alpha + int(s[i:i+2], 16) * (1 - alpha)) for i in (0, 2, 4))
    return "#" + out

# ---- sequential / diverging ---------------------------------------------------------
SEQ_TEAL = ["#E0F2F1", "#B2DFDB", "#80CBC4", "#4DB6AC", "#26A69A", "#0D9488", "#0B7A70", "#08635B"]
DIVERGING = (INDIGO, "#D8DADF", CORAL)   # cool pole, neutral midpoint, warm pole

# ---- type & geometry (ICLR: \textwidth = 5.5 in) ------------------------------------
TEXTWIDTH_IN = 5.5
COLW_IN = 2.65            # half-width panel
FONT_SERIF = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
SIZE_TICK, SIZE_LABEL, SIZE_TITLE, SIZE_PANEL = 7.5, 8.5, 9.5, 10

RC = {
    "font.family": "serif", "font.serif": FONT_SERIF, "mathtext.fontset": "stix",
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": SIZE_TITLE, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 6,
    "axes.labelsize": SIZE_LABEL, "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_TICK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.6, "lines.markersize": 5.5, "lines.markeredgewidth": 1.2, "lines.markeredgecolor": SURF,
    "patch.linewidth": 0, "errorbar.capsize": 2,
    "legend.frameon": False, "legend.fontsize": SIZE_TICK, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
    "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
    "text.color": INK,
}

def use():
    """Apply the design system to matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)

def panel_label(ax, text, x=-0.02, y=1.06):
    """Bold panel letter/number at the top-left, outside the axes (ICLR style: (a) (b) or ① ②)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=SIZE_PANEL, fontweight="bold",
            ha="right", va="bottom", color=INK)

def bars(ax, x, heights, color=TEAL, width=0.55, **kw):
    """Thin bars, no edge stroke; the surface gap comes from width < 1."""
    return ax.bar(x, heights, width=width, color=color, edgecolor="none", linewidth=0, **kw)

def ci(ax, x, lo, hi, color=INK2, lw=1.0):
    """Interval as a thin vertical rule (no caps larger than the mark)."""
    return ax.vlines(x, lo, hi, color=color, linewidth=lw, zorder=3)

def zero_line(ax, y=0.0, color=INK3):
    return ax.axhline(y, color=color, linewidth=0.8, zorder=1)

def direct_label(ax, x, y, text, dy=3, color=INK2, size=SIZE_TICK, ha="center"):
    """Selective direct label above a mark (use sparingly: the extreme, the endpoint, the one that matters)."""
    return ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points", ha=ha, va="bottom",
                       fontsize=size, color=color)

def save(fig, path):
    """SVG with real <text> (svg.fonttype none) and PDF with embedded TrueType (fonttype 42)."""
    p = str(path)
    fig.savefig(p if p.endswith(".svg") else p + ".svg")
    fig.savefig((p[:-4] if p.endswith(".svg") else p) + ".pdf")
