"""Shared matplotlib style for the 2026-10-10 Figure 1 redraw.
Import: `import sys; sys.path.insert(0, '<this dir or a copy in code/>'); from fig1_rc import apply, C`.
Rules: white background, grey = reference/convention, one accent (C.ACC),
optional second accent (C.ACC2) only when the figure compares exactly two things."""
import matplotlib as mpl

class C:
    INK = '#1a1a1a'      # text, primary marks
    REF = '#8c8c8c'      # reference lines, conventions, "as stored", null bands (lines)
    BAND = '#e6e6e6'     # null / noise / CI bands (fills)
    ACC = '#c0392b'      # the one accent: this paper's measurement / "after control"
    ACC2 = '#1f5fa8'     # second accent, only for an explicit two-way contrast
    GRID = '#ececec'

def apply(base_pt=8.0):
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': base_pt, 'axes.titlesize': base_pt + 0.5, 'axes.labelsize': base_pt,
        'xtick.labelsize': base_pt - 0.5, 'ytick.labelsize': base_pt - 0.5, 'legend.fontsize': base_pt - 0.5,
        'axes.edgecolor': C.INK, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.grid': False, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.0, 'lines.markersize': 3.5,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
    })
