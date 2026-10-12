"""Schematic glyphs, written straight to SVG with ElementTree (html-deck method; adapted from the
user's HY scientific_glyphs.py). Schematics carry NO axes, ticks or numbers -- anything that reads
as data is plotted from receipts with matplotlib and composed alongside (see compose.py).

Canvas convention: width 1600 px for a full-\textwidth figure (prints at 5.5 in => 1 px = 0.2475 pt,
so a 34 px label prints at ~8.5 pt). Use text(size=...) in *printed points*; it converts.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from figstyle import TEAL, CORAL, INDIGO, AMBER, VIOLET, GREY, INK, INK2, INK3, GRID, SURF, tint_hex, FONT_SERIF

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
ICONS = json.loads((Path(__file__).with_name("lucide_icons.json")).read_text())
PX_PER_PT = 1 / 0.2475          # for a 1600 px canvas printed at 5.5 in
FONT = ", ".join(f"'{f}'" if " " in f else f for f in FONT_SERIF)

def element(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{NS}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attrs.items()})

def canvas(width=1600, height=900):
    svg = ET.Element(f"{{{NS}}}svg", width=str(width), height=str(height), viewBox=f"0 0 {width} {height}")
    defs = element(svg, "defs")
    for name, color in (("arrow", INK), ("arrow-teal", TEAL), ("arrow-coral", CORAL), ("arrow-indigo", INDIGO),
                        ("arrow-grey", GREY)):
        m = element(defs, "marker", id=name, viewBox="0 0 10 10", refX="9", refY="5", markerWidth="7",
                    markerHeight="7", orient="auto-start-reverse")
        element(m, "path", d="M0 0 L10 5 L0 10 z", fill=color)
    element(svg, "rect", x=0, y=0, width=width, height=height, fill=SURF)
    return svg

def pt(size_pt):
    return round(size_pt * PX_PER_PT, 1)

def text(parent, x, y, s, size=8.5, color=INK, weight="normal", anchor="start", italic=False, family=FONT):
    t = element(parent, "text", x=x, y=y, font_family=family, font_size=pt(size), fill=color,
                font_weight=weight, text_anchor=anchor, font_style="italic" if italic else "normal")
    t.text = s
    return t

def panel(parent, x, y, w, h, hue, title=None, number=None, alpha=0.08, r=18):
    """Tinted panel background with an optional numbered title. hue = the panel's semantic colour."""
    g = element(parent, "g")
    element(g, "rect", x=x, y=y, width=w, height=h, rx=r, fill=tint_hex(hue, alpha), stroke="none")
    if number is not None:
        element(g, "circle", cx=x + 34, cy=y + 34, r=18, fill=SURF, stroke=hue, stroke_width=2.5)
        text(g, x + 34, y + 41, str(number), size=8.5, color=hue, weight="bold", anchor="middle")
    if title:
        text(g, x + (64 if number is not None else 24), y + 41, title, size=9, weight="bold")
    return g

def arrow(parent, x1, y1, x2, y2, color=INK, width=3, marker="arrow", dashed=False):
    a = element(parent, "path", d=f"M{x1} {y1} L{x2} {y2}", stroke=color, stroke_width=width, fill="none",
                marker_end=f"url(#{marker})", stroke_linecap="round")
    if dashed:
        a.set("stroke-dasharray", "10 8")
    return a

def icon(parent, name, x, y, size=48, color=INDIGO, stroke=1.75):
    """Lucide icon by its kebab-case name (offline JSON; 2,112 icons)."""
    g = element(parent, "g", transform=f"translate({x} {y}) scale({size / 24})", fill="none", stroke=color,
                stroke_width=stroke, stroke_linecap="round", stroke_linejoin="round")
    for tag, attrs in ICONS[name]:
        ET.SubElement(g, f"{{{NS}}}{tag}", {k: str(v) for k, v in attrs.items()})
    return g

def model_stack(parent, x, y, w=120, h=64, color=INDIGO, layers=3):
    """A stack of learned layers -- deliberately not an architecture."""
    g = element(parent, "g")
    for i in range(layers - 1, -1, -1):
        dx, dy = 6 * i, -5 * i
        element(g, "rect", x=x + dx, y=y + dy, width=w - 6 * (layers - 1), height=h, rx=6,
                fill=tint_hex(color, 0.12) if i == 0 else SURF, stroke=color, stroke_width=2.2 if i == 0 else 1.4)
    return g

def feature_grid(parent, x, y, cols=8, rows=5, cell=16, gap=4, color=INDIGO, lit=(), lit_color=None):
    """An SAE dictionary as a grid of cells; `lit` = iterable of (col,row) to highlight. No numbers."""
    g = element(parent, "g")
    lit = set(lit)
    for r in range(rows):
        for c in range(cols):
            on = (c, r) in lit
            element(g, "rect", x=x + c * (cell + gap), y=y + r * (cell + gap), width=cell, height=cell, rx=3,
                    fill=(lit_color or color) if on else tint_hex(color, 0.10), stroke=color if on else "none",
                    stroke_width=1.2)
    return g

def document(parent, x, y, w=90, h=110, color=INK2, lines=4, label=None):
    g = element(parent, "g")
    element(g, "path", d=f"M{x} {y} h{w-22} l22 22 v{h-22} h-{w} z", fill=SURF, stroke=color, stroke_width=2)
    element(g, "path", d=f"M{x+w-22} {y} v22 h22", fill="none", stroke=color, stroke_width=2)
    for i in range(lines):
        element(g, "rect", x=x + 14, y=y + 40 + i * 14, width=w - 28 - (18 if i == lines - 1 else 0), height=5, rx=2.5,
                fill=GRID)
    if label:
        text(g, x + w / 2, y + h + 26, label, size=8, color=INK2, anchor="middle", italic=True)
    return g

def check(parent, x, y, size=28, color=TEAL):
    return element(parent, "path", d=f"M{x} {y+size*0.55} l{size*0.3} {size*0.3} l{size*0.6} -{size*0.7}",
                   fill="none", stroke=color, stroke_width=4, stroke_linecap="round", stroke_linejoin="round")

def cross(parent, x, y, size=24, color=GREY):
    g = element(parent, "g", stroke=color, stroke_width=4, stroke_linecap="round")
    element(g, "path", d=f"M{x} {y} l{size} {size}"); element(g, "path", d=f"M{x+size} {y} l-{size} {size}")
    return g

def band(parent, x, y, w, h, color=AMBER, alpha=0.35, label=None):
    """A neutral 'null band' / range, no ticks."""
    g = element(parent, "g")
    element(g, "rect", x=x, y=y, width=w, height=h, rx=4, fill=tint_hex(color, alpha), stroke="none")
    if label:
        text(g, x + w / 2, y - 10, label, size=7.5, color=INK2, anchor="middle", italic=True)
    return g

def write(svg, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(p, encoding="utf-8", xml_declaration=True)
    return p

def raster(parent, png_path, x=0, y=0, w=None, h=None):
    """Embed a PNG (e.g. the approved Gemini schematic) as a data URI so the SVG is
    self-contained for Chrome/rsvg. Returns (w, h) in px as used."""
    import base64
    from PIL import Image
    png_path = str(png_path)
    iw, ih = Image.open(png_path).size
    w = w or iw; h = h or (ih * w / iw)
    data = base64.b64encode(open(png_path, "rb").read()).decode()
    el = element(parent, "image", x=x, y=y, width=w, height=h, preserveAspectRatio="none")
    el.set("{http://www.w3.org/1999/xlink}href", "data:image/png;base64," + data)
    el.set("href", "data:image/png;base64," + data)
    return w, h


def slot(parent, x, y, w, h, fill="#FFFFFF", stroke=None, rx=10):
    """White cover over a raster data-slot before an mpl panel is placed on top."""
    kw = dict(x=x, y=y, width=w, height=h, rx=rx, fill=fill)
    if stroke:
        kw.update(stroke=stroke, stroke_width=1.5)
    else:
        kw.update(stroke="none")
    return element(parent, "rect", **kw)
