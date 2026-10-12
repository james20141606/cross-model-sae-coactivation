"""Compose a schematic SVG (glyphs.py) with matplotlib data panels (figstyle.py) into one figure,
then export PDF with embedded fonts via rsvg-convert. Usage:

    from compose import place, export
    fig = place("layout.svg", [("panel_a.svg", x, y, None), ...], out="fig1.svg")   # scale=None -> MPL_SCALE
    export("fig1.svg")      # -> fig1.pdf (+ fig1.png preview)
"""
import subprocess, pathlib, re
from svgutils import compose as C

# A matplotlib SVG is in points (72/in). The schematic canvas is 1600 px for 5.5 in, so a data panel
# saved at figsize (W_in, H_in) lands at its true printed size when scaled by MPL_SCALE.
TEXTWIDTH_IN, COLW_IN = 5.5, 2.65
MPL_SCALE = 1600 / (TEXTWIDTH_IN * 72)   # = 4.04 px per pt on the canonical 1600-px canvas

def _px(v):
    return float(re.sub(r"[a-z]+$", "", str(v)))


def place(layout_svg, panels, out, width=None, height=None, width_in=TEXTWIDTH_IN):
    """Overlay mpl panel SVGs (in pt) on a layout SVG (in px).

    scale=None -> the panel lands at its true printed size for a layout whose full
    width prints at `width_in` inches: scale = layout_px / (width_in * 72). For the
    canonical 1600-px canvas that is MPL_SCALE (4.04); for a 3072-px Gemini raster
    it is 7.76. Pass width_in=COLW_IN for a single-column figure."""
    base = C.SVG(str(layout_svg))
    w, h = width or base.width, height or base.height
    auto = _px(w) / (width_in * 72)
    items = [base]
    for path, x, y, scale in panels:
        p = C.SVG(str(path)).scale(scale if scale else auto).move(x, y)
        items.append(p)
    fig = C.Figure(w, h, *items)
    fig.save(str(out))
    return pathlib.Path(out)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _svg_size_px(svg_text):
    """(w, h) in px from width/height attrs, else from viewBox."""
    head = re.search(r"<svg[^>]*>", svg_text).group(0)
    def num(attr):
        m = re.search(attr + r'="([0-9.]+)(px)?"', head)
        return float(m.group(1)) if m else None
    w, h = num("width"), num("height")
    if w is None or h is None:
        vb = re.search(r'viewBox="([^"]+)"', head).group(1).split()
        w, h = float(vb[2]), float(vb[3])
    return w, h


def svg_to_pdf(svg, pdf):
    """SVG -> PDF through headless Chrome (Skia).

    rsvg-convert/cairo 1.18 writes every SPACE glyph through an unnamed
    embedded Type 3 font ([none] in pdffonts); Skia embeds CID TrueType
    subsets only. The page is sized to the SVG canvas (px -> 0.75 pt) with
    zero margins, so the PDF has exactly the SVG's aspect ratio and LaTeX
    scaling works as before. Falls back to rsvg-convert if Chrome is absent.
    """
    svg, pdf = pathlib.Path(svg), pathlib.Path(pdf)
    if not pathlib.Path(CHROME).exists():
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf), str(svg)], check=True)
        return pdf
    txt = svg.read_text()
    w, h = _svg_size_px(txt)
    body = re.sub(r"^\s*<\?xml[^>]*\?>", "", txt).strip()
    wrap = svg.with_suffix(".print.html")
    wrap.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"@page{{size:{w}px {h}px;margin:0}}html,body{{margin:0;padding:0}}svg{{display:block}}"
        "</style></head><body>" + body + "</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf.resolve()}", wrap.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    wrap.unlink()
    return pdf


def check_fonts(pdf):
    """Return the pdffonts table; raise if any Type 3 font is present."""
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout
    if "Type 3" in out:
        raise RuntimeError(f"Type 3 font in {pdf}:\n{out}")
    return out


def export(svg, pdf=None, png=None, png_width=2400, audit=True, width_in=5.5, min_pt=7.0):
    """SVG -> PDF (+PNG preview). Runs audit.py first (layout: overlaps, overflow, crossings,
    text < min_pt at the printed width) and refuses to export a figure with audit errors;
    pass audit=False only for a deliberate exception, and say why in the ELN."""
    svg = pathlib.Path(svg)
    pdf = pathlib.Path(pdf) if pdf else svg.with_suffix(".pdf")
    png = pathlib.Path(png) if png else svg.with_suffix(".png")
    if audit:
        import importlib.util
        spec = importlib.util.spec_from_file_location("figaudit", pathlib.Path(__file__).with_name("audit.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        d, out = mod.run(svg, width_in, min_pt)
        apng = mod.annotate(svg, d, out)
        errs = [o for o in out if o[0] == "error"]
        if errs:
            kinds = {}
            for o in errs:
                kinds[o[1]] = kinds.get(o[1], 0) + 1
            lines = "\n".join(f"  {k:10} {m}" for _, k, m, _ in errs[:25])
            raise RuntimeError(f"audit: {len(errs)} layout errors {kinds} in {svg.name}; see {apng.name}\n{lines}")
    svg_to_pdf(svg, pdf)
    check_fonts(pdf)
    subprocess.run(["rsvg-convert", "-w", str(png_width), "-o", str(png), str(svg)], check=True)
    return pdf, png
