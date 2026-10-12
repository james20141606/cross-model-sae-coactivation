#!/usr/bin/env python3
"""Layout audit for a composed figure SVG, using headless Chrome for real text metrics.

Reports (error unless noted):
  text-text     two text boxes intersect
  overflow      a text box whose centre lies inside a filled shape is not fully inside it
                (cell labels wider than their cell, labels spilling out of a panel)
  collision     a text box intersects a filled shape it is not inside (legend over bars);
                warning when the shape is a large container (panel / figure background)
  crossing      a stroked line/path passes through a text box (connectors through titles);
                warning when the stroke is hairline (< 1.2 px effective, e.g. a gridline)
  outside       a text box leaves the canvas
  too-small     printed size < --min-pt (default 7) at --width-in (default 5.5 in text width)

Mark a subtree `data-audit="skip"` to exclude it. Writes <stem>.audit.png (boxes drawn on
an rsvg render) and prints a table. Exit status 1 when any error is present.

    python audit.py fig.svg [--width-in 5.5] [--min-pt 7] [--json out.json] [--quiet]
"""
import argparse, json, pathlib, re, subprocess, sys, tempfile

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

JS = r"""
(function(){
  const svg=document.querySelector('svg');
  let W=parseFloat(svg.getAttribute('width'))||0, H=parseFloat(svg.getAttribute('height'))||0;
  const vb=svg.getAttribute('viewBox');
  if(vb){const p=vb.trim().split(/[\s,]+/).map(Number); W=p[2]; H=p[3];}
  svg.setAttribute('width',W); svg.setAttribute('height',H); svg.style.display='block';
  const root=svg.getBoundingClientRect();
  function skip(el){for(let e=el;e&&e!==svg;e=e.parentNode){if(e.getAttribute&&e.getAttribute('data-audit')==='skip')return true;}return false;}
  function rawbb(el){const b=el.getBoundingClientRect();return [b.left-root.left,b.top-root.top,b.width,b.height];}
  function clipbb(el){ // nearest ancestor-or-self clip-path (mpl clips bars/markers to the axes): intersect with it
    for(let e=el;e&&e!==svg;e=e.parentNode){
      const cp=e.getAttribute&&(e.getAttribute('clip-path')||(getComputedStyle(e).clipPath||''));
      const m=cp&&cp.match(/url\(["']?#([^"')]+)/); if(!m)continue;
      const c=document.getElementById(m[1]); if(!c)continue;
      const r=c.querySelector('rect'); if(!r)continue;
      const ctm=e.getScreenCTM(); const pt=svg.createSVGPoint();
      pt.x=parseFloat(r.getAttribute('x')||0); pt.y=parseFloat(r.getAttribute('y')||0); const p1=pt.matrixTransform(ctm);
      pt.x+=parseFloat(r.getAttribute('width')||0); pt.y+=parseFloat(r.getAttribute('height')||0); const p2=pt.matrixTransform(ctm);
      return [Math.min(p1.x,p2.x)-root.left,Math.min(p1.y,p2.y)-root.top,Math.abs(p2.x-p1.x),Math.abs(p2.y-p1.y)];
    }
    return null;}
  function bb(el){const b=rawbb(el); const c=clipbb(el); if(!c)return b;
    const x0=Math.max(b[0],c[0]),y0=Math.max(b[1],c[1]),x1=Math.min(b[0]+b[2],c[0]+c[2]),y1=Math.min(b[1]+b[3],c[1]+c[3]);
    return [x0,y0,Math.max(0,x1-x0),Math.max(0,y1-y0)];}
  function scaleOf(el){const m=el.getScreenCTM();return Math.sqrt(Math.abs(m.a*m.d-m.b*m.c));}
  const texts=[],shapes=[],strokes=[];
  svg.querySelectorAll('text').forEach(t=>{
    if(skip(t))return; if(t.closest('defs'))return;
    const s=(t.textContent||'').replace(/\s+/g,' ').trim(); if(!s)return;
    const cs=getComputedStyle(t); const fs=parseFloat(cs.fontSize)*scaleOf(t);
    texts.push({s:s.slice(0,60),b:bb(t),px:fs});
  });
  svg.querySelectorAll('rect,circle,ellipse,polygon,path,line,polyline,use').forEach(e=>{
    if(skip(e))return; if(e.closest('defs,clipPath,marker,mask,pattern,symbol'))return;
    const cs=getComputedStyle(e); const b=bb(e);
    const fill=cs.fill, stroke=cs.stroke;
    const fo=parseFloat(cs.fillOpacity===''?1:cs.fillOpacity)*parseFloat(cs.opacity===''?1:cs.opacity);
    const so=parseFloat(cs.strokeOpacity===''?1:cs.strokeOpacity)*parseFloat(cs.opacity===''?1:cs.opacity);
    const filled=fill&&fill!=='none'&&fo>0.05;
    const stroked=stroke&&stroke!=='none'&&so>0.05;
    const white=/^rgb\((2[4-5][0-9]), (2[4-5][0-9]), (2[4-5][0-9])\)$/.test(fill);
    if(filled&&!white&&b[2]*b[3]>0) shapes.push({tag:e.tagName,b:b,area:b[2]*b[3]});
    if(stroked&&!filled&&(e.tagName==='line'||e.tagName==='path'||e.tagName==='polyline')){
      let pts=[]; let sw=parseFloat(cs.strokeWidth)*scaleOf(e);
      try{const L=e.getTotalLength(); const m=e.getScreenCTM(); const n=Math.max(2,Math.ceil(L/3));
        for(let i=0;i<=n;i++){const p=e.getPointAtLength(L*i/n); const q=p.matrixTransform(m); pts.push([+(q.x-root.left).toFixed(1),+(q.y-root.top).toFixed(1)]);}
      }catch(err){}
      if(pts.length) strokes.push({tag:e.tagName,pts:pts,sw:sw,b:b});
    }
  });
  document.getElementById('audit').textContent=JSON.stringify({W:W,H:H,texts:texts,shapes:shapes,strokes:strokes});
})();
"""


def measure(svg_path):
    txt = pathlib.Path(svg_path).read_text()
    body = re.sub(r"^\s*<\?xml[^>]*\?>", "", txt).strip()
    html = ("<!doctype html><html><head><meta charset='utf-8'><style>html,body{margin:0;padding:0}</style></head>"
            "<body>" + body + "<pre id='audit'></pre><script>" + JS + "</script></body></html>")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=str(pathlib.Path(svg_path).parent)) as f:
        f.write(html); tmp = pathlib.Path(f.name)
    try:
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--window-size=2400,2400",
                            "--virtual-time-budget=3000", "--dump-dom", tmp.resolve().as_uri()],
                           capture_output=True, text=True, timeout=120)
    finally:
        tmp.unlink()
    m = re.search(r'<pre id="audit">(.*?)</pre>', r.stdout, re.S)
    if not m:
        raise RuntimeError("Chrome returned no audit payload:\n" + r.stderr[-2000:])
    import html as _h
    return json.loads(_h.unescape(m.group(1)))


def inter(a, b, pad=0.0):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or ay + ah + pad <= by or by + bh + pad <= ay)


def inside(a, b, pad=0.0):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return ax >= bx - pad and ay >= by - pad and ax + aw <= bx + bw + pad and ay + ah <= by + bh + pad


def centre_in(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    cx, cy = ax + aw / 2, ay + ah / 2
    return bx <= cx <= bx + bw and by <= cy <= by + bh


def glyph_box(b):
    """Chrome reports the font box (ascent+descent); shrink to roughly the ink box."""
    x, y, w, h = b
    return [x, y + 0.14 * h, w, h * (1 - 0.14 - 0.20)]


def run(svg, width_in=5.5, min_pt=7.0):
    d = measure(svg)
    W, H = d["W"], d["H"]
    pt_per_px = width_in * 72 / W
    canvas_area = W * H
    T = d["texts"]; S = d["shapes"]; K = d["strokes"]
    out = []
    for t in T:
        t["g"] = glyph_box(t["b"])
    for i, t in enumerate(T):
        b = t["g"]
        if not inside(b, [0, 0, W, H], 0.5):
            out.append(("error", "outside", t["s"], b))
        pt = t["px"] * pt_per_px
        if pt < min_pt - 0.05:
            out.append(("error", "too-small", f'{t["s"]} ({pt:.1f} pt)', b))
        for j in range(i + 1, len(T)):
            if inter(b, T[j]["g"], -1.0):
                out.append(("error", "text-text", f'{t["s"]} × {T[j]["s"]}', b))
        for s in S:
            big = s["area"] > 0.02 * canvas_area
            if centre_in(b, s["b"]):
                if not inside(b, s["b"], 1.5):
                    out.append(("error", "overflow", f'{t["s"]} ⊄ {s["tag"]} {[round(v) for v in s["b"]]}', b))
            elif inter(b, s["b"], -1.0):
                out.append(("warn" if big else "error", "collision", f'{t["s"]} ∩ {s["tag"]} {[round(v) for v in s["b"]]}', b))
        for k in K:
            if not inter(b, k["b"], 0):
                continue
            x, y, w, h = b
            if any(x <= px <= x + w and y <= py <= y + h for px, py in k["pts"]):
                out.append(("warn" if k["sw"] < 1.2 else "error", "crossing", f'{t["s"]} ← {k["tag"]} sw={k["sw"]:.1f}', b))
    return d, out


def annotate(svg, d, out, png_width=None):
    from PIL import Image, ImageDraw
    svg = pathlib.Path(svg)
    png_width = png_width or int(d["W"] * 1.5)
    png = svg.with_suffix(".audit.png")
    subprocess.run(["rsvg-convert", "-w", str(png_width), "-o", str(png), str(svg)], check=True)
    im = Image.open(png).convert("RGB"); dr = ImageDraw.Draw(im)
    k = png_width / d["W"]
    for lvl, kind, msg, b in out:
        x, y, w, h = [v * k for v in b]
        col = (220, 38, 38) if lvl == "error" else (234, 138, 0)
        dr.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], outline=col, width=3)
        dr.text((x, y - 12), kind, fill=col)
    im.save(png)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svg"); ap.add_argument("--width-in", type=float, default=5.5)
    ap.add_argument("--min-pt", type=float, default=7.0); ap.add_argument("--json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    d, out = run(a.svg, a.width_in, a.min_pt)
    png = annotate(a.svg, d, out)
    errs = [o for o in out if o[0] == "error"]; warns = [o for o in out if o[0] == "warn"]
    if not a.quiet:
        for lvl, kind, msg, b in sorted(out, key=lambda o: (o[0] != "error", o[1])):
            print(f'{lvl:5} {kind:10} @({b[0]:.0f},{b[1]:.0f} {b[2]:.0f}×{b[3]:.0f})  {msg}')
    print(f'{pathlib.Path(a.svg).name}: {len(d["texts"])} texts, {len(errs)} errors, {len(warns)} warnings → {png.name}')
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"errors": errs, "warnings": warns}, ensure_ascii=False, indent=1))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
