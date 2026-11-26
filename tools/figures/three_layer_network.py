import math, sys
W, H = 860, 470
INK, EDGE = "#404452", "#8e8ea0"
R = 36
FONT = "'STIX Two Text','Latin Modern Roman','Times New Roman',Times,serif"
cols = {"x": 70, 1: 270, 2: 490, 3: 680, "y": 800}
def ys(n, gap):
    top = H / 2 - (n - 1) * gap / 2
    return [top + i * gap for i in range(n)]
y_in = ys(4, 112)
y1 = ys(4, 112)
y2 = ys(3, 112)
y3 = ys(1, 112)
out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
       '<title id="t">A neural network with three layers</title>',
       '<desc id="d">Four input features feed a first layer of four units, which feeds a second layer of three units, which feeds a third layer of one unit that outputs the prediction.</desc>',
       f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1 L10 5 L0 9 z" fill="{EDGE}"/></marker></defs>']
def arrow(x1, y1_, x2, y2_, pad1, pad2):
    d = math.hypot(x2 - x1, y2_ - y1_)
    ux, uy = (x2 - x1) / d, (y2_ - y1_) / d
    out.append(f'<line x1="{x1 + ux * pad1:.1f}" y1="{y1_ + uy * pad1:.1f}" x2="{x2 - ux * pad2:.1f}" y2="{y2_ - uy * pad2:.1f}" stroke="{EDGE}" stroke-width="1.4" marker-end="url(#arrow)"/>')
def label(cx, cy, base, sup, sub, size=27):
    small = round(size * 0.56)
    bx = cx - (size * 0.42 if (sup or sub) else 0)
    out.append(f'<text x="{bx:.1f}" y="{cy + size * 0.3:.1f}" font-family="{FONT}" font-style="italic" font-size="{size}" fill="{INK}" text-anchor="middle">{base}</text>')
    sx = bx + size * 0.3
    if sup:
        out.append(f'<text x="{sx:.1f}" y="{cy - size * 0.2:.1f}" font-family="{FONT}" font-size="{small}" fill="{INK}">{sup}</text>')
    if sub:
        out.append(f'<text x="{sx:.1f}" y="{cy + size * 0.52:.1f}" font-family="{FONT}" font-size="{small}" fill="{INK}">{sub}</text>')
for ya in y_in:
    for yb in y1:
        arrow(cols["x"], ya, cols[1], yb, 34, R + 3)
for ya in y1:
    for yb in y2:
        arrow(cols[1], ya, cols[2], yb, R + 3, R + 3)
for ya in y2:
    arrow(cols[2], ya, cols[3], y3[0], R + 3, R + 3)
arrow(cols[3], y3[0], cols["y"], y3[0], R + 3, 30)
for layer, yy in ((1, y1), (2, y2), (3, y3)):
    for j, y in enumerate(yy, start=1):
        out.append(f'<circle cx="{cols[layer]}" cy="{y:.1f}" r="{R}" fill="#ffffff" stroke="{INK}" stroke-width="2.4"/>')
        label(cols[layer], y, "a", f"[{layer}]", str(j))
for j, y in enumerate(y_in, start=1):
    label(cols["x"] - 8, y, "x", "(<tspan font-style=\"italic\">i</tspan>)", str(j))
label(cols["y"] + 8, y3[0], "ŷ", "(<tspan font-style=\"italic\">i</tspan>)", "", size=30)
out.append("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
