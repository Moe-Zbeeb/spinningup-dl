import random
import sys

random.seed(11)
W, H = 1200, 640
BG, FG = "#840132", "#ffffff"
layers = [7, 10, 10, 10, 7, 4]
x0, x1 = 330, 900
xs = [x0 + i * (x1 - x0) / (len(layers) - 1) for i in range(len(layers))]
gap = 50


def ys(n):
    top = H / 2 - (n - 1) * gap / 2
    return [top + i * gap for i in range(n)]


pos = [[(xs[i], y) for y in ys(n)] for i, n in enumerate(layers)]
active = [{1, 3, 5}, {1, 4, 5, 8}, {2, 4, 7}, {1, 3, 6, 8}, {1, 3, 5}, {1}]
out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
    '<title id="t">A neural network reading a handwritten digit</title>',
    '<desc id="d">A small image of the digit seven feeds a six-layer network. One path of units lights up and the output distribution peaks on a single class.</desc>',
    f'<rect width="{W}" height="{H}" fill="{BG}"/>',
]
digit = ["0000000", "0111110", "0000110", "0001100", "0011000", "0011000", "0000000"]
cell = 28
gx, gy = 60, H / 2 - 7 * cell / 2
for r, row in enumerate(digit):
    for c, v in enumerate(row):
        o = 1 if v == "1" else 0.13
        out.append(f'<rect x="{gx + c * cell:.1f}" y="{gy + r * cell:.1f}" width="{cell - 4}" height="{cell - 4}" fill="{FG}" fill-opacity="{o}"/>')
gright = gx + 7 * cell - 4
for x, y in pos[0]:
    out.append(f'<path d="M{gright + 10:.1f} {H / 2:.1f} C{gright + 60:.1f} {H / 2:.1f} {x - 60:.1f} {y:.1f} {x - 13:.1f} {y:.1f}" fill="none" stroke="{FG}" stroke-opacity="0.3" stroke-width="1.2"/>')
quiet, loud = [], []
for i in range(len(layers) - 1):
    for a, (xa, ya) in enumerate(pos[i]):
        for b, (xb, yb) in enumerate(pos[i + 1]):
            if a in active[i] and b in active[i + 1]:
                o = random.uniform(0.6, 1.0)
                loud.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="{FG}" stroke-opacity="{o:.2f}" stroke-width="2"/>')
            else:
                o = random.uniform(0.05, 0.2)
                quiet.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="{FG}" stroke-opacity="{o:.2f}" stroke-width="1"/>')
out += quiet + loud
for i, layer in enumerate(pos):
    for j, (x, y) in enumerate(layer):
        if j in active[i]:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{FG}"/>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" fill="{BG}" stroke="{FG}" stroke-opacity="0.6" stroke-width="1.6"/>')
bx = 975
probs = [0.07, 0.86, 0.12, 0.05]
for (x, y), p in zip(pos[-1], probs):
    hot = p == max(probs)
    out.append(f'<line x1="{x + 13:.1f}" y1="{y:.1f}" x2="{bx - 12}" y2="{y:.1f}" stroke="{FG}" stroke-opacity="{1 if hot else 0.3}" stroke-width="{2 if hot else 1.2}"/>')
    out.append(f'<rect x="{bx}" y="{y - 11:.1f}" width="165" height="22" fill="{FG}" fill-opacity="0.13"/>')
    out.append(f'<rect x="{bx}" y="{y - 11:.1f}" width="{165 * p:.1f}" height="22" fill="{FG}" fill-opacity="{1 if hot else 0.55}"/>')
out.append("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
