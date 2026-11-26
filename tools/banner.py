import math
import random
import sys

W, H = 1200, 600
NAVY = "#404452"
BLUE = "#2e95d3"
SKY = "#8fcaf0"
LAVENDER = "#b7aef0"
PINK = "#f19fd3"
MINT = "#5fd6b5"
WHITE = "#ffffff"

rng = random.Random(7)
out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
    '<title id="t">Spinning Up in Deep Learning</title>',
    '<desc id="d">An abstract illustration of a deep network: a pixel image of a digit passes through stacks of convolutional feature maps and a fully connected network to a set of output probabilities, under a falling training curve.</desc>',
    "<defs>",
    '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e9e7f5"/><stop offset="1" stop-color="#dbe9f6"/></linearGradient>',
    f'<linearGradient id="plane" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{SKY}" stop-opacity="0.95"/><stop offset="1" stop-color="{LAVENDER}" stop-opacity="0.85"/></linearGradient>',
    f'<linearGradient id="plane2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{LAVENDER}" stop-opacity="0.95"/><stop offset="1" stop-color="{PINK}" stop-opacity="0.8"/></linearGradient>',
    f'<linearGradient id="plane3" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{MINT}" stop-opacity="0.9"/><stop offset="1" stop-color="{SKY}" stop-opacity="0.9"/></linearGradient>',
    "</defs>",
    f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
]

for i in range(0, W + 1, 40):
    for j in range(0, H + 1, 40):
        out.append(f'<circle cx="{i}" cy="{j}" r="1.3" fill="{NAVY}" fill-opacity="0.10"/>')

pts = []
for k in range(121):
    x = 250 + k * (W - 290) / 120
    y = 120 + 360 * math.exp(-k / 18) + rng.uniform(-9, 9) * math.exp(-k / 60)
    pts.append(f"{x:.1f},{H - y:.1f}")
out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{WHITE}" stroke-width="5" stroke-opacity="0.75" stroke-linejoin="round"/>')
out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{BLUE}" stroke-width="1.6" stroke-opacity="0.45" stroke-linejoin="round"/>')

digit = [
    "00000000",
    "00011000",
    "00111000",
    "00011000",
    "00011000",
    "00011000",
    "00111100",
    "00000000",
]
cell, gx, gy = 19, 70, 220
for r, row in enumerate(digit):
    for c, v in enumerate(row):
        if v == "1":
            fill, op = NAVY, 0.95
        else:
            fill, op = WHITE, 0.85
        out.append(f'<rect x="{gx + c * cell}" y="{gy + r * cell}" width="{cell - 3}" height="{cell - 3}" rx="3" fill="{fill}" fill-opacity="{op}"/>')
out.append(f'<rect x="{gx - 8}" y="{gy - 8}" width="{8 * cell + 13}" height="{8 * cell + 13}" rx="10" fill="none" stroke="{NAVY}" stroke-opacity="0.35" stroke-width="1.5"/>')


def plane(x, y, w, h, skew, fill):
    return (f'<path d="M{x} {y + skew} L{x + w} {y} L{x + w} {y + h} L{x} {y + h + skew} Z" '
            f'fill="{fill}" stroke="{NAVY}" stroke-opacity="0.55" stroke-width="1.3"/>')


stacks = [(300, 170, 70, 230, 34, 4, 16, "url(#plane)"), (440, 205, 52, 165, 24, 6, 12, "url(#plane2)"), (560, 232, 38, 112, 16, 8, 9, "url(#plane3)")]
for x0, y0, w, h, skew, n, step, fill in stacks:
    for k in range(n):
        out.append(plane(x0 + k * step, y0 - k * 4, w, h, skew, fill))
        for q in range(3):
            fx = x0 + k * step + w * (0.25 + 0.25 * q)
            fy = y0 - k * 4 + h * (0.3 + 0.2 * ((q + k) % 3)) + skew * 0.5
            out.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="2.2" fill="{WHITE}" fill-opacity="0.8"/>')
out.append(f'<path d="M{gx + 8 * cell + 8} {gy + 4 * cell} C 270 {gy + 4 * cell}, 280 290, 300 290" fill="none" stroke="{NAVY}" stroke-opacity="0.4" stroke-width="1.6"/>')
for a, b in ((420, 440), (556, 560)):
    out.append(f'<path d="M{a} 290 L{b} 290" stroke="{NAVY}" stroke-opacity="0.4" stroke-width="1.6"/>')

layers = [6, 8, 8, 4]
xs = [720, 830, 940, 1050]
gap = 42


def ys(n):
    top = 300 - (n - 1) * gap / 2
    return [top + i * gap for i in range(n)]


pos = [[(x, y) for y in ys(n)] for x, n in zip(xs, layers)]
hot = [{1, 3, 4}, {2, 3, 6}, {1, 4, 5}, {1}]
for (x, y) in pos[0]:
    out.append(f'<path d="M640 290 C 680 290, {x - 40} {y}, {x - 10} {y}" fill="none" stroke="{NAVY}" stroke-opacity="0.18" stroke-width="1.1"/>')
quiet, loud = [], []
for i in range(len(layers) - 1):
    for a, (xa, ya) in enumerate(pos[i]):
        for b, (xb, yb) in enumerate(pos[i + 1]):
            mx = (xa + xb) / 2
            path = f'M{xa} {ya} C {mx} {ya}, {mx} {yb}, {xb} {yb}'
            if a in hot[i] and b in hot[i + 1]:
                color = PINK if (a + b) % 2 else BLUE
                loud.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" stroke-opacity="0.9"/>')
            else:
                quiet.append(f'<path d="{path}" fill="none" stroke="{NAVY}" stroke-width="1" stroke-opacity="{rng.uniform(0.07, 0.18):.2f}"/>')
out += quiet + loud
for i, layer in enumerate(pos):
    for j, (x, y) in enumerate(layer):
        if j in hot[i]:
            out.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{WHITE}" stroke="{BLUE}" stroke-width="3"/>')
            out.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{PINK if (i + j) % 2 else BLUE}"/>')
        else:
            out.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{WHITE}" fill-opacity="0.9" stroke="{NAVY}" stroke-opacity="0.45" stroke-width="1.5"/>')

bx = 1085
probs = [0.06, 0.82, 0.08, 0.04]
for (x, y), p in zip(pos[-1], probs):
    top = p == max(probs)
    out.append(f'<rect x="{bx}" y="{y - 9}" width="90" height="18" rx="9" fill="{WHITE}" fill-opacity="0.85"/>')
    out.append(f'<rect x="{bx}" y="{y - 9}" width="{max(18, 90 * p):.1f}" height="18" rx="9" fill="{PINK if top else LAVENDER}" fill-opacity="{1 if top else 0.8}"/>')

tokens = 9
for k in range(tokens):
    x = 300 + k * 52
    y = 500
    out.append(f'<rect x="{x}" y="{y}" width="34" height="34" rx="9" fill="{WHITE}" fill-opacity="0.9" stroke="{NAVY}" stroke-opacity="0.35" stroke-width="1.3"/>')
    out.append(f'<rect x="{x + 9}" y="{y + 9}" width="16" height="16" rx="4" fill="{[BLUE, MINT, LAVENDER, PINK][k % 4]}" fill-opacity="0.85"/>')
    if k:
        out.append(f'<path d="M{x - 18} {y + 17} L{x - 2} {y + 17}" stroke="{NAVY}" stroke-opacity="0.4" stroke-width="1.5"/>')
for a, b in ((1, 5), (2, 7), (4, 8)):
    xa, xb = 317 + a * 52, 317 + b * 52
    out.append(f'<path d="M{xa} 498 Q {(xa + xb) / 2} {455 - (b - a) * 6} {xb} 498" fill="none" stroke="{PINK}" stroke-opacity="0.7" stroke-width="1.8"/>')

grid_x, grid_y, s = 820, 470, 13
for i in range(6):
    for j in range(6):
        v = max(0.08, 1 - abs(i - j) * 0.45) if i != 5 - j else 0.95
        v = 0.92 if i == j else (0.35 if abs(i - j) == 1 else 0.08)
        out.append(f'<rect x="{grid_x + j * s}" y="{grid_y + i * s}" width="{s - 2}" height="{s - 2}" rx="2" fill="{BLUE}" fill-opacity="{v:.2f}"/>')

out.append("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
