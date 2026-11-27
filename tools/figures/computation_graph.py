import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import BLUE, INK, MAROON, ORANGE

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "_static" / "img" / "neural_networks" / "computation_graph.svg"
MATH = "STIX Two Text, Latin Modern Roman, Times New Roman, Times, serif"
SANS = "Lato, Helvetica Neue, Helvetica, Arial, sans-serif"
LOCAL_FILL = "#fdf0e4"
GLOBAL_FILL = "#fbeef2"


def box(x, y, w, h, lines, stroke=INK, fill="#ffffff", size=20, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash_attr}/>']
    for k, line in enumerate(lines):
        ty = y + h / 2 + (k - (len(lines) - 1) / 2) * (size + 6) + size * 0.35
        out.append(f'<text x="{x + w / 2}" y="{ty:.1f}" font-family="{MATH}" font-style="italic" font-size="{size}" fill="#000" text-anchor="middle">{line}</text>')
    return out


def arrow(x1, y1, x2, y2, color, dash=None, width=2.2):
    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    bx, by = x2 - ux * 10, y2 - uy * 10
    px, py = -uy * 5, ux * 5
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return [f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>',
            f'<path d="M{x2:.1f} {y2:.1f} L{bx + px:.1f} {by + py:.1f} L{bx - px:.1f} {by - py:.1f} z" fill="{color}"/>']


def band(y, h, title, color):
    return [f'<rect x="8" y="{y}" width="1084" height="{h}" rx="12" fill="{color}" fill-opacity="0.05"/>',
            f'<text x="22" y="{y + 21}" font-family="{SANS}" font-size="15" font-weight="700" fill="{color}">{title}</text>']


def main():
    W, H = 1100, 610
    b = []
    b += band(30, 210, "Forward pass: compute the values", BLUE)
    b += band(252, 96, "Local gradients: each node on its own", ORANGE)
    b += [f'<rect x="8" y="360" width="1084" height="192" rx="12" fill="{MAROON}" fill-opacity="0.05"/>',
          f'<text x="1078" y="540" font-family="{SANS}" font-size="15" font-weight="700" fill="{MAROON}" text-anchor="end">Global gradients: ∂ℒ / ∂(variable) = upstream global × local</text>']

    b += box(30, 80, 60, 40, ["w"])
    b += box(30, 140, 60, 40, ["x"])
    b += box(30, 200, 60, 40, ["b"], stroke=INK)
    b += box(220, 140, 180, 40, ["z = w<tspan font-size='13' dy='-8'>⊤</tspan><tspan dy='8'>x + b</tspan>"])
    b += box(520, 140, 140, 40, ["a = σ(z)"])
    b += box(870, 140, 140, 40, ["ℒ(a, y)"])
    b += box(910, 60, 60, 36, ["y"], size=18)
    b += arrow(90, 100, 220, 152, BLUE)
    b += arrow(90, 160, 220, 160, BLUE)
    b += arrow(90, 220, 220, 168, BLUE)
    b += arrow(400, 160, 520, 160, BLUE)
    b += arrow(660, 160, 870, 160, BLUE)
    b += arrow(940, 96, 940, 140, BLUE)

    b += box(205, 286, 210, 50, ["∂z/∂w = x,  ∂z/∂b = 1"], stroke=ORANGE, fill=LOCAL_FILL, size=17)
    b += box(490, 286, 200, 50, ["∂a/∂z = a(1 − a)"], stroke=ORANGE, fill=LOCAL_FILL, size=17)
    b += box(790, 286, 290, 50, ["∂ℒ/∂a = −y/a + (1 − y)/(1 − a)"], stroke=ORANGE, fill=LOCAL_FILL, size=17)
    for cx in (310, 590, 940):
        b += [f'<line x1="{cx}" y1="180" x2="{cx}" y2="286" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="3 3"/>']

    b += box(690, 400, 300, 64, ["∂ℒ/∂a = 1 · ∂ℒ/∂a", "= −y/a + (1 − y)/(1 − a)"], stroke=MAROON, fill=GLOBAL_FILL, size=17, dash="7 4")
    b += box(370, 400, 250, 64, ["∂ℒ/∂z = ∂ℒ/∂a · ∂a/∂z", "= a − y"], stroke=MAROON, fill=GLOBAL_FILL, size=17, dash="7 4")
    b += box(30, 400, 250, 64, ["∂ℒ/∂w = ∂ℒ/∂z · ∂z/∂w", "= x (a − y)"], stroke=MAROON, fill=GLOBAL_FILL, size=17, dash="7 4")
    b += box(30, 478, 250, 64, ["∂ℒ/∂b = ∂ℒ/∂z · ∂z/∂b", "= a − y"], stroke=MAROON, fill=GLOBAL_FILL, size=17, dash="7 4")

    b += arrow(690, 432, 620, 432, MAROON, dash="7 5")
    b += arrow(370, 432, 280, 432, MAROON, dash="7 5")
    b += arrow(370, 444, 280, 508, MAROON, dash="7 5")
    b += arrow(940, 336, 900, 400, ORANGE, width=1.8)
    b += arrow(590, 336, 530, 400, ORANGE, width=1.8)
    b += arrow(250, 336, 200, 400, ORANGE, width=1.8)

    b += [f'<line x1="90" y1="582" x2="130" y2="582" stroke="{BLUE}" stroke-width="2.2"/>',
          f'<text x="138" y="587" font-family="{SANS}" font-size="15" fill="{INK}">forward pass</text>',
          f'<rect x="270" y="573" width="34" height="18" rx="4" fill="{LOCAL_FILL}" stroke="{ORANGE}" stroke-width="1.8"/>',
          f'<text x="312" y="587" font-family="{SANS}" font-size="15" fill="{INK}">local gradient: depends only on that node</text>',
          f'<line x1="660" y1="582" x2="700" y2="582" stroke="{MAROON}" stroke-width="2.2" stroke-dasharray="7 5"/>',
          f'<text x="708" y="587" font-family="{SANS}" font-size="15" fill="{INK}">global gradient: flows back from the loss</text>']

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
           '<title id="t">Local and global gradients in the computation graph of logistic regression</title>',
           '<desc id="d">Three bands. The forward pass computes z from w, x, and b, then a equals sigma of z, then the loss. Under each node, a local gradient box gives the derivative of that node alone: dz/dw equals x and dz/db equals 1, da/dz equals a times one minus a, and dL/da. Below, global gradient boxes flow backward from the loss, each the upstream global gradient times the local gradient: dL/da, then dL/dz equals a minus y, then dL/dw equals x times a minus y and dL/db equals a minus y.</desc>']
    OUT.write_text("\n".join(svg + b + ["</svg>"]) + "\n")


if __name__ == "__main__":
    main()
