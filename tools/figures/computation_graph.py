import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import BLUE, INK, MAROON

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "_static" / "img" / "neural_networks" / "computation_graph.svg"
MATH = "STIX Two Text, Latin Modern Roman, Times New Roman, Times, serif"
SANS = "Lato, Helvetica Neue, Helvetica, Arial, sans-serif"


def box(x, y, w, h, label, stroke=INK):
    return [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#ffffff" stroke="{stroke}" stroke-width="2"/>',
            f'<text x="{x + w / 2}" y="{y + h / 2 + 7}" font-family="{MATH}" font-style="italic" font-size="21" fill="#000" text-anchor="middle">{label}</text>']


def head(x2, y2, ux, uy, color):
    bx, by = x2 - ux * 10, y2 - uy * 10
    px, py = -uy * 5, ux * 5
    return f'<path d="M{x2:.1f} {y2:.1f} L{bx + px:.1f} {by + py:.1f} L{bx - px:.1f} {by - py:.1f} z" fill="{color}"/>'


def line(x1, y1, x2, y2, color=BLUE):
    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    return [f'<line x1="{x1}" y1="{y1}" x2="{x2 - ux * 9:.1f}" y2="{y2 - uy * 9:.1f}" stroke="{color}" stroke-width="2.2"/>', head(x2, y2, ux, uy, color)]


def curve(x1, y1, cx, cy, x2, y2, color=MAROON):
    d = math.hypot(x2 - cx, y2 - cy)
    ux, uy = (x2 - cx) / d, (y2 - cy) / d
    return [f'<path d="M{x1} {y1} Q{cx} {cy} {x2 - ux * 9:.1f} {y2 - uy * 9:.1f}" fill="none" stroke="{color}" stroke-width="2.2" stroke-dasharray="7 5"/>', head(x2, y2, ux, uy, color)]


def label(x, y, text, color=MAROON, size=17, anchor="middle"):
    return [f'<text x="{x}" y="{y}" font-family="{MATH}" font-style="italic" font-size="{size}" fill="{color}" text-anchor="{anchor}">{text}</text>']


def frac(x, y, num, den, color=MAROON, size=16):
    return [f'<text x="{x}" y="{y - 6}" font-family="{MATH}" font-style="italic" font-size="{size}" fill="{color}" text-anchor="middle">{num}</text>',
            f'<line x1="{x - 16}" y1="{y}" x2="{x + 16}" y2="{y}" stroke="{color}" stroke-width="1.2"/>',
            f'<text x="{x}" y="{y + 16}" font-family="{MATH}" font-style="italic" font-size="{size}" fill="{color}" text-anchor="middle">{den}</text>']


def main():
    W, H = 1000, 380
    body = []
    body += box(30, 60, 60, 44, "w")
    body += box(30, 158, 60, 44, "x")
    body += box(30, 256, 60, 44, "b")
    body += box(200, 158, 170, 44, "z = w<tspan font-size='14' dy='-8'>⊤</tspan><tspan dy='8'>x + b</tspan>")
    body += box(490, 158, 130, 44, "a = σ(z)")
    body += box(800, 158, 130, 44, "ℒ(a, y)")
    body += box(835, 290, 60, 44, "y")
    body += line(90, 82, 200, 172)
    body += line(90, 180, 200, 180)
    body += line(90, 278, 200, 188)
    body += line(370, 180, 490, 180)
    body += line(620, 180, 800, 180)
    body += line(865, 290, 865, 202)
    body += curve(800, 166, 710, 95, 620, 166)
    body += label(710, 116, "∂ℒ / ∂a = −y / a + (1 − y) / (1 − a)", size=18)
    body += curve(490, 166, 430, 105, 370, 166)
    body += label(430, 124, "∂a / ∂z = a(1 − a)", size=18)
    body += label(285, 236, "dz = a − y", size=19)
    body += curve(212, 158, 160, 70, 90, 70)
    body += label(155, 52, "dw = x dz", size=18)
    body += curve(212, 202, 160, 290, 90, 290)
    body += label(155, 328, "db = dz", size=18)
    body.append(f'<line x1="250" y1="360" x2="290" y2="360" stroke="{BLUE}" stroke-width="2.2"/>')
    body.append(f'<text x="298" y="365" font-family="{SANS}" font-size="15" fill="{INK}">forward pass: compute the values</text>')
    body.append(f'<line x1="560" y1="360" x2="600" y2="360" stroke="{MAROON}" stroke-width="2.2" stroke-dasharray="7 5"/>')
    body.append(f'<text x="608" y="365" font-family="{SANS}" font-size="15" fill="{INK}">backward pass: chain rule</text>')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
           '<title id="t">Computation graph of logistic regression</title>',
           '<desc id="d">Inputs w, x, and b feed z equals w transpose x plus b, which feeds a equals sigma of z, which feeds the loss L of a and y. Solid blue arrows run forward from left to right. Dashed maroon arrows run backward from the loss, labeled with the derivative of the loss with respect to a, the derivative of a with respect to z, dz equals a minus y, dw equals x dz, and db equals dz.</desc>']
    OUT.write_text("\n".join(svg + body + ["</svg>"]) + "\n")


if __name__ == "__main__":
    main()
