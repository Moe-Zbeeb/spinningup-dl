import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import BLUE, GRAY, GREEN, INK, MAROON, ORANGE, plt, save, setup, tidy

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs" / "_static" / "img" / "sequence_models"
FONT = "Lato, Helvetica Neue, Helvetica, Arial, sans-serif"
MATH = "STIX Two Text, Latin Modern Roman, Times New Roman, Times, serif"
LIGHT = "#eef6fb"


def box(x, y, w, h, text, fill=LIGHT, stroke=INK, size=15, bold=False, italic=False, rx=8):
    weight = ' font-weight="700"' if bold else ""
    style = ' font-style="italic"' if italic else ""
    family = MATH if italic else FONT
    lines = text.split("\n")
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>']
    for k, line in enumerate(lines):
        ty = y + h / 2 + (k - (len(lines) - 1) / 2) * (size + 3) + size * 0.35
        out.append(f'<text x="{x + w / 2}" y="{ty:.1f}" font-family="{family}" font-size="{size}"{weight}{style} fill="#000" text-anchor="middle">{line}</text>')
    return out


def text(x, y, s, size=15, color="#000", anchor="middle", bold=False, italic=False):
    weight = ' font-weight="700"' if bold else ""
    style = ' font-style="italic"' if italic else ""
    family = MATH if italic else FONT
    return [f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}"{weight}{style} fill="{color}" text-anchor="{anchor}">{s}</text>']


def arrow(x1, y1, x2, y2, color=INK, width=1.8, dash=None):
    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    bx, by = x2 - ux * 9, y2 - uy * 9
    px, py = -uy * 4.5, ux * 4.5
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return [f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>',
            f'<path d="M{x2:.1f} {y2:.1f} L{bx + px:.1f} {by + py:.1f} L{bx - px:.1f} {by - py:.1f} z" fill="{color}"/>']


def sup(base, s):
    return f'{base}<tspan font-size="11" dy="-6">⟨{s}⟩</tspan>'


def doc(name, w, h, title, desc, body):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">',
             f'<title id="t">{title}</title>', f'<desc id="d">{desc}</desc>'] + body + ["</svg>"]
    (OUT / name).write_text("\n".join(parts) + "\n")


def ner_notation():
    words = ["Layla", "and", "Karim", "visited", "Beirut", "in", "March"]
    labels = [1, 0, 1, 0, 0, 0, 0]
    body = []
    for t, (w, y) in enumerate(zip(words, labels), start=1):
        x = 20 + (t - 1) * 128
        body += text(x + 58, 28, sup("x", t), size=17, italic=True)
        body += box(x, 40, 116, 44, w, fill="#ffffff")
        body += box(x, 104, 116, 44, str(y), fill="#fbeef2" if y else LIGHT, stroke=MAROON if y else INK, bold=bool(y))
        body += text(x + 58, 172, sup("y", t), size=17, italic=True)
    doc("ner_notation.svg", 920, 186, "Inputs and labels for a named entity recognition example",
        "The sentence Layla and Karim visited Beirut in March, one word per box labeled x 1 to x 7, with a label below each word: 1 under Layla and Karim, which are people's names, and 0 under every other word.", body)


def rnn_unrolled():
    body = []
    xs = [150, 330, 510, 760]
    body += box(10, 120, 80, 50, sup("a", 0), fill="#ffffff", italic=True, size=18)
    for k, x in enumerate(xs):
        t = ["1", "2", "3", "T_x"][k]
        label_t = "T" if k == 3 else t
        body += box(x, 110, 110, 70, "RNN cell", fill=LIGHT)
        body += box(x + 20, 230, 70, 44, sup("x", label_t), fill="#ffffff", italic=True, size=18)
        body += box(x + 20, 20, 70, 44, sup("ŷ", label_t), fill="#ffffff", italic=True, size=18)
        body += arrow(x + 55, 230, x + 55, 182)
        body += arrow(x + 55, 110, x + 55, 66)
        body += text(x + 82, 210, "W_ax", size=12, color=GRAY)
        body += text(x + 80, 92, "W_ya", size=12, color=GRAY)
    body += arrow(90, 145, 150, 145)
    body += text(120, 136, "W_aa", size=12, color=GRAY)
    for a, b in ((260, 330), (440, 510)):
        body += arrow(a, 145, b, 145)
        body += text((a + b) / 2, 136, "W_aa", size=12, color=GRAY)
    body += arrow(620, 145, 650, 145)
    body += text(690, 150, "· · ·", size=22)
    body += arrow(720, 145, 760, 145)
    for k, x in enumerate(xs[:3]):
        body += text(x + 125 + (0 if k < 2 else 10), 165, sup("a", k + 1), size=14, italic=True)
    body += text(445, 300, "the same weights W_aa, W_ax, W_ya are used at every time step", size=15, color=MAROON, bold=True)
    doc("rnn_unrolled.svg", 900, 312, "A recurrent neural network unrolled in time",
        "A row of identical RNN cells. Each cell takes the input x at its time step from below and the previous activation a from the left, outputs a prediction y hat above, and passes its activation to the next cell. The same weights are used at every step.", body)


def rnn_types():
    body = []
    panels = [("One to one", [1], [1]), ("One to many", [1, 0, 0, 0], [1, 1, 1, 1]), ("Many to one", [1, 1, 1, 1], [0, 0, 0, 1]),
              ("Many to many, same length", [1, 1, 1, 1], [1, 1, 1, 1]), ("Many to many, encoder-decoder", [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1])]
    x0 = 10
    for title, ins, outs in panels:
        n = len(ins)
        cells = n * 46 - 14
        width = max(cells + 20, len(title) * 9.5)
        pad = (width - cells) / 2 - 10
        body += text(x0 + width / 2, 22, title, size=17, bold=True)
        for k in range(n):
            cx = x0 + 10 + pad + k * 46
            body.append(f'<rect x="{cx}" y="80" width="32" height="32" rx="6" fill="{LIGHT}" stroke="{INK}" stroke-width="1.6"/>')
            if ins[k]:
                body.append(f'<rect x="{cx + 6}" y="140" width="20" height="20" rx="4" fill="#ffffff" stroke="{BLUE}" stroke-width="1.6"/>')
                body += arrow(cx + 16, 140, cx + 16, 114, color=BLUE, width=1.4)
            if outs[k]:
                body.append(f'<rect x="{cx + 6}" y="34" width="20" height="20" rx="4" fill="#ffffff" stroke="{MAROON}" stroke-width="1.6"/>')
                body += arrow(cx + 16, 80, cx + 16, 56, color=MAROON, width=1.4)
            if k > 0:
                body += arrow(cx - 14, 96, cx, 96, width=1.4)
        x0 += width + 60
    body += text(x0 / 2, 196, "blue: inputs  ·  maroon: outputs  ·  arrows between cells carry the activation forward", size=18, color=INK)
    doc("rnn_types.svg", x0, 208, "Types of recurrent neural network architectures",
        "Five small diagrams: one to one with a single input and output; one to many with one input and four outputs; many to one with four inputs and one output at the end; many to many with an output at every input; and an encoder-decoder with three inputs followed by three outputs.", body)


def rnn_gradients():
    data = json.loads((HERE / "data" / "sequence_models_gradients.json").read_text())
    fig, ax = plt.subplots(figsize=(7.4, 4))
    steps = np.array(data["steps_back"])
    for kind, color, label in (("RNN", MAROON, "plain RNN"), ("LSTM", BLUE, "LSTM")):
        v = np.maximum(np.array(data[kind]), 1e-30)
        ax.plot(steps, v, color=color, lw=2.4, label=label)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 2)
    ax.set_xlim(0, 60)
    ax.set_xlabel("how many steps back the input is")
    ax.set_ylabel("size of its gradient\n(relative to the last input)")
    ax.legend(loc="upper right")
    tidy(ax)
    save(fig, OUT / "rnn_gradients.svg")


def lstm_cell():
    body = []
    body.append(f'<rect x="120" y="40" width="620" height="270" rx="16" fill="#fafafa" stroke="{INK}" stroke-width="1.4"/>')
    body += text(140, 66, "LSTM cell", size=15, bold=True, anchor="start")
    body += arrow(20, 100, 225, 100, width=2.4)
    body += text(60, 90, sup("c", "t−1"), size=17, italic=True)
    body.append(f'<circle cx="250" cy="100" r="22" fill="#ffffff" stroke="{MAROON}" stroke-width="2"/>')
    body += text(250, 107, "×", size=22, color=MAROON)
    body += arrow(272, 100, 448, 100, width=2.4)
    body.append(f'<circle cx="470" cy="100" r="22" fill="#ffffff" stroke="{GREEN}" stroke-width="2"/>')
    body += text(470, 107, "+", size=24, color=GREEN)
    body += arrow(492, 100, 850, 100, width=2.4)
    body += text(800, 90, sup("c", "t"), size=17, italic=True)
    body += text(360, 132, "memory cell: the long-term path", size=13, color=INK)
    sub = lambda base, k: f'{base}<tspan font-size="12" dy="4">{k}</tspan>'
    gates = [(170, sub("Γ", "f"), "forget", MAROON), (300, sub("Γ", "u"), "update", GREEN), (430, "c̃", "candidate", BLUE), (560, sub("Γ", "o"), "output", ORANGE)]
    for x, sym, name, color in gates:
        body += box(x, 210, 100, 44, f"{name}", fill="#ffffff", stroke=color, size=14)
        body += text(x + 50, 280, sym, size=17, italic=True, color=color)
    body.append(f'<polyline points="220,210 220,180 250,180 250,124" fill="none" stroke="{MAROON}" stroke-width="1.8"/>')
    body += arrow(250, 140, 250, 124, color=MAROON)
    body.append(f'<circle cx="420" cy="160" r="20" fill="#ffffff" stroke="{GREEN}" stroke-width="2"/>')
    body += text(420, 167, "×", size=20, color=GREEN)
    body += arrow(355, 210, 406, 174, color=GREEN)
    body += arrow(475, 210, 434, 174, color=BLUE)
    body += arrow(436, 146, 458, 120, color=GREEN)
    body += box(620, 140, 70, 36, "tanh", fill="#ffffff", size=13)
    body += arrow(655, 100, 655, 140)
    body.append(f'<circle cx="655" cy="196" r="0" fill="none"/>')
    body.append(f'<circle cx="720" cy="196" r="18" fill="#ffffff" stroke="{ORANGE}" stroke-width="2"/>')
    body += text(720, 203, "×", size=20, color=ORANGE)
    body += arrow(690, 158, 707, 182, color=INK)
    body += arrow(660, 222, 703, 202, color=ORANGE)
    body += arrow(738, 196, 850, 196, width=2.4)
    body += text(800, 186, sup("a", "t"), size=17, italic=True)
    body += arrow(20, 330, 250, 330, width=2)
    body += text(60, 350, sup("a", "t−1") + ", " + sup("x", "t"), size=16, italic=True)
    body.append(f'<line x1="250" y1="330" x2="610" y2="330" stroke="{INK}" stroke-width="2"/>')
    for x in (220, 350, 480, 610):
        body.append(f'<line x1="{x}" y1="330" x2="{x}" y2="256" stroke="{INK}" stroke-width="1.6"/>')
    doc("lstm_cell.svg", 870, 370, "The LSTM cell",
        "The memory cell c runs straight across the top. The forget gate multiplies the old memory, the update gate times the candidate is added to it, giving the new memory. The new memory goes through tanh and is multiplied by the output gate to give the activation a. All four gates read the previous activation and the current input.", body)


def brnn():
    body = []
    xs = [60, 260, 460, 660]
    for k, x in enumerate(xs, start=1):
        body += box(x, 40, 110, 40, sup("ŷ", k), fill="#ffffff", italic=True, size=17)
        body += box(x, 110, 110, 40, "forward", fill=LIGHT, stroke=BLUE, size=13)
        body += box(x, 180, 110, 40, "backward", fill="#fbeef2", stroke=MAROON, size=13)
        body += box(x, 250, 110, 40, sup("x", k), fill="#ffffff", italic=True, size=17)
        body += arrow(x + 30, 250, x + 30, 152, color=BLUE, width=1.4)
        body += arrow(x + 80, 250, x + 80, 222, color=MAROON, width=1.4)
        body += arrow(x + 30, 110, x + 30, 82, color=BLUE, width=1.4)
        body += arrow(x + 80, 180, x + 80, 82, color=MAROON, width=1.4)
    for a, b in zip(xs, xs[1:]):
        body += arrow(a + 110, 130, b, 130, color=BLUE, width=2)
        body += arrow(b, 200, a + 110, 200, color=MAROON, width=2)
    body += text(420, 318, "each prediction sees the whole sequence: the past through the forward layer, the future through the backward one", size=14, color=INK)
    doc("brnn.svg", 840, 330, "A bidirectional RNN",
        "Four time steps. A forward layer passes activations left to right and a backward layer passes them right to left. Each output y hat reads from both layers at its time step.", body)


def deep_rnn():
    body = []
    xs = [80, 270, 460, 650]
    for k, x in enumerate(xs, start=1):
        body += box(x, 20, 100, 40, sup("ŷ", k), fill="#ffffff", italic=True, size=17)
        for layer, y in ((3, 90), (2, 170), (1, 250)):
            body += box(x, y, 100, 44, f"a[{layer}]⟨{k}⟩", fill=LIGHT, size=14)
        body += box(x, 330, 100, 40, sup("x", k), fill="#ffffff", italic=True, size=17)
        for y1, y2 in ((330, 296), (250, 216), (170, 136), (90, 62)):
            body += arrow(x + 50, y1, x + 50, y2, width=1.4)
    for a, b in zip(xs, xs[1:]):
        for y in (112, 192, 272):
            body += arrow(a + 100, y, b, y, width=1.6)
    body += text(30, 116, "layer 3", size=13, color=INK, anchor="middle")
    body += text(30, 196, "layer 2", size=13, color=INK, anchor="middle")
    body += text(30, 276, "layer 1", size=13, color=INK, anchor="middle")
    doc("deep_rnn.svg", 780, 380, "A deep RNN with three layers",
        "Three recurrent layers stacked on top of each other over four time steps. Each layer passes activations to the right in time and up to the layer above. The top layer produces the outputs y hat.", body)


def word_embeddings():
    data = json.loads((HERE / "data" / "sequence_models_word2vec.json").read_text())
    colors = {"fruit": ORANGE, "drink": GREEN, "animal": MAROON, "city": BLUE, "number": "#6e6e80"}
    fig, ax = plt.subplots(figsize=(8, 5.6))
    xy = np.array(data["xy"])
    for g, color in colors.items():
        m = np.array([gg == g for gg in data["groups"]])
        ax.scatter(xy[m, 0], xy[m, 1], color=color, s=40, label=g, zorder=3)
    span = xy.max(0) - xy.min(0)
    center = xy.mean(0)
    for g, color in colors.items():
        m = np.array([gg == g for gg in data["groups"]])
        c = xy[m].mean(0)
        names = [w for w, gg in zip(data["words"], data["groups"]) if gg == g]
        direction = (c - center) / (np.linalg.norm(c - center) + 1e-9)
        tx, ty = c + direction * 0.12 * span.max()
        ha = "left" if direction[0] >= 0 else "right"
        ax.text(tx, ty, f"{g}:\n" + ", ".join(names[:3]) + ",\n" + ", ".join(names[3:]), fontsize=11, color=color, ha=ha, va="center")
    pad = 0.45 * span
    ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
    ax.set_ylim(xy[:, 1].min() - pad[1] * 0.75, xy[:, 1].max() + pad[1] * 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    tidy(ax, grid=False)
    save(fig, OUT / "word_embeddings.svg")


def embedding_matrix():
    body = []
    body.append(f'<rect x="20" y="30" width="340" height="150" fill="{LIGHT}" stroke="{INK}" stroke-width="1.6"/>')
    body.append(f'<rect x="230" y="30" width="18" height="150" fill="{MAROON}" fill-opacity="0.35" stroke="{MAROON}" stroke-width="1.6"/>')
    body += text(190, 210, "E: 300 × 10,000", size=15, bold=True)
    body += text(239, 22, "column j", size=13, color=MAROON)
    body += text(385, 112, "·", size=34)
    body.append(f'<rect x="410" y="10" width="22" height="200" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
    body.append(f'<rect x="410" y="128" width="22" height="10" fill="{MAROON}"/>')
    body += text(470, 137, "1 in", size=12, color=MAROON)
    body += text(470, 151, "row j", size=12, color=MAROON)
    body += text(421, 232, "o_j: 10,000 × 1", size=14, bold=True, anchor="middle")
    body += text(530, 112, "=", size=30)
    body.append(f'<rect x="570" y="30" width="22" height="150" fill="{MAROON}" fill-opacity="0.35" stroke="{MAROON}" stroke-width="1.6"/>')
    body += text(600, 210, "e_j: 300 × 1", size=14, bold=True, anchor="start")
    doc("embedding_matrix.svg", 720, 245, "Looking up an embedding with the embedding matrix",
        "A wide embedding matrix E of size 300 by 10,000 times a one-hot vector o j that has a single 1 in row j gives the column j of E, the 300-dimensional embedding e j.", body)


def negative_sampling():
    words = ["the", "of", "and", "glass", "orange", "juice", "durian"]
    counts = np.array([60000, 35000, 28000, 900, 400, 350, 5], dtype=float)
    dists = [("by frequency", counts, GRAY), ("frequency$^{3/4}$", counts ** 0.75, MAROON), ("uniform", np.ones_like(counts), BLUE)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.26
    x = np.arange(len(words))
    for k, (name, w, color) in enumerate(dists):
        p = w / w.sum()
        ax.bar(x + (k - 1) * width, p, width=width, color=color, label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(words)
    ax.set_ylabel("probability of being\nsampled as a negative")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    tidy(ax)
    save(fig, OUT / "negative_sampling.svg")


def debias():
    g = np.array([1.0, 0.0])
    neutral = lambda e: e - (e @ g) / (g @ g) * g
    babysitter = np.array([0.45, 0.80])
    gm, gf = np.array([0.75, 0.40]), np.array([-0.55, 0.62])
    mu = (gm + gf) / 2
    mu_orth = neutral(mu)
    scale = math.sqrt(1 - mu_orth @ mu_orth)
    gm_new = np.array([scale, 0]) + mu_orth
    gf_new = np.array([-scale, 0]) + mu_orth
    nb = neutral(babysitter)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, title in zip(axes, ("Neutralize", "Equalize")):
        ax.axhline(0, color=GRAY, lw=1)
        ax.axvline(0, color=GRAY, lw=1)
        ax.annotate("", xy=(1.15, 0), xytext=(-1.15, 0), arrowprops=dict(arrowstyle="<->", color=MAROON, lw=1.6))
        ax.text(1.12, -0.12, "bias direction (gender)", ha="right", fontsize=11, color=MAROON)
        ax.text(0.03, 1.08, "non-bias direction", fontsize=11, color=INK)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.25, 1.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        tidy(ax, title, grid=False)
    ax = axes[0]
    ax.plot(*babysitter, "o", color=GRAY, ms=9)
    ax.text(babysitter[0] + 0.04, babysitter[1] + 0.03, "babysitter (before)", fontsize=11, color=GRAY)
    ax.annotate("", xy=nb, xytext=babysitter, arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.plot(*nb, "o", color=BLUE, ms=9)
    ax.text(nb[0] + 0.04, nb[1] - 0.09, "babysitter (after)", fontsize=11, color=BLUE)
    ax = axes[1]
    for old, new, name in ((gm, gm_new, "grandmother"), (gf, gf_new, "grandfather")):
        ax.plot(*old, "o", color=GRAY, ms=8)
        ax.annotate("", xy=new, xytext=old, arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        ax.plot(*new, "o", color=ORANGE, ms=9)
        ax.text(new[0], new[1] - 0.12, name, fontsize=11, color=ORANGE, ha="center")
    ax.plot(*nb, "o", color=BLUE, ms=9)
    ax.text(nb[0] + 0.04, nb[1] + 0.03, "babysitter", fontsize=11, color=BLUE)
    for new in (gm_new, gf_new):
        ax.plot([new[0], nb[0]], [new[1], nb[1]], color=BLUE, lw=1, ls=(0, (3, 2)))
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "debias.svg")


def sentiment_models():
    body = []
    body += text(10, 24, "Averaging model", size=16, bold=True, anchor="start")
    words = ["the", "service", "was", "awful"]
    for k, w in enumerate(words):
        x = 20 + k * 120
        body += box(x, 40, 100, 36, w, fill="#ffffff", size=14)
        body += box(x, 100, 100, 36, "embedding", fill=LIGHT, size=13)
        body += arrow(x + 50, 76, x + 50, 98)
        body += arrow(x + 50, 136, 550, 175, width=1.2)
    body += box(500, 160, 110, 40, "average", fill="#f4f4f6", size=14)
    body += arrow(610, 180, 650, 180)
    body += box(650, 160, 100, 40, "softmax", fill="#fbeef2", stroke=MAROON, size=14)
    body += arrow(750, 180, 790, 180)
    body += box(790, 160, 70, 40, "ŷ", fill="#ffffff", italic=True, size=18)
    body += text(10, 254, "RNN model", size=16, bold=True, anchor="start")
    for k, w in enumerate(words):
        x = 20 + k * 150
        body += box(x, 270, 100, 36, w, fill="#ffffff", size=14)
        body += box(x, 330, 100, 36, "embedding", fill=LIGHT, size=13)
        body += box(x, 390, 100, 40, "RNN", fill=LIGHT, size=14)
        body += arrow(x + 50, 306, x + 50, 328)
        body += arrow(x + 50, 366, x + 50, 388)
        if k:
            body += arrow(x - 50, 410, x, 410)
    body += arrow(570, 410, 650, 410)
    body += box(650, 390, 100, 40, "softmax", fill="#fbeef2", stroke=MAROON, size=14)
    body += arrow(750, 410, 790, 410)
    body += box(790, 390, 70, 40, "ŷ", fill="#ffffff", italic=True, size=18)
    doc("sentiment_models.svg", 870, 440, "Two models for sentiment classification",
        "Top: each word of the review the service was awful is looked up as an embedding, the embeddings are averaged, and a softmax predicts the rating. Bottom: the embeddings feed an RNN one word at a time, and a softmax on the last activation predicts the rating.", body)


def seq2seq():
    body = []
    body += text(10, 24, "Machine translation", size=16, bold=True, anchor="start")
    src = ["le", "chat", "dort"]
    tgt = ["the", "cat", "sleeps", "&lt;EOS&gt;"]
    for k, w in enumerate(src):
        x = 20 + k * 110
        body += box(x, 100, 90, 44, "encoder", fill=LIGHT, size=13)
        body += box(x, 170, 90, 34, w, fill="#ffffff", size=14)
        body += arrow(x + 45, 170, x + 45, 146)
        if k:
            body += arrow(x - 20, 122, x, 122)
    body += arrow(330, 122, 400, 122, color=MAROON, width=2.4)
    body += text(365, 112, "a sentence", size=12, color=MAROON)
    body += text(365, 156, "encoding", size=12, color=MAROON)
    for k, w in enumerate(tgt):
        x = 400 + k * 110
        body += box(x, 100, 90, 44, "decoder", fill="#fbeef2", stroke=MAROON, size=13)
        body += box(x, 40, 90, 34, w, fill="#ffffff", size=14)
        body += arrow(x + 45, 100, x + 45, 76)
        if k:
            body += arrow(x - 20, 122, x, 122)
    body += text(10, 254, "Image captioning", size=16, bold=True, anchor="start")
    body += box(20, 300, 150, 50, "image", fill="#ffffff", size=14)
    body += arrow(170, 325, 220, 325)
    body += box(220, 300, 140, 50, "ConvNet", fill=LIGHT, size=14)
    body += arrow(360, 325, 400, 325, color=MAROON, width=2.4)
    for k, w in enumerate(["a", "cat", "on", "grass"]):
        x = 400 + k * 110
        body += box(x, 300, 90, 50, "decoder", fill="#fbeef2", stroke=MAROON, size=13)
        body += box(x, 268, 90, 26, w, fill="#ffffff", size=13)
        if k:
            body += arrow(x - 20, 325, x, 325)
    doc("seq2seq.svg", 850, 370, "Sequence-to-sequence models",
        "Top: an encoder RNN reads the French sentence le chat dort and passes its final activation to a decoder RNN that writes the cat sleeps end of sentence. Bottom: a ConvNet encodes an image and a decoder RNN writes the caption a cat on grass.", body)


def beam_tree():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    nodes = {
        "root": (0, 0, "start", None),
        "the": (1, 1.2, "the", 0.5), "a": (1, 0, "a", 0.4), "eos0": (1, -1.2, "<EOS>", 0.1),
        "the_cat": (2, 2.0, "cat", 0.15), "the_dog": (2, 1.2, "dog", 0.15), "the_eos": (2, 0.4, "<EOS>", 0.2),
        "a_cat": (2, -0.3, "cat", 0.36), "a_eos": (2, -0.9, "<EOS>", 0.04),
        "a_cat_eos": (3, -0.3, "<EOS>", 0.36), "the_cat_eos": (3, 2.0, "<EOS>", 0.15), "the_dog_eos": (3, 1.2, "<EOS>", 0.15),
    }
    edges = [("root", "the"), ("root", "a"), ("root", "eos0"), ("the", "the_cat"), ("the", "the_dog"), ("the", "the_eos"),
             ("a", "a_cat"), ("a", "a_eos"), ("a_cat", "a_cat_eos"), ("the_cat", "the_cat_eos"), ("the_dog", "the_dog_eos")]
    greedy = {("root", "the"), ("the", "the_eos")}
    best = {("root", "a"), ("a", "a_cat"), ("a_cat", "a_cat_eos")}
    for a, b in edges:
        xa, ya = nodes[a][:2]
        xb, yb = nodes[b][:2]
        color, lw = (BLUE, 2.6) if (a, b) in best else (ORANGE, 2.6) if (a, b) in greedy else (GRAY, 1.2)
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, zorder=1)
    for key, (x, y, label, p) in nodes.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=INK), zorder=3)
        if p is not None:
            ax.text(x, y - 0.32, f"P = {p:g}", ha="center", fontsize=10, color=INK)
    ax.text(0.4, -1.75, "orange: greedy search picks “the”, then must stop: P = 0.20", fontsize=12, color=ORANGE)
    ax.text(0.4, -2.15, "blue: beam search with B = 2 keeps “a” and finds “a cat”: P = 0.36", fontsize=12, color=BLUE)
    ax.set_xlim(-0.4, 3.5)
    ax.set_ylim(-2.4, 2.5)
    ax.axis("off")
    save(fig, OUT / "beam_search.svg")


def attention_heatmap():
    data = json.loads((HERE / "data" / "sequence_models_attention.json").read_text())
    alpha = np.array(data["alpha"])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(alpha, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(data["input"])))
    ax.set_xticklabels([str(v) for v in data["input"]])
    ax.set_yticks(range(len(data["output"])))
    ax.set_yticklabels([str(v) for v in data["output"]])
    ax.set_xlabel("input sequence (encoder step $t'$)")
    ax.set_ylabel("output sequence (decoder step $t$)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"attention weight $\alpha^{\langle t, t' \rangle}$")
    tidy(ax, grid=False)
    save(fig, OUT / "attention_heatmap.svg")
    return data["accuracy"]


def attention_model():
    body = []
    xs = [40, 170, 300, 430]
    words = ["x⟨1⟩", "x⟨2⟩", "x⟨3⟩", "x⟨4⟩"]
    for k, (x, w) in enumerate(zip(xs, words), start=1):
        body += box(x, 330, 90, 36, w, fill="#ffffff", italic=True, size=16)
        body += box(x, 260, 90, 44, "BRNN", fill=LIGHT, size=13)
        body += arrow(x + 45, 330, x + 45, 306)
        body += text(x + 45, 248, sup("a", k), size=15, italic=True)
    for k, x in enumerate(xs, start=1):
        body += arrow(x + 45, 236, 500, 160, color=BLUE, width=1.3)
        body += text((x + 45 + 500) / 2 - 20, (236 + 160) / 2 + 4, f"α⟨t,{k}⟩", size=12, italic=True, color=BLUE)
    body.append(f'<circle cx="520" cy="150" r="26" fill="#ffffff" stroke="{BLUE}" stroke-width="2"/>')
    body += text(520, 157, "Σ", size=20, color=BLUE)
    body += arrow(546, 150, 618, 150, color=BLUE, width=2)
    body += text(582, 140, sup("c", "t"), size=15, italic=True, color=BLUE)
    body += box(620, 120, 120, 60, "decoder\nstep t", fill="#fbeef2", stroke=MAROON, size=14)
    body += box(650, 30, 60, 36, sup("ŷ", "t"), fill="#ffffff", italic=True, size=16)
    body += arrow(680, 120, 680, 68)
    body += arrow(560, 230, 620, 180, color=MAROON, width=1.6)
    body += text(530, 250, sup("s", "t−1"), size=15, italic=True, color=MAROON)
    body += arrow(740, 150, 800, 150, color=MAROON, width=1.6)
    body += text(810, 155, sup("s", "t"), size=15, italic=True, color=MAROON, anchor="start")
    doc("attention_model.svg", 860, 380, "The attention model",
        "A bidirectional RNN encoder reads x 1 to x 4 and produces activations a 1 to a 4. Attention weights alpha t comma t prime combine them into a context c t, which feeds the decoder at step t together with the previous decoder state s t minus 1. The decoder outputs y hat t.", body)


def ctc_figure():
    frames = "hhh_eee_ll_lll__ooo"
    keep = []
    prev = None
    for ch in frames:
        keep.append(ch != prev and ch != "_")
        prev = ch
    fig, ax = plt.subplots(figsize=(11, 2.6))
    for i, (ch, k) in enumerate(zip(frames, keep)):
        face = "#fbeef2" if k else ("#f4f4f6" if ch == "_" else LIGHT)
        ax.add_patch(plt.Rectangle((i, 1), 0.92, 0.92, facecolor=face, edgecolor=MAROON if k else INK, lw=1.8 if k else 1))
        ax.text(i + 0.46, 1.46, ch, ha="center", va="center", fontsize=14, fontweight="bold" if k else "normal")
    ax.text(-0.2, 1.46, "RNN output\nper frame", ha="right", va="center", fontsize=12)
    ax.text(-0.2, 0.25, "after collapsing", ha="right", va="center", fontsize=12)
    ax.text(len(frames) / 2, 0.25, "h e l l o", ha="center", va="center", fontsize=18, color=MAROON, fontweight="bold")
    ax.set_xlim(-3, len(frames) + 0.3)
    ax.set_ylim(-0.2, 2.1)
    ax.axis("off")
    save(fig, OUT / "ctc.svg")


def trigger_word():
    rng = np.random.default_rng(3)
    T, F = 400, 40
    spec = rng.gamma(1.2, 0.3, (F, T))
    for start, length, kind in ((60, 45, "other"), (170, 50, "trigger"), (300, 45, "trigger")):
        band = np.exp(-((np.arange(F)[:, None] - (12 if kind == "trigger" else 25)) ** 2) / 40)
        spec[:, start:start + length] += 2.2 * band * (0.8 + 0.4 * rng.random(length))
    labels = np.zeros(T)
    for end in (220, 345):
        labels[end:end + 30] = 1
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].imshow(spec, aspect="auto", origin="lower", cmap="Blues")
    axes[0].set_yticks([])
    axes[0].set_ylabel("frequency")
    for start, length, label in ((60, 45, "another word"), (170, 50, "trigger word"), (300, 45, "trigger word")):
        axes[0].text(start + length / 2, F + 1, label, ha="center", va="bottom", fontsize=11, color=MAROON if label == "trigger word" else INK)
    axes[0].set_ylim(0, F + 7)
    axes[1].fill_between(np.arange(T), labels, step="mid", color=MAROON, alpha=0.6)
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].set_yticks([0, 1])
    axes[1].set_ylabel("label")
    axes[1].set_xlabel("time")
    axes[1].set_xticks([])
    for ax in axes:
        tidy(ax, grid=False)
    fig.tight_layout(h_pad=0.6)
    save(fig, OUT / "trigger_word.svg")


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    ner_notation()
    rnn_unrolled()
    rnn_types()
    rnn_gradients()
    lstm_cell()
    brnn()
    deep_rnn()
    word_embeddings()
    embedding_matrix()
    negative_sampling()
    debias()
    sentiment_models()
    seq2seq()
    beam_tree()
    print("attention accuracy:", attention_heatmap())
    attention_model()
    ctc_figure()
    trigger_word()


if __name__ == "__main__":
    main()
