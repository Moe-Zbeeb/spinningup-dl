import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import BLUE, GRAY, GREEN, INK, MAROON, ORANGE, plt, save, setup, tidy

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs" / "_static" / "img" / "convnets"
FONT = "Lato, Helvetica Neue, Helvetica, Arial, sans-serif"
LIGHT = "#eef6fb"


def conv2d(image, kernel):
    f = kernel.shape[0]
    h, w = image.shape[0] - f + 1, image.shape[1] - f + 1
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(image[i:i + f, j:j + f] * kernel)
    return out


def draw_grid(ax, values, x0, y0, cell=1.0, cmap="Greys_r", vmin=None, vmax=None, fmt="{:g}", fontsize=12, highlight=None):
    values = np.asarray(values, dtype=float)
    n_r, n_c = values.shape
    lo = values.min() if vmin is None else vmin
    hi = values.max() if vmax is None else vmax
    cm = plt.get_cmap(cmap)
    for i in range(n_r):
        for j in range(n_c):
            v = values[i, j]
            t = 0.5 if hi == lo else (v - lo) / (hi - lo)
            color = cm(0.25 + 0.6 * t)
            ax.add_patch(plt.Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, facecolor=color, edgecolor="#ffffff", lw=1.2))
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, fmt.format(v), ha="center", va="center", fontsize=fontsize, color="#ffffff" if lum < 0.5 else "#000000")
    if highlight is not None:
        r, c, h, w = highlight
        ax.add_patch(plt.Rectangle((x0 + c * cell, y0 - (r + h) * cell), w * cell, h * cell, fill=False, edgecolor=MAROON, lw=2.6))


def conv_example():
    image = np.array([[10] * 3 + [0] * 3] * 6, dtype=float)
    kernel = np.array([[1, 0, -1]] * 3, dtype=float)
    out = conv2d(image, kernel)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    draw_grid(ax, image, 0, 6, highlight=(0, 0, 3, 3))
    ax.text(6.8, 3, r"$\ast$", fontsize=30, ha="center", va="center")
    draw_grid(ax, kernel, 7.6, 4.5, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.text(11.4, 3, "=", fontsize=30, ha="center", va="center")
    draw_grid(ax, out, 12.2, 5, cmap="Blues", vmin=-10, vmax=30, highlight=(0, 0, 1, 1))
    ax.text(3, 6.5, "image (6 × 6)", ha="center", fontsize=13)
    ax.text(9.1, 5.0, "filter (3 × 3)", ha="center", fontsize=13)
    ax.text(14.2, 5.5, "output (4 × 4)", ha="center", fontsize=13)
    ax.set_xlim(-0.3, 16.5)
    ax.set_ylim(-0.4, 7.1)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, OUT / "conv_example.svg")


def scene(size=96):
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), 0.85)
    img[(xx > 20) & (xx < 64) & (yy > 44) & (yy < 86)] = 0.35
    roof = (yy > 18) & (yy <= 44) & (np.abs(xx - 42) < (yy - 18) * 0.95)
    img[roof] = 0.15
    img[(xx > 36) & (xx < 48) & (yy > 62) & (yy < 86)] = 0.95
    img[(xx - 78) ** 2 + (yy - 22) ** 2 < 100] = 0.55
    img[yy >= 86] = 0.6
    return img


def edge_detection():
    img = scene()
    sobel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    vert = np.abs(conv2d(img, sobel_v))
    horiz = np.abs(conv2d(img, sobel_v.T))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, data, title, cmap in zip(axes, (img, vert, horiz), ("Image", "Vertical edge filter", "Horizontal edge filter"), ("gray", "Blues", "Blues")):
        ax.imshow(data, cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        tidy(ax, title, grid=False)
    save(fig, OUT / "edge_detection.svg")


def padding_stride():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    n, p = 5, 1
    for i in range(n + 2 * p):
        for j in range(n + 2 * p):
            pad = i < p or j < p or i >= n + p or j >= n + p
            ax.add_patch(plt.Rectangle((j, -i - 1), 1, 1, facecolor="#e4e4e8" if pad else LIGHT, edgecolor="#ffffff", lw=1.5))
            ax.text(j + 0.5, -i - 0.5, "0" if pad else "", ha="center", va="center", fontsize=11, color=GRAY)
    ax.add_patch(plt.Rectangle((0, -3), 3, 3, fill=False, edgecolor=MAROON, lw=2.8))
    ax.set_title("Padding p = 1: a 5 × 5 image becomes 7 × 7", fontsize=14, fontweight="bold", color="#000000")
    ax.text(3.5, -7.8, "a 3 × 3 filter now fits in 5 × 5 positions,\nso the output stays 5 × 5", ha="center", fontsize=12)
    ax = axes[1]
    n = 7
    for i in range(n):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j, -i - 1), 1, 1, facecolor=LIGHT, edgecolor="#ffffff", lw=1.5))
    for (r, c), color, ls in (((0, 0), MAROON, "-"), ((0, 2), ORANGE, "--"), ((0, 4), GREEN, ":")):
        ax.add_patch(plt.Rectangle((c + 0.06, -r - 3 + 0.06), 2.88, 2.88, fill=False, edgecolor=color, lw=2.6, ls=ls))
    ax.set_title("Stride s = 2: the filter jumps 2 pixels", fontsize=14, fontweight="bold", color="#000000")
    ax.text(3.5, -7.8, "on a 7 × 7 image a 3 × 3 filter fits\n3 times per row, so the output is 3 × 3", ha="center", fontsize=12)
    for ax in axes:
        ax.set_xlim(-0.3, 7.3)
        ax.set_ylim(-8.7, 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(w_pad=3)
    save(fig, OUT / "padding_stride.svg")


def cuboid(ax, x, y, w, h, d, face=LIGHT, edge=INK, label=None, below=None, label_size=12):
    dx, dy = d * 0.6, d * 0.45
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)]
    side = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    for poly, shade in ((side, 0.88), (top, 0.94), (front, 1.0)):
        ax.add_patch(plt.Polygon(poly, closed=True, facecolor=face, edgecolor=edge, lw=1.3, alpha=shade))
    if label:
        ax.text(x + w / 2 + dx / 2, y + h + dy + 0.25, label, ha="center", va="bottom", fontsize=label_size)
    if below:
        ax.text(x + w / 2, y - 0.3, below, ha="center", va="top", fontsize=label_size - 1, color=INK)


def conv_volume():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    cuboid(ax, 0, 0, 3.0, 3.0, 1.6, label="input 6 × 6 × 3")
    ax.text(4.6, 1.5, r"$\ast$", fontsize=26, ha="center", va="center")
    cuboid(ax, 5.3, 1.9, 1.5, 1.5, 1.6, face="#fbeef2", edge=MAROON, below=None, label="filter 1: 3 × 3 × 3")
    cuboid(ax, 5.3, -0.9, 1.5, 1.5, 1.6, face="#fdf0e4", edge=ORANGE, label=None, below="filter 2: 3 × 3 × 3")
    ax.text(8.9, 1.5, "=", fontsize=26, ha="center", va="center")
    ax.add_patch(plt.Rectangle((9.7, 1.7), 2, 2, facecolor="#fbeef2", edgecolor=MAROON, lw=1.3))
    ax.add_patch(plt.Rectangle((9.7, -0.7), 2, 2, facecolor="#fdf0e4", edgecolor=ORANGE, lw=1.3))
    ax.text(10.7, 4.0, "two 4 × 4 outputs", ha="center", fontsize=12)
    ax.annotate("", xy=(13.4, 1.5), xytext=(12.2, 1.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(12.8, 1.9, "stack", ha="center", fontsize=12)
    cuboid(ax, 13.8, 0.5, 2.0, 2.0, 0.9, face="#f4eff2", label="output 4 × 4 × 2")
    ax.set_xlim(-0.3, 17.2)
    ax.set_ylim(-1.8, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, OUT / "conv_volume.svg")


def pooling():
    x = np.array([[1, 3, 2, 1], [2, 9, 1, 1], [1, 3, 2, 3], [5, 6, 1, 2]], dtype=float)
    mx = x.reshape(2, 2, 2, 2).max(axis=(1, 3))
    av = x.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    colors = [["#d8ebf7", "#fbe3ea"], ["#fde7d4", "#d9f2ea"]]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i in range(4):
        for j in range(4):
            ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1, facecolor=colors[i // 2][j // 2], edgecolor="#ffffff", lw=1.5))
            ax.text(j + 0.5, 3.5 - i, f"{x[i, j]:g}", ha="center", va="center", fontsize=14)
    for k, (data, x0, title) in enumerate(((mx, 6.2, "max pooling"), (av, 10.2, "average pooling"))):
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((x0 + j, 2 - i), 1, 1, facecolor=colors[i][j], edgecolor="#ffffff", lw=1.5))
                ax.text(x0 + j + 0.5, 2.5 - i, f"{data[i, j]:g}", ha="center", va="center", fontsize=14)
        ax.text(x0 + 1, 3.3, title, ha="center", fontsize=13)
    ax.annotate("", xy=(5.9, 2), xytext=(4.3, 2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(5.1, 2.3, "f = 2, s = 2", ha="center", fontsize=12)
    ax.text(2, 4.4, "input 4 × 4", ha="center", fontsize=13)
    ax.set_xlim(-0.3, 12.5)
    ax.set_ylim(-0.3, 4.9)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, OUT / "pooling.svg")


def lenet_like():
    fig, ax = plt.subplots(figsize=(12, 3.8))
    vols = [(32, 3, "32 × 32 × 3", "input"), (28, 6, "28 × 28 × 6", "conv 5 × 5"), (14, 6, "14 × 14 × 6", "max pool"),
            (10, 16, "10 × 10 × 16", "conv 5 × 5"), (5, 16, "5 × 5 × 16", "max pool")]
    x = 0.0
    for size, depth, label, op in vols:
        h = 0.5 + size * 0.09
        d = 0.25 + depth * 0.07
        cuboid(ax, x, 1.6 - h / 2, h, h, d, label=label, below=op, label_size=11)
        x += h + d * 0.6 + 0.85
    for k, (units, label) in enumerate(((400, "400"), (120, "FC 120"), (84, "FC 84"), (10, "softmax 10"))):
        h = 0.6 + units * 0.006
        ax.add_patch(plt.Rectangle((x, 1.6 - h / 2), 0.32, h, facecolor=LIGHT, edgecolor=INK, lw=1.3))
        ax.text(x + 0.16, (1.6 - h / 2 - 0.3) if k % 2 == 0 else (1.6 + h / 2 + 0.3), label, ha="center", va="top" if k % 2 == 0 else "bottom", fontsize=11)
        x += 1.6
    ax.set_xlim(-0.3, x)
    ax.set_ylim(-0.5, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, OUT / "lenet_like.svg")


def resnet_depth():
    data = json.loads((HERE / "data" / "convnets_resnet_experiment.json").read_text())
    fig, ax = plt.subplots(figsize=(7.4, 4))
    ax.plot(data["depths"], data["plain"], color=MAROON, lw=2.4, marker="o", label="plain network")
    ax.plot(data["depths"], data["residual"], color=BLUE, lw=2.4, marker="o", label="residual network")
    ax.axhline(np.log(2), color=GRAY, lw=1.2, ls=(0, (4, 3)))
    ax.text(102, np.log(2) + 0.02, "chance level", ha="right", fontsize=12, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(data["depths"])
    ax.set_xticklabels([str(d) for d in data["depths"]])
    ax.set_xlabel("number of layers")
    ax.set_ylabel("training loss after training")
    ax.set_ylim(-0.03, 0.8)
    ax.legend(loc="center left")
    tidy(ax)
    save(fig, OUT / "resnet_depth.svg")


def svg_box(x, y, w, h, text, fill=LIGHT, stroke=INK, size=15, bold=False):
    weight = ' font-weight="700"' if bold else ""
    lines = text.split("\n")
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>']
    for k, line in enumerate(lines):
        ty = y + h / 2 + (k - (len(lines) - 1) / 2) * (size + 3) + size * 0.35
        out.append(f'<text x="{x + w / 2}" y="{ty:.1f}" font-family="{FONT}" font-size="{size}"{weight} fill="#000" text-anchor="middle">{line}</text>')
    return out


def svg_arrow(x1, y1, x2, y2, color=INK):
    import math

    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    bx, by = x2 - ux * 10, y2 - uy * 10
    px, py = -uy * 5, ux * 5
    return [f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="1.8"/>',
            f'<path d="M{x2:.1f} {y2:.1f} L{bx + px:.1f} {by + py:.1f} L{bx - px:.1f} {by - py:.1f} z" fill="{color}"/>']


def svg_doc(path, w, h, title, desc, body):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">',
             f'<title id="t">{title}</title>', f'<desc id="d">{desc}</desc>'] + body + ["</svg>"]
    path.write_text("\n".join(parts) + "\n")


def resnet_block():
    body = []
    body += svg_box(10, 95, 90, 46, "a[l]", fill="#ffffff")
    body += svg_box(150, 95, 140, 46, "linear + ReLU")
    body += svg_box(340, 95, 90, 46, "a[l+1]", fill="#ffffff")
    body += svg_box(480, 95, 120, 46, "linear")
    body += svg_box(650, 95, 60, 46, "+", fill="#fbeef2", stroke=MAROON, size=22, bold=True)
    body += svg_box(760, 95, 80, 46, "ReLU")
    body += svg_box(890, 95, 100, 46, "a[l+2]", fill="#ffffff")
    for x1, x2 in ((100, 150), (290, 340), (430, 480), (600, 650), (710, 760), (840, 890)):
        body += svg_arrow(x1, 118, x2, 118)
    body.append(f'<path d="M55 95 C55 20, 680 20, 680 88" fill="none" stroke="{MAROON}" stroke-width="2.4"/>')
    body.append(f'<path d="M674 80 L680 92 L686 80 z" fill="{MAROON}"/>')
    body.append(f'<text x="500" y="196" font-family="{FONT}" font-size="16" font-weight="700" fill="{MAROON}" text-anchor="middle">shortcut: a[l] skips ahead and is added before the last ReLU</text>')
    body.append(f'<text x="540" y="162" font-family="{FONT}" font-size="14" fill="{INK}" text-anchor="middle">z[l+2]</text>')
    svg_doc(OUT / "resnet_block.svg", 1000, 210, "A residual block",
            "The activation a of layer l passes through a linear layer and ReLU, then another linear layer, and the result is added to a of layer l carried along a shortcut, before a final ReLU produces a of layer l plus 2.", body)


def bottleneck():
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6))
    ax = axes[0]
    cuboid(ax, 0, 0, 2.2, 2.2, 3.0, label="28 × 28 × 192")
    ax.annotate("", xy=(6.3, 1.1), xytext=(4.4, 1.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(5.35, 1.5, "conv 5 × 5,\n32 filters", ha="center", fontsize=12)
    cuboid(ax, 6.6, 0, 2.2, 2.2, 0.7, label="28 × 28 × 32")
    ax.text(10.5, 1.1, "120.4 million\nmultiplications", fontsize=14, va="center", color=MAROON, fontweight="bold")
    ax = axes[1]
    cuboid(ax, 0, 0, 2.2, 2.2, 3.0, label="28 × 28 × 192")
    ax.annotate("", xy=(6.3, 1.1), xytext=(4.4, 1.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(5.35, 1.5, "conv 1 × 1,\n16 filters", ha="center", fontsize=12)
    cuboid(ax, 6.6, 0, 2.2, 2.2, 0.45, face="#fbeef2", edge=MAROON, label="28 × 28 × 16")
    ax.annotate("", xy=(11.4, 1.1), xytext=(9.5, 1.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(10.45, 1.5, "conv 5 × 5,\n32 filters", ha="center", fontsize=12)
    cuboid(ax, 11.7, 0, 2.2, 2.2, 0.7, label="28 × 28 × 32")
    ax.text(15.0, 1.1, "2.4 + 10.0 = 12.4\nmillion", fontsize=14, va="center", color=BLUE, fontweight="bold")
    for ax in axes:
        ax.set_xlim(-0.3, 18)
        ax.set_ylim(-0.5, 4.3)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].set_title("Direct", fontsize=14, fontweight="bold", color="#000000", loc="left")
    axes[1].set_title("With a 1 × 1 bottleneck", fontsize=14, fontweight="bold", color="#000000", loc="left")
    fig.tight_layout(h_pad=1)
    save(fig, OUT / "bottleneck.svg")


def inception_module():
    body = []
    body += svg_box(10, 170, 150, 60, "previous\nactivation\n28 × 28 × 192", fill="#ffffff", size=14)
    rows = [(30, [("1 × 1 conv, 64", LIGHT)]), (120, [("1 × 1 conv, 96", "#fbeef2"), ("3 × 3 conv, 128", LIGHT)]),
            (210, [("1 × 1 conv, 16", "#fbeef2"), ("5 × 5 conv, 32", LIGHT)]), (300, [("3 × 3 max pool,\nsame", "#f4f4f6"), ("1 × 1 conv, 32", "#fbeef2")])]
    for y, boxes in rows:
        body += svg_arrow(160, 200, 235, y + 23)
        if len(boxes) == 1:
            body += svg_box(240, y, 340, 46, boxes[0][0], fill=boxes[0][1])
            body += svg_arrow(580, y + 23, 655, 200)
        else:
            body += svg_box(240, y, 150, 46, boxes[0][0], fill=boxes[0][1], size=14)
            body += svg_arrow(390, y + 23, 430, y + 23)
            body += svg_box(430, y, 150, 46, boxes[1][0], fill=boxes[1][1], size=14)
            body += svg_arrow(580, y + 23, 655, 200)
    body += svg_box(660, 170, 150, 60, "concatenate\nchannels", fill="#ffffff", size=14, bold=True)
    body += svg_arrow(810, 200, 850, 200)
    body += svg_box(850, 170, 140, 60, "28 × 28 × 256", fill="#ffffff", size=14)
    body.append(f'<text x="500" y="378" font-family="{FONT}" font-size="14" fill="{MAROON}" text-anchor="middle">red: 1 × 1 convolutions that shrink the number of channels</text>')
    svg_doc(OUT / "inception_module.svg", 1000, 390, "An inception module",
            "The previous activation feeds four parallel branches: a 1 by 1 convolution; a 1 by 1 convolution followed by a 3 by 3 convolution; a 1 by 1 convolution followed by a 5 by 5 convolution; and a 3 by 3 max pool followed by a 1 by 1 convolution. Their outputs are concatenated along the channels.", body)


def color_scene(size=128):
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3))
    img[:] = (0.62, 0.8, 0.95)
    img[yy > 92] = (0.45, 0.72, 0.38)
    body = ((xx - 60) / 26) ** 2 + ((yy - 82) / 16) ** 2 < 1
    head = (xx - 88) ** 2 + (yy - 62) ** 2 < 13 ** 2
    ear1 = (yy < 52) & (yy > 40) & (np.abs(xx - 81) < (yy - 40) * 0.45)
    ear2 = (yy < 52) & (yy > 40) & (np.abs(xx - 95) < (yy - 40) * 0.45)
    tail = (np.abs(yy - (80 - (xx - 34) * 0.9)) < 3) & (xx > 18) & (xx < 36)
    cat = body | head | ear1 | ear2 | tail
    img[cat] = (0.93, 0.6, 0.25)
    img[((xx - 84) ** 2 + (yy - 60) ** 2 < 5) | ((xx - 93) ** 2 + (yy - 60) ** 2 < 5)] = (0.1, 0.1, 0.1)
    img[(xx - 22) ** 2 + (yy - 20) ** 2 < 120] = (1.0, 0.9, 0.3)
    return img


def sample(img, fn):
    size = img.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    sx, sy = fn(xx, yy)
    sx = np.clip(np.round(sx), 0, size - 1).astype(int)
    sy = np.clip(np.round(sy), 0, size - 1).astype(int)
    return img[sy, sx]


def augmentation():
    img = color_scene()
    s = img.shape[0]
    c = (s - 1) / 2
    ang = np.deg2rad(15)
    versions = [
        ("Original", img),
        ("Mirrored", img[:, ::-1]),
        ("Random crop", sample(img, lambda x, y: 40 + x * 0.65, ) if False else sample(img, lambda x, y: (40 + x * 0.65, 30 + y * 0.65))),
        ("Rotated 15°", sample(img, lambda x, y: (c + (x - c) * np.cos(ang) - (y - c) * np.sin(ang), c + (x - c) * np.sin(ang) + (y - c) * np.cos(ang)))),
        ("Sheared", sample(img, lambda x, y: (x + 0.3 * (y - c), y))),
        ("Color shifted", np.clip(img * np.array([0.8, 1.05, 1.25]) + np.array([0.0, 0.02, 0.05]), 0, 1)),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.7))
    for ax, (title, data) in zip(axes, versions):
        ax.imshow(data, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        tidy(ax, grid=False)
        ax.set_title(title, fontsize=13, color="#000000")
    fig.tight_layout(w_pad=0.6)
    save(fig, OUT / "augmentation.svg")


def data_spectrum():
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.annotate("", xy=(10, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
    for x, name in ((2.2, "object\ndetection"), (5.0, "image\nrecognition"), (7.8, "speech\nrecognition")):
        ax.plot(x, 0, "o", color=BLUE, ms=10)
        ax.text(x, 0.35, name, ha="center", va="bottom", fontsize=13)
    ax.text(0, -0.45, "little data:\nmore hand-engineering,\nspecialized architectures", ha="left", va="top", fontsize=12, color=MAROON)
    ax.text(10, -0.45, "lots of data:\nsimpler algorithms,\nless hand-engineering", ha="right", va="top", fontsize=12, color=BLUE)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-1.7, 1.4)
    ax.axis("off")
    save(fig, OUT / "data_spectrum.svg")


def localization():
    img = color_scene()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.imshow(img, extent=(0, 1, 1, 0))
    x0, y0, x1, y1 = 18 / 128, 38 / 128, 103 / 128, 100 / 128
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=MAROON, lw=2.6))
    bx, by = (x0 + x1) / 2, (y0 + y1) / 2
    ax.plot(bx, by, "o", color=MAROON, ms=7)
    ax.annotate("", xy=(x1, y1 + 0.06), xytext=(x0, y1 + 0.06), arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(bx, y1 + 0.09, f"$b_w = {x1 - x0:.2f}$", ha="center", va="top", fontsize=12)
    ax.annotate("", xy=(x1 + 0.05, y1), xytext=(x1 + 0.05, y0), arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(x1 + 0.07, by, f"$b_h = {y1 - y0:.2f}$", va="center", fontsize=12)
    ax.text(0.02, 1.16, f"center $(b_x, b_y) = ({bx:.2f}, {by:.2f})$", fontsize=12, color="#000000")
    ax.set_xlim(-0.02, 1.25)
    ax.set_ylim(1.2, -0.02)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    tidy(ax, "Classification with localization", grid=False)
    ax = axes[1]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 + 0.32 * np.cos(t), 0.5 + 0.42 * np.sin(t), color=INK, lw=2)
    pts = [(0.37, 0.4), (0.43, 0.4), (0.57, 0.4), (0.63, 0.4), (0.5, 0.55), (0.4, 0.7), (0.5, 0.74), (0.6, 0.7), (0.3, 0.3), (0.7, 0.3)]
    ax.plot([0.36, 0.44], [0.4, 0.4], color=INK, lw=2)
    ax.plot([0.56, 0.64], [0.4, 0.4], color=INK, lw=2)
    ax.plot([0.5, 0.48, 0.52], [0.45, 0.55, 0.55], color=INK, lw=1.6)
    ax.plot([0.4, 0.5, 0.6], [0.7, 0.74, 0.7], color=INK, lw=2)
    ax.plot([0.28, 0.44], [0.3, 0.28], color=INK, lw=2)
    ax.plot([0.56, 0.72], [0.28, 0.3], color=INK, lw=2)
    for k, (x, y) in enumerate(pts, start=1):
        ax.plot(x, y, "o", color=MAROON, ms=8)
        ax.text(x + 0.015, y - 0.02, str(k), fontsize=10, color=MAROON)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    tidy(ax, "Landmark detection", grid=False)
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "localization.svg")


def sliding_windows():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    n = 16
    for i in range(n):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j * 0.3, -(i + 1) * 0.3), 0.3, 0.3, facecolor=LIGHT, edgecolor="#ffffff", lw=0.8))
    wins = [((0, 0), MAROON, "-"), ((0, 2), ORANGE, "--"), ((2, 0), GREEN, "-."), ((2, 2), BLUE, ":")]
    for (r, c), color, ls in wins:
        off = 0.04 * (r + c) / 2
        ax.add_patch(plt.Rectangle((c * 0.3 + off, -(r + 14) * 0.3 - off), 14 * 0.3, 14 * 0.3, fill=False, edgecolor=color, lw=2.4, ls=ls))
    ax.text(2.4, 0.25, "16 × 16 image: four 14 × 14 windows, stride 2", ha="center", fontsize=13)
    ax.annotate("", xy=(8.2, -2.4), xytext=(5.4, -2.4), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.text(6.8, -2.0, "one pass of\nthe ConvNet", ha="center", fontsize=12)
    for (r, c), color, _ in wins:
        ax.add_patch(plt.Rectangle((8.6 + c / 2 * 1.2, -1.2 - r / 2 * 1.2 - 1.2), 1.2, 1.2, facecolor=color, alpha=0.35, edgecolor=color, lw=2))
    ax.text(9.8, -0.75, "2 × 2 output:\none prediction per window", ha="center", fontsize=12)
    ax.set_xlim(-0.2, 11.6)
    ax.set_ylim(-5.2, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, OUT / "sliding_windows.svg")


def yolo_grid():
    img = color_scene()
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.imshow(img, extent=(0, 3, 3, 0))
    for k in range(1, 3):
        ax.axvline(k, color="#ffffff", lw=2)
        ax.axhline(k, color="#ffffff", lw=2)
    x0, y0, x1, y1 = 18 / 128 * 3, 38 / 128 * 3, 103 / 128 * 3, 100 / 128 * 3
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=MAROON, lw=2.6))
    ax.add_patch(plt.Rectangle((1, 1), 1, 1, facecolor=MAROON, alpha=0.18, edgecolor=MAROON, lw=2.4))
    ax.plot(cx, cy, "o", color=MAROON, ms=8)
    ax.text(1.04, 1.12, "this cell holds\nthe box's center", fontsize=11, color="#000000", va="top")
    ax.text(0.05, 2.95, f"cell-relative: $b_x = {cx - 1:.2f}$, $b_y = {cy - 1:.2f}$, $b_w = {x1 - x0:.2f}$, $b_h = {y1 - y0:.2f}$", fontsize=11, color="#000000", va="bottom",
            bbox=dict(facecolor="#ffffff", edgecolor="none", alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    tidy(ax, grid=False)
    save(fig, OUT / "yolo_grid.svg")
    return cx - 1, cy - 1, x1 - x0, y1 - y0


def iou_figure():
    a, b = (1, 1, 5, 4), (3, 2, 7, 6)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.add_patch(plt.Rectangle((a[0], a[1]), a[2] - a[0], a[3] - a[1], facecolor=BLUE, alpha=0.18, edgecolor=BLUE, lw=2.4))
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], facecolor=ORANGE, alpha=0.18, edgecolor=ORANGE, lw=2.4))
    ax.add_patch(plt.Rectangle((3, 2), 2, 2, facecolor=MAROON, alpha=0.55, edgecolor=MAROON, lw=2))
    ax.text(1.2, 3.6, "prediction\narea 12", fontsize=12, color=BLUE, va="top")
    ax.text(5.2, 5.6, "ground truth\narea 16", fontsize=12, color=ORANGE, va="top")
    ax.text(4, 3, "4", fontsize=14, color="#ffffff", ha="center", va="center", fontweight="bold")
    ax.text(4, 0.35, "IoU = 4 / (12 + 16 − 4) = 0.17", ha="center", fontsize=14, color="#000000")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6.6)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    save(fig, OUT / "iou.svg")


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms_figure():
    boxes = [(10, 10, 50, 60), (12, 14, 52, 58), (8, 8, 46, 62), (70, 20, 110, 70), (72, 18, 108, 66), (30, 70, 40, 80)]
    scores = [0.9, 0.75, 0.6, 0.8, 0.7, 0.3]
    order = [i for i in np.argsort(scores)[::-1] if scores[i] > 0.6]
    kept = []
    while order:
        best = order.pop(0)
        kept.append(best)
        order = [i for i in order if box_iou(boxes[best], boxes[i]) < 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, show, title in zip(axes, (range(len(boxes)), kept), ("Before: every predicted box", "After non-max suppression")):
        stagger = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 0}
        for i in show:
            b = boxes[i]
            color = MAROON if i in kept else GRAY
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, lw=2.2 if i in kept else 1.6))
            if i == 5:
                ax.text(b[0], b[3] + 6, f"{scores[i]:.2f}", fontsize=11, color=color)
            else:
                ax.text(b[2] + 1.5, b[1] + 6 + 13 * stagger[i] - (b[1] - 10 if i < 3 else b[1] - 20), f"{scores[i]:.2f}", fontsize=11, color=color)
        ax.set_xlim(0, 128)
        ax.set_ylim(92, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        tidy(ax, title, grid=False)
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "nms.svg")
    return kept


def anchor_boxes():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0.6, 0.3), 0.8, 2.2, fill=False, edgecolor=BLUE, lw=2.6))
    ax.add_patch(plt.Rectangle((2.0, 0.9), 2.2, 0.9, fill=False, edgecolor=ORANGE, lw=2.6))
    ax.text(1.0, 2.75, "anchor 1", ha="center", fontsize=12, color=BLUE)
    ax.text(3.1, 2.05, "anchor 2", ha="center", fontsize=12, color=ORANGE)
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 3.2)
    tidy(ax, "Two anchor shapes", grid=False)
    ax = axes[1]
    ax.add_patch(plt.Rectangle((0, 0), 6, 4, facecolor="#f4f4f6", edgecolor=INK, lw=1.2))
    ax.add_patch(plt.Rectangle((2, 1.33), 2, 1.34, facecolor="none", edgecolor=INK, lw=1.2, ls=(0, (3, 2))))
    ax.add_patch(plt.Rectangle((2.6, 0.4), 0.8, 3.0, fill=False, edgecolor=BLUE, lw=2.6))
    ax.add_patch(plt.Rectangle((1.2, 1.4), 3.6, 1.2, fill=False, edgecolor=ORANGE, lw=2.6))
    ax.plot(3.0, 1.9, "o", color=BLUE, ms=8)
    ax.plot(3.0, 2.0, "o", color=ORANGE, ms=8)
    ax.text(3.0, 3.55, "person: matches anchor 1", ha="center", fontsize=12, color=BLUE)
    ax.text(3.0, 0.15, "car: matches anchor 2", ha="center", fontsize=12, color=ORANGE)
    ax.text(4.1, 2.75, "same grid cell", fontsize=11, color=INK)
    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(-0.1, 4.1)
    tidy(ax, "Two objects centered in one cell", grid=False)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "anchor_boxes.svg")


def triplet():
    body = []
    for k, (name, y, fill) in enumerate((("Anchor A", 30, LIGHT), ("Positive P", 120, "#e3f4ee"), ("Negative N", 210, "#fbeef2"))):
        body += svg_box(10, y, 130, 50, name, fill=fill)
        body += svg_arrow(140, y + 25, 200, y + 25)
        body += svg_box(200, y, 170, 50, "same ConvNet", fill="#ffffff")
        body += svg_arrow(370, y + 25, 430, y + 25)
        label = ("f(A)", "f(P)", "f(N)")[k]
        body += svg_box(430, y, 90, 50, label, fill="#ffffff")
    body.append(f'<rect x="580" y="20" width="400" height="250" rx="10" fill="#fafafa" stroke="{INK}" stroke-width="1.2"/>')
    pa, pp, pn = (700, 150), (760, 115), (880, 190)
    body.append(f'<circle cx="{pa[0]}" cy="{pa[1]}" r="95" fill="none" stroke="{GRAY}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    body.append(f'<line x1="{pa[0]}" y1="{pa[1]}" x2="{pp[0]}" y2="{pp[1]}" stroke="{GREEN}" stroke-width="2.4"/>')
    body.append(f'<line x1="{pa[0]}" y1="{pa[1]}" x2="{pn[0]}" y2="{pn[1]}" stroke="{MAROON}" stroke-width="2.4"/>')
    for (x, y), color, name in ((pa, BLUE, "f(A)"), (pp, GREEN, "f(P)"), (pn, MAROON, "f(N)")):
        body.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{color}"/>')
        body.append(f'<text x="{x + 12}" y="{y - 10}" font-family="{FONT}" font-size="15" fill="#000">{name}</text>')
    body.append(f'<text x="780" y="255" font-family="{FONT}" font-size="14" fill="{INK}" text-anchor="middle">training pulls f(P) in and pushes f(N) out, by a margin α</text>')
    body.append(f'<text x="780" y="45" font-family="{FONT}" font-size="15" font-weight="700" fill="#000" text-anchor="middle">encoding space</text>')
    svg_doc(OUT / "triplet.svg", 1000, 290, "Triplet loss",
            "An anchor image, a positive image of the same person, and a negative image of another person go through the same ConvNet. In the space of encodings, the positive's encoding is close to the anchor's and the negative's is far away, outside a dashed circle.", body)


def gram():
    size = 48
    yy, xx = np.mgrid[0:size, 0:size]
    stripes = (np.sin(xx / 1.6) > 0).astype(float)
    checks = (((xx // 6) + (yy // 6)) % 2).astype(float)
    filters = {
        "vertical": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float),
        "horizontal": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float),
        "diagonal": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=float),
        "blob": np.ones((3, 3)) / 9,
    }
    names = list(filters)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4), gridspec_kw={"width_ratios": [1, 1.25]})
    for row, (title, img) in enumerate((("Stripes", stripes), ("Checkerboard", checks))):
        acts = np.stack([np.maximum(conv2d(img, filters[n]), 0) for n in names])
        flat = acts.reshape(len(names), -1)
        G = flat @ flat.T
        G = G / G.max()
        ax = axes[row, 0]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        tidy(ax, title, grid=False)
        ax = axes[row, 1]
        im = ax.imshow(G, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
        ax.set_yticklabels(names, fontsize=11)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=10, color="#ffffff" if G[i, j] > 0.55 else "#000000")
        tidy(ax, "Gram matrix", grid=False)
    fig.tight_layout(h_pad=2, w_pad=2)
    save(fig, OUT / "gram.svg")


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    conv_example()
    edge_detection()
    padding_stride()
    conv_volume()
    pooling()
    lenet_like()
    resnet_depth()
    resnet_block()
    bottleneck()
    inception_module()
    augmentation()
    data_spectrum()
    localization()
    sliding_windows()
    print("yolo:", [round(v, 2) for v in yolo_grid()])
    iou_figure()
    print("nms kept:", nms_figure())
    anchor_boxes()
    triplet()
    gram()


if __name__ == "__main__":
    main()
