import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import BLUE, GRAY, GREEN, INK, MAROON, ORANGE, plt, save, setup, tidy

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "_static" / "img" / "structuring_projects"
FONT = "Lato, Helvetica Neue, Helvetica, Arial, sans-serif"


def human_level():
    t = np.linspace(0, 10, 400)
    bayes, human = 0.96, 0.9
    acc = bayes - (bayes - 0.3) * np.exp(-0.45 * t)
    fig, ax = plt.subplots(figsize=(7.4, 4))
    ax.plot(t, acc, color=BLUE, lw=2.6, label="model accuracy")
    ax.axhline(human, color=ORANGE, lw=1.8, ls=(0, (5, 3)))
    ax.axhline(bayes, color=MAROON, lw=1.8, ls=(0, (2, 2)))
    ax.text(0.15, human - 0.045, "human-level performance", color=ORANGE, fontsize=12)
    ax.text(0.15, bayes + 0.012, "best possible accuracy (Bayes optimal error)", color=MAROON, fontsize=12)
    cross = t[np.argmax(acc >= human)]
    ax.axvline(cross, color=GRAY, lw=1, ls=(0, (1, 2)))
    ax.text(cross + 0.15, 0.45, "progress slows\nafter passing humans", fontsize=12)
    ax.set_ylim(0.25, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("time spent on the project")
    ax.set_ylabel("accuracy")
    tidy(ax, grid=False)
    save(fig, OUT / "human_level.svg")


def error_gaps():
    levels = [("Human-level\nerror", 4.0), ("Training\nerror", 7.0), ("Training-dev\nerror", 10.0), ("Dev\nerror", 12.0), ("Test\nerror", 12.5)]
    gaps = ["avoidable bias", "variance", "data mismatch", "overfitting\nto the dev set"]
    colors = [MAROON, BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = np.arange(len(levels))
    for x, (name, value) in zip(xs, levels):
        ax.bar(x, value, width=0.55, color="#d7d8e0")
        ax.text(x, value + 0.3, f"{value:g}%", ha="center", fontsize=13, color="#000000")
    for i, (gap, color) in enumerate(zip(gaps, colors)):
        lo, hi = levels[i][1], levels[i + 1][1]
        xm = i + 0.5
        ax.annotate("", xy=(xm, hi), xytext=(xm, lo), arrowprops=dict(arrowstyle="<->", color=color, lw=2.2, shrinkA=0, shrinkB=0))
        ax.plot([i + 0.275, xm], [lo, lo], color=color, lw=1, ls=(0, (2, 2)))
        ax.plot([xm, i + 1 - 0.275], [hi, hi], color=color, lw=1, ls=(0, (2, 2)))
        ax.text(xm + 0.05, (lo + hi) / 2 + 3.2, gap, color=color, fontsize=12, ha="center", va="center", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _ in levels], fontsize=12)
    ax.set_yticks([])
    ax.set_ylim(0, 17.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    tidy(ax, grid=False)
    save(fig, OUT / "error_gaps.svg")


def error_analysis():
    dev_error = 10.0
    cats = [("Dog misclassified as cat", 8), ("Great cat (lion, tiger)", 43), ("Blurry image", 61), ("Incorrectly labeled", 6)]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    names = [c for c, _ in cats][::-1]
    shares = [s for _, s in cats][::-1]
    bars = ax.barh(names, shares, color=BLUE, height=0.55)
    for bar, s in zip(bars, shares):
        best = dev_error * (1 - s / 100)
        ax.text(s + 1.2, bar.get_y() + bar.get_height() / 2, f"{s}% of errors: fixing all of them lowers dev error from 10% to {best:.1f}%", va="center", fontsize=11.5)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("share of the 100 misclassified dev examples (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    tidy(ax, grid=False)
    save(fig, OUT / "error_analysis.svg")


def data_split():
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2))
    a, b = 200, 10
    options = [
        ("Option 1: shuffle everything together", [("train", 205.0, 200 / 210), ("dev", 2.5, 200 / 210), ("test", 2.5, 200 / 210)]),
        ("Option 2: dev and test from the target distribution", [("train", 205.0, 200 / 205), ("dev", 2.5, 0.0), ("test", 2.5, 0.0)]),
    ]
    widths = {"train": 6.4, "dev": 1.6, "test": 1.6}
    for ax, (title, sets) in zip(axes, options):
        x = 0.0
        for name, size, frac_a in sets:
            w = widths[name]
            ax.barh(0, w * frac_a, left=x, color=GRAY, height=0.6)
            ax.barh(0, w * (1 - frac_a), left=x + w * frac_a, color=BLUE, height=0.6)
            ax.add_patch(plt.Rectangle((x, -0.3), w, 0.6, fill=False, edgecolor=INK, lw=1.4))
            label = f"{name}\n{size:g}k"
            ax.text(x + w / 2, 0.52, label, ha="center", va="bottom", fontsize=12)
            x += w + 0.25
        ax.set_xlim(-0.1, 10.2)
        ax.set_ylim(-0.45, 1.25)
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", color="#000000", loc="left")
    axes[0].text(10.2, -0.42, "", fontsize=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["distribution A: 200k web images", "distribution B: 10k app images (the target)"], loc="lower center", ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=2.2, rect=(0, 0.06, 1, 1))
    save(fig, OUT / "data_split.svg")


def network_svg(path, title, desc, layers, width, height, x0, gap_x, gap_y, styles, extra=None, labels=None):
    import math

    R = 20
    cols = [x0 + i * gap_x for i in range(len(layers))]
    pos = []
    for c, n in zip(cols, layers):
        top = height / 2 - (n - 1) * gap_y / 2
        pos.append([(c, top + k * gap_y) for k in range(n)])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
             f'<title id="t">{title}</title>', f'<desc id="d">{desc}</desc>']
    for i in range(len(layers) - 1):
        for (xa, ya) in pos[i]:
            for (xb, yb) in pos[i + 1]:
                d = math.hypot(xb - xa, yb - ya)
                ux, uy = (xb - xa) / d, (yb - ya) / d
                color = styles(i + 1)["edge"]
                parts.append(f'<line x1="{xa + ux * R:.1f}" y1="{ya + uy * R:.1f}" x2="{xb - ux * R:.1f}" y2="{yb - uy * R:.1f}" stroke="{color}" stroke-opacity="0.8" stroke-width="1.3"/>')
    for i, layer in enumerate(pos):
        st = styles(i)
        for (x, y) in layer:
            parts.append(f'<circle cx="{x}" cy="{y}" r="{R}" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="2.2"/>')
    if labels:
        for (x, y), text in zip(pos[-1], labels):
            parts.append(f'<text x="{x + 32}" y="{y + 5}" font-family="{FONT}" font-size="16" fill="{INK}">{text}</text>')
    if extra:
        parts += extra(pos)
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def transfer_learning():
    def styles(i):
        if i == 4:
            return {"fill": "#ffffff", "stroke": MAROON, "edge": MAROON}
        return {"fill": "#eef6fb", "stroke": INK, "edge": GRAY}

    def extra(pos):
        x_last = pos[4][0][0]
        x_first = pos[0][0][0]
        x_prev = pos[3][0][0]
        return [
            f'<rect x="{x_first - 40}" y="40" width="{x_prev - x_first + 80}" height="330" rx="10" fill="none" stroke="{BLUE}" stroke-width="2" stroke-dasharray="6 4"/>',
            f'<text x="{(x_first + x_prev) / 2}" y="30" font-family="{FONT}" font-size="17" font-weight="700" fill="{BLUE}" text-anchor="middle">Pre-trained on task A (e.g. a million everyday photos)</text>',
            f'<text x="{(x_first + x_prev) / 2}" y="395" font-family="{FONT}" font-size="15" fill="{INK}" text-anchor="middle">small dataset: keep frozen · large dataset: fine-tune as well</text>',
            f'<text x="{x_last}" y="30" font-family="{FONT}" font-size="17" font-weight="700" fill="{MAROON}" text-anchor="middle">New output layer</text>',
            f'<text x="{x_last}" y="395" font-family="{FONT}" font-size="15" fill="{INK}" text-anchor="middle">trained on task B (e.g. X-rays)</text>',
        ]

    network_svg(OUT / "transfer_learning.svg", "Transfer learning",
                "A network with four layers pre-trained on task A, inside a dashed box, followed by a new output layer drawn in a different color that is trained on task B.",
                [4, 5, 5, 4, 2], 820, 410, 90, 150, 62, styles, extra)


def multi_task():
    def styles(i):
        if i == 3:
            return {"fill": "#ffffff", "stroke": MAROON, "edge": INK}
        return {"fill": "#eef6fb", "stroke": INK, "edge": GRAY}

    def extra(pos):
        x_first, x_prev = pos[0][0][0], pos[2][0][0]
        return [
            f'<rect x="{x_first - 40}" y="22" width="{x_prev - x_first + 80}" height="350" rx="10" fill="none" stroke="{BLUE}" stroke-width="2" stroke-dasharray="6 4"/>',
            f'<text x="{(x_first + x_prev) / 2}" y="396" font-family="{FONT}" font-size="16" font-weight="700" fill="{BLUE}" text-anchor="middle">shared layers learn features useful for every task</text>',
            f'<text x="{pos[3][0][0] + 40}" y="396" font-family="{FONT}" font-size="16" font-weight="700" fill="{MAROON}" text-anchor="middle">one sigmoid per task</text>',
        ]

    network_svg(OUT / "multi_task.svg", "Multi-task learning",
                "A network whose shared hidden layers, inside a dashed box, feed four sigmoid output units labeled pedestrian, car, stop sign, and traffic light.",
                [4, 5, 5, 4], 820, 410, 90, 170, 75, styles, extra,
                labels=["pedestrian?", "car?", "stop sign?", "traffic light?"])


def end_to_end():
    def box(x, y, w, text, fill="#eef6fb", stroke=INK, bold=False):
        weight = ' font-weight="700"' if bold else ""
        return [f'<rect x="{x}" y="{y}" width="{w}" height="46" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>',
                f'<text x="{x + w / 2}" y="{y + 29}" font-family="{FONT}" font-size="15"{weight} fill="#000" text-anchor="middle">{text}</text>']

    def arrow(x1, x2, y):
        return [f'<line x1="{x1}" y1="{y}" x2="{x2 - 8}" y2="{y}" stroke="{INK}" stroke-width="1.8"/>',
                f'<path d="M{x2 - 9} {y - 5} L{x2} {y} L{x2 - 9} {y + 5} z" fill="{INK}"/>']

    W, H = 900, 250
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
             '<title id="t">A pipeline versus end-to-end learning for speech recognition</title>',
             '<desc id="d">Top: audio goes through hand-designed features, then phonemes, then words, then the transcript. Bottom: one neural network maps audio directly to the transcript.</desc>',
             f'<text x="10" y="26" font-family="{FONT}" font-size="17" font-weight="700" fill="#000">Pipeline</text>',
             f'<text x="10" y="156" font-family="{FONT}" font-size="17" font-weight="700" fill="#000">End-to-end</text>']
    xs = [10, 190, 370, 550, 730]
    names = ["audio", "features (MFCC)", "phonemes", "words", "transcript"]
    for i, (x, name) in enumerate(zip(xs, names)):
        parts += box(x, 40, 150, name, fill="#ffffff" if i in (0, 4) else "#eef6fb")
        if i < 4:
            parts += arrow(x + 150, xs[i + 1], 63)
    parts += box(10, 170, 150, "audio", fill="#ffffff")
    parts += arrow(160, 330, 193)
    parts += box(330, 170, 230, "one neural network", fill="#fbeef2", stroke=MAROON, bold=True)
    parts += arrow(560, 730, 193)
    parts += box(730, 170, 150, "transcript", fill="#ffffff")
    parts.append("</svg>")
    (OUT / "end_to_end.svg").write_text("\n".join(parts) + "\n")


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    human_level()
    error_gaps()
    error_analysis()
    data_split()
    transfer_learning()
    multi_task()
    end_to_end()


if __name__ == "__main__":
    main()
