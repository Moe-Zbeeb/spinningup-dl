import sys

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2e95d3"
INK = "#404452"
GRID = "#e4e4e8"

plt.rcParams.update({
    "svg.fonttype": "path",
    "svg.hashsalt": "spinningup-dl",
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "font.size": 14,
})

z = np.linspace(-6, 6, 601)
panels = [
    ("Perceptron", np.where(z < 0, 0.0, 1.0), r"$\Phi(z)$", (0.55, 0.55), [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    ("Sigmoid", 1 / (1 + np.exp(-z)), r"$\dfrac{1}{1+e^{-z}}$", (0.52, 0.2), [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    ("Tanh", np.tanh(z), r"$\tanh(z)$", (0.52, 0.52), [-1.0, -0.5, 0, 0.5, 1.0]),
    ("ReLU", np.maximum(0, z), r"$\max(0,\ z)$", (0.08, 0.68), [0, 2, 4, 6]),
    ("Leaky ReLU", np.where(z < 0, 0.1 * z, z), "$0.1z$ if $z < 0$\n$z$ if $z \\geq 0$", (0.06, 0.62), [0, 2, 4, 6]),
    ("ELU", np.where(z < 0, np.exp(z) - 1, z), "$e^{z} - 1$ if $z < 0$\n$z$ if $z \\geq 0$", (0.06, 0.62), [0, 2, 4, 6]),
]

fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
for ax, (title, y, label, where, yticks) in zip(axes.flat, panels):
    if title == "Perceptron":
        ax.plot(z[z < 0], y[z < 0], color=BLUE, lw=2.4, solid_capstyle="round")
        ax.plot(z[z >= 0], y[z >= 0], color=BLUE, lw=2.4, solid_capstyle="round")
        ax.plot([0, 0], [0, 1], color=BLUE, lw=2.4, ls=(0, (2, 2)))
    else:
        ax.plot(z, y, color=BLUE, lw=2.4, solid_capstyle="round")
    ax.set_title(title, fontsize=16, fontweight="bold", color="#000000", pad=8)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks(yticks)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.text(where[0], where[1], label, transform=ax.transAxes, fontsize=16, linespacing=1.5)

fig.tight_layout(h_pad=2.2, w_pad=2.0)
fig.savefig(sys.argv[1], transparent=True, metadata={"Date": None})
