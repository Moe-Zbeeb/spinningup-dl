import sys

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2e95d3"
INK = "#404452"
GRID = "#e4e4e8"
UNIT_COLORS = ["#2e95d3", "#840132", "#e86c09", "#10a37f"]

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


def make_data(rng, m=400):
    X = rng.uniform(-1, 1, size=(2, m))
    Y = ((X[0] * X[1]) > 0).astype(float).reshape(1, m)
    return X, Y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(W1, b1, W2, b2, X, Y, steps, lr):
    m = X.shape[1]
    history, losses = [], []
    for _ in range(steps):
        Z1 = W1 @ X + b1
        A1 = np.tanh(Z1)
        Z2 = W2 @ A1 + b2
        A2 = sigmoid(Z2)
        losses.append(float(-np.mean(Y * np.log(A2 + 1e-12) + (1 - Y) * np.log(1 - A2 + 1e-12))))
        history.append(W1[:, 0].copy())
        dZ2 = A2 - Y
        dW2 = dZ2 @ A1.T / m
        db2 = dZ2.sum(axis=1, keepdims=True) / m
        dZ1 = (W2.T @ dZ2) * (1 - A1 ** 2)
        dW1 = dZ1 @ X.T / m
        db1 = dZ1.sum(axis=1, keepdims=True) / m
        W1, b1, W2, b2 = W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2
    return np.array(history), losses


def main(path):
    rng = np.random.default_rng(0)
    X, Y = make_data(rng)
    n_h, steps, lr = 4, 2000, 1.0
    const = train(np.full((n_h, 2), 0.5), np.zeros((n_h, 1)), np.full((1, n_h), 0.5), np.zeros((1, 1)), X, Y, steps, lr)
    rand = train(rng.standard_normal((n_h, 2)) * 0.01, np.zeros((n_h, 1)), rng.standard_normal((1, n_h)) * 0.01, np.zeros((1, 1)), X, Y, steps, lr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (hist, losses), title in zip(axes, (const, rand), ("Every weight starts at 0.5", "Weights start small and random")):
        for j in range(n_h):
            ax.plot(hist[:, j], color=UNIT_COLORS[j], lw=2.2, label=f"unit {j + 1}")
        ax.set_title(title, fontsize=15, fontweight="bold", color="#000000", pad=10)
        ax.set_xlabel("training step")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8, labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.text(0.97, 0.47, f"final loss {losses[-1]:.2f}", transform=ax.transAxes, ha="right", va="center", fontsize=13)
    axes[0].set_ylabel(r"first weight of each unit, $w_{j1}^{[1]}$")
    axes[1].legend(frameon=False, fontsize=12, loc="upper left")
    axes[0].text(0.97, 0.62, "all four units follow\nthe same curve", transform=axes[0].transAxes, ha="right", va="center", fontsize=13)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, transparent=True, metadata={"Date": None})
    print("constant: units identical at end:", bool(np.all(const[0][-1] == const[0][-1][0])), "loss", round(const[1][-1], 3))
    print("random: final first weights", np.round(rand[0][-1], 2), "loss", round(rand[1][-1], 3))


if __name__ == "__main__":
    main(sys.argv[1])
