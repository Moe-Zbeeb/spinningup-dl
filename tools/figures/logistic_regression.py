import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import BLUE, GRAY, INK, MAROON, plt, save, setup, tidy

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "_static" / "img" / "neural_networks"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    setup()
    rng = np.random.default_rng(0)
    m = 200
    X = np.concatenate([rng.normal([-1.0, -0.6], 0.75, (m // 2, 2)), rng.normal([1.0, 0.8], 0.75, (m // 2, 2))]).T
    Y = np.concatenate([np.zeros(m // 2), np.ones(m // 2)])[None, :]
    w, b, alpha = np.zeros((2, 1)), 0.0, 0.5
    costs = []
    for _ in range(300):
        A = sigmoid(w.T @ X + b)
        costs.append(float(-np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))))
        dZ = A - Y
        w -= alpha * (X @ dZ.T) / m
        b -= alpha * float(dZ.mean())
    accuracy = float(((sigmoid(w.T @ X + b) > 0.5) == Y).mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    g = np.linspace(-3.5, 3.5, 300)
    G1, G2 = np.meshgrid(g, g)
    P = sigmoid(w[0, 0] * G1 + w[1, 0] * G2 + b)
    cs = ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    ax.contour(G1, G2, P, levels=[0.5], colors=INK, linewidths=2)
    ax.scatter(X[0, Y[0] == 0], X[1, Y[0] == 0], s=22, color=BLUE, label="$y = 0$", edgecolor="#ffffff", linewidth=0.5)
    ax.scatter(X[0, Y[0] == 1], X[1, Y[0] == 1], s=22, color=MAROON, label="$y = 1$", edgecolor="#ffffff", linewidth=0.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\hat{y} = \sigma(w^{\top} x + b)$")
    tidy(ax, "Predicted probability", grid=False)
    ax = axes[1]
    ax.plot(costs, color=BLUE, lw=2.4)
    ax.set_xlabel("iteration of gradient descent")
    ax.set_ylabel("cost $J(w, b)$")
    ax.set_ylim(0, 0.75)
    ax.axhline(costs[-1], color=GRAY, lw=1, ls=(0, (3, 2)))
    ax.text(298, costs[-1] + 0.03, f"{costs[-1]:.3f}", ha="right", fontsize=12)
    tidy(ax, "Training", grid=True)
    fig.tight_layout(w_pad=3)
    save(fig, OUT / "logistic_regression.svg")
    print(f"final cost {costs[-1]:.4f}, accuracy {accuracy:.3f}, w {w.ravel().round(3)}, b {b:.3f}")


if __name__ == "__main__":
    main()
