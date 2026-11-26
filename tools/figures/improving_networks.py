import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import BLUE, GRAY, GREEN, INK, MAROON, ORANGE, SERIES, plt, save, setup, tidy

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "_static" / "img" / "improving_networks"


def poly_features(x, degree):
    return np.polynomial.legendre.legvander(x, degree)


def bias_variance():
    rng = np.random.default_rng(3)
    truth = lambda x: np.sin(2.6 * x) + 0.3 * x
    x = np.sort(rng.uniform(-1, 1, 18))
    y = truth(x) + rng.normal(0, 0.18, x.size)
    grid = np.linspace(-1, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, degree, title in zip(axes, (1, 4, 15), ("High bias (underfitting)", "Just right", "High variance (overfitting)")):
        coef = np.linalg.lstsq(poly_features(x, degree), y, rcond=None)[0]
        ax.plot(grid, truth(grid), color=GRAY, lw=1.5, ls=(0, (4, 3)), label="true function")
        ax.plot(grid, poly_features(grid, degree) @ coef, color=BLUE, lw=2.4, label="model")
        ax.scatter(x, y, s=28, color=INK, zorder=3, label="training data")
        ax.set_ylim(-2.2, 2.2)
        ax.set_xticks([])
        ax.set_yticks([])
        tidy(ax, title, grid=False)
    axes[0].legend(loc="lower right", fontsize=11)
    fig.tight_layout(w_pad=1.5)
    save(fig, OUT / "bias_variance.svg")


def regularization_tanh():
    z = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(-0.5, 0.5, color=BLUE, alpha=0.12, lw=0)
    ax.plot(z, np.tanh(z), color=BLUE, lw=2.4, label=r"$\tanh(z)$")
    ax.plot(z, z, color=GRAY, lw=1.5, ls=(0, (4, 3)), label=r"$z$")
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel(r"$z$")
    ax.annotate("small weights keep $z$ here,\nwhere tanh is almost linear", xy=(0.45, -0.35), xytext=(1.2, -0.9), fontsize=12, arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.legend(loc="upper left")
    tidy(ax)
    save(fig, OUT / "regularization_tanh.svg")


def dropout_diagram():
    import math

    W, H = 900, 380
    R = 22
    layers = [3, 5, 5, 1]

    def panel(x0, dropped, title):
        cols = [x0 + 40 + i * 120 for i in range(len(layers))]
        pos = []
        for c, n in zip(cols, layers):
            top = 200 - (n - 1) * 58 / 2
            pos.append([(c, top + k * 58) for k in range(n)])
        parts = [f'<text x="{x0 + 220}" y="34" font-family="Lato, Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#000" text-anchor="middle">{title}</text>']
        for i in range(len(layers) - 1):
            for a, (xa, ya) in enumerate(pos[i]):
                for b, (xb, yb) in enumerate(pos[i + 1]):
                    off = (i, a) in dropped or (i + 1, b) in dropped
                    d = math.hypot(xb - xa, yb - ya)
                    ux, uy = (xb - xa) / d, (yb - ya) / d
                    color, opacity = (GRAY, 0.18) if off else (INK, 0.75)
                    parts.append(f'<line x1="{xa + ux * R:.1f}" y1="{ya + uy * R:.1f}" x2="{xb - ux * R:.1f}" y2="{yb - uy * R:.1f}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="1.4"/>')
        for i, layer in enumerate(pos):
            for k, (x, y) in enumerate(layer):
                if (i, k) in dropped:
                    parts.append(f'<circle cx="{x}" cy="{y}" r="{R}" fill="#ffffff" stroke="{GRAY}" stroke-opacity="0.5" stroke-width="2" stroke-dasharray="4 3"/>')
                    parts.append(f'<path d="M{x - 9} {y - 9} L{x + 9} {y + 9} M{x + 9} {y - 9} L{x - 9} {y + 9}" stroke="{MAROON}" stroke-width="2.4" stroke-linecap="round"/>')
                else:
                    parts.append(f'<circle cx="{x}" cy="{y}" r="{R}" fill="#ffffff" stroke="{INK}" stroke-width="2.2"/>')
        return parts

    body = panel(0, {(1, 1), (1, 4), (2, 0), (2, 2)}, "Iteration t")
    body += panel(460, {(1, 0), (1, 2), (2, 3), (0, 1)}, "Iteration t + 1")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
           '<title id="t">Dropout on two consecutive iterations</title>',
           '<desc id="d">The same network drawn twice. On each iteration a different random set of units is crossed out, and their connections are faded.</desc>']
    svg += body + ["</svg>"]
    (OUT / "dropout.svg").write_text("\n".join(svg) + "\n")


def early_stopping():
    rng = np.random.default_rng(7)
    truth = lambda x: np.sin(2.6 * x) + 0.3 * x
    x_tr = rng.uniform(-1, 1, 16)
    y_tr = truth(x_tr) + rng.normal(0, 0.25, x_tr.size)
    x_dev = rng.uniform(-1, 1, 400)
    y_dev = truth(x_dev) + rng.normal(0, 0.25, x_dev.size)
    hidden = 200
    W1 = rng.normal(0, 4.0, (hidden, 1))
    b1 = rng.uniform(-4, 4, (hidden, 1))
    feats = lambda x: np.tanh(W1 @ x.reshape(1, -1) + b1)
    F_tr, F_dev = feats(x_tr), feats(x_dev)
    w = np.zeros((1, hidden))
    lr = 1.0 / np.linalg.eigvalsh(F_tr @ F_tr.T / x_tr.size).max()
    steps = 200000
    tr, dev = [], []
    for _ in range(steps):
        err = w @ F_tr - y_tr
        tr.append(float(np.mean(err ** 2)))
        dev.append(float(np.mean((w @ F_dev - y_dev) ** 2)))
        w -= lr * (err @ F_tr.T) / x_tr.size
    tr, dev = np.array(tr), np.array(dev)
    it = np.arange(1, steps + 1)
    best = int(np.argmin(dev))
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(it, tr, color=BLUE, lw=2.4, label="training cost")
    ax.plot(it, dev, color=MAROON, lw=2.4, label="dev cost")
    ax.axvline(it[best], color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(it[best] * 1.15, 0.72, "stop here", fontsize=12)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("cost")
    ax.legend(loc="upper right")
    tidy(ax)
    save(fig, OUT / "early_stopping.svg")
    return best, tr[best], dev[best], tr[-1], dev[-1]


def gd_path(grad, start, lr, steps):
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        p = p - lr * grad(p)
        path.append(p.copy())
    return np.array(path)


def normalization_contours():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    cases = [
        ("Unnormalized features", np.array([1.0, 25.0]), 0.075, (-9, 1.6)),
        ("Normalized features", np.array([1.0, 1.0]), 0.9, (-9, 6)),
    ]
    for ax, (title, h, lr, start) in zip(axes, cases):
        grad = lambda p, h=h: h * p
        path = gd_path(grad, start, lr, 30)
        lim = 10
        g = np.linspace(-lim, lim, 300)
        W1, W2 = np.meshgrid(g, g)
        J = 0.5 * (h[0] * W1 ** 2 + h[1] * W2 ** 2)
        levels = np.geomspace(J[J > 0].min() * 40, J.max(), 9)
        ax.contour(W1, W2, J, levels=levels, colors=GRAY, linewidths=1)
        ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=2, marker="o", ms=4)
        ax.plot(0, 0, marker="*", ms=14, color=MAROON)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$w_1$")
        ax.set_ylabel(r"$w_2$")
        tidy(ax, title, grid=False)
        ax.text(0.03, 0.04, f"{len(path) - 1} steps shown", transform=ax.transAxes, fontsize=12)
    fig.tight_layout(w_pad=3)
    save(fig, OUT / "normalization_contours.svg")


def vanishing_exploding():
    rng = np.random.default_rng(0)
    n, L = 100, 50
    a0 = rng.standard_normal((n, 64))
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for scale, color in zip((1.2, 1.0, 0.8), (MAROON, BLUE, ORANGE)):
        a = a0.copy()
        norms = [np.sqrt(np.mean(a ** 2))]
        for _ in range(L):
            W = rng.standard_normal((n, n)) * scale / np.sqrt(n)
            a = W @ a
            norms.append(np.sqrt(np.mean(a ** 2)))
        ax.plot(range(L + 1), norms, color=color, lw=2.4, label=f"weights {scale} times the balanced scale")
    ax.set_yscale("log")
    ax.set_xlabel("layer $l$")
    ax.set_ylabel(r"typical size of $a^{[l]}$")
    ax.legend(loc="upper left", fontsize=11)
    tidy(ax)
    save(fig, OUT / "vanishing_exploding.svg")


def minibatch_costs():
    rng = np.random.default_rng(1)
    m, n = 4096, 2
    X = rng.standard_normal((m, n))
    w_true = np.array([2.0, -3.0])
    p = 1 / (1 + np.exp(-(X @ w_true + 0.5)))
    y = (rng.uniform(size=m) < p).astype(float)

    def cost(w, b):
        q = 1 / (1 + np.exp(-(X @ w + b)))
        return float(-np.mean(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (size, lr, color, title) in zip(axes, ((m, 0.5, BLUE, f"Batch (size {m})"), (64, 0.5, ORANGE, "Mini-batch (size 64)"), (1, 0.05, MAROON, "Stochastic (size 1)"))):
        w, b = np.zeros(n), 0.0
        costs = []
        order = rng.permutation(m)
        pos = 0
        for _ in range(300):
            if pos + size > m:
                order, pos = rng.permutation(m), 0
            idx = order[pos:pos + size]
            pos += size
            q = 1 / (1 + np.exp(-(X[idx] @ w + b)))
            costs.append(float(-np.mean(y[idx] * np.log(q + 1e-12) + (1 - y[idx]) * np.log(1 - q + 1e-12))))
            w -= lr * X[idx].T @ (q - y[idx]) / size
            b -= lr * float(np.mean(q - y[idx]))
        ax.plot(costs, color=color, lw=1.2 if size == 1 else 1.8)
        ax.plot(np.convolve(costs, np.ones(25) / 25, mode="valid"), color=INK, lw=1.6, ls=(0, (4, 2))) if size < m else None
        ax.set_xlabel("update")
        ax.set_ylim(0, 1.6)
        tidy(ax, title)
    axes[0].set_ylabel("cost on the current batch")
    fig.tight_layout(w_pad=1.5)
    save(fig, OUT / "minibatch_costs.svg")


def ewma():
    rng = np.random.default_rng(4)
    days = np.arange(1, 181)
    temp = 18 + 9 * np.sin((days - 40) / 180 * 2 * np.pi) + rng.normal(0, 2.4, days.size)

    def average(beta, correct=False):
        v, out = 0.0, []
        for t, x in enumerate(temp, start=1):
            v = beta * v + (1 - beta) * x
            out.append(v / (1 - beta ** t) if correct else v)
        return np.array(out)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax = axes[0]
    ax.scatter(days, temp, s=10, color=GRAY, label="daily temperature")
    for beta, color in ((0.5, ORANGE), (0.9, BLUE), (0.98, MAROON)):
        ax.plot(days, average(beta), color=color, lw=2.2, label=fr"$\beta = {beta}$, about {round(1 / (1 - beta))} days")
    ax.set_xlabel("day")
    ax.legend(loc="lower center", fontsize=11)
    tidy(ax, "Larger β averages over more days")
    ax = axes[1]
    ax.scatter(days, temp, s=10, color=GRAY)
    ax.plot(days, average(0.98), color=MAROON, lw=2.2, label="without bias correction")
    ax.plot(days, average(0.98, correct=True), color=GREEN, lw=2.2, label="with bias correction")
    ax.set_xlabel("day")
    ax.legend(loc="lower center", fontsize=11)
    tidy(ax, "Bias correction fixes the start (β = 0.98)")
    axes[0].set_ylabel("°C")
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "ewma.svg")


def optimizer_paths(names, filename):
    h = np.array([0.5, 18.0])
    grad = lambda p: h * p
    start = np.array([-9.0, 2.5])
    steps = 40

    def run(name):
        p = start.copy()
        v = np.zeros(2)
        s = np.zeros(2)
        path = [p.copy()]
        for t in range(1, steps + 1):
            g = grad(p)
            if name == "Gradient descent":
                p = p - 0.105 * g
            elif name == "Momentum":
                v = 0.9 * v + 0.1 * g
                p = p - 0.105 * v
            elif name == "RMSprop":
                s = 0.9 * s + 0.1 * g ** 2
                p = p - 0.5 * g / (np.sqrt(s) + 1e-8)
            elif name == "Adam":
                v = 0.9 * v + 0.1 * g
                s = 0.999 * s + 0.001 * g ** 2
                v_hat = v / (1 - 0.9 ** t)
                s_hat = s / (1 - 0.999 ** t)
                p = p - 0.5 * v_hat / (np.sqrt(s_hat) + 1e-8)
            path.append(p.copy())
        return np.array(path)

    lim = 10
    g = np.linspace(-lim, lim, 300)
    W1, W2 = np.meshgrid(g, np.linspace(-4, 4, 200))
    J = 0.5 * (h[0] * W1 ** 2 + h[1] * W2 ** 2)
    rows = 2 if len(names) == 4 else 1
    cols = len(names) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.4 * rows), squeeze=False)
    finals = {}
    for ax, name in zip(axes.flat, names):
        path = run(name)
        finals[name] = float(np.linalg.norm(path[-1]))
        ax.contour(W1, W2, J, levels=np.geomspace(0.5, J.max(), 9), colors=GRAY, linewidths=1)
        ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=2, marker="o", ms=3.5)
        ax.plot(0, 0, marker="*", ms=14, color=MAROON)
        ax.set_xlim(-lim, lim / 2)
        ax.set_ylim(-4, 4)
        ax.set_xlabel(r"$w$")
        ax.set_ylabel(r"$b$")
        tidy(ax, name, grid=False)
    fig.tight_layout(w_pad=2.5)
    save(fig, OUT / filename)
    return finals


def lr_decay():
    epochs = np.arange(0, 51)
    a0 = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(epochs, a0 / (1 + 0.2 * epochs), color=BLUE, lw=2.4, label=r"$\alpha_0 / (1 + 0.2\,t)$")
    ax.plot(epochs, a0 * 0.95 ** epochs, color=MAROON, lw=2.4, label=r"$0.95^{t}\,\alpha_0$")
    ax.plot(epochs[1:], a0 / np.sqrt(epochs[1:]), color=ORANGE, lw=2.4, label=r"$\alpha_0 / \sqrt{t}$")
    ax.step(epochs, a0 * 0.5 ** (epochs // 10), where="post", color=GREEN, lw=2.4, label="staircase: halve every 10 epochs")
    ax.set_xlabel(r"epoch $t$")
    ax.set_ylabel(r"learning rate $\alpha$")
    ax.set_ylim(0, 0.21)
    ax.legend(loc="upper right")
    tidy(ax)
    save(fig, OUT / "lr_decay.svg")


def saddle_point():
    g = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(g, g)
    fig = plt.figure(figsize=(11, 4.6))
    for i, (title, Z) in enumerate((("Local minimum", X ** 2 + Y ** 2), ("Saddle point", X ** 2 - Y ** 2))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="Blues", linewidth=0, antialiased=True, alpha=0.9, rstride=2, cstride=2)
        ax.scatter([0], [0], [0], color=MAROON, s=60, depthshade=False)
        ax.set_title(title, fontsize=15, fontweight="bold", color="#000000")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel(r"$w_1$")
        ax.set_ylabel(r"$w_2$")
        ax.set_zlabel(r"$J$")
        ax.view_init(elev=28, azim=-55)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_alpha(0)
    fig.tight_layout()
    save(fig, OUT / "saddle_point.svg")


def grid_vs_random():
    rng = np.random.default_rng(2)
    g = np.linspace(0, 1, 300)
    A, E = np.meshgrid(g, g)
    score = np.exp(-((A - 0.72) ** 2) / 0.012) + 0.05 * E
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    grid_pts = np.array([(a, e) for a in (1 / 6, 0.5, 5 / 6) for e in (1 / 6, 0.5, 5 / 6)])
    rand_pts = rng.uniform(0.04, 0.96, (9, 2))
    for ax, pts, title in zip(axes, (grid_pts, rand_pts), ("Grid search", "Random search")):
        ax.imshow(score, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", alpha=0.65, aspect="auto")
        ax.scatter(pts[:, 0], pts[:, 1], s=60, color=MAROON, zorder=3)
        for a in pts[:, 0]:
            ax.plot([a, a], [-0.07, -0.02], color=MAROON, lw=2, clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"learning rate $\alpha$ (matters a lot)", labelpad=26)
        ax.set_ylabel(r"$\epsilon$ (barely matters)")
        tidy(ax, title, grid=False)
        ax.text(0.03, 0.9, f"{len(np.unique(np.round(pts[:, 0], 6)))} distinct values of α", transform=ax.transAxes, fontsize=12, color="#000000")
    fig.tight_layout(w_pad=3)
    save(fig, OUT / "grid_vs_random.svg")


def batch_norm():
    rng = np.random.default_rng(5)
    z = rng.normal(4, 3, 4000)
    mu, var = z.mean(), z.var()
    z_norm = (z - mu) / np.sqrt(var + 1e-8)
    z_tilde = 0.5 * z_norm - 1.5
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    data = (
        (z, r"$z^{(i)}$: mean 4, standard deviation 3"),
        (z_norm, r"$z^{(i)}_{\mathrm{norm}}$: mean 0, standard deviation 1"),
        (z_tilde, r"$\tilde{z}^{(i)}$ with $\gamma = 0.5$, $\beta = -1.5$"),
    )
    for ax, (values, title) in zip(axes, data):
        ax.hist(values, bins=60, range=(-8, 16), color=BLUE, alpha=0.85)
        ax.set_yticks([])
        ax.set_xlim(-8, 16)
        tidy(ax, grid=False)
        ax.set_title(title, fontsize=13, color="#000000", pad=8)
    fig.tight_layout(w_pad=1.5)
    save(fig, OUT / "batch_norm.svg")


def softmax_figure():
    z = np.array([2.0, 1.0, -1.0, 3.0])
    t = np.exp(z)
    soft = t / t.sum()
    hard = (z == z.max()).astype(float)
    labels = ["class 1", "class 2", "class 3", "class 4"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, values, title, color in zip(axes, (z, soft, hard), (r"Scores $z^{[L]}$", "Softmax", "Hardmax"), (GRAY, BLUE, MAROON)):
        bars = ax.bar(labels, values, color=color, width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.08 if v >= 0 else -0.3) * (3 if title.startswith("Scores") else 0.35), f"{v:.2f}" if not title.startswith("Scores") else f"{v:.0f}", ha="center", fontsize=12)
        ax.axhline(0, color=INK, lw=0.8)
        tidy(ax, title, grid=False)
        ax.tick_params(axis="x", labelsize=11)
    axes[0].set_ylim(-2.4, 3.8)
    axes[1].set_ylim(0, 1.15)
    axes[2].set_ylim(0, 1.15)
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "softmax.svg")
    return soft


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    bias_variance()
    regularization_tanh()
    dropout_diagram()
    print("early stopping (best step, train, dev, final train, final dev):", early_stopping())
    normalization_contours()
    vanishing_exploding()
    minibatch_costs()
    ewma()
    print("momentum:", optimizer_paths(["Gradient descent", "Momentum"], "momentum_paths.svg"))
    print("rmsprop:", optimizer_paths(["Gradient descent", "RMSprop"], "rmsprop_paths.svg"))
    print("all:", optimizer_paths(["Gradient descent", "Momentum", "RMSprop", "Adam"], "optimizer_paths.svg"))
    lr_decay()
    saddle_point()
    grid_vs_random()
    batch_norm()
    print("softmax:", np.round(softmax_figure(), 3))


if __name__ == "__main__":
    main()
