import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_progress(path):
    lines = pathlib.Path(path).read_text().strip().splitlines()
    header = lines[0].split("\t")
    rows = [[float(v) if v not in ("", "nan") else float("nan") for v in line.split("\t")] for line in lines[1:]]
    return {k: np.array([r[i] for r in rows]) for i, k in enumerate(header)}


def find_runs(logdir):
    for progress in sorted(pathlib.Path(logdir).rglob("progress.txt")):
        run = progress.parent
        config_path = run / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        yield run, config, read_progress(progress)


def group_name(run, config, logdir):
    name = config.get("exp_name") or run.parent.name
    return name


def make_plots(logdirs, xaxis="Epoch", values=("TestAccuracy",), legend=None, smooth=1, output=None, show=False):
    groups = {}
    for i, logdir in enumerate(logdirs):
        for run, config, data in find_runs(logdir):
            name = legend[i] if legend else group_name(run, config, logdir)
            groups.setdefault(name, []).append(data)
    if not groups:
        raise SystemExit(f"No progress.txt files found under {', '.join(map(str, logdirs))}")
    figures = []
    for value in values:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name, runs in groups.items():
            runs = [r for r in runs if value in r and xaxis in r]
            if not runs:
                continue
            n = min(len(r[xaxis]) for r in runs)
            x = runs[0][xaxis][:n]
            ys = np.stack([r[value][:n] for r in runs])
            if smooth > 1:
                kernel = np.ones(smooth)
                ys = np.stack([np.convolve(y, kernel, "same") / np.convolve(np.ones_like(y), kernel, "same") for y in ys])
            mean, std = ys.mean(0), ys.std(0)
            label = f"{name} ({len(runs)} run{'s' if len(runs) > 1 else ''})"
            ax.plot(x, mean, lw=2, label=label)
            if len(runs) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(value)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        figures.append(fig)
        if output:
            out = pathlib.Path(output)
            target = out if len(values) == 1 and out.suffix else out.with_name(f"{out.stem}_{value}{out.suffix or '.png'}")
            fig.savefig(target, dpi=150)
            print(f"Saved {target}")
    if show:
        matplotlib.use("TkAgg", force=True)
        plt.show()
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot training curves saved by spinup_dl.")
    parser.add_argument("logdir", nargs="+")
    parser.add_argument("--xaxis", "-x", default="Epoch")
    parser.add_argument("--value", "-y", nargs="+", default=["TestAccuracy"])
    parser.add_argument("--legend", "-l", nargs="*")
    parser.add_argument("--smooth", "-s", type=int, default=1)
    parser.add_argument("--output", "-o", default="plot.png")
    args = parser.parse_args(argv)
    make_plots(args.logdir, args.xaxis, args.value, args.legend, args.smooth, args.output)


if __name__ == "__main__":
    main()
