import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt

BLUE = "#2e95d3"
INK = "#404452"
GRID = "#e4e4e8"
MAROON = "#840132"
ORANGE = "#e86c09"
GREEN = "#10a37f"
GRAY = "#8e8ea0"
SERIES = [BLUE, MAROON, ORANGE, GREEN]


def setup():
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
        "legend.frameon": False,
        "legend.fontsize": 12,
    })


def tidy(ax, title=None, grid=True):
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color="#000000", pad=10)
    if grid:
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save(fig, path):
    fig.savefig(path, transparent=True, metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
