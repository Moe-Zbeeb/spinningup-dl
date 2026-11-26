import pathlib
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import INK, plt, setup

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "_static" / "img" / "authors"
PEOPLE = [
    ("ammar_mohanna", "Ammar Mohanna", "AUB–FAF Manager", (150, 80, 1450, 1380)),
    ("mohamad_zbeeb", "Mohamad Zbeeb", "Main Author", (100, 60, 700, 660)),
    ("mohamad_bazzi", "Mohamad Bazzi", "Main Author", (56, 20, 456, 420)),
]


def portrait(src, box, size=600):
    img = Image.open(src).convert("RGB").crop(box).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    mask = mask.resize((size, size), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def main(sources):
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for (slug, name, role, box), src in zip(PEOPLE, sources):
        img = portrait(src, box)
        img.save(OUT / f"{slug}.png")
        images.append((img, name, role))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6))
    for ax, (img, name, role) in zip(axes, images):
        ax.imshow(img)
        ax.set_xlim(-10, 610)
        ax.set_ylim(800, -10)
        ax.text(300, 680, name, ha="center", fontsize=17, fontweight="bold", color="#000000")
        ax.text(300, 750, role, ha="center", fontsize=14, color=INK)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "authors.png", dpi=150, transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1:4])
