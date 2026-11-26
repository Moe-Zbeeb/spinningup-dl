import hashlib
import json
import math
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "spinup_dl" / "pretrained.json"

MODELS = {
    "mlp_mnist": ("mlp", "TestAccuracy"),
    "lenet_mnist": ("cnn --arch lenet", "TestAccuracy"),
    "resnet_cifar10": ("cnn --arch resnet --dataset cifar10 --augment True --lr_schedule cosine", "TestAccuracy"),
    "lstm_shakespeare": ("rnn --cell lstm", "TestPerplexity"),
    "transformer_shakespeare": ("transformer", "TestPerplexity"),
    "vae_mnist": ("vae", "TestLoss"),
    "diffusion_mnist": ("diffusion", "TestLoss"),
}


def last_row(progress):
    lines = progress.read_text().strip().splitlines()
    return dict(zip(lines[0].split("\t"), map(float, lines[-1].split("\t"))))


def describe(metric, row):
    epochs = int(row["Epoch"])
    if metric == "TestAccuracy":
        return f"{100 * row[metric]:.1f}% test accuracy after {epochs} epochs"
    if metric == "TestPerplexity":
        return f"test perplexity {row[metric]:.2f} after {epochs} epochs"
    return f"test loss {row[metric]:.4g} after {epochs} epochs"


def main(runs_dir, out_dir, tag):
    runs_dir, out_dir = pathlib.Path(runs_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = {"tag": tag, "models": {}}
    for name, (command, metric) in MODELS.items():
        run = runs_dir / name / f"{name}_s0"
        row = last_row(run / "progress.txt")
        target = out_dir / f"{name}.pt"
        shutil.copyfile(run / "pyt_save" / "model.pt", target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        registry["models"][name] = {
            "file": target.name,
            "sha256": digest,
            "result": describe(metric, row),
            "command": f"python -m spinup_dl.run {command} --epochs {int(row['Epoch'])}",
            "minutes": math.ceil(row["Time"] / 60),
        }
        print(f"{name:<24}{target.stat().st_size / 1e6:6.1f} MB  {registry['models'][name]['result']}")
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")
    print(f"Wrote {REGISTRY}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
