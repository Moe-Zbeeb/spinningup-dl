import json
import math
import pathlib

import torch
import torch.nn as nn

OUT = pathlib.Path(__file__).resolve().parent / "data" / "convnets_resnet_experiment.json"


def spirals(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * 3 * math.pi + 0.5
    label = torch.randint(0, 2, (n,), generator=g)
    sign = 1 - 2 * label.float()
    x = torch.stack([sign * t * torch.cos(t), sign * t * torch.sin(t)], dim=1) / 10
    return x + 0.02 * torch.randn(n, 2, generator=g), label


class Block(nn.Module):
    def __init__(self, width, residual):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        self.residual = residual

    def forward(self, a):
        z = self.fc2(torch.relu(self.fc1(a)))
        return torch.relu(z + a) if self.residual else torch.relu(z)


def make(depth, residual, width=32):
    layers = [nn.Linear(2, width), nn.ReLU()]
    layers += [Block(width, residual) for _ in range(depth // 2)]
    layers += [nn.Linear(width, 2)]
    return nn.Sequential(*layers)


def train(depth, residual, x, y, steps=1500, seed=0):
    torch.manual_seed(seed)
    model = make(depth, residual)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(nn.functional.cross_entropy(model(x), y)), float((model(x).argmax(1) == y).float().mean())


def main():
    x, y = spirals(1000)
    depths = [2, 8, 20, 32, 56, 100]
    results = {"depths": depths, "plain": [], "residual": []}
    for d in depths:
        for kind in ("plain", "residual"):
            losses = [train(d, kind == "residual", x, y, seed=s)[0] for s in range(3)]
            results[kind].append(sum(losses) / len(losses))
            print(d, kind, round(results[kind][-1], 4), flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
