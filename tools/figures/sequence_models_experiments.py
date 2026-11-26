import json
import pathlib
import random

import torch
import torch.nn as nn

DATA = pathlib.Path(__file__).resolve().parent / "data"


def gradient_experiment():
    T = 100
    results = {"steps_back": list(range(T))}
    for kind in ("RNN", "LSTM"):
        norms = torch.zeros(T)
        for seed in range(20):
            torch.manual_seed(seed)
            cell = {"RNN": nn.RNN, "LSTM": nn.LSTM}[kind](8, 32, batch_first=True)
            if kind == "LSTM":
                with torch.no_grad():
                    cell.bias_ih_l0[32:64].fill_(1.0)
            x = torch.randn(1, T, 8, requires_grad=True)
            out, _ = cell(x)
            out[0, -1].sum().backward()
            norms += x.grad[0].norm(dim=1).flip(0)
        norms /= 20
        results[kind] = (norms / norms[0]).tolist()
        print("gradient", kind, [round(v, 5) for v in results[kind][::20]], flush=True)
    (DATA / "sequence_models_gradients.json").write_text(json.dumps(results, indent=2) + "\n")


def word2vec_experiment():
    rng = random.Random(0)
    groups = {
        "fruit": ["apple", "orange", "banana", "mango", "grape", "peach"],
        "drink": ["juice", "tea", "coffee", "milk", "water"],
        "animal": ["cat", "dog", "horse", "rabbit", "lion", "tiger"],
        "city": ["beirut", "paris", "tokyo", "cairo", "london", "rome"],
        "number": ["one", "two", "three", "four", "five", "six"],
    }
    templates = [
        ("i ate a fresh {fruit} this morning", ["fruit"]),
        ("she bought {number} kilos of {fruit} at the market", ["number", "fruit"]),
        ("a glass of {fruit} {drink} please", ["fruit", "drink"]),
        ("he drinks {drink} every day", ["drink"]),
        ("the {animal} slept near the door", ["animal"]),
        ("we saw {number} {animal}s at the zoo", ["number", "animal"]),
        ("my {animal} likes to run", ["animal"]),
        ("they flew to {city} last summer", ["city"]),
        ("the streets of {city} were busy", ["city"]),
        ("we spent {number} days in {city}", ["number", "city"]),
        ("i counted to {number} slowly", ["number"]),
    ]
    sentences = []
    for _ in range(30000):
        t, slots = rng.choice(templates)
        words = {s: rng.choice(groups[s]) for s in slots}
        sentences.append(t.format(**words).replace("s at", " at").split())
    vocab = sorted({w for s in sentences for w in s})
    idx = {w: i for i, w in enumerate(vocab)}
    counts = torch.zeros(len(vocab))
    pairs = []
    for s in sentences:
        ids = [idx[w] for w in s]
        for i, c in enumerate(ids):
            counts[c] += 1
            for j in range(max(0, i - 2), min(len(ids), i + 3)):
                if j != i:
                    pairs.append((c, ids[j]))
    pairs = torch.tensor(pairs)
    noise = counts ** 0.75
    noise /= noise.sum()
    torch.manual_seed(0)
    dim, k = 16, 5
    E = nn.Embedding(len(vocab), dim)
    theta = nn.Embedding(len(vocab), dim)
    nn.init.normal_(E.weight, std=0.1)
    nn.init.zeros_(theta.weight)
    opt = torch.optim.Adam(list(E.parameters()) + list(theta.parameters()), lr=0.01)
    for step in range(3000):
        batch = pairs[torch.randint(0, len(pairs), (512,))]
        c, t = batch[:, 0], batch[:, 1]
        neg = torch.multinomial(noise, 512 * k, replacement=True).view(512, k)
        ec = E(c)
        pos = nn.functional.logsigmoid((theta(t) * ec).sum(1))
        negs = nn.functional.logsigmoid(-(theta(neg) * ec.unsqueeze(1)).sum(2)).sum(1)
        loss = -(pos + negs).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    words = [w for g in groups.values() for w in g]
    W = E.weight.detach()[[idx[w] for w in words]]
    W = W - W.mean(0)
    U, S, V = torch.linalg.svd(W, full_matrices=False)
    xy = (W @ V[:2].T).tolist()
    label = {w: g for g, ws in groups.items() for w in ws}
    out = {"words": words, "groups": [label[w] for w in words], "xy": xy}
    (DATA / "sequence_models_word2vec.json").write_text(json.dumps(out, indent=2) + "\n")
    print("word2vec done, loss", round(float(loss), 3), flush=True)


class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, 32)
        self.encoder = nn.GRU(32, hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.GRUCell(32 + 2 * hidden, hidden)
        self.score = nn.Sequential(nn.Linear(hidden + 2 * hidden, 32), nn.Tanh(), nn.Linear(32, 1))
        self.out = nn.Linear(hidden + 2 * hidden, vocab)
        self.start = vocab

    def forward(self, x, y):
        enc, _ = self.encoder(self.embed(x))
        b, tx, _ = enc.shape
        s = torch.zeros(b, self.decoder.hidden_size)
        prev = torch.full((b,), self.start, dtype=torch.long)
        logits, alphas = [], []
        for t in range(y.shape[1]):
            e = self.score(torch.cat([s.unsqueeze(1).expand(b, tx, -1), enc], dim=2)).squeeze(2)
            alpha = torch.softmax(e, dim=1)
            context = (alpha.unsqueeze(2) * enc).sum(1)
            s = self.decoder(torch.cat([self.embed(prev), context], dim=1), s)
            logits.append(self.out(torch.cat([s, context], dim=1)))
            alphas.append(alpha)
            prev = y[:, t]
        return torch.stack(logits, 1), torch.stack(alphas, 1)


def attention_experiment():
    torch.manual_seed(0)
    vocab, length = 10, 8
    model = Seq2SeqAttention(vocab)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(2500):
        x = torch.randint(0, vocab, (64, length))
        y = x.flip(1)
        logits, _ = model(x, y)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
        y = x.flip(1)
        logits, alphas = model(x, y)
        test_x = torch.randint(0, vocab, (500, length))
        test_logits, _ = model(test_x, test_x.flip(1))
        acc = float((test_logits.argmax(2) == test_x.flip(1)).float().mean())
    out = {"input": x[0].tolist(), "output": y[0].tolist(), "alpha": alphas[0].tolist(), "accuracy": acc}
    (DATA / "sequence_models_attention.json").write_text(json.dumps(out, indent=2) + "\n")
    print("attention accuracy", round(acc, 3), flush=True)


if __name__ == "__main__":
    import sys

    DATA.mkdir(exist_ok=True)
    chosen = sys.argv[1:] or ["gradient", "word2vec", "attention"]
    for name in chosen:
        globals()[f"{name}_experiment"]()
