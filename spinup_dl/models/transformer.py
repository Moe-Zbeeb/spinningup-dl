import torch
import torch.nn as nn

from spinup_dl.data import text_dataset
from spinup_dl.models.common import get_device, set_seed
from spinup_dl.models.language import train_language_model
from spinup_dl.utils.logx import Logger


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length=128, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = {"vocab_size": vocab_size, "context_length": context_length, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "dropout": dropout}
        self.context_length = context_length
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        t = x.size(1)
        if t > self.context_length:
            raise ValueError(f"sequence of length {t} is longer than the context length {self.context_length}")
        pos = torch.arange(t, device=x.device)
        h = self.drop(self.token(x) + self.position(pos))
        return self.head(self.ln(self.blocks(h)))

    @torch.no_grad()
    def generate(self, ids, length, temperature=1.0):
        self.eval()
        for _ in range(length):
            logits = self(ids[:, -self.context_length:])
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
        return ids


def transformer(dataset="tiny_shakespeare", context_length=128, d_model=128, n_heads=4, n_layers=4, dropout=0.1,
                batch_size=64, epochs=10, steps_per_epoch=200, lr=1e-3, weight_decay=0.01, optimizer="adamw", grad_clip=1.0,
                sample_length=300, prompt="\n", seed=0, device="auto", logger_kwargs=dict()):
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())
    set_seed(seed)
    data = text_dataset(dataset)
    model = Transformer(data.vocab_size, context_length, d_model, n_heads, n_layers, dropout)
    train_language_model(model, data, logger, epochs, steps_per_epoch, batch_size, context_length, lr, weight_decay, optimizer,
                         grad_clip, get_device(device), sample_length, prompt)
    logger.close()
    return model
