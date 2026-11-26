import torch
import torch.nn as nn

from spinup_dl.data import text_dataset
from spinup_dl.models.common import get_device, set_seed
from spinup_dl.models.language import train_language_model
from spinup_dl.utils.logx import Logger

CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=256, num_layers=2, cell="lstm", dropout=0.0):
        super().__init__()
        self.config = {"vocab_size": vocab_size, "embed_dim": embed_dim, "hidden_size": hidden_size, "num_layers": num_layers, "cell": cell, "dropout": dropout}
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = CELLS[cell](embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None, return_state=False):
        out, state = self.rnn(self.embed(x), state)
        logits = self.head(out)
        return (logits, state) if return_state else logits

    @torch.no_grad()
    def generate(self, ids, length, temperature=1.0):
        self.eval()
        logits, state = self(ids, return_state=True)
        out = ids
        for _ in range(length):
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
            logits, state = self(nxt, state, return_state=True)
        return out


def rnn(dataset="tiny_shakespeare", cell="lstm", embed_dim=64, hidden_size=256, num_layers=2, dropout=0.0, seq_len=128,
        batch_size=64, epochs=10, steps_per_epoch=200, lr=2e-3, weight_decay=0.0, optimizer="adam", grad_clip=1.0,
        sample_length=300, prompt="\n", seed=0, device="auto", logger_kwargs=dict()):
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())
    set_seed(seed)
    data = text_dataset(dataset)
    model = CharRNN(data.vocab_size, embed_dim, hidden_size, num_layers, cell, dropout)
    train_language_model(model, data, logger, epochs, steps_per_epoch, batch_size, seq_len, lr, weight_decay, optimizer,
                         grad_clip, get_device(device), sample_length, prompt)
    logger.close()
    return model
