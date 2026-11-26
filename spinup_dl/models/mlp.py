import torch.nn as nn

from spinup_dl.data import flat_dim, image_datasets
from spinup_dl.models.common import ACTIVATIONS, get_device, set_seed, train_classifier
from spinup_dl.utils.logx import Logger


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden_sizes=(256, 256), out_dim=10, activation="relu", dropout=0.0):
        super().__init__()
        self.config = {"in_dim": in_dim, "hidden_sizes": list(hidden_sizes), "out_dim": out_dim, "activation": activation, "dropout": dropout}
        layers, prev = [nn.Flatten()], in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mlp(dataset="mnist", hidden_sizes=(256, 256), activation="relu", dropout=0.0, epochs=10, batch_size=128,
        lr=1e-3, weight_decay=0.0, optimizer="adam", seed=0, device="auto", num_workers=0, logger_kwargs=dict()):
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())
    set_seed(seed)
    train_set, test_set, info = image_datasets(dataset)
    model = MLP(flat_dim(info["shape"]), hidden_sizes, info["classes"], activation, dropout)
    train_classifier(model, train_set, test_set, logger, epochs, batch_size, lr, weight_decay, optimizer, get_device(device), num_workers)
    logger.close()
    return model
