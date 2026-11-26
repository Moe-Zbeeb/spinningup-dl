import time

import torch
import torch.nn as nn
from torchvision.utils import save_image

from spinup_dl.data import flat_dim, image_datasets
from spinup_dl.models.common import count_parameters, get_device, loaders, make_optimizer, set_seed
from spinup_dl.utils.logx import Logger


class VAE(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), hidden_size=400, latent_dim=20):
        super().__init__()
        self.config = {"image_shape": list(image_shape), "hidden_size": hidden_size, "latent_dim": latent_dim}
        self.image_shape = tuple(image_shape)
        d = flat_dim(image_shape)
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(d, hidden_size), nn.ReLU())
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, d))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z)).view(-1, *self.image_shape)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        return self.decode(torch.randn(n, self.config["latent_dim"], device=device))


def vae_loss(x, x_hat, mu, logvar):
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kl, recon, kl


def vae(dataset="mnist", hidden_size=400, latent_dim=20, epochs=10, batch_size=128, lr=1e-3, weight_decay=0.0,
        optimizer="adam", seed=0, device="auto", num_workers=0, logger_kwargs=dict()):
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())
    set_seed(seed)
    device = get_device(device)
    train_set, test_set, info = image_datasets(dataset)
    model = VAE(info["shape"], hidden_size, latent_dim).to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    train_loader, test_loader = loaders(train_set, test_set, batch_size, num_workers)
    logger.log(f"Number of parameters: {count_parameters(model)}")
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, _, _ = vae_loss(x, x_hat, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        model.eval()
        t_loss = t_recon = t_kl = 0.0
        m = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                loss, recon, kl = vae_loss(x, *model(x))
                t_loss += loss.item() * x.size(0)
                t_recon += recon.item() * x.size(0)
                t_kl += kl.item() * x.size(0)
                m += x.size(0)
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("TrainLoss", total / n)
        logger.log_tabular("TestLoss", t_loss / m)
        logger.log_tabular("TestReconstruction", t_recon / m)
        logger.log_tabular("TestKL", t_kl / m)
        logger.log_tabular("Time", time.time() - start)
        logger.dump_tabular()
        logger.save_model(model)
        save_image(model.sample(64, device), logger.output_dir / f"samples_epoch{epoch}.png", nrow=8)
    logger.close()
    return model
