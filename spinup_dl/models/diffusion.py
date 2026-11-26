import math
import time

import torch
import torch.nn as nn
from torchvision.utils import save_image

from spinup_dl.data import image_datasets
from spinup_dl.models.common import count_parameters, get_device, loaders, make_optimizer, set_seed
from spinup_dl.utils.logx import Logger


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class Denoiser(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.t_dim = 4 * c
        self.time_mlp = nn.Sequential(nn.Linear(c, self.t_dim), nn.SiLU(), nn.Linear(self.t_dim, self.t_dim))
        self.base = c
        self.inp = nn.Conv2d(in_channels, c, 3, padding=1)
        self.down1 = ResBlock(c, c, self.t_dim)
        self.down2 = ResBlock(c, 2 * c, self.t_dim)
        self.mid = ResBlock(2 * c, 2 * c, self.t_dim)
        self.up2 = ResBlock(4 * c, c, self.t_dim)
        self.up1 = ResBlock(2 * c, c, self.t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, in_channels, 3, padding=1))
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.base))
        h0 = self.inp(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        m = self.mid(self.pool(h2), t_emb)
        u2 = self.up2(torch.cat([self.upsample(m), h2], dim=1), t_emb)
        u1 = self.up1(torch.cat([self.upsample(u2), h1], dim=1), t_emb)
        return self.out(u1)


class Diffusion(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), base_channels=32, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.config = {"image_shape": list(image_shape), "base_channels": base_channels, "timesteps": timesteps, "beta_start": beta_start, "beta_end": beta_end}
        self.image_shape = tuple(image_shape)
        self.timesteps = timesteps
        self.net = Denoiser(image_shape[0], base_channels)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def loss(self, x0):
        x0 = 2 * x0 - 1
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t][:, None, None, None]
        x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise
        return nn.functional.mse_loss(self.net(x_t, t), noise)

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        x = torch.randn(n, *self.image_shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.net(x, t)
            alpha, a_bar, beta = self.alphas[i], self.alpha_bars[i], self.betas[i]
            x = (x - beta / (1 - a_bar).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        return ((x.clamp(-1, 1) + 1) / 2)


def diffusion(dataset="mnist", base_channels=32, timesteps=1000, beta_start=1e-4, beta_end=0.02, epochs=10, batch_size=128,
              lr=2e-4, weight_decay=0.0, optimizer="adam", sample_every=1, num_samples=64, seed=0, device="auto", num_workers=0,
              logger_kwargs=dict()):
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())
    set_seed(seed)
    device = get_device(device)
    train_set, test_set, info = image_datasets(dataset)
    if info["shape"][1] % 4:
        raise ValueError("image height and width must be divisible by 4")
    model = Diffusion(info["shape"], base_channels, timesteps, beta_start, beta_end).to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    train_loader, test_loader = loaders(train_set, test_set, batch_size, num_workers)
    logger.log(f"Number of parameters: {count_parameters(model)}")
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            loss = model.loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        model.eval()
        t_total, m = 0.0, 0
        with torch.no_grad():
            state = torch.get_rng_state()
            torch.manual_seed(0)
            for x, _ in test_loader:
                x = x.to(device)
                t_total += model.loss(x).item() * x.size(0)
                m += x.size(0)
            torch.set_rng_state(state)
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("TrainLoss", total / n)
        logger.log_tabular("TestLoss", t_total / m)
        logger.log_tabular("Time", time.time() - start)
        logger.dump_tabular()
        logger.save_model(model)
        if sample_every > 0 and (epoch % sample_every == 0 or epoch == epochs):
            save_image(model.sample(num_samples, device), logger.output_dir / f"samples_epoch{epoch}.png", nrow=8)
    logger.close()
    return model
