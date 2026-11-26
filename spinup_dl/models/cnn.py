import torch
import torch.nn as nn

from spinup_dl.data import image_datasets
from spinup_dl.models.common import get_device, set_seed, train_classifier
from spinup_dl.utils.logx import Logger


class LeNet(nn.Module):
    def __init__(self, in_channels=1, image_size=28, num_classes=10):
        super().__init__()
        self.config = {"in_channels": in_channels, "image_size": image_size, "num_classes": num_classes}
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_channels, image_size, image_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, a):
        z = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(a)))))
        return torch.relu(z + self.shortcut(a))


class ResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, widths=(32, 64, 128), blocks_per_stage=2):
        super().__init__()
        self.config = {"in_channels": in_channels, "num_classes": num_classes, "widths": list(widths), "blocks_per_stage": blocks_per_stage}
        self.stem = nn.Sequential(nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        blocks, prev = [], widths[0]
        for i, w in enumerate(widths):
            for j in range(blocks_per_stage):
                blocks.append(ResidualBlock(prev, w, stride=2 if (j == 0 and i > 0) else 1))
                prev = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(prev, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def cnn(arch="lenet", dataset="mnist", widths=(32, 64, 128), blocks_per_stage=2, augment=False, epochs=10, batch_size=128,
        lr=1e-3, weight_decay=0.0, optimizer="adam", seed=0, device="auto", num_workers=0, logger_kwargs=dict()):
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())
    set_seed(seed)
    train_set, test_set, info = image_datasets(dataset, augment=augment)
    channels, size = info["shape"][0], info["shape"][1]
    if arch == "lenet":
        model = LeNet(channels, size, info["classes"])
    elif arch == "resnet":
        model = ResNet(channels, info["classes"], widths, blocks_per_stage)
    else:
        raise ValueError(f"unknown arch '{arch}'; choose lenet or resnet")
    train_classifier(model, train_set, test_set, logger, epochs, batch_size, lr, weight_decay, optimizer, get_device(device), num_workers)
    logger.close()
    return model
