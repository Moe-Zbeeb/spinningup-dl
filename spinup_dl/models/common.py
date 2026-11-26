import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device="auto"):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer '{name}'; choose adam, adamw, or sgd")


def loaders(train_set, test_set, batch_size, num_workers=0):
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=num_workers)
    return train, test


def evaluate_classifier(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            count += y.numel()
    return total_loss / count, correct / count


def train_classifier(model, train_set, test_set, logger, epochs, batch_size, lr, weight_decay, optimizer, device, num_workers=0, lr_schedule="constant"):
    model.to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    train_loader, test_loader = loaders(train_set, test_set, batch_size, num_workers)
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_schedule '{lr_schedule}'; choose constant or cosine")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader)) if lr_schedule == "cosine" else None
    logger.log(f"Number of parameters: {count_parameters(model)}")
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            count += y.numel()
        test_loss, test_acc = evaluate_classifier(model, test_loader, device)
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("TrainLoss", total_loss / count)
        logger.log_tabular("TrainAccuracy", correct / count)
        logger.log_tabular("TestLoss", test_loss)
        logger.log_tabular("TestAccuracy", test_acc)
        logger.log_tabular("Time", time.time() - start)
        logger.dump_tabular()
        logger.save_model(model)
    return model
