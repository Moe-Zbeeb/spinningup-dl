import math
import time

import torch
import torch.nn as nn

from spinup_dl.models.common import count_parameters, make_optimizer


@torch.no_grad()
def evaluate_lm(model, data, device, batch_size, seq_len, batches=20):
    model.eval()
    g = torch.Generator().manual_seed(1234)
    total = 0.0
    for _ in range(batches):
        x, y = data.batch("test", batch_size, seq_len, generator=g)
        logits = model(x.to(device))
        total += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)).item()
    return total / batches


def train_language_model(model, data, logger, epochs, steps_per_epoch, batch_size, seq_len, lr, weight_decay, optimizer,
                         grad_clip, device, sample_length, prompt):
    model.to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    logger.log(f"Number of parameters: {count_parameters(model)}")
    extra = {"chars": "".join(data.chars)}
    start = time.time()
    samples = open(logger.output_dir / "samples.txt", "w")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            x, y = data.batch("train", batch_size, seq_len)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item()
        test_loss = evaluate_lm(model, data, device, batch_size, seq_len)
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("TrainLoss", total / steps_per_epoch)
        logger.log_tabular("TestLoss", test_loss)
        logger.log_tabular("TestPerplexity", math.exp(test_loss))
        logger.log_tabular("Time", time.time() - start)
        logger.dump_tabular()
        logger.save_model(model, extra=extra)
        if sample_length > 0:
            ids = data.encode(prompt) or [0]
            text = data.decode(model.generate(torch.tensor([ids], device=device), sample_length)[0].tolist())
            samples.write(f"===== epoch {epoch} =====\n{text}\n\n")
            samples.flush()
    samples.close()
    return model
