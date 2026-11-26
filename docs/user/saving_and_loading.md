# Experiment Outputs

```{contents} Table of Contents
:depth: 2
```

## Model Outputs

Every run writes its results to its own folder:

```text
data/
└── lenet_mnist/
    └── lenet_mnist_s0/
        ├── config.json
        ├── progress.txt
        └── pyt_save/
            └── model.pt
```

| File | Contents |
| --- | --- |
| `config.json` | Every argument the trainer was called with, so that you can see exactly how a result was produced and run it again |
| `progress.txt` | The metrics of every epoch, one row per epoch, separated by tabs |
| `pyt_save/model.pt` | The trained model's weights and configuration, overwritten after every epoch |

The generative and language models save samples too, so that you can watch them improve:

| Model | Extra files |
| --- | --- |
| RNN, Transformer | `samples.txt`, a passage of generated text after every epoch |
| VAE, Diffusion | `samples_epoch1.png`, `samples_epoch2.png`, and so on: a grid of 64 generated images after each epoch |

### What Is Logged

The columns of `progress.txt` depend on the model.

| Models | Columns |
| --- | --- |
| MLP, CNN | `Epoch`, `TrainLoss`, `TrainAccuracy`, `TestLoss`, `TestAccuracy`, `Time` |
| RNN, Transformer | `Epoch`, `TrainLoss`, `TestLoss`, `TestPerplexity`, `Time` |
| VAE | `Epoch`, `TrainLoss`, `TestLoss`, `TestReconstruction`, `TestKL`, `Time` |
| Diffusion | `Epoch`, `TrainLoss`, `TestLoss`, `Time` |

`Time` is the number of seconds since training started. The losses are averages per example, and the train metrics are averaged over the epoch while the model was still changing, so they lag slightly behind the test metrics.

The same table is printed to the terminal after every epoch. This is the first epoch of the MLP on MNIST:

```text
---------------------------------
|         Epoch |           1 |
|     TrainLoss |      0.3263 |
| TrainAccuracy |      0.9098 |
|      TestLoss |      0.1613 |
|  TestAccuracy |      0.9526 |
|          Time |       3.137 |
---------------------------------
```

## Save Directory Location

By default, results go into `data/` inside the repository, in a folder named after the experiment and, inside it, one folder per seed. The [Running Experiments](running.md) page explains how to change the location with `--data_dir` or the `SPINUP_DL_DATA` environment variable, and how to add a date stamp to the folder names.

## Loading and Running Trained Models

`load_model` rebuilds a saved model from its folder:

```python
from spinup_dl import load_model

model = load_model("data/lenet_mnist/lenet_mnist_s0")
```

The model is returned in evaluation mode, on the CPU unless you pass `device="cuda"` or `device="mps"`.

### Classifying Images

```python
import torch
from torchvision import datasets, transforms

test = datasets.MNIST("data/datasets", train=False, download=True, transform=transforms.ToTensor())
x, y = test[0]
with torch.no_grad():
    probs = torch.softmax(model(x.unsqueeze(0)), dim=1)
print("predicted", probs.argmax().item(), "true", y)
```

### Generating Text

A language model saves its character vocabulary with it, in `model.extra["chars"]`:

```python
import torch
from spinup_dl import load_model

model = load_model("data/lstm_shakespeare/lstm_shakespeare_s0")
chars = model.extra["chars"]
prompt = torch.tensor([[chars.index(c) for c in "ROMEO:"]])
out = model.generate(prompt, length=300, temperature=0.8)
print("".join(chars[i] for i in out[0].tolist()))
```

A lower `temperature` makes the text more predictable, and a higher one makes it more varied.

### Generating Images

```python
from torchvision.utils import save_image
from spinup_dl import load_model

model = load_model("data/vae_mnist/vae_mnist_s0")
save_image(model.sample(64), "vae_samples.png", nrow=8)
```

Diffusion models have the same `sample` method. They take longer, because sampling runs the denoising network once for each of the `timesteps` steps, 1000 by default.
