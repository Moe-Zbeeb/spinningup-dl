# Models

```{contents} Table of Contents
:depth: 2
```

## What's Included

The companion code for this site is a small Python package, `spinup_dl`, with standalone PyTorch implementations of the following models:

| Model | Trainer | Task | Default data |
| --- | --- | --- | --- |
| [MLP](../models/mlp.md) | `mlp` | Classify images | MNIST |
| [ConvNet](../models/cnn.md) | `cnn` | Classify images | MNIST |
| [RNN](../models/rnn.md) | `rnn` | Model text | Shakespeare |
| [Transformer](../models/transformer.md) | `transformer` | Model text | Shakespeare |
| [VAE](../models/vae.md) | `vae` | Generate images | MNIST |
| [Diffusion](../models/diffusion.md) | `diffusion` | Generate images | MNIST |

The trainers are imported from the package, as in `from spinup_dl import cnn`. The ConvNet trainer builds either a LeNet or a ResNet, and the RNN trainer a plain RNN, a GRU, or an LSTM. The two text models are character-level language models trained on Tiny Shakespeare, about a million characters of Shakespeare's plays.

Every model trains on a laptop CPU in minutes, uses a GPU automatically when there is one, and can also run on Fashion-MNIST, CIFAR-10, or any text file of your own.

## Why These Models?

These six models cover the main ideas of the [Introduction to Deep Learning](../intro/neural_networks.md), from the simplest network to the families behind modern systems.

- **The multilayer perceptron** is the network of [Part 1](../intro/neural_networks.md): forward propagation, backpropagation, and activation functions, with nothing else in the way. It is the baseline every other model is measured against.
- **The ConvNets** put [Part 4](../intro/convnets.md) into practice. LeNet shows the classic pattern of convolution and pooling; the ResNet adds residual blocks and batch normalization, the ingredients of most modern vision networks.
- **The recurrent networks** are the models of [Part 5](../intro/sequence_models.md). Training the same character-level language model with a plain RNN, a GRU, and an LSTM shows the difference that gating makes.
- **The Transformer** solves the same language modeling task with attention instead of recurrence. It is the architecture behind today's large language models, so it is worth comparing directly with the RNNs.
- **The variational autoencoder** and **the diffusion model** are generative: instead of labeling data, they learn to produce new data. They represent two different ideas, compressing data into a latent space and learning to reverse a noising process.

The implementations favor clarity over speed. Each one is short enough to read in one sitting, and uses only plain PyTorch, so what you read on the model's page is what runs.

## Code Format

Every model lives in one file under `spinup_dl/models/` and follows the same layout.

1. **A model class**, such as `MLP` or `Transformer`, an ordinary `nn.Module`. It stores its constructor arguments in `self.config`, so a saved model can be rebuilt exactly.
2. **A trainer function** with the same name as the file, such as `mlp` or `transformer`. Its arguments are the experiment's hyperparameters, each with a sensible default. It
   - creates a logger and saves the experiment's configuration,
   - sets the random seed,
   - loads the dataset,
   - builds the model and optimizer,
   - runs the training loop, logging the metrics of each epoch,
   - and saves the model after every epoch.

The two image classifiers share their training loop in `spinup_dl/models/common.py`, and the two language models share theirs in `spinup_dl/models/language.py`. The remaining code is utilities: `spinup_dl/data.py` loads the datasets, `spinup_dl/utils/logx.py` records experiments, `spinup_dl/utils/run_utils.py` runs grids of experiments, `spinup_dl/utils/plot.py` plots results, and `spinup_dl/utils/load.py` loads saved models.

```{admonition} You Should Know
The code is meant to be read and changed. To try an idea, such as a different activation function or a new kind of block, copy a model file, edit it, and train it the same way. The [Running Experiments](running.md) page shows how to compare your version with the original over several random seeds.
```
