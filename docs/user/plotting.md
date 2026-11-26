# Plotting Results

```{contents} Table of Contents
:depth: 2
```

## The Plot Command

The companion code includes a plotting utility for training curves:

```bash
python -m spinup_dl.utils.plot [logdir ...] [--value VALUE ...] [--xaxis XAXIS] [--legend NAME ...] [--smooth N] [--output FILE]
```

It searches each `logdir` for every `progress.txt` underneath it, groups the runs by experiment name, and plots each group as one curve: the average over its runs, with a shaded band of one standard deviation when there are several seeds.

| Argument | Meaning |
| --- | --- |
| `logdir` | One or more folders to search for results |
| `--value`, `-y` | The columns to plot, such as `TestAccuracy` or `TestLoss`. Several values make several plots. Defaults to `TestAccuracy` |
| `--xaxis`, `-x` | The column for the horizontal axis. Defaults to `Epoch` |
| `--legend`, `-l` | Names for the curves, one per `logdir`, instead of the experiment names |
| `--smooth`, `-s` | Smooth each curve with a moving average over this many points. Defaults to 1, no smoothing |
| `--output`, `-o` | File to save the plot to. Defaults to `plot.png`; with several values, the value name is added to the file name |

## Examples

Compare two experiments on test accuracy:

```bash
python -m spinup_dl.utils.plot data/mlp_mnist data/lenet_mnist --value TestAccuracy --output mnist.png
```

```{figure} ../_static/img/user/plot_example.png
:alt: Test accuracy against epoch for an MLP and a LeNet trained on MNIST for three epochs. Both rise above 97 percent, with the LeNet slightly higher.
:width: 80%

The output of the command above, for real three-epoch runs of the MLP and the LeNet on MNIST.
```

Plot a grid of experiments over several seeds, with a curve per setting and a band showing the spread between seeds:

```bash
python -m spinup_dl.utils.plot data/ --value TestLoss TestAccuracy --output sweep.png
```

Plot a language model's test loss against wall-clock time instead of epochs:

```bash
python -m spinup_dl.utils.plot data/lstm_shakespeare data/transformer_shakespeare --value TestLoss --xaxis Time
```

## Plotting in Your Own Code

`progress.txt` is a plain tab-separated file, so you can also load it yourself. `read_progress` returns a dictionary of NumPy arrays, one per column:

```python
import matplotlib.pyplot as plt
from spinup_dl.utils.plot import read_progress

progress = read_progress("data/lenet_mnist/lenet_mnist_s0/progress.txt")
plt.plot(progress["Epoch"], progress["TestAccuracy"])
plt.show()
```
