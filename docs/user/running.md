# Running Experiments

```{contents} Table of Contents
:depth: 2
```

One of the best ways to learn deep learning is to run the models, change them, and see what happens. The companion code makes this easy, from the command line or from your own Python scripts.

## Launching from the Command Line

Every model can be trained with

```bash
python -m spinup_dl.run [model] [arguments]
```

where `[model]` is `mlp`, `cnn`, `rnn`, `transformer`, `vae`, or `diffusion`, and each argument is written as `--name value`. For example, to train a LeNet on Fashion-MNIST for 5 epochs with a learning rate of 0.0005:

```bash
python -m spinup_dl.run cnn --arch lenet --dataset fashion_mnist --epochs 5 --lr 5e-4
```

Values are read as Python literals, so numbers, `True` and `False`, and lists all work. Tuples and lists are written without spaces, for example `--hidden_sizes [512,256]`.

To see every argument a model accepts, with its default value, run

```bash
python -m spinup_dl.run cnn --help
```

Each model's page under Models Docs explains its arguments in detail.

### Running a Grid of Experiments

Give an argument several values, and the runner trains one model for every combination:

```bash
python -m spinup_dl.run mlp --hidden_sizes [128] [256,256] --lr 1e-3 3e-4 --seed 0 1 2 --exp_name mlp_sweep
```

This runs $2 \times 2 \times 3 = 12$ experiments, one after another: two architectures, two learning rates, and three random seeds each.

### Special Arguments

A few arguments control the experiment rather than the model.

| Argument | Meaning |
| --- | --- |
| `--exp_name` | Name of the experiment, used for its output folder. Defaults to the model's name |
| `--data_dir` | Folder in which to save the results. Defaults to `data/` in the repository |
| `--datestamp` | Add the date and time to the output folder names, so that runs never overwrite each other |
| `--seed` | Random seed. Give several values to repeat the experiment |
| `--device` | `cpu`, `cuda`, or `mps`. Defaults to `auto`, which picks the fastest available |

Output folders are named after the experiment and after every argument that takes more than one value, so the runs of a grid are easy to tell apart. The grid above writes to folders such as `data/mlp_sweep_hidden_sizes128_lr0.001/mlp_sweep_hidden_sizes128_lr0.001_s0/`. The [Experiment Outputs](saving_and_loading.md) page describes what goes in them.

```{admonition} You Should Know
Always compare models over several random seeds. Two runs of the same model with different seeds can differ by more than a real improvement, especially early in training or on small datasets. Three seeds is a reasonable minimum; the [Plotting Results](plotting.md) page shows how to plot their average and spread.
```

## Launching from Scripts

Every model is also a Python function, so you can train it from your own code:

```python
from spinup_dl import cnn
from spinup_dl.utils.logx import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs("lenet_fashion", seed=0)
model = cnn(arch="lenet", dataset="fashion_mnist", epochs=5, seed=0, logger_kwargs=logger_kwargs)
```

The function returns the trained model. `setup_logger_kwargs` builds the output folder from the experiment's name and seed, the same way the command line does.

### Using ExperimentGrid

For grids of experiments in a script, use `ExperimentGrid`. Add each argument with the list of values to try, and call `run` with the model to train:

```python
from spinup_dl import ExperimentGrid, transformer

grid = ExperimentGrid(name="tf_depth")
grid.add("n_layers", [2, 4, 6])
grid.add("epochs", 5)
grid.add("seed", [0, 1, 2])
grid.run(transformer)
```

This trains Transformers with 2, 4, and 6 layers, each with three seeds. Arguments with a single value, like `epochs` here, apply to every run. To make an argument appear in the folder names even when it has a single value, add it with `in_name=True`.
