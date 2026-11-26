# Installation

```{contents} Table of Contents
:depth: 2
```

## Installing Python

The companion code needs Python 3.10 or newer. Check which version you have with

```bash
python3 --version
```

If you need a newer one, install it from [python.org](https://www.python.org/downloads/) or with a package manager such as Homebrew or `apt`. Then create a **virtual environment**, a private folder of packages for this project, so that its dependencies do not interfere with anything else on your computer:

```bash
python3 -m venv ~/venvs/spinup-dl
source ~/venvs/spinup-dl/bin/activate
```

Run the `source` line again in every new terminal before using the code. On Windows, the second line is `~\venvs\spinup-dl\Scripts\activate`.

## Installing PyTorch

The models are written in [PyTorch](https://pytorch.org/). The plain install below works on every platform and uses the CPU, or Apple's GPU on Macs with Apple silicon:

```bash
pip install torch torchvision
```

If you have an NVIDIA GPU, use the command that the [PyTorch website](https://pytorch.org/get-started/locally/) gives for your system and CUDA version instead.

## Installing Spinning Up in Deep Learning

Download the code and install it as an editable package, so that changes you make to the code take effect immediately:

```bash
git clone https://github.com/Moe-Zbeeb/spinningup-dl.git
cd spinningup-dl
pip install -e .
```

This installs `spinup_dl` together with NumPy and Matplotlib.

## Check Your Install

Train a tiny model on a few hundred randomly generated images. It downloads nothing and takes a few seconds:

```bash
python -m spinup_dl.run mlp --dataset fake --epochs 2
```

You should see a table for each of the two epochs, ending with a line like

```text
Logging data to .../spinningup-dl/data/mlp/mlp_s0/progress.txt
```

near the top of the output. If that works, try a real dataset. This downloads MNIST, about 12 MB, and trains for three epochs, which takes seconds on a recent laptop:

```bash
python -m spinup_dl.run mlp --epochs 3
```

The test accuracy should be around 97%.

## Running the Tests

The repository includes tests that train every model briefly on random data and check that saving, loading, the command-line runner, and plotting all work:

```bash
pip install -e ".[test]"
pytest
```

## Where Things Are Stored

Datasets are downloaded to `data/datasets/` and experiment results are written to `data/`, both inside the repository folder. To store them somewhere else, set the `SPINUP_DL_DATA` environment variable before running:

```bash
export SPINUP_DL_DATA=~/spinup_dl_data
```

The default location is set in `spinup_dl/user_config.py`.
