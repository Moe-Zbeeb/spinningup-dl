# Multilayer Perceptron

```{contents} Table of Contents
:depth: 2
```

## Background

A **multilayer perceptron**, or **MLP**, is the network of [Part 1](../intro/neural_networks.md#neural-networks): a stack of fully connected layers, each followed by an activation function, ending in an output layer. Every input feature is connected to every unit of the first layer, and every unit of one layer to every unit of the next. It is the simplest deep network, and the baseline every other model is measured against.

On images, an MLP first flattens the image into one long vector of pixels, so it ignores which pixels are next to each other. That is exactly the knowledge that [convolutional networks](cnn.md) build in, and comparing the two on the same dataset shows how much it is worth.

### Quick Facts

- Input: a fixed-size vector, here a flattened image. Output: one score per class.
- It makes no assumption about the structure of the input, so it treats pixels, words, and table columns alike.
- With enough hidden units, even one hidden layer can approximate any continuous function on a bounded domain.
- Use it for tabular data, as a baseline, and as a component inside larger models.

### Key Equations

For a network with $L$ layers, forward propagation computes, for $l = 1, \dots, L$,

$$
\begin{aligned}
z^{[l]} &= W^{[l]} a^{[l-1]} + b^{[l]}, \\
a^{[l]} &= g\left(z^{[l]}\right),
\end{aligned}
$$ (eq-mlp-forward)

where $a^{[0]} = x$ is the flattened input, $W^{[l]}$ and $b^{[l]}$ are the weights and bias of layer $l$, and $g$ is the activation function, ReLU by default. The last layer has no activation; its outputs $z^{[L]}$ are the class scores, turned into probabilities by a softmax, $\hat{y} = \mathrm{softmax}\left(z^{[L]}\right)$. Training minimizes the cross-entropy loss averaged over a mini-batch of $m$ examples,

$$
J = -\frac{1}{m} \sum_{i=1}^{m} \log \hat{y}^{(i)}_{y^{(i)}},
$$

the negative log of the probability the network gives to the correct class $y^{(i)}$, using [backward propagation](../intro/neural_networks.md#backward-propagation) and the [Adam](../intro/improving_networks.md#adam-adaptive-moment-estimation) optimizer.

### Pseudocode

```text
Algorithm 1  Training an MLP classifier

Input: training set, layer sizes, learning rate α, number of epochs
Initialize the weights W[l] randomly and the biases b[l] to zero
for epoch = 1, 2, ..., epochs do
    for each mini-batch (X, Y) of the shuffled training set do
        Flatten each image in X into a vector
        Compute the class scores with forward propagation, equation (1)
        Compute the cross-entropy loss J
        Compute the gradients dW[l], db[l] with backward propagation
        Update every W[l] and b[l] with Adam, learning rate α
    end for
    Log the training metrics and the loss and accuracy on the test set
    Save the model
end for
```

## Documentation

```python
spinup_dl.mlp(dataset="mnist", hidden_sizes=(256, 256), activation="relu", dropout=0.0, epochs=10, batch_size=128,
              lr=1e-3, weight_decay=0.0, optimizer="adam", seed=0, device="auto", num_workers=0, logger_kwargs=dict())
```

Trains an MLP classifier and returns it.

| Argument | Meaning |
| --- | --- |
| `dataset` | `mnist`, `fashion_mnist`, `cifar10`, or `fake`, a few hundred random images for testing |
| `hidden_sizes` | Number of units in each hidden layer. `(256, 256)` is two hidden layers of 256 units |
| `activation` | `relu`, `tanh`, `sigmoid`, or `gelu` |
| `dropout` | Probability of dropping each hidden unit during training. 0 turns dropout off |
| `epochs` | Number of passes through the training set |
| `batch_size` | Mini-batch size |
| `lr` | Learning rate |
| `weight_decay` | L2 regularization strength, applied by the optimizer |
| `optimizer` | `adam`, `adamw`, or `sgd`, which uses momentum 0.9 |
| `seed` | Random seed |
| `device` | `auto`, `cpu`, `cuda`, or `mps` |
| `num_workers` | Number of background processes that load data |
| `logger_kwargs` | Where to save the results; see [Running Experiments](../user/running.md) |

The model class is `spinup_dl.models.mlp.MLP(in_dim, hidden_sizes, out_dim, activation, dropout)`.

### What to Expect

```bash
python -m spinup_dl.run mlp --epochs 20
```

This run took 39 seconds and reached 97.8% test accuracy on MNIST. Its training accuracy reached 99.7%: the gap between the two is the overfitting that [regularization](../intro/improving_networks.md#regularization) addresses.

The trained model from this run is published, so you can use it without training:

```python
from spinup_dl import load_pretrained

model = load_pretrained("mlp_mnist")
```

### Things to Try

- Compare `--hidden_sizes [64]`, `[256,256]`, and `[1024,1024,1024]`. Bigger networks fit the training set better; watch whether the test accuracy follows.
- Turn on `--dropout 0.3` with a large network and compare the gap between `TrainAccuracy` and `TestAccuracy`.
- Switch `--activation sigmoid` and see how much slower training gets, as [Part 1](../intro/neural_networks.md#activation-functions) predicts.

## References

### Relevant Papers

- [Learning Representations by Back-Propagating Errors](https://doi.org/10.1038/323533a0), Rumelhart, Hinton, and Williams, 1986.
- [Multilayer Feedforward Networks Are Universal Approximators](https://doi.org/10.1016/0893-6080(89)90020-8), Hornik, Stinchcombe, and White, 1989.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), Kingma and Ba, 2014.

### Other Public Implementations

- [PyTorch tutorial: Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html), an MLP on Fashion-MNIST.
- [scikit-learn's MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), for tabular data without a deep learning framework.
