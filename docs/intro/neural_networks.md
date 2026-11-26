# Part 1: Neural Networks and Deep Learning

```{contents} Table of Contents
:depth: 2
```

```{admonition} Coming soon
This part is being written. Its sections are listed below and will be filled in one at a time.
```

Welcome to Part 1 of our introduction to deep learning! This part builds up the basic neural network, one piece at a time. In this part, we aim to acquaint you with

- logistic regression, seen as a neural network,
- how a neural network computes its output and learns its parameters,
- the activation functions its units apply,
- and the random initialization of its parameters.

## Logistic Regression as a Neural Network

*Coming soon.*

## Neural Networks

A neural network is built from layers of units. Each layer takes the activations of the layer before it and computes activations of its own, until the last layer produces the prediction.

```{figure} ../_static/img/neural_networks/three_layer_network.svg
:alt: Four input features feed a first layer of four units, which feeds a second layer of three units, which feeds a third layer of one unit that outputs the prediction.
:width: 100%

A neural network with three layers. Unit $j$ of layer $l$ is written $a_j^{[l]}$, and it takes every activation of the layer before it.
```

The superscript in square brackets, $[l]$, names a layer, and the superscript in round brackets, $(i)$, names a training example. The network takes the input features $x_1^{(i)}, \dots, x_4^{(i)}$ of example $i$ and returns its prediction $\hat{y}^{(i)}$.

Training repeats three computations for every layer $l$: forward propagation, backward propagation, and the parameter update. All three are written in vectorized form, so they process the whole training set at once.

### Forward Propagation

**Forward propagation** goes through the layers in order. For layer $l$, it computes

$$
\begin{aligned}
Z^{[l]} &= W^{[l]} A^{[l-1]} + b^{[l]}, \\
A^{[l]} &= g^{[l]}\left(Z^{[l]}\right),
\end{aligned}
$$ (eq-neural-networks-forward)

where $W^{[l]}$ and $b^{[l]}$ are the weights and the bias of layer $l$, $g^{[l]}$ is its activation function, and $A^{[l-1]}$ holds the activations of the layer before it, with $A^{[0]} = X$, the input. Each column of $X$, $Z^{[l]}$, and $A^{[l]}$ belongs to one of the $m$ training examples.

### Backward Propagation

**Backward propagation** goes through the layers in reverse. For layer $l$, it computes

$$
\begin{aligned}
dZ^{[l]} &= dA^{[l]} \ast g^{[l]\prime}\left(Z^{[l]}\right), \\
dW^{[l]} &= \frac{1}{m} \, dZ^{[l]} \left(A^{[l-1]}\right)^{\top}, \\
db^{[l]} &= \frac{1}{m} \, \texttt{np.sum}\left(dZ^{[l]}, \texttt{axis=1}, \texttt{keepdims=True}\right), \\
dA^{[l-1]} &= \left(W^{[l]}\right)^{\top} dZ^{[l]},
\end{aligned}
$$

where $dA^{[l]}$ is the gradient of the cost with respect to $A^{[l]}$, and likewise for $dZ^{[l]}$, $dW^{[l]}$, and $db^{[l]}$. The symbol $\ast$ is element-wise multiplication, and $g^{[l]\prime}$ is the derivative of the activation function. The sum in the third line adds up the columns of $dZ^{[l]}$, one per example, and keeps the result as a column vector. The $Z^{[l]}$ and $A^{[l-1]}$ on the right-hand side are the ones computed in {eq}`eq-neural-networks-forward`. The last line hands $dA^{[l-1]}$ to the layer before, which is how the computation moves backward through the network.

### Parameter Update

Once the gradients are known, gradient descent updates the parameters of layer $l$:

$$
\begin{aligned}
W^{[l]} &\leftarrow W^{[l]} - \alpha \cdot dW^{[l]}, \\
b^{[l]} &\leftarrow b^{[l]} - \alpha \cdot db^{[l]},
\end{aligned}
$$

where $\alpha$ is the learning rate.

## Activation Functions

Every layer passes $Z^{[l]}$ through its **activation function** $g^{[l]}$, as in the [forward propagation](#forward-propagation) step. This section covers four common choices: the sigmoid, tanh, the ReLU, and the leaky ReLU.

### Sigmoid

The **sigmoid**, or logistic function, is usually used for the output layer only, because its output lies between 0 and 1. It is not used for the hidden layers, because its derivative is close to 0 for values of $z$ far from the origin. The function and its derivative are

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right).
$$

### Tanh

The **tanh** function is better than the sigmoid for hidden layers, even though it is only a shifted and rescaled sigmoid: $\tanh(z) = 2\sigma(2z) - 1$. Its output lies between $-1$ and $1$, so when its inputs have a mean of zero, its outputs have a mean close to zero as well, which can be useful for the computation that follows. The function and its derivative are

$$
\tanh(z) = \frac{2}{1 + e^{-2z}} - 1, \qquad \tanh'(z) = 1 - \tanh(z)^2.
$$

### ReLU

The **rectified linear unit**, or **ReLU**, $\mathrm{ReLU}(z) = \max(0, z)$, is the de facto standard for hidden units today. Its derivative is easy to compute, and it does not suffer from the slow convergence of gradient descent that a derivative close to 0 causes. The function and its derivative are

$$
\mathrm{ReLU}(z) = \begin{cases} 0 & \text{if } z < 0, \\ z & \text{if } z \geq 0, \end{cases} \qquad \mathrm{ReLU}'(z) = \begin{cases} 0 & \text{if } z < 0, \\ 1 & \text{if } z \geq 0. \end{cases}
$$

```{admonition} You Should Know
The derivative of the ReLU is not defined at $z = 0$, where the function has a corner. We work around this with a convention on what the derivative is at 0. The formula above sets it to 1.
```

### Leaky ReLU

The **leaky ReLU**, $\max(0.01z, z)$, is similar to the ReLU, but instead of being zero for values of $z$ below 0, it has a small positive slope there. Its derivative is therefore never 0, which makes it easier for an optimizer such as gradient descent to converge. In practice, the ordinary ReLU is still the standard. The function and its derivative are

$$
\mathrm{LReLU}(z) = \begin{cases} 0.01z & \text{if } z < 0, \\ z & \text{if } z \geq 0, \end{cases} \qquad \mathrm{LReLU}'(z) = \begin{cases} 0.01 & \text{if } z < 0, \\ 1 & \text{if } z \geq 0. \end{cases}
$$

```{figure} ../_static/img/neural_networks/activation_functions.svg
:alt: Six plots of activation functions over z from minus 6 to 6. The perceptron is a step from 0 to 1 at z equals 0. The sigmoid rises smoothly from 0 to 1, and tanh from minus 1 to 1. ReLU is zero for negative z and rises linearly after. Leaky ReLU has a small slope for negative z. ELU curves smoothly from minus 1 up to the line z.
:width: 100%

Activation functions. The perceptron step and the ELU are shown for comparison. The leaky ReLU is drawn with a slope of 0.1 for $z < 0$ so that the slope is visible.
```

## Random Initialization

Before the first step of gradient descent, every weight and bias needs a starting value. Zero looks like the natural choice. This section explains why zero fails for the weights, why it is fine for the biases, and why the weights should start small.

### The Symmetry Problem

Two units are **symmetric** if they do exactly the same computation. A symmetric unit is wasted: whatever it computes, its twin computes too.

To see how symmetry arises, take a hidden layer whose units all start with the same weights and the same bias, for example all zeros, and follow one step of training.

1. **Forward.** Unit $j$ computes $z_j = w_j x + b_j$, where $w_j$ is row $j$ of $W^{[l]}$. If every row $w_j$ and every bias $b_j$ is the same, every unit gets the same $z_j$ on every example, and so the same activation $a_j$.
2. **Backward.** The units feed the next layer through the same weights, so the cost cannot tell them apart, and [backward propagation](#backward-propagation) hands each of them the same gradient. The rows of $dW^{[l]}$ are identical, and so are the entries of $db^{[l]}$.
3. **Update.** Identical parameters minus identical steps are still identical.

After the update the units are exactly where they started: symmetric. The same three steps repeat on the next iteration, and the one after, so the units stay symmetric throughout all backprop iterations, however long you train. A hidden layer of 100 symmetric units computes nothing that a single unit could not.

Think of the hidden units as a team of people asked to describe photographs. If everyone starts with the same instructions and gets the same feedback after every attempt, nobody ever specializes: you get one opinion, repeated. To get a team where one person watches for edges, another for colors, and another for textures, the members have to start out different.

**Example: Watching Symmetry in PyTorch.** Here, a hidden layer of four units starts with every parameter equal to 0.5. After one backward pass, the gradient of its weights has four identical rows:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x, y = torch.randn(64, 3), torch.randn(64, 1)

model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
for p in model.parameters():
    nn.init.constant_(p, 0.5)

loss = nn.functional.mse_loss(model(x), y)
loss.backward()
print(model[0].weight.grad)
```

```text
tensor([[0.1355, 0.0580, 0.2716],
        [0.1355, 0.0580, 0.2716],
        [0.1355, 0.0580, 0.2716],
        [0.1355, 0.0580, 0.2716]])
```

Every unit is about to take exactly the same step.

### Breaking the Symmetry

We avoid symmetry by using **random initialization** rather than zero initialization. When each row of $W^{[l]}$ starts at different random values, each unit computes a slightly different $z_j$, receives a slightly different gradient, and takes a slightly different step. Gradient descent then pushes the units further apart, each toward whatever feature helps the cost most from where it started. This is called **breaking the symmetry**.

```{figure} ../_static/img/neural_networks/symmetry_breaking.svg
:alt: Two plots of the first weight of each of four hidden units over 2000 training steps. On the left, where every weight starts at 0.5, the four curves lie exactly on top of each other and the final loss is 0.55. On the right, where the weights start small and random, the curves separate after about 150 steps, one going up to 3 and three going down to between minus 2.6 and minus 3.4, and the final loss is 0.06.
:width: 100%

A network with four tanh hidden units, trained by gradient descent to tell whether the two coordinates of a point have the same sign. With every weight at 0.5 (left), the four units follow one curve for all 2000 steps, so the network acts as if it had a single unit and cannot solve the problem. With small random weights (right), the units separate and the loss falls to 0.06.
```

The figure shows both halves of the story. On the left, the units never separate, no matter how long training runs. On the right, the differences between the units are too small to see at first, but they grow until each unit has a role of its own.

**Example: Random Initialization in PyTorch.** To initialize a layer the way described here, draw the weights from a standard normal distribution scaled by 0.01, and set the biases to zero:

```python
import torch.nn as nn

layer = nn.Linear(3, 4)
nn.init.normal_(layer.weight, mean=0.0, std=0.01)
nn.init.zeros_(layer.bias)
print(layer.weight)
```

A normal distribution with standard deviation 0.01 is the same as `0.01 * torch.randn(...)`.

### Why the Bias Can Start at Zero

The bias $b$ does not need random initialization like the weights $W$. Symmetry needs units that are identical in every way. Once the rows of $W^{[l]}$ differ, the units already compute different $z_j$, even when every $b_j$ is zero. Their gradients differ, including their entries of $db^{[l]}$, so the biases drift apart on their own during training. Starting every bias at zero is the usual choice.

### Why the Weights Should Be Small

$W$ should be initialized to small random values, for example standard normal values multiplied by 0.01. Small weights allow faster convergence of gradient descent when the activation functions are sigmoid or tanh.

The reason is the shape of those functions. As the [Activation Functions](#activation-functions) section showed, sigmoid and tanh flatten out when $z$ is far from 0, and their derivatives there are close to 0. A unit whose $z$ lands in a flat region is **saturated**: gradient descent changes its weights very slowly, because every gradient that passes through it is multiplied by $g'(z) \approx 0$. The table shows how quickly the derivatives fall:

| $z$ | $\tanh'(z)$ | $\sigma'(z)$ |
| ---: | ---: | ---: |
| 0 | 1.0 | 0.25 |
| 0.32 | 0.91 | 0.24 |
| 3 | 0.0099 | 0.045 |
| 5 | 0.00018 | 0.0067 |

Large weights push $z$ into the flat region. Suppose a unit has 1000 inputs, each of size about 1, and its weights are drawn from a standard normal distribution. Then $z = w_1 x_1 + \dots + w_{1000} x_{1000}$ is a sum of 1000 random terms, and its typical size is $\sqrt{1000} \approx 32$, since the variances of independent terms add up. About 92% of such units start with $|z| > 3$, where $\tanh'$ is below 0.01. Multiply the weights by 0.01 and the typical size of $z$ drops to about 0.32, on the steep part of the curve, where $\tanh'(z) \approx 0.91$ and learning is fast.

```{admonition} You Should Know
The factor 0.01 is a rule of thumb that works well for networks with one or two hidden layers. In a deep network, weights that are too small shrink the signal a little more at every layer, which slows learning in a different way. The flat start of the right panel in the figure above is a mild version of this. Part 2 shows how to choose the scale of the weights from the number of inputs to each layer.
```
