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

*Coming soon.*
