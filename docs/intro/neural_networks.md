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

*Coming soon.*

## Random Initialization

*Coming soon.*
