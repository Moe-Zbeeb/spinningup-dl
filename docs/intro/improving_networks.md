# Part 2: Improving Deep Neural Networks

```{contents} Table of Contents
:depth: 2
```

Welcome to Part 2 of our introduction to deep learning! [Part 1](neural_networks.md) showed how a neural network computes its output and learns its parameters. This part is about making that learning work well in practice: training faster, generalizing better, and choosing the many settings that the equations leave open.[^sefidian] In this part, we aim to acquaint you with

- how to split your data and read the errors your model makes,
- regularization, the family of techniques that fights overfitting,
- the numerical problems of deep networks and how initialization and gradient checking guard against them,
- the optimization algorithms that replace plain gradient descent, from mini-batches to Adam,
- how to tune hyperparameters,
- batch normalization,
- and classification with more than two classes.

## Training, Dev (Cross-Validation), and Test Sets

Before you train anything, split your data into three sets, each with its own job.

- The **training set** is what gradient descent sees. It is used to learn the model's parameters, $W$ and $b$.
- The **dev set**, also called the **cross-validation set**, is used to compare models while you develop them: which architecture, which learning rate, which regularization. Choosing these settings is called tuning the **hyperparameters**.
- The **test set** is held back until the end. Because no decision was ever based on it, it gives an unbiased estimate of how the final model will do on new data. Some teams skip it and keep only a dev set, but then the reported error is no longer unbiased, since the dev set has been used to make choices.

How much data goes into each set depends on how much you have. With a small dataset, say a few thousand examples, the traditional splits are 70/30 (train and test) or 60/20/20 (train, dev, and test). With a million examples, a dev set of 10,000 is already plenty to tell two models apart, so splits like 98/1/1 are common.

```{admonition} You Should Know
The dev and test sets must come from the same distribution, the distribution you actually care about. If your training data comes from somewhere else, for example photos scraped from the web while your users upload phone photos, make sure the dev set matches the test set, not the training set. Otherwise you spend months tuning the model toward the wrong target.
```

**Example: A 98/1/1 Split.** PyTorch can split a dataset at random in one call:

```python
import torch
from torch.utils.data import TensorDataset, random_split

X, y = torch.randn(100_000, 20), torch.randint(0, 2, (100_000,))
data = TensorDataset(X, y)
train, dev, test = random_split(data, [0.98, 0.01, 0.01], generator=torch.Generator().manual_seed(0))
print(len(train), len(dev), len(test))
```

```text
98000 1000 1000
```

## Bias and Variance

Once a model is trained, its errors on the training set and on the dev set tell you what is wrong with it.

A model with **high bias** does badly even on the data it was trained on: it is too simple to capture the pattern, which is called **underfitting**. A model with **high variance** does well on the training set but much worse on the dev set: it has memorized the training examples, noise included, instead of learning the pattern, which is called **overfitting**.

```{figure} ../_static/img/improving_networks/bias_variance.svg
:alt: Three plots of the same eighteen noisy training points drawn from a smooth wavy curve. On the left, a straight line misses the curve. In the middle, a smooth curve follows the true function. On the right, a wiggly curve passes through every point and swings far outside the data between them.
:width: 100%

Three models fit to the same training data. The straight line underfits, the curve in the middle fits the pattern, and the wiggly curve chases the noise in each point.
```

To judge whether an error is high, compare it with the **base error**: the best error anyone could hope for on the task, often estimated by how well humans do. Suppose humans make almost no mistakes. Then

| Training error | Dev error | Diagnosis |
| ---: | ---: | --- |
| 1% | 10% | High variance |
| 14% | 15% | High bias |
| 14% | 28% | High bias and high variance |
| 0.5% | 1% | Low bias and low variance |

The gap between the base error and the training error measures bias. The gap between the training error and the dev error measures variance. High bias and high variance can happen at once, when a model is too rigid in some regions of the input and too flexible in others.

Each problem has its own fixes. For high bias, try them roughly in this order:

1. A bigger network, the most reliable fix. With good regularization it does not increase variance.
2. Training for longer.
3. A different architecture.
4. A hyperparameter search.
5. More informative input features.

For high variance:

1. More training data, the most reliable fix. It does not increase bias.
2. Regularization, including dropout, early stopping, and data augmentation, all covered below.
3. A less complex model.

```{admonition} You Should Know
Classical machine learning talks about a **bias-variance tradeoff**: reducing one tends to increase the other. Deep learning largely escapes it. A bigger network reduces bias without hurting variance, as long as you regularize, and more data reduces variance without hurting bias. As long as you can afford both, you can drive both down.
```

## Regularization

**Regularization** reduces overfitting by adding a penalty on large weights to the cost function. Gradient descent then has to balance fitting the training data against keeping the weights small.

For logistic regression, the **L2 regularization** adds the squared length of the weight vector:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \L\left(\hat{y}^{(i)}, y^{(i)}\right) + \frac{\lambda}{2m} \|w\|_2^2,
$$

where $\|w\|_2^2 = \sum_j w_j^2$, and $\lambda$ is the **regularization parameter**, a hyperparameter that sets the strength of the penalty. The **L1 regularization** uses the sum of absolute values instead:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \L\left(\hat{y}^{(i)}, y^{(i)}\right) + \frac{\lambda}{2m} \|w\|_1,
$$

where $\|w\|_1 = \sum_j |w_j|$. L1 tends to push many weights to exactly zero, which makes $w$ sparse. In practice, L2 is used far more often. The bias $b$ is usually left out of the penalty: it is a single number per unit, and penalizing it makes little difference.

For a neural network, the penalty sums over the weight matrices of all $L$ layers:

$$
J\left(W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]}\right) = \frac{1}{m} \sum_{i=1}^{m} \L\left(\hat{y}^{(i)}, y^{(i)}\right) + \frac{\lambda}{2m} \sum_{l=1}^{L} \left\|W^{[l]}\right\|_F^2,
$$

where the **Frobenius norm** $\|W^{[l]}\|_F^2 = \sum_{i=1}^{n^{[l]}} \sum_{j=1}^{n^{[l-1]}} \left(W^{[l]}_{ij}\right)^2$ is the sum of the squares of every entry, and $n^{[l]}$ is the number of units in layer $l$.

### Weight Decay

The penalty adds $\frac{\lambda}{m} W^{[l]}$ to the gradient of each weight matrix, so the gradient descent update becomes

$$
W^{[l]} \leftarrow W^{[l]} - \alpha \left( dW^{[l]}_{\text{backprop}} + \frac{\lambda}{m} W^{[l]} \right) = \left(1 - \frac{\alpha \lambda}{m}\right) W^{[l]} - \alpha \, dW^{[l]}_{\text{backprop}},
$$

where $dW^{[l]}_{\text{backprop}}$ is the gradient that [backward propagation](neural_networks.md#backward-propagation) computes without the penalty. Before every step, the weights are multiplied by a number slightly less than 1. This is why L2 regularization is also called **weight decay**.

### Why Regularization Reduces Overfitting

There are two ways to see it.

First, make $\lambda$ very large. Gradient descent then drives many weights close to zero, and units whose weights are near zero barely contribute. The network behaves like a much smaller network, and a smaller network cannot memorize as much. A well-chosen $\lambda$ sits between the two extremes: it trims the network's effective size just enough to stop it from fitting noise.

Second, look at a single tanh unit. Small weights mean a small $z$, and near $z = 0$ the tanh function is almost a straight line.

```{figure} ../_static/img/improving_networks/regularization_tanh.svg
:alt: The tanh curve next to the line z. A shaded band between z equals minus 0.5 and 0.5 marks the region where the two nearly coincide.
:width: 70%

Regularization keeps the weights, and so $z$, small. In that range tanh is nearly linear.
```

If every unit works in its nearly linear range, the whole network computes something close to a linear function. A nearly linear function is simple, and a simple function cannot bend around every noisy training point the way the wiggly curve in the [bias and variance figure](#bias-and-variance) does.

**Example: L2 Regularization in PyTorch.** You can add the penalty to the loss yourself. Here `mse_loss` already averages over the $m$ examples, matching the $\frac{1}{m}$ in the cost:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x, y = torch.randn(256, 10), torch.randn(256, 1)
model = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 1))

lambd, m = 0.7, x.shape[0]
penalty = sum((p ** 2).sum() for name, p in model.named_parameters() if "weight" in name)
loss = nn.functional.mse_loss(model(x), y) + lambd / (2 * m) * penalty
loss.backward()
```

PyTorch optimizers also accept a `weight_decay` argument that applies the decay directly in the update. Note that it applies to every parameter you pass it, biases included, and that its value plays the role of $\lambda / m$.

## Dropout

**Dropout** is a regularization technique that, on every training iteration, switches off a random subset of the units in each layer. The switched-off units output zero for that iteration, in both the forward and the backward pass, so they neither contribute to the prediction nor get updated. On the next iteration a new random subset is switched off. At test time no units are dropped.

```{figure} ../_static/img/improving_networks/dropout.svg
:alt: The same network drawn twice. On each iteration a different random set of hidden units is crossed out and their connections are faded.
:width: 100%

Dropout on two consecutive iterations. Each iteration trains a different thinned-out network.
```

Each layer can have its own **keep probability**, written `keep_prob`: the chance that a unit stays on. A layer with many parameters, which is more likely to overfit, can get a lower `keep_prob`, and the input layer usually keeps every unit.

### Inverted Dropout

The most common implementation is **inverted dropout**. For layer 3 with `keep_prob = 0.8`, it draws a random mask, zeroes the dropped units, and then divides by `keep_prob`:

```python
import torch

torch.manual_seed(0)
keep_prob = 0.8
a3 = torch.rand(1000, 64)

d3 = torch.rand_like(a3) < keep_prob
a3_dropped = a3 * d3 / keep_prob

print(f"kept {d3.float().mean():.2f} of the units")
print(f"mean activation before {a3.mean():.3f}, after {a3_dropped.mean():.3f}")
```

```text
kept 0.80 of the units
mean activation before 0.500, after 0.500
```

Dropping 20% of the units would shrink the input to the next layer by 20% on average. Dividing by 0.8 scales the surviving activations back up, so the expected value of `a3` stays the same, as the output shows. Thanks to this, nothing needs to change at test time: you simply run the network without dropout.

In practice, use `nn.Dropout`, which does exactly this. Its argument `p` is the probability of dropping a unit, $1 - \text{keep\_prob}$, and it is only active while the model is in training mode:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Linear(256, 10),
)
model.train()
model.eval()
```

Call `model.train()` before training and `model.eval()` before evaluating.

### Why Dropout Works

There are two ways to see it.

First, every iteration trains a smaller network, and smaller networks overfit less. Training many thinned-out networks that share weights acts a little like averaging an ensemble of models.

Second, look at a single unit in the next layer. Any of its inputs can disappear on any iteration, so it cannot rely on one input too heavily. The safest strategy is to spread its weight across all of its inputs, giving each a little. Spreading the weight out shrinks the squared norm of the weights, which is the same effect as L2 regularization.

```{admonition} You Should Know
With dropout on, the cost $J$ is no longer well defined, because every iteration computes it on a different random network. It does not have to decrease steadily from one iteration to the next, so a plot of $J$ cannot tell you whether gradient descent is working. A common practice is to turn dropout off, check that $J$ decreases, and then turn it back on.
```

## Other Regularization Methods

### Early Stopping

As training goes on, the training cost keeps falling, but the dev cost usually falls only up to a point and then rises again as the network starts fitting noise. **Early stopping** keeps the parameters from the iteration where the dev cost was lowest.

```{figure} ../_static/img/improving_networks/early_stopping.svg
:alt: Training cost and dev cost over two hundred thousand training steps on a log scale. Both fall quickly at first. The training cost keeps falling to almost zero. The dev cost bottoms out near 0.09 around step two hundred, stays flat, then climbs to almost 0.6.
:width: 80%

A model trained on 16 noisy points. The dev cost is lowest around step 200. Training on to the end drives the training cost to almost zero while the dev cost rises sixfold.
```

Early stopping works for the same reason as L2 regularization. The weights start small and grow during training, so stopping early leaves them at a moderate size.

```{admonition} You Should Know
Early stopping has a drawback. Training a model involves two separate jobs: making the cost $J$ small, and keeping the model from overfitting. It is easier to work on them one at a time, with separate tools for each, an idea called **orthogonalization**. Early stopping couples the two jobs, because the moment you stop, you also stop optimizing $J$. L2 regularization keeps them apart: you can train as long as you like and adjust $\lambda$ separately, at the price of trying several values of $\lambda$.
```

### Data Augmentation

More data is the most reliable cure for overfitting, but data is often expensive. **Data augmentation** creates new training examples from existing ones by applying changes that do not alter the label: flipping an image horizontally, cropping it, rotating it slightly, or distorting it a little. A flipped cat is still a cat. Augmented examples are not as informative as new, independent ones, but they cost almost nothing, and they teach the network that the label does not depend on these changes. Augmentation matters most when the dataset is small. More advanced variants use generative models to synthesize entirely new examples.

## Normalizing Training Sets

Training is faster when every input feature has a similar scale. **Normalizing** the inputs takes two steps. First subtract the mean,

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}, \qquad x \leftarrow x - \mu,
$$

so that every feature has mean zero. Then divide by the standard deviation,

$$
\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} \left(x^{(i)}\right)^2, \qquad x \leftarrow x / \sigma,
$$

where the square and the division are element-wise, so each feature gets its own $\sigma$. After the two steps every feature has mean 0 and variance 1.

### Why Normalization Speeds Up Training

Suppose feature $x_1$ ranges from 1 to 1000 and feature $x_2$ from 0 to 1. A small change in $w_1$ then moves the prediction far more than the same change in $w_2$, so the cost is much steeper in the $w_1$ direction than in the $w_2$ direction. Its contours are long, thin ellipses.

```{figure} ../_static/img/improving_networks/normalization_contours.svg
:alt: Two contour plots of a cost function with gradient descent paths. On the left the contours are thin ellipses and the path zig-zags across the narrow direction while creeping along the long one. On the right the contours are circles and the path heads straight to the minimum.
:width: 100%

Thirty steps of gradient descent on an elongated cost (left) and on the same problem after normalization (right). On the left, the largest learning rate that does not diverge in the steep direction is tiny for the shallow direction.
```

On elongated contours, gradient descent is stuck with a dilemma. A learning rate large enough to make progress along the shallow direction overshoots in the steep one and diverges, so you need a small learning rate and many steps. On round contours, the gradient points at the minimum from everywhere, and you can take large steps.

```{admonition} You Should Know
Compute $\mu$ and $\sigma$ on the training set only, and use those same values to normalize the dev and test sets. Every example must go through the same transformation, or the network sees the dev data on a different scale from the data it was trained on.
```

**Example: Normalizing with Training Statistics.**

```python
import torch

torch.manual_seed(0)
X_train = torch.randn(1000, 2) * torch.tensor([300.0, 0.3]) + torch.tensor([500.0, 0.5])
X_dev = torch.randn(200, 2) * torch.tensor([300.0, 0.3]) + torch.tensor([500.0, 0.5])

mu = X_train.mean(dim=0)
sigma = X_train.std(dim=0)
X_train = (X_train - mu) / sigma
X_dev = (X_dev - mu) / sigma
```

## Vanishing and Exploding Gradients

In a very deep network, activations and gradients can grow or shrink exponentially with depth. When they grow, the gradients **explode**: updates become huge and the numbers overflow. When they shrink, the gradients **vanish**: the early layers barely learn.

To see why, take a deep network with linear activations, $g(z) = z$, and set every bias to zero. Its output is a product of all the weight matrices:

$$
\hat{y} = W^{[L]} W^{[L-1]} \cdots W^{[2]} W^{[1]} x.
$$

If every $W^{[l]}$ is $1.5$ times the identity matrix, then $\hat{y} = 1.5^{L} x$. With $L = 50$ layers, that is about $6 \times 10^{8}$ times $x$. If every $W^{[l]}$ is $0.5$ times the identity, then $\hat{y} = 0.5^{50} x \approx 9 \times 10^{-16} x$. Weights only slightly larger than 1 make activations explode, and weights only slightly smaller than 1 make them vanish.

```{figure} ../_static/img/improving_networks/vanishing_exploding.svg
:alt: Typical size of the activations, on a log scale, against layer number from 0 to 50, for three weight scales. With weights 1.2 times the balanced scale, the size grows to about ten thousand. With the balanced scale it stays near one. With 0.8 times the balanced scale it shrinks to about one ten-thousandth.
:width: 80%

Activations in a 50-layer network with random weights. Scaling the weights 20% above or below a balanced value changes the size of the last layer's activations by a factor of about 10,000 in either direction.
```

The same happens to the gradients on the way back, because backward propagation multiplies by the transposes of the same matrices. Exploding gradients make training numerically unstable. Vanishing gradients leave the early layers almost frozen, so the network trains slowly or not at all.

## Weight Initialization for Deep Networks

Careful initialization is a partial solution to vanishing and exploding gradients. The idea is to choose the scale of the random weights so that each layer's activations stay at about the same size as its inputs.

A unit computes $z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n$, where $n = n^{[l-1]}$ is the number of inputs to layer $l$. If the inputs have variance 1 and the weights are independent with variance $\mathrm{Var}(w)$, then $z$ has variance $n \, \mathrm{Var}(w)$. The more inputs a unit has, the smaller each weight should be. Setting

$$
\mathrm{Var}(w) = \frac{1}{n^{[l-1]}}
$$

keeps the variance of $z$ at 1. This choice suits tanh and is called **Xavier initialization**. A ReLU sets about half of its inputs to zero, which halves the variance, so for ReLU layers the variance is doubled to $\frac{2}{n^{[l-1]}}$, which is called **He initialization**. A third option, $\frac{2}{n^{[l-1]} + n^{[l]}}$, balances the forward and the backward pass and is also used with tanh.

In code, you draw standard normal values and multiply them by the square root of the variance:

| Activation | Weight variance | Scale factor for `randn` |
| --- | --- | --- |
| tanh | $\frac{1}{n^{[l-1]}}$ | $\sqrt{1 / n^{[l-1]}}$ |
| tanh (alternative) | $\frac{2}{n^{[l-1]} + n^{[l]}}$ | $\sqrt{2 / (n^{[l-1]} + n^{[l]})}$ |
| ReLU | $\frac{2}{n^{[l-1]}}$ | $\sqrt{2 / n^{[l-1]}}$ |

This is the balanced scale in the figure above, the one that keeps the activations flat across 50 layers. It replaces the fixed factor of 0.01 from [Part 1](neural_networks.md#random-initialization), which only works for shallow networks.

**Example: Initialization in PyTorch.** PyTorch has both rules built in:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
layer = nn.Linear(512, 256)

nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
print(f"ReLU: std {layer.weight.std():.4f}, sqrt(2/512) = {(2 / 512) ** 0.5:.4f}")

nn.init.xavier_normal_(layer.weight)
print(f"tanh: std {layer.weight.std():.4f}, sqrt(2/(512+256)) = {(2 / 768) ** 0.5:.4f}")
```

```text
ReLU: std 0.0624, sqrt(2/512) = 0.0625
tanh: std 0.0511, sqrt(2/(512+256)) = 0.0510
```

## Numerical Approximation of Gradients

You can estimate the derivative of a function without calculus by evaluating it at nearby points. The **one-sided difference** uses one nearby point,

$$
f'(\theta) \approx \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon},
$$

and the **two-sided difference** uses one on each side,

$$
f'(\theta) \approx \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon},
$$

where $\epsilon$ is a small number. The two-sided difference is far more accurate: its error shrinks like $\epsilon^2$, while the one-sided error shrinks only like $\epsilon$. Since $\epsilon$ is less than 1, $\epsilon^2$ is much smaller than $\epsilon$.

**Example: Comparing the Two Differences.** Take $f(\theta) = \theta^3$ at $\theta = 1$, where the true derivative is 3:

```python
f = lambda t: t ** 3
t, true = 1.0, 3.0
for eps in (0.1, 0.01):
    one = (f(t + eps) - f(t)) / eps
    two = (f(t + eps) - f(t - eps)) / (2 * eps)
    print(f"eps={eps}: one-sided error {abs(one - true):.6f}, two-sided error {abs(two - true):.6f}")
```

```text
eps=0.1: one-sided error 0.310000, two-sided error 0.010000
eps=0.01: one-sided error 0.030100, two-sided error 0.000100
```

Dividing $\epsilon$ by 10 divides the one-sided error by about 10, but the two-sided error by 100.

## Gradient Checking

**Gradient checking** uses the two-sided difference to test whether your implementation of backward propagation is correct. It catches bugs that would otherwise only show up as a model that mysteriously trains badly.

1. Reshape $W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]}$ into one long vector $\theta$, so the cost becomes a function $J(\theta)$.
2. Reshape $dW^{[1]}, db^{[1]}, \dots, dW^{[L]}, db^{[L]}$ the same way into a vector $d\theta$, the gradient your backpropagation computes.
3. For every component $i$, estimate the gradient numerically:

$$
d\theta_{\text{approx}}[i] = \frac{J(\theta_1, \dots, \theta_i + \epsilon, \dots) - J(\theta_1, \dots, \theta_i - \epsilon, \dots)}{2\epsilon}.
$$

4. Compare the two vectors with

$$
\frac{\left\| d\theta_{\text{approx}} - d\theta \right\|_2}{\left\| d\theta_{\text{approx}} \right\|_2 + \left\| d\theta \right\|_2},
$$

where $\|\cdot\|_2$ is the Euclidean length of a vector. Dividing by the lengths makes the result independent of the scale of the gradient.

With $\epsilon = 10^{-7}$, read the result as follows.

| Result | Meaning |
| --- | --- |
| About $10^{-7}$ or less | Great: the implementation is almost certainly correct |
| About $10^{-5}$ | Suspicious: check the components carefully |
| About $10^{-3}$ or more | There is almost certainly a bug |

If the check fails, look at which components of $d\theta$ differ most from $d\theta_{\text{approx}}$. If they all belong to $db^{[3]}$, say, the bug is in the code for that gradient.

**Example: Gradient Checking a Small Network.** This compares PyTorch's automatic gradient of a two-layer network against the numerical estimate. It uses 64-bit floats, because the check needs more precision than 32 bits give:

```python
import torch

torch.manual_seed(0)
X, y = torch.randn(20, 3, dtype=torch.float64), torch.randn(20, 1, dtype=torch.float64)
theta = torch.randn(3 * 4 + 4 + 4 + 1, dtype=torch.float64, requires_grad=True)


def cost(theta):
    W1, b1 = theta[:12].view(4, 3), theta[12:16].view(4, 1)
    W2, b2 = theta[16:20].view(1, 4), theta[20:21].view(1, 1)
    A1 = torch.tanh(W1 @ X.T + b1)
    return ((W2 @ A1 + b2 - y.T) ** 2).mean()


cost(theta).backward()
dtheta = theta.grad

eps = 1e-7
dtheta_approx = torch.zeros_like(dtheta)
with torch.no_grad():
    for i in range(theta.numel()):
        plus, minus = theta.clone(), theta.clone()
        plus[i] += eps
        minus[i] -= eps
        dtheta_approx[i] = (cost(plus) - cost(minus)) / (2 * eps)

diff = (dtheta_approx - dtheta).norm() / (dtheta_approx.norm() + dtheta.norm())
print(f"difference: {diff:.1e}")
```

```text
difference: 1.2e-09
```

```{admonition} You Should Know
A few rules make gradient checking reliable.

- Use it only to debug, never during training. It evaluates the cost twice for every parameter, which is far too slow.
- If the cost includes a regularization term, include it in $J$ when you check, since the gradient includes it too.
- Turn dropout off while checking. With dropout, $J$ changes randomly from one evaluation to the next.
- Run the check once at initialization and again after some training. Some bugs only show up once the weights have grown away from their small starting values.
```

## Mini-Batch Gradient Descent

The gradient descent of Part 1 processes the whole training set before taking a single step. This is **batch gradient descent**. With millions of examples, every step takes a long time.

**Mini-batch gradient descent** splits the training set into small **mini-batches**, written $X^{\{1\}}, X^{\{2\}}, \dots$ with labels $Y^{\{1\}}, Y^{\{2\}}, \dots$, and takes one gradient descent step on each. One pass through all the mini-batches is called an **epoch**. With 5 million examples in mini-batches of 1000, an epoch makes 5000 steps instead of one.

The **mini-batch size** decides where you land between two extremes.

- **Size $m$** is batch gradient descent. Each step uses all the data, so it heads in a reliable direction, but it takes too long per step on a large dataset.
- **Size 1** is **stochastic gradient descent**: every example is its own mini-batch. Steps are cheap but noisy, the cost never settles exactly at the minimum, and you lose the speed-up from vectorization, because you process one example at a time.
- **In between** is usually best. You get the speed of vectorization and make progress long before seeing the whole training set. The path is noisier than batch gradient descent and may wander around the minimum without settling, which a slowly decreasing learning rate fixes.

```{figure} ../_static/img/improving_networks/minibatch_costs.svg
:alt: Three plots of the cost on the current batch over 300 updates. With the full batch the cost falls smoothly. With mini-batches of 64 it falls along the same trend but jitters up and down. With single examples it jumps around wildly.
:width: 100%

The cost on the current batch during logistic regression on 4096 examples. The dashed lines show the trend. Batch gradient descent falls smoothly. Mini-batch and stochastic gradient descent follow the same downward trend with more and more noise, but each of their steps costs a tiny fraction of a batch step.
```

With mini-batches, the cost does not fall at every step. It trends downward with noise, because some mini-batches are harder than others.

```{admonition} You Should Know
For a small training set, about 2000 examples or fewer, just use batch gradient descent. Otherwise, typical mini-batch sizes are powers of two, such as 64, 128, 256, and 512, because memory on CPUs and GPUs is organized in powers of two. Make sure one mini-batch fits in that memory.
```

**Example: Mini-Batches in PyTorch.** A `DataLoader` shuffles the data and hands it out one mini-batch at a time:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

X, y = torch.randn(10_000, 20), torch.randint(0, 2, (10_000,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

for X_batch, y_batch in loader:
    pass

print(len(loader), "mini-batches per epoch; the last one has", len(X_batch), "examples")
```

```text
157 mini-batches per epoch; the last one has 16 examples
```

## Exponentially Weighted Moving Averages

The faster optimization algorithms in the next sections are built on one tool: the **exponentially weighted moving average**, a way to smooth a noisy sequence $\theta_1, \theta_2, \dots$. It keeps a running value $V_t$ and updates it with every new data point:

$$
V_t = \beta V_{t-1} + (1 - \beta) \theta_t,
$$ (eq-improving-networks-ewma)

where $V_0 = 0$ and $\beta$ is between 0 and 1. Each new point gets weight $1 - \beta$, and everything that came before keeps weight $\beta$.

Unrolling {eq}`eq-improving-networks-ewma` shows that $V_t$ is a weighted sum of all past points, with weights that shrink exponentially into the past: $(1 - \beta)$ for $\theta_t$, $(1 - \beta)\beta$ for $\theta_{t-1}$, $(1 - \beta)\beta^2$ for $\theta_{t-2}$, and so on. Together the weights add up to 1, apart from a small correction near the start that we come back to below.

### How Far Back It Looks

$V_t$ is roughly an average over the last $\frac{1}{1 - \beta}$ points. With $\beta = 0.9$, that is about 10 points. The reason is that $\beta^{10} = 0.9^{10} \approx 0.35$, so a point 10 steps old has lost about two thirds of its weight. In general, with $\beta = 1 - \varepsilon$,

$$
(1 - \varepsilon)^{1/\varepsilon} \approx \frac{1}{e} \approx 0.37,
$$

so after $\frac{1}{\varepsilon} = \frac{1}{1 - \beta}$ steps, a point's weight has dropped to about a third.

```{figure} ../_static/img/improving_networks/ewma.svg
:alt: Two plots of noisy daily temperatures over 180 days with smoothed curves. On the left, beta 0.5 follows the noise closely, beta 0.9 is smooth and on track, and beta 0.98 is very smooth but lags behind and starts far too low. On the right, the beta 0.98 curve with bias correction starts on the data, while the uncorrected curve starts near zero.
:width: 100%

Smoothing daily temperatures. A larger $\beta$ averages over more days, which gives a smoother curve but reacts later to changes (left). Bias correction fixes the start of the curve (right).
```

The left panel shows the tradeoff. With $\beta = 0.5$, averaging over about 2 days, the curve follows every bit of noise. With $\beta = 0.98$, averaging over about 50 days, the curve is very smooth but lags behind the data, because it gives so much weight to old temperatures.

### Bias Correction

Because $V_0 = 0$, the first values of $V_t$ are far too small. With $\beta = 0.98$, $V_1 = 0.02\,\theta_1$, only 2% of the first temperature. The early weights do not add up to 1 yet. **Bias correction** divides by exactly the missing amount:

$$
V_t^{\text{corrected}} = \frac{V_t}{1 - \beta^t}.
$$

At $t = 1$ this divides by $0.02$, giving back $\theta_1$. As $t$ grows, $\beta^t$ goes to 0 and the correction disappears, which is what the right panel shows.

**Example: A Moving Average with Bias Correction.**

```python
def ewma(values, beta, bias_correction=True):
    v, out = 0.0, []
    for t, theta in enumerate(values, start=1):
        v = beta * v + (1 - beta) * theta
        out.append(v / (1 - beta ** t) if bias_correction else v)
    return out


temps = [20.0, 22.0, 21.0, 23.0]
print([round(v, 2) for v in ewma(temps, 0.9, bias_correction=False)])
print([round(v, 2) for v in ewma(temps, 0.9)])
```

```text
[2.0, 4.0, 5.7, 7.43]
[20.0, 21.05, 21.03, 21.61]
```

Without the correction, the average of temperatures around 21 degrees starts at 2.

## Gradient Descent with Momentum

**Gradient descent with momentum** almost always trains faster than plain gradient descent. Instead of stepping along the current gradient, it steps along an exponentially weighted moving average of recent gradients. On each iteration, compute $dW$ and $db$ on the current mini-batch, then

$$
\begin{aligned}
V_{dW} &= \beta V_{dW} + (1 - \beta)\, dW, & W &\leftarrow W - \alpha V_{dW}, \\
V_{db} &= \beta V_{db} + (1 - \beta)\, db, & b &\leftarrow b - \alpha V_{db},
\end{aligned}
$$

where $V_{dW}$ and $V_{db}$ start at zero and have the same shapes as $W$ and $b$.

### Why Momentum Helps

Picture a cost whose contours are long ellipses, as in the normalization figure. Gradient descent zig-zags across the narrow direction while creeping along the long one.

```{figure} ../_static/img/improving_networks/momentum_paths.svg
:alt: Two contour plots of an elongated cost with 40-step paths. Gradient descent zig-zags sharply up and down while moving slowly to the right. Momentum makes a few wide smooth swings and ends closer to the minimum.
:width: 100%

Forty steps on an elongated cost with the same learning rate. Momentum averages away most of the up-and-down zig-zag and ends 0.4 from the minimum, against 1.0 for plain gradient descent.
```

Averaging fixes this. In the narrow direction, the gradient flips sign on every step, so its average is close to zero and the oscillation dies down. In the long direction, every gradient points the same way, so the average keeps the full step. The result is a smoother path, and because the oscillations are damped, you can often use a larger learning rate.

A physical picture helps. Think of the parameters as a ball rolling down the cost surface. The gradient $dW$ is the push from the slope, $V_{dW}$ is the ball's velocity, and $\beta$ acts like friction that keeps the ball from accelerating without limit. A ball with momentum is not thrown off course by every bump.

```{admonition} You Should Know
- $\beta = 0.9$, an average over about the last 10 gradients, works well in almost every case. It is rarely worth tuning.
- Bias correction is usually skipped for momentum. After about 10 steps the average has warmed up anyway.
- A common variant drops the $1 - \beta$ factor: $V_{dW} = \beta V_{dW} + dW$. This makes $V_{dW}$ about $\frac{1}{1 - \beta}$ times larger, so the learning rate has to shrink by the factor $1 - \beta$ to compensate. PyTorch's `SGD` with `momentum=0.9` uses this variant.
```

## RMSprop (Root Mean Square Prop)

**RMSprop** gives every parameter its own effective learning rate. It keeps an exponentially weighted moving average of the squared gradients, and divides each step by its square root. On each iteration,

$$
\begin{aligned}
S_{dW} &= \beta S_{dW} + (1 - \beta)\, dW^2, & W &\leftarrow W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon}, \\
S_{db} &= \beta S_{db} + (1 - \beta)\, db^2, & b &\leftarrow b - \alpha \frac{db}{\sqrt{S_{db}} + \epsilon},
\end{aligned}
$$

where the squares, the square roots, and the division are all element-wise. The small constant $\epsilon$, often $10^{-8}$, keeps the division safe when $S$ is close to zero.

### Why RMSprop Helps

$\sqrt{S_{dW}}$ is the **root mean square** of recent gradients, measured separately for every parameter. Dividing by it rescales each parameter's step by how large its gradients usually are.

- A parameter with large gradients, such as the steep direction where gradient descent oscillates, gets divided by a large number, so its steps shrink and the oscillation is damped.
- A parameter with small gradients, such as the shallow direction where gradient descent creeps, gets divided by a small number, so its steps grow.

```{figure} ../_static/img/improving_networks/rmsprop_paths.svg
:alt: Two contour plots of an elongated cost with 40-step paths. Gradient descent zig-zags up and down. RMSprop moves smoothly down to the long axis and then straight along it to the minimum.
:width: 100%

Forty steps on the elongated cost. RMSprop takes evenly sized steps in both directions, so it neither oscillates nor creeps.
```

In the figure the steep direction is labeled $b$ and the shallow one $w$, but in a real network the steep and shallow directions are mixtures of many parameters. RMSprop adapts each parameter separately, wherever its gradients happen to be large or small. Because it damps the oscillations, it also allows a larger learning rate.

## Adam (Adaptive Moment Estimation)

**Adam** combines momentum and RMSprop, with bias correction for both. It keeps both moving averages and uses the first in the numerator of the step and the second in the denominator. On iteration $t$, compute $dW$ and $db$ on the current mini-batch, then

$$
\begin{aligned}
V_{dW} &= \beta_1 V_{dW} + (1 - \beta_1)\, dW, & V_{db} &= \beta_1 V_{db} + (1 - \beta_1)\, db, \\
S_{dW} &= \beta_2 S_{dW} + (1 - \beta_2)\, dW^2, & S_{db} &= \beta_2 S_{db} + (1 - \beta_2)\, db^2, \\
V_{dW}^{\text{corrected}} &= \frac{V_{dW}}{1 - \beta_1^t}, & V_{db}^{\text{corrected}} &= \frac{V_{db}}{1 - \beta_1^t}, \\
S_{dW}^{\text{corrected}} &= \frac{S_{dW}}{1 - \beta_2^t}, & S_{db}^{\text{corrected}} &= \frac{S_{db}}{1 - \beta_2^t}, \\
W &\leftarrow W - \alpha \frac{V_{dW}^{\text{corrected}}}{\sqrt{S_{dW}^{\text{corrected}}} + \epsilon}, & b &\leftarrow b - \alpha \frac{V_{db}^{\text{corrected}}}{\sqrt{S_{db}^{\text{corrected}}} + \epsilon}.
\end{aligned}
$$

The name comes from the two averages: $V$ estimates the mean of the gradient, its first **moment**, and $S$ estimates the mean of its square, the second moment.

Adam has four hyperparameters, and three of them almost never need tuning.

| Hyperparameter | Role | Usual value |
| --- | --- | --- |
| $\alpha$ | Learning rate | Needs tuning |
| $\beta_1$ | Momentum average | 0.9 |
| $\beta_2$ | RMSprop average | 0.999 |
| $\epsilon$ | Keeps the division safe | $10^{-8}$ |

Bias correction matters more here than for momentum. With $\beta_2 = 0.999$, the average $S$ looks back about 1000 steps, so without correction it would be far too small for hundreds of steps, and the very first updates would be huge.

```{figure} ../_static/img/improving_networks/optimizer_paths.svg
:alt: Four contour plots of the same elongated cost with 40-step paths for gradient descent, momentum, RMSprop, and Adam. Gradient descent zig-zags. Momentum swings smoothly. RMSprop goes straight along the long axis. Adam curves smoothly to the minimum with a small final loop.
:width: 100%

The four optimizers on the same elongated cost, 40 steps each. Adam combines the smooth averaging of momentum with the even step sizes of RMSprop.
```

Adam works well across a wide range of problems and architectures, which is why it is the usual first choice.

**Example: The Optimizers in PyTorch.** All three methods are built in. PyTorch calls RMSprop's $\beta$ `alpha`:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)

momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)
adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

## Learning Rate Decay

Early in training, large steps make fast progress. Near the minimum, they make mini-batch gradient descent bounce around instead of settling. **Learning rate decay** lowers the learning rate as training goes on, so the steps shrink as the parameters approach the minimum.

A common schedule divides the initial learning rate $\alpha_0$ by a factor that grows with the epoch number $t$:

$$
\alpha = \frac{1}{1 + \text{decay\_rate} \times t} \, \alpha_0,
$$

where $\text{decay\_rate}$ is a new hyperparameter. Other schedules are also used:

- **exponential decay**, $\alpha = 0.95^{t} \, \alpha_0$,
- $\alpha = \frac{k}{\sqrt{t}} \, \alpha_0$, for some constant $k$,
- a **staircase**, which keeps the learning rate fixed for a number of epochs and then cuts it, for example in half,
- and **manual decay**, where you watch a long training run and lower the learning rate by hand when progress stalls.

```{figure} ../_static/img/improving_networks/lr_decay.svg
:alt: Learning rate against epoch from 0 to 50 for four schedules starting at 0.2. The 1 over 1 plus 0.2 t schedule and the 1 over square root of t schedule drop quickly then flatten. The exponential schedule falls more steadily. The staircase halves every 10 epochs.
:width: 80%

Four decay schedules, all starting from $\alpha_0 = 0.2$.
```

```{admonition} You Should Know
Learning rate decay is lower on the list of hyperparameters to try. Get a good fixed learning rate first, and add decay only afterward.
```

**Example: A Decay Schedule in PyTorch.** A scheduler updates the optimizer's learning rate once per epoch. With $\alpha_0 = 0.2$ and $\text{decay\_rate} = 1$:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
decay_rate = 1.0
schedule = lambda epoch: 1 / (1 + decay_rate * epoch)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

for epoch in range(4):
    print(epoch, round(optimizer.param_groups[0]["lr"], 4))
    optimizer.step()
    scheduler.step()
```

```text
0 0.2
1 0.1
2 0.0667
3 0.05
```

## Local Optima and Saddle Points

A common worry is that gradient descent gets stuck in a bad local minimum. That picture comes from drawing the cost over two parameters, where local minima look everywhere. In the high-dimensional spaces of real networks, it is misleading.

At any point where the gradient is zero, the cost can curve up or down along each direction. A **local minimum** needs it to curve up along every direction at once. A **saddle point** curves up along some directions and down along others.

```{figure} ../_static/img/improving_networks/saddle_point.svg
:alt: Two 3D surfaces with a dot at the center. On the left, a bowl that curves up in every direction. On the right, a saddle that curves up along one axis and down along the other.
:width: 90%

A local minimum curves up in every direction. A saddle point curves up in some and down in others. Both have zero gradient at the center.
```

A rough way to see why saddle points dominate: if each direction were equally likely to curve up or down, the chance that all of 20,000 directions curve up would be $2^{-20000}$, effectively zero. So most points with zero gradient in a large network are saddle points, not local minima, and getting stuck in a bad local minimum is unlikely.

The real problem is **plateaus**: large regions where the gradient is close to zero, often around saddle points. Gradient descent crawls across them slowly before finding a direction that leads down. Momentum, RMSprop, and Adam help, because they keep moving and rescale small gradients.

## Hyperparameter Tuning Process

Training a network involves many hyperparameters. They are not equally important. A reasonable order of priority is:

1. the learning rate $\alpha$, the most important by far,
2. the momentum term $\beta$, with 0.9 as a good default,
3. the mini-batch size,
4. the number of hidden units,
5. the number of layers,
6. learning rate decay,
7. and Adam's $\beta_1$, $\beta_2$, and $\epsilon$, which almost never need tuning beyond their defaults.

### Random Search, Not Grid Search

The obvious way to search is a **grid**: try every combination of a few values for each hyperparameter. A better way is to choose the combinations at random.

```{figure} ../_static/img/improving_networks/grid_vs_random.svg
:alt: Two plots of nine trial points over a learning rate axis and an epsilon axis, with a shaded band showing that performance depends almost only on the learning rate. The grid search points line up in three columns, so they test only three learning rates. The random search points are scattered and test nine different learning rates.
:width: 100%

Nine trials of two hyperparameters, where only the learning rate matters. The grid tests just 3 learning rates; random search tests 9.
```

The reason is that you rarely know in advance which hyperparameters matter. Suppose you search over $\alpha$ and $\epsilon$ with 9 trials. On a 3 by 3 grid, you only ever try 3 values of $\alpha$, and the other 6 trials are wasted on $\epsilon$, which barely changes anything. With random combinations, every trial tests a new value of $\alpha$. With more hyperparameters, the space becomes a high-dimensional cube, and a grid fine enough to cover it is far too expensive.

### Coarse to Fine

Search in stages. First sample widely across the whole range. Then find the region where the best results cluster, zoom in on it, and sample more densely there.

### Choosing the Right Scale

Sampling at random does not mean sampling uniformly on a linear scale. Suppose $\alpha$ could be anywhere from $0.0001$ to $1$. Uniform sampling would put 90% of the trials between $0.1$ and $1$, and only 10% across the four orders of magnitude below. Sample on a **log scale** instead: draw $r$ uniformly from $[-4, 0]$ and set $\alpha = 10^{r}$. Then each of the ranges $0.0001$ to $0.001$, $0.001$ to $0.01$, and so on gets the same share of trials.

The same logic applies to the averaging parameters $\beta$, $\beta_1$, and $\beta_2$, but on $1 - \beta$. To search from $0.9$ to $0.999$, draw $r$ uniformly from $[-3, -1]$ and set $\beta = 1 - 10^{r}$. The reason is that the average looks back about $\frac{1}{1 - \beta}$ steps, which is very sensitive near 1. Moving $\beta$ from $0.9$ to $0.9005$ barely changes anything, about 10 steps either way. Moving it from $0.999$ to $0.9995$ changes the window from about 1000 steps to 2000.

**Example: Sampling on a Log Scale.**

```python
import torch

torch.manual_seed(0)
r = -4 * torch.rand(5)
alpha = 10 ** r

r = -3 + 2 * torch.rand(5)
beta = 1 - 10 ** r

print("alpha:", [f"{a:.5f}" for a in alpha.tolist()])
print("beta: ", [f"{b:.4f}" for b in beta.tolist()])
```

```text
alpha: ['0.01035', '0.00085', '0.44268', '0.29640', '0.05893']
beta:  ['0.9815', '0.9904', '0.9379', '0.9918', '0.9816']
```

### Pandas and Caviar

How you search also depends on how much computing power you have.

- **Babysitting one model**, the **panda** approach, like a panda raising a single cub: train one model, watch its learning curve every day, and nudge the hyperparameters as it goes. This needs little hardware and suits very large models that you can only afford to train once.
- **Training many models in parallel**, the **caviar** approach, like a fish laying thousands of eggs: launch many models with different hyperparameters at once and keep the best. This needs a lot of hardware, and when you have it, it is the better choice.

## Batch Normalization

[Normalizing the inputs](#normalizing-training-sets) speeds up training of the first layer. **Batch normalization** applies the same idea inside the network: it normalizes the values $z$ that feed every layer's activation function, so that each layer sees inputs on a consistent scale.

### How It Works

Take one layer and one mini-batch, with values $z^{(1)}, \dots, z^{(m)}$ for a given unit, where $m$ is now the mini-batch size. Batch normalization computes their mean and variance and normalizes them:

$$
\mu = \frac{1}{m} \sum_{i} z^{(i)}, \qquad \sigma^2 = \frac{1}{m} \sum_{i} \left(z^{(i)} - \mu\right)^2, \qquad z^{(i)}_{\text{norm}} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}},
$$

where $\epsilon$ keeps the division safe. Now every unit has mean 0 and variance 1. That is not always what we want: a sigmoid unit, say, might work better with inputs spread out more widely. So batch normalization adds two learnable parameters per unit, $\gamma$ and $\beta$, and computes

$$
\tilde{z}^{(i)} = \gamma \, z^{(i)}_{\text{norm}} + \beta,
$$

which goes into the activation function in place of $z^{(i)}$. Gradient descent learns $\gamma$ and $\beta$ like any other parameters, so the network chooses the mean and spread that work best for each unit. If it learns $\gamma = \sqrt{\sigma^2 + \epsilon}$ and $\beta = \mu$, it undoes the normalization entirely.

```{figure} ../_static/img/improving_networks/batch_norm.svg
:alt: Three histograms. The raw values z are centered at 4 with a wide spread. The normalized values are centered at 0 with standard deviation 1. The final values, with gamma 0.5 and beta minus 1.5, are centered at minus 1.5 and narrower.
:width: 100%

One unit's values across a batch: as computed (left), normalized (middle), and rescaled by the learned $\gamma$ and $\beta$ (right).
```

The parameter $\beta$ here is unrelated to the $\beta$ of momentum or Adam; the two just share a letter.

```{admonition} You Should Know
- Normalizing $z$, before the activation function, is the usual choice. Some work normalizes $a$, after the activation, instead.
- With batch normalization, the bias $b^{[l]}$ is no longer needed. Subtracting the mean cancels any constant added to $z$, and $\beta^{[l]}$ takes over the job of the bias.
- At test time you may predict on a single example, which has no batch to average over. Instead, keep an exponentially weighted moving average of $\mu$ and $\sigma^2$ across the mini-batches during training, and use those averages at test time.
```

### Why Batch Normalization Works

There are three ways to see it.

First, it does for every layer what input normalization does for the first: it puts the values on a similar scale, so the cost is better shaped and larger learning rates work.

Second, it makes each layer more robust to changes in the layers before it. When the distribution of a model's inputs changes, for example because new data looks different from old data, the model usually has to be retrained. This is called **covariate shift**. Inside a network, every hidden layer faces a version of this problem: as training updates the earlier layers, the values arriving at a later layer keep shifting, and the later layer has to keep chasing them. Batch normalization guarantees that, however the earlier layers change, the inputs to each layer keep the mean and variance set by its $\gamma$ and $\beta$. Later layers get firmer ground to stand on and can learn more independently. This matters most deep in the network, where the effects of all the earlier changes pile up.

Third, it has a slight regularization effect. Each mini-batch's mean and variance are computed from only that mini-batch, so they are a little noisy, which adds a little noise to every unit's value, much like dropout does. The effect is small, and it shrinks as the mini-batch grows. Do not rely on batch normalization as a regularizer.

**Example: Batch Normalization in PyTorch.** The layer before a `BatchNorm1d` can drop its bias:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(20, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
    nn.Linear(64, 1),
)
z = model[1](model[0](torch.randn(128, 20) * 5 + 3))
print(f"mean {z.mean():.3f}, std {z.std(unbiased=False):.3f}")
```

```text
mean 0.000, std 1.000
```

At the start of training $\gamma = 1$ and $\beta = 0$, so the output is exactly normalized. As with dropout, call `model.eval()` at test time, which switches batch normalization to the moving averages.

## Multi-Class Classification

So far the output layer has had a single unit for a yes-or-no answer. To choose among $C$ classes, the output layer gets $C$ units, one per class. Unit $j$ outputs the probability that the input belongs to class $j$, so the $C$ outputs must all be between 0 and 1 and must add up to 1.

### Softmax

The **softmax** activation function turns the output layer's $z^{[L]}$ into such probabilities. It exponentiates every entry and divides by their sum:

$$
t = e^{z^{[L]}}, \qquad a^{[L]}_j = \frac{t_j}{\sum_{k=1}^{C} t_k},
$$

where the exponential is element-wise. Exponentiating makes every entry positive, and dividing by the sum makes them add up to 1. Unlike the activation functions of [Part 1](neural_networks.md#activation-functions), softmax takes a whole vector and returns a whole vector, because each output depends on all the entries of $z^{[L]}$.

The name contrasts it with **hardmax**, which outputs 1 for the largest entry of $z^{[L]}$ and 0 everywhere else. Softmax is a soft version of the same idea: the largest entry gets the largest probability, but the others keep some.

```{figure} ../_static/img/improving_networks/softmax.svg
:alt: Three bar charts for four classes. The scores are 2, 1, minus 1, and 3. Softmax turns them into probabilities 0.24, 0.09, 0.01, and 0.66. Hardmax gives 0, 0, 0, and 1.
:width: 100%

Scores for four classes (left), their softmax probabilities (middle), and hardmax (right).
```

With $C = 2$, softmax reduces to logistic regression: the two probabilities are a sigmoid output and one minus it. Softmax is the generalization of logistic regression to more than two classes.

### The Loss

Write the true label $y$ as a vector with a 1 in the position of the correct class and 0 everywhere else. The loss for one example is

$$
\L(\hat{y}, y) = -\sum_{j=1}^{C} y_j \log \hat{y}_j,
$$

where $\hat{y} = a^{[L]}$. Only the term for the correct class survives, so the loss is $-\log$ of the probability the network gives to the right answer. Making the loss small means making that probability large. The cost averages the loss over the $m$ examples,

$$
J = \frac{1}{m} \sum_{i=1}^{m} \L\left(\hat{y}^{(i)}, y^{(i)}\right),
$$

and backward propagation starts from a simple gradient at the output layer:

$$
dz^{[L]} = \hat{y} - y.
$$

**Example: Softmax and Its Gradient in PyTorch.** PyTorch's `cross_entropy` takes the scores $z^{[L]}$ directly and applies softmax internally. Its gradient matches $\hat{y} - y$:

```python
import torch
import torch.nn.functional as F

z = torch.tensor([[2.0, 1.0, -1.0, 3.0]], requires_grad=True)
y = torch.tensor([3])

y_hat = torch.softmax(z, dim=1)
loss = F.cross_entropy(z, y)
loss.backward()

print("y_hat:", y_hat.detach().round(decimals=3))
print("dz:   ", z.grad.round(decimals=3))
```

```text
y_hat: tensor([[0.2420, 0.0890, 0.0120, 0.6570]])
dz:    tensor([[ 0.2420,  0.0890,  0.0120, -0.3430]])
```

The correct class is the fourth, so $y = (0, 0, 0, 1)$, and $dz$ is $\hat{y}$ with 1 subtracted from its fourth entry.

## Recap

In this part, you learned how to diagnose a model from its training and dev errors and what to do about high bias or high variance. You saw how L2 regularization, dropout, early stopping, and data augmentation fight overfitting. You learned how input normalization, careful initialization, and gradient checking make deep networks trainable and correct. You met the optimizers that replace plain gradient descent, from mini-batches through momentum and RMSprop to Adam, along with learning rate decay. Finally, you saw how to tune hyperparameters, how batch normalization stabilizes deep networks, and how softmax extends classification to many classes. Part 3 moves from training a single model to running a whole machine learning project.

[^sefidian]: The topics and their order follow Amir Masoud Sefidian, [Coursera Deep Learning Specialization Notes: Improving Deep Neural Networks](https://www.sefidian.com/wp-content/uploads/2023/02/Improving-Deep-Neural-Networks.pdf), 2022, which summarizes the second course of Andrew Ng's Deep Learning Specialization.
