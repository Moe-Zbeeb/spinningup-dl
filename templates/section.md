# Section Template

```{contents} Table of Contents
:depth: 2
```

## How to Use This Template

A topic is one `##` section of a part page. The section below, on gradient clipping, is a complete example of the shape every topic follows. Copy its structure, not its words.

1. An opening paragraph that says what the topic is and why the reader needs it now.
2. The idea, one step at a time. Each new term is bold where it is defined. Each equation is introduced by the sentence before it, and the sentence after it explains any symbol the reader has not met yet.
3. `###` subsections when the topic has distinct parts.
4. A short PyTorch example, led in by a bold "Example" label, when the notes contain code or the maintainer asks for one.
5. "You Should Know" boxes for conventions, caveats, and pointers to further reading.

## Gradient Clipping

When gradients become very large, a single update can throw the parameters far away from anywhere useful, and the loss jumps or turns into `nan`. **Gradient clipping** limits the size of the gradient before each update, so that one bad batch cannot undo the progress made so far.

### Clipping by Norm

The most common form rescales the whole gradient when its norm exceeds a threshold $c$. If $g$ is the gradient of the loss with respect to all the parameters, we replace it by

$$
\hat{g} = \min\left(1, \frac{c}{\|g\|_2}\right) g,
$$ (eq-section-clip)

where $\|g\|_2$ is the Euclidean norm of $g$. When the norm is below $c$, nothing changes. When it is above $c$, {eq}`eq-section-clip` shrinks the gradient to norm exactly $c$ and keeps its direction.

```{admonition} You Should Know
Clipping by norm changes the length of the gradient but not its direction. Clipping each coordinate separately, called **clipping by value**, changes both. This is why clipping by norm is the usual default.
```

### Clipping in Practice

**Example: Clipping in a Training Step.** In PyTorch, clipping is one line between the backward pass and the optimizer step:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(32, 10), torch.randn(32, 1)

loss = nn.functional.mse_loss(model(x), y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

The threshold `max_norm` plays the role of $c$. It is a hyperparameter: a reasonable way to choose it is to log the gradient norm for a few hundred steps and set the threshold a little above its typical value.

```{admonition} You Should Know
Clipping treats the symptom, not the cause. If the gradients explode on every step, look first at the learning rate and at how the weights are initialized.
```
