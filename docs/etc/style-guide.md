---
orphan: true
---

# Page Elements

```{contents} Table of Contents
:depth: 2
```

This page shows every element a content page can use, written the way it should appear in a real page. Open its source next to the rendered page and copy what you need. It is not linked from the sidebar.

Like every content page on Spinning Up, it opens with a table of contents.

## Text

Body text is set in Lato at 16px with a 24px line height, the same as OpenAI's Spinning Up. Use **bold** for a term at the point where it is defined, *italics* for emphasis, and `inline code` for anything a reader would type: a function such as `torch.nn.Linear`, a shape such as `(batch, features)`, or a file name.

Links to other pages are written with relative paths, like this one to [Part 1](../intro/neural_networks.md). External links look the same: [PyTorch documentation](https://pytorch.org/docs/).

- Bulleted lists are for parallel items with no order.
- Keep each item to a sentence or two.

1. Numbered lists are for steps that happen in order.
2. The forward pass comes before the loss.
3. The loss comes before the backward pass.

> Block quotes are for quoting a paper or a person, never for emphasis.

## Math

Inline math goes between single dollar signs: a linear layer computes $z = Wx + b$ with $W \in \R^{m \times n}$.

Display math goes between double dollar signs. Add a label when you want to refer back to it.

$$
\L(\theta) = - \frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)
$$ (eq-style-guide-nll)

Equation {eq}`eq-style-guide-nll` is the negative log-likelihood. A label is `eq-`, then this page's file name with hyphens, then a short name. Multi-line derivations use `aligned` inside the double dollar signs:

$$
\begin{aligned}
\nabla_\theta \L(\theta) &= \nabla_\theta \E_{(x, y) \sim \mathcal{D}} \big[ \ell(f_\theta(x), y) \big] \\
&\approx \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \, \ell(f_\theta(x_i), y_i)
\end{aligned}
$$

A function defined piece by piece uses `cases`:

$$
g'(z) = \begin{cases} 0 & \text{if } z < 0, \\ 1 & \text{if } z \geq 0. \end{cases}
$$

The macros `\R`, `\E`, `\L`, `\argmax`, and `\argmin` are defined for the whole site.

## Callout Boxes

The default box is "You Should Know". Use it for a convention the reader will meet elsewhere, a caveat, a common mistake, or a pointer to further reading.

```{admonition} You Should Know
The softmax function turns a vector of scores into a probability distribution:

$$
\mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

Subtracting the same constant from every score leaves the output unchanged.
```

An optional derivation that the main text can skip goes in a sidebar. On a wide screen it sits in the right margin, outside the text column. Place it at the top of its section and start its title with "(Optional)".

```{sidebar} (Optional) Deriving the Softmax Shift
Subtracting $c$ from every score multiplies the numerator and the denominator by the same factor $e^{-c}$:

$$
\frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_k}}{\sum_j e^{z_j}}.
$$
```

The text runs at full width beside it. On a narrower screen, the sidebar is a full-width box above the text instead.

```{note}
Notes are asides: useful, but the page reads fine without them.
```

```{tip}
Tips are practical advice, such as checking tensor shapes with `assert` before debugging anything else.
```

```{warning}
Warnings are for mistakes that cost real time, such as forgetting `optimizer.zero_grad()` and accumulating gradients across steps.
```

## Code

Code blocks name their language so they are highlighted. Every block has a copy button.

```python
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(784, 256, 10)
logits = model(torch.randn(32, 784))
print(logits.shape)
```

Shell commands use `bash`:

```bash
python -m pip install torch
python train.py --epochs 10 --lr 3e-4
```

Pseudocode uses a plain `text` block:

```text
Algorithm 1  Minibatch stochastic gradient descent

Input: initial parameters θ, learning rate α, batch size B
for step = 1, 2, ... do
    sample a batch {(x_i, y_i)} of size B from the dataset
    compute g = (1/B) Σ ∇θ ℓ(f_θ(x_i), y_i)
    update θ ← θ − α g
end for
```

## Tables

| Layer | Output shape | Parameters |
| --- | --- | ---: |
| Linear 784 → 256 | (batch, 256) | 200,960 |
| ReLU | (batch, 256) | 0 |
| Linear 256 → 10 | (batch, 10) | 2,570 |

## Figures

Figures live in `docs/_static/img/` and are placed with the `figure` directive so they get a caption.

```{figure} ../_static/img/banner.svg
:alt: A neural network reading a handwritten digit.
:width: 100%

A network reading a handwritten digit. Captions say what the reader should notice.
```

## Definitions

Loss
: A number that measures how wrong the model is on a batch of data. Training makes it smaller.

Epoch
: One full pass over the training set.

## Footnotes and References

Cite a paper with a footnote.[^adam] The footnote text can sit anywhere in the file; by convention it goes at the bottom.

[^adam]: Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2015.
