# Information Theory and Optimization Background

```{contents} Table of Contents
:depth: 2
```

```{admonition} Coming soon
This page is being written one section at a time. More sections will be added below.
```

Welcome to the background for the course! Most of the cost functions used to train neural networks come from a few ideas in information theory, and training itself is an optimization problem. This page introduces those ideas before you meet them in the [Introduction to Deep Learning](../intro/neural_networks.md), so that the cost functions there look like natural choices rather than formulas to memorize.

## Information Content

Given a random variable $X$ and a possible outcome $x_i$ with probability $p_X(x_i) = p_i$, the **information content** of that outcome, also called its **self-information** or **surprisal**, is

$$
I(p_i) = -\log(p_i).
$$

An outcome with probability 1 has information content 0: seeing it tells you nothing you did not already know. As $p_i$ shrinks toward 0, $I(p_i)$ grows, so the rarer an outcome, the more surprising it is to observe. The base of the logarithm sets the unit: with $\log_2$, information is measured in **bits**, and with the natural logarithm, $\ln$, in **nats**. A fair coin landing heads, with $p = 0.5$, carries exactly 1 bit.

**Example: A Classifier's Prediction.** An image classifier that sorts photos into three classes outputs, through its softmax, a probability for each class. Its prediction for one photo is a random variable $Y$ whose outcomes are the classes:

| Outcome $y_i$ | Probability $p_i$ | $I(p_i)$ in bits | $I(p_i)$ in nats |
| --- | ---: | ---: | ---: |
| cat | 0.7 | 0.515 | 0.357 |
| dog | 0.2 | 2.322 | 1.609 |
| bird | 0.1 | 3.322 | 2.303 |

The model expects "cat", so finding that the photo really shows a cat surprises it little, while a bird would surprise it most. Now suppose the true label is "cat". The model's surprisal at the true label, $-\ln 0.7 \approx 0.357$ nats, is exactly the cross-entropy loss that [Part 2](../intro/improving_networks.md#the-loss) uses to train classifiers. Training a classifier means making it as little surprised as possible by the correct answers.
