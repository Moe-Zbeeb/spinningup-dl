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

## Entropy

The **entropy** of a random variable $X$ with $K$ possible outcomes is the expected value of the information content of its outcomes:

$$
H(X) = H(p) \equiv \E\left[I(p_i)\right] = \sum_{i=1}^{K} p_i \log\left(\frac{1}{p_i}\right) = -\sum_{i=1}^{K} p_i \log(p_i),
$$

where $p = (p_1, \dots, p_K)$ lists the probabilities of the outcomes, and each outcome's information content $I(p_i)$ is weighted by how often that outcome occurs.

```{admonition} Intuition
The entropy of a random variable is its expected information content: the amount of surprise you should expect, on average, from observing it.
```

### Coding Scheme Interpretation

Before we move on, there is another way to interpret entropy. It uses the idea of a **coding scheme**, which for now you can think of as a way to represent the outcomes of a random variable as strings of binary digits.

Imagine you work at a company with three types of employees: full-timers, part-timers, and contractors. Your friend Claude, who sits in the other room, wants to be notified every time an employee enters the office, by a binary string sent to a pager. Since a pager message should be short, Claude asks you to use as few bits as possible.

Should you give the three employee types codes of the same length? Not really. Full-timers are very common, so you give them the shortest code. Part-timers are less common, so they get a slightly longer code. Contractors are rare, so they get the longest code. Common outcomes are sent often, so making their codes short saves the most bits.

Shannon found the function that gives the optimal number of bits to encode each outcome:

$$
\ell_i = \log_2\left(\frac{1}{p_i}\right) = -\log_2(p_i),
$$

where $\ell_i$ is the length, in bits, of the code for an outcome with probability $p_i$. This is exactly the information content $I(p_i)$ measured in bits: the more surprising an outcome, the longer its code.

Averaged over the outcomes, weighting each code length by how often its outcome occurs, these optimal codes use $\sum_i p_i \ell_i = -\sum_i p_i \log_2(p_i)$ bits per observation, which is the entropy of $X$ measured in bits. So entropy is the expected number of bits needed to encode an observation of $X$ under the optimal coding scheme.

## Cross-Entropy

We saw that entropy is the expected number of bits needed to encode an observation of $X$ under the optimal coding scheme. In contrast, **cross-entropy** is the expected number of bits needed to encode an observation of $X$ under the wrong coding scheme.

Call $q$ the wrong probability distribution that is used to build the coding scheme. Then we use $-\log(q_i)$ bits to encode the $i$-th possible value of $X$. Although the code is built from $q$, the observations are still distributed according to $p$, so we take the expected value over $p$:

$$
H(p, q) = \E_{p}\left[I(q_i)\right] = -\sum_{i} p_i \log(q_i).
$$

From this interpretation, it follows that

$$
\begin{aligned}
H(p, q) &> H(p) \quad \text{for all } q \neq p, \\
H(p, p) &= H(p).
\end{aligned}
$$

Using the wrong code always costs extra bits, and using the right one costs exactly the entropy.

```{admonition} Side Note
Log loss is often called cross-entropy loss. Indeed, it is the cross-entropy between the distribution of the true labels and the predictions.
```

