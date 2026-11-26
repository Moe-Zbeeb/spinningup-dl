# Part 5: Sequence Models

```{contents} Table of Contents
:depth: 2
```

Welcome to Part 5 of our introduction to deep learning! So far, every input has had a fixed size: a vector of features, or an image of a given height and width. Much of the world's data is not like that. A sentence can have three words or thirty, a recording can last a second or an hour, and what a word means depends on the words around it. **Sequence models** are built for data that unfolds in order.[^sefidian] In this part, we aim to acquaint you with

- recurrent neural networks, and how they read a sequence one step at a time,
- language models, and the GRU and LSTM units that let networks remember over long stretches,
- word embeddings, the vectors that capture what words mean, and how to learn them,
- sequence-to-sequence models for translation, with beam search and the BLEU score,
- the attention mechanism,
- and speech recognition.

## Applications

Sequence models handle problems where the input, the output, or both are sequences:

- **speech recognition**: audio in, a transcript out,
- **music generation**: perhaps nothing, or a genre, in, a sequence of notes out,
- **sentiment classification**: a review in, a star rating out,
- **DNA sequence analysis**: a sequence of bases in, labels for the parts that code for proteins out,
- **machine translation**: a sentence in one language in, the same sentence in another out,
- **video activity recognition**: a sequence of frames in, the activity out,
- and **named entity recognition**: a sentence in, the names of people, places, and organizations in it out.

The input and output can have the same length, as in named entity recognition, or different lengths, as in translation, and sometimes only one of them is a sequence.

## Notation

As a running example, take **named entity recognition**: deciding which words in a sentence are names of people.

```{figure} ../_static/img/sequence_models/ner_notation.svg
:alt: The sentence Layla and Karim visited Beirut in March, one word per box labeled x 1 to x 7, with a label below each word: 1 under Layla and Karim, which are people's names, and 0 under every other word.
:width: 100%

A sentence of 7 words and its labels: 1 for a person's name, 0 otherwise.
```

We write the elements of the input as $x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \dots, x^{\langle t \rangle}, \dots$, and those of the output as $y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, \dots$, where the superscript $\langle t \rangle$ is the position, or **time step**, in the sequence. Combined with the notation of earlier parts:

| Symbol | Meaning |
| --- | --- |
| $x^{(i)\langle t \rangle}$ | Element $t$ of the input of training example $i$ |
| $y^{(i)\langle t \rangle}$ | Element $t$ of the output of training example $i$ |
| $T_x^{(i)}$ | Length of the input of example $i$, here 7 |
| $T_y^{(i)}$ | Length of the output of example $i$, here also 7 |

Different examples can have different lengths, which is why $T_x$ carries the superscript $(i)$.

### Representing Words

To feed words to a network, first fix a **vocabulary**, or **dictionary**: a list of all the words the model knows. A real vocabulary has 10,000 to a million words. Each word is then represented by a **one-hot vector**: a vector as long as the vocabulary, with a 1 in the word's position and 0 everywhere else. Words that are not in the vocabulary are replaced by a special token, `<UNK>`, for "unknown".

**Example: One-Hot Vectors.** With a toy vocabulary of 9 tokens:

```python
import torch

vocab = ["<UNK>", "<EOS>", "and", "beirut", "in", "karim", "layla", "march", "visited"]
index = {word: i for i, word in enumerate(vocab)}

sentence = "Layla and Karim visited Beirut in March".lower().split()
ids = torch.tensor([index.get(w, index["<UNK>"]) for w in sentence])
one_hot = torch.nn.functional.one_hot(ids, num_classes=len(vocab))
print(ids.tolist())
print(one_hot[0].tolist())
```

```text
[6, 2, 5, 8, 3, 4, 7]
[0, 0, 0, 0, 0, 0, 1, 0, 0]
```

"layla" is word 6 of the vocabulary, so its one-hot vector has a 1 in position 6.

## Recurrent Neural Networks (RNN)

Why not use an ordinary fully connected network, with one input per word? There are two problems.

- Inputs and outputs can have different lengths in different examples. Padding every sentence to some maximum length is wasteful and still breaks on a longer one.
- An ordinary network does not share what it learns across positions. If it learns that "Layla" is a name when it appears first, it has to learn it again, with different weights, for every other position.

This is the same problem that convolutions solved for images, and recurrent networks solve it in a similar way: by reusing the same weights at every position.

### Forward Propagation

A **recurrent neural network**, or **RNN**, reads the sequence one element at a time. At each time step, it combines the current input $x^{\langle t \rangle}$ with its own activation from the previous step, $a^{\langle t-1 \rangle}$:

$$
\begin{aligned}
a^{\langle t \rangle} &= g\left(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a\right), \\
\hat{y}^{\langle t \rangle} &= g\left(W_{ya} a^{\langle t \rangle} + b_y\right),
\end{aligned}
$$

where $a^{\langle 0 \rangle}$ is usually a vector of zeros. The activation $a^{\langle t \rangle}$ summarizes everything the network has read so far, and it is passed on to the next step. The activation function in the first line is usually tanh, and the one in the second depends on the output: sigmoid for a yes-or-no label, softmax for a choice among words.

```{figure} ../_static/img/sequence_models/rnn_unrolled.svg
:alt: A row of identical RNN cells. Each cell takes the input x at its time step from below and the previous activation a from the left, outputs a prediction y hat above, and passes its activation to the next cell. The same weights are used at every step.
:width: 100%

An RNN unrolled in time. The same weights are used at every step, so the network can read sequences of any length.
```

The weights $W_{aa}$, $W_{ax}$, $W_{ya}$ and the biases are **shared** across all time steps. The prediction at step $t$ can use information from $x^{\langle 1 \rangle}$ through $x^{\langle t \rangle}$, but not from later inputs. In "Layla visited Beirut", the network must decide whether "Layla" is a name before it has seen the rest of the sentence. [Bidirectional RNNs](#bidirectional-rnns-brnn) fix this.

The notation is often simplified by placing $W_{aa}$ and $W_{ax}$ side by side in one matrix $W_a = \begin{bmatrix} W_{aa} & W_{ax} \end{bmatrix}$, and stacking $a^{\langle t-1 \rangle}$ on top of $x^{\langle t \rangle}$:

$$
a^{\langle t \rangle} = g\left(W_a \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_a\right), \qquad \hat{y}^{\langle t \rangle} = g\left(W_y a^{\langle t \rangle} + b_y\right).
$$

With a vocabulary of 10,000 words and 100 hidden units, $x^{\langle t \rangle}$ is $10{,}000 \times 1$ and $a^{\langle t \rangle}$ is $100 \times 1$, so $W_{aa}$ is $100 \times 100$, $W_{ax}$ is $100 \times 10{,}000$, and $W_a$ is $100 \times 10{,}100$.

**Example: One RNN Step, Two Ways.** The stacked form computes exactly the same thing:

```python
import torch

torch.manual_seed(0)
n_a, n_x = 4, 3
W_aa, W_ax, b_a = torch.randn(n_a, n_a), torch.randn(n_a, n_x), torch.randn(n_a, 1)
a_prev, x_t = torch.randn(n_a, 1), torch.randn(n_x, 1)

a_t = torch.tanh(W_aa @ a_prev + W_ax @ x_t + b_a)

W_a = torch.cat([W_aa, W_ax], dim=1)
a_t_stacked = torch.tanh(W_a @ torch.cat([a_prev, x_t], dim=0) + b_a)

print(W_a.shape, torch.allclose(a_t, a_t_stacked))
```

```text
torch.Size([4, 7]) True
```

### Backpropagation Through Time

For named entity recognition, each time step makes a yes-or-no prediction, so the loss at step $t$ is the logistic loss,

$$
\L^{\langle t \rangle}\left(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}\right) = -y^{\langle t \rangle} \log \hat{y}^{\langle t \rangle} - \left(1 - y^{\langle t \rangle}\right) \log\left(1 - \hat{y}^{\langle t \rangle}\right),
$$

and the loss for the whole sequence adds up the losses at every step:

$$
\L(\hat{y}, y) = \sum_{t=1}^{T_y} \L^{\langle t \rangle}\left(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}\right).
$$

Backpropagation runs through the unrolled network from right to left, backward in time, which is why it is called **backpropagation through time**. Because the same weights are used at every step, their gradient is the sum of their gradients at every step. And the gradient with respect to $a^{\langle t \rangle}$ gets contributions from two places: from the loss at step $t$, and from step $t + 1$, which used $a^{\langle t \rangle}$ as an input.

### Different Types of RNNs

Inputs and outputs can be arranged in several ways.[^karpathy]

```{figure} ../_static/img/sequence_models/rnn_types.svg
:alt: Five small diagrams: one to one with a single input and output; one to many with one input and four outputs; many to one with four inputs and one output at the end; many to many with an output at every input; and an encoder-decoder with three inputs followed by three outputs.
:width: 100%

The main RNN architectures.
```

- **Many to many, same length** ($T_x = T_y$): an output at every input, as in named entity recognition.
- **Many to many, different lengths** ($T_x \neq T_y$): as in machine translation. An **encoder** reads the whole input, then a **decoder** produces the output. [Basic Sequence Models](#basic-sequence-models) covers this.
- **Many to one**: read a whole sequence and produce one output at the end, as in sentiment classification.
- **One to many**: one input, or none, and a sequence of outputs, as in music generation. Each output is usually fed back in as the next input.
- **One to one**: an ordinary network, listed for completeness.

## Language Model and Sequence Generation

A **language model** assigns probabilities to sentences. It answers two related questions.

1. How likely is a given sentence? A speech recognizer hearing "I need two pairs of socks" must decide between that and "I need two pears of socks", which sound the same. A good language model gives the first a much higher probability.
2. Given the beginning of a sentence, how likely is each possible next word?

### Training a Language Model

To train an RNN as a language model, start with a large **corpus**: a body of text, such as a collection of news articles. **Tokenize** it, splitting it into words, and add an `<EOS>` token at the end of each sentence, so that the model can learn when sentences end. Replace rare words outside the vocabulary with `<UNK>`.

Then train the RNN to predict each word from the ones before it. At step 1, the input is $x^{\langle 1 \rangle} = 0$ and the output $\hat{y}^{\langle 1 \rangle}$ is a softmax over the whole vocabulary: the model's guess of the probability of every possible first word. At step 2, the input is the true first word, $x^{\langle 2 \rangle} = y^{\langle 1 \rangle}$, and the output is the probability of every possible second word given the first. In general, $x^{\langle t \rangle} = y^{\langle t-1 \rangle}$: each step is told the correct previous word and asked to predict the next one. The loss at each step is the softmax loss, and the loss for the sentence sums them:

$$
\L^{\langle t \rangle}\left(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}\right) = -\sum_{i} y^{\langle t \rangle}_i \log \hat{y}^{\langle t \rangle}_i, \qquad \L = \sum_{t} \L^{\langle t \rangle}\left(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}\right).
$$

Once trained, the model gives the probability of a whole sentence by the chain rule of probability:

$$
P\left(y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, y^{\langle 3 \rangle}\right) = P\left(y^{\langle 1 \rangle}\right) P\left(y^{\langle 2 \rangle} \mid y^{\langle 1 \rangle}\right) P\left(y^{\langle 3 \rangle} \mid y^{\langle 1 \rangle}, y^{\langle 2 \rangle}\right),
$$

where each factor is read off the softmax output at one time step.

**Example: The Probability of a Sentence.** With made-up conditional probabilities for each step, a toy model scores "the cat slept":

```python
import math

p_first = {"the": 0.20, "a": 0.15}
p_next = {("the", "cat"): 0.05, ("cat", "slept"): 0.10, ("slept", "<EOS>"): 0.40}

sentence = ["the", "cat", "slept", "<EOS>"]
prob = p_first[sentence[0]]
for prev, word in zip(sentence, sentence[1:]):
    prob *= p_next[(prev, word)]
print(f"P(sentence) = {prob:.6f}, log P = {math.log(prob):.3f}")
```

```text
P(sentence) = 0.000400, log P = -7.824
```

Even a plausible sentence has a small probability, because there are so many possible sentences. In practice, we work with the logarithm, which turns the product into a sum and avoids numbers too small to store.

### Sampling Novel Sequences

A trained language model can also write. To **sample** a sentence:

1. Start with $x^{\langle 1 \rangle} = 0$ and $a^{\langle 0 \rangle} = 0$, and compute the softmax output $\hat{y}^{\langle 1 \rangle}$, a probability for every word.
2. Pick a word at random according to those probabilities, for example with `torch.multinomial`.
3. Feed the chosen word in as $x^{\langle 2 \rangle}$, compute $\hat{y}^{\langle 2 \rangle}$, and sample the second word.
4. Repeat until the model samples `<EOS>`, or until the sentence reaches a chosen length. If it samples `<UNK>`, you can reject it and sample again.

**Example: Sampling Words from a Softmax.**

```python
import torch

torch.manual_seed(0)
vocab = ["the", "cat", "dog", "slept", "ran", "<EOS>"]
y_hat = torch.softmax(torch.tensor([2.0, 0.5, 0.4, 0.1, 0.1, -1.0]), dim=0)

samples = [vocab[i] for i in torch.multinomial(y_hat, num_samples=8, replacement=True)]
print([round(p, 2) for p in y_hat.tolist()])
print(samples)
```

```text
[0.56, 0.13, 0.11, 0.08, 0.08, 0.03]
['ran', 'dog', 'the', 'ran', 'cat', 'dog', 'the', 'the']
```

Sampling rather than always taking the most likely word is what makes the generated text varied. "the" has a probability of 0.56, but other words still come up.

A language model can also work with characters instead of words. A **character-level** model never meets an unknown word, since every word is made of known characters. But its sequences are much longer, which makes it more expensive to train and worse at capturing how words far apart in a sentence relate to each other. Word-level models are more common, though character-level models are used for text full of unusual words.

### Vanishing Gradients with RNNs

Basic RNNs are bad at **long-range dependencies**. Consider "The cats, which had already eaten all the food we left out this morning, were full." Whether the verb is "was" or "were" depends on "cats", a dozen words earlier. A basic RNN struggles to carry that information so far.

The cause is [vanishing gradients](improving_networks.md#vanishing-and-exploding-gradients), the same problem that affects very deep networks. An RNN unrolled over 100 time steps is a 100-layer network, and the gradient that reaches early steps has been multiplied by the same weights many times over.

```{figure} ../_static/img/sequence_models/rnn_gradients.svg
:alt: The size of the gradient with respect to an input, relative to the most recent input, against how many steps back the input is, on a log scale. For a plain RNN the gradient falls below one ten-billionth within 40 steps. For an LSTM it falls far more slowly, staying above one ten-thousandth at 60 steps.
:width: 80%

A real measurement on randomly initialized networks, averaged over 20 of each. The gradient of the final output with respect to each input shrinks exponentially with distance, but far faster for a plain RNN than for an LSTM.
```

Exploding gradients can happen too, but they are easier to deal with. They show up as huge numbers or `nan` values, so they are easy to spot, and **gradient clipping**, rescaling the gradient whenever its norm passes a threshold, controls them. Vanishing gradients are the harder problem, and the GRU and the LSTM were designed to solve it.

### Gated Recurrent Unit (GRU)

The **gated recurrent unit**, or **GRU**, adds a **memory cell** $c$ to the RNN.[^gru] For the GRU, the memory cell is the activation itself, $c^{\langle t \rangle} = a^{\langle t \rangle}$; the separate letter matters for the LSTM, where the two differ. The memory cell can hold on to a piece of information, such as whether the subject of the sentence is singular or plural, for as long as it is needed.

At every step, the GRU computes a **candidate** new value for the memory,

$$
\tilde{c}^{\langle t \rangle} = \tanh\left(W_c \begin{bmatrix} \Gamma_r \ast c^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_c\right),
$$

and then decides how much of the memory to replace with it. The decisions are made by **gates**, written $\Gamma$, the capital Greek letter gamma, for gate. Each gate is a vector of numbers between 0 and 1, computed by a sigmoid:

$$
\begin{aligned}
\Gamma_u &= \sigma\left(W_u \begin{bmatrix} c^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_u\right), \\
\Gamma_r &= \sigma\left(W_r \begin{bmatrix} c^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_r\right).
\end{aligned}
$$

The **update gate** $\Gamma_u$ decides whether to replace the memory with the candidate:

$$
c^{\langle t \rangle} = \Gamma_u \ast \tilde{c}^{\langle t \rangle} + \left(1 - \Gamma_u\right) \ast c^{\langle t-1 \rangle},
$$

where $\ast$ is element-wise multiplication. The **relevance gate** $\Gamma_r$ decides how much the old memory matters when computing the candidate. $\Gamma_u$, $\Gamma_r$, $c^{\langle t \rangle}$, and $\tilde{c}^{\langle t \rangle}$ all have the same size, the number of hidden units, so each unit has its own gates.

The key is the last equation. When $\Gamma_u$ is close to 0, $c^{\langle t \rangle} \approx c^{\langle t-1 \rangle}$: the memory is copied forward unchanged. In the sentence about the cats, a unit could set its memory to "plural" at "cats", keep $\Gamma_u \approx 0$ through the whole clause in between, and still have the information when it reaches "were". Because the sigmoid saturates, the gate is often very close to 0 or 1. And because a nearly unchanged memory also passes its gradient backward nearly unchanged, the GRU largely solves the vanishing gradient problem.

```{admonition} You Should Know
PyTorch's `nn.GRU` uses a slightly different but equivalent convention: its update gate $z$ weights the old memory rather than the candidate, $h_t = (1 - z_t) \ast n_t + z_t \ast h_{t-1}$, and it applies the relevance gate, which it calls the reset gate $r$, after multiplying the old memory by its weights. When reading papers or code, check which convention is in use.
```

## LSTM

The **long short-term memory** unit, or **LSTM**, is older than the GRU, more complex, and at least as widely used.[^lstm] Compared with the GRU, it makes three changes.

- The memory cell $c^{\langle t \rangle}$, the long-term memory, is now separate from the activation $a^{\langle t \rangle}$, the short-term output.
- The single update gate is split in two: an **update gate** $\Gamma_u$ decides how much of the candidate to add, and a **forget gate** $\Gamma_f$ decides how much of the old memory to keep. They no longer have to add up to 1.
- An **output gate** $\Gamma_o$ decides how much of the memory to reveal as the activation.

There is no relevance gate. The equations are

$$
\begin{aligned}
\tilde{c}^{\langle t \rangle} &= \tanh\left(W_c \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_c\right), \\
\Gamma_u &= \sigma\left(W_u \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_u\right), \\
\Gamma_f &= \sigma\left(W_f \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_f\right), \\
\Gamma_o &= \sigma\left(W_o \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_o\right), \\
c^{\langle t \rangle} &= \Gamma_u \ast \tilde{c}^{\langle t \rangle} + \Gamma_f \ast c^{\langle t-1 \rangle}, \\
a^{\langle t \rangle} &= \Gamma_o \ast \tanh\left(c^{\langle t \rangle}\right).
\end{aligned}
$$

```{figure} ../_static/img/sequence_models/lstm_cell.svg
:alt: The memory cell c runs straight across the top. The forget gate multiplies the old memory, the update gate times the candidate is added to it, giving the new memory. The new memory goes through tanh and is multiplied by the output gate to give the activation a. All four gates read the previous activation and the current input.
:width: 100%

The LSTM cell. The memory cell runs straight across the top, changed only by the forget gate's multiplication and the update gate's addition.
```

The line along the top of the figure is the heart of the LSTM. Information on it is only scaled by the forget gate and added to by the update gate, so with $\Gamma_f$ close to 1 and $\Gamma_u$ close to 0, the memory, and its gradient, can travel across many time steps almost unchanged. This is why the LSTM in the vanishing gradient figure keeps its gradient so much longer than the plain RNN.

Some LSTM variants add **peephole connections**, which let the gates also look at $c^{\langle t-1 \rangle}$. In these, element $i$ of $c^{\langle t-1 \rangle}$ affects only element $i$ of each gate.

### GRU or LSTM?

There is no consensus on which is better; each wins on some problems. The GRU is simpler, with two gates instead of three, so it is faster to run and easier to scale to bigger networks. The LSTM is more flexible and has the longer track record, which makes it the more common default. Both are very good at long-range dependencies.

**Example: The LSTM Equations Match PyTorch.** Computing one step by hand with the weights of a PyTorch `LSTMCell` gives the same result. PyTorch stores the four weight matrices stacked, in the order update, forget, candidate, output:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
n_x, n_a = 3, 4
cell = nn.LSTMCell(n_x, n_a)
x_t, a_prev, c_prev = torch.randn(1, n_x), torch.randn(1, n_a), torch.randn(1, n_a)

z = x_t @ cell.weight_ih.T + cell.bias_ih + a_prev @ cell.weight_hh.T + cell.bias_hh
gamma_u, gamma_f, c_tilde, gamma_o = z.chunk(4, dim=1)
gamma_u, gamma_f, gamma_o = torch.sigmoid(gamma_u), torch.sigmoid(gamma_f), torch.sigmoid(gamma_o)
c_tilde = torch.tanh(c_tilde)

c_t = gamma_u * c_tilde + gamma_f * c_prev
a_t = gamma_o * torch.tanh(c_t)

a_ref, c_ref = cell(x_t, (a_prev, c_prev))
print(torch.allclose(a_t, a_ref, atol=1e-6), torch.allclose(c_t, c_ref, atol=1e-6))
```

```text
True True
```

## Bidirectional RNNs (BRNN)

The RNNs so far only look backward: the prediction at step $t$ cannot use anything after it. Sometimes the later words are exactly what you need. In "Jordan flew to Beirut on Monday" and "Jordan borders Syria to the north", the first word alone cannot tell you whether "Jordan" is a person or a country; the words after it can.

A **bidirectional RNN**, or **BRNN**, runs two RNNs over the sequence: a forward one from left to right and a backward one from right to left. Each time step then has two activations, $\overrightarrow{a}^{\langle t \rangle}$ from the forward RNN and $\overleftarrow{a}^{\langle t \rangle}$ from the backward one, and the prediction uses both:

$$
\hat{y}^{\langle t \rangle} = g\left(W_y \begin{bmatrix} \overrightarrow{a}^{\langle t \rangle} \\ \overleftarrow{a}^{\langle t \rangle} \end{bmatrix} + b_y\right).
$$

```{figure} ../_static/img/sequence_models/brnn.svg
:alt: Four time steps. A forward layer passes activations left to right and a backward layer passes them right to left. Each output y hat reads from both layers at its time step.
:width: 100%

A bidirectional RNN. Every prediction combines information from the past and from the future.
```

The two directions can use plain RNN units, GRUs, or LSTMs; bidirectional LSTMs are especially common, and are a strong default for many language tasks. The price is that a BRNN needs the entire sequence before it can make any prediction. That is fine for text that is already written, but not for real-time speech recognition, where you cannot wait for the speaker to finish before transcribing the first word.

## Deep RNNs

For harder problems, RNN layers can be stacked. In a **deep RNN**, the activations of layer $l$ at time $t$ serve as the inputs to layer $l + 1$ at time $t$. We write $a^{[l]\langle t \rangle}$ for the activation of layer $l$ at time $t$. Each layer has its own weights, shared across time, so that for example

$$
a^{[2]\langle 3 \rangle} = g\left(W_a^{[2]} \begin{bmatrix} a^{[2]\langle 2 \rangle} \\ a^{[1]\langle 3 \rangle} \end{bmatrix} + b_a^{[2]}\right).
$$

```{figure} ../_static/img/sequence_models/deep_rnn.svg
:alt: Three recurrent layers stacked on top of each other over four time steps. Each layer passes activations to the right in time and up to the layer above. The top layer produces the outputs y hat.
:width: 90%

A deep RNN with three layers.
```

Deep RNNs are much shallower than ConvNets: three layers is already a lot, because each layer is itself unrolled over many time steps. A common variant puts a deep stack of ordinary, non-recurrent layers on top of each output instead of more recurrent layers. Deep RNNs can use GRU or LSTM units, and can be bidirectional.

**Example: GRU and LSTM Layers in PyTorch.** A batch of 32 sequences, each 20 steps long, with 300 features per step:

```python
import torch
import torch.nn as nn

x = torch.randn(32, 20, 300)

gru = nn.GRU(input_size=300, hidden_size=128, batch_first=True)
lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True)

out, h = gru(x)
print("GRU:", tuple(out.shape), tuple(h.shape))
out, (h, c) = lstm(x)
print("deep bidirectional LSTM:", tuple(out.shape), tuple(h.shape))
```

```text
GRU: (32, 20, 128) (1, 32, 128)
deep bidirectional LSTM: (32, 20, 256) (6, 32, 128)
```

The bidirectional output has $2 \times 128 = 256$ features per step, the two directions side by side, and the final hidden state has $3 \times 2 = 6$ entries, one per layer and direction.

## NLP and Word Embeddings

**Natural language processing**, or **NLP**, is the field of making computers work with human language. One idea transformed it: representing words by learned vectors.

### The Problem with One-Hot Vectors

One-hot vectors treat every word as unrelated to every other. The one-hot vectors of any two different words are orthogonal and equally far apart, so "apple" is no closer to "orange" than it is to "king". A model that has learned that "I would like a glass of orange juice" is likely gets no help with "I would like a glass of apple juice", even though the two fruits play the same role.

### Featurized Representations

Imagine instead describing every word by a list of features, such as how much it relates to gender, royalty, age, or food:

| Feature | man | woman | king | queen | apple | mango |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gender | $-1.00$ | $0.98$ | $-0.94$ | $0.96$ | $0.00$ | $0.01$ |
| Royal | $0.02$ | $0.01$ | $0.93$ | $0.95$ | $0.01$ | $0.00$ |
| Age | $0.60$ | $0.58$ | $0.70$ | $0.68$ | $0.03$ | $0.02$ |
| Food | $0.03$ | $0.01$ | $0.02$ | $0.01$ | $0.95$ | $0.97$ |

Now "apple" and "mango" have nearly identical vectors, and anything a model learns about one carries over to the other. Such a vector is called a **word embedding**, and we write the embedding of word number 456 as $e_{456}$, or simply $e_{\text{apple}}$. In practice, embeddings have a few hundred dimensions and are learned from text rather than designed by hand, so their dimensions do not have neat names like these. To look at them, embeddings are often projected down to two dimensions with an algorithm such as **t-SNE**, which shows related words gathered together.

### Using Word Embeddings

Embeddings make [transfer learning](structuring_projects.md#transfer-learning) easy for language:

1. Learn word embeddings from a very large body of text, say a billion words from the internet, or download embeddings that someone else has trained.
2. Use them as the input representation for your task, which might have only 100,000 words of labeled training data, such as named entity recognition.
3. Optionally, fine-tune the embeddings on your task's data, if you have enough of it.

A model that has never seen "durian" in its labeled data, but knows from the embeddings that "durian" is close to "mango", can still guess correctly that "a durian farmer" is a person. Embeddings are also much more compact: a few hundred dense numbers instead of a 10,000-dimensional one-hot vector.

### Properties of Word Embeddings

Embeddings capture **analogies**.[^mikolov] In the table above, the difference between "man" and "woman" is mostly in the gender feature, and so is the difference between "king" and "queen":

$$
e_{\text{man}} - e_{\text{woman}} \approx e_{\text{king}} - e_{\text{queen}}.
$$

So to answer "man is to woman as king is to what?", look for the word $w$ whose embedding is most similar to $e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}$:

$$
w = \argmax_{w} \; \text{sim}\left(e_w, \; e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}\right).
$$

The most common similarity measure is the **cosine similarity**, the cosine of the angle between two vectors:

$$
\text{sim}(u, v) = \frac{u^{\top} v}{\|u\|_2 \, \|v\|_2}.
$$

It is 1 for vectors pointing the same way, 0 for perpendicular vectors, and $-1$ for opposite ones. Another option is the negative of the squared Euclidean distance, $-\|u - v\|^2$, negated because a distance grows as vectors become less similar. The two differ in how they treat the lengths of the vectors: cosine similarity ignores length and compares only direction.

**Example: Solving an Analogy.** With the embeddings from the table, plus two more words:

```python
import torch

words = ["man", "woman", "king", "queen", "boy", "girl", "apple", "mango"]
E = torch.tensor([
    [-1.00, 0.02, 0.60, 0.03],
    [ 0.98, 0.01, 0.58, 0.01],
    [-0.94, 0.93, 0.70, 0.02],
    [ 0.96, 0.95, 0.68, 0.01],
    [-0.97, 0.01, 0.10, 0.04],
    [ 0.99, 0.02, 0.12, 0.02],
    [ 0.00, 0.01, 0.03, 0.95],
    [ 0.01, 0.00, 0.02, 0.97],
])
e = {w: E[i] for i, w in enumerate(words)}


def cosine(u, v):
    return (u @ v) / (u.norm() * v.norm())


target = e["king"] - e["man"] + e["woman"]
candidates = [w for w in words if w not in ("king", "man", "woman")]
scores = {w: round(float(cosine(e[w], target)), 3) for w in candidates}
print(max(scores, key=scores.get), scores)
```

```text
queen {'queen': 0.999, 'boy': -0.617, 'girl': 0.732, 'apple': 0.02, 'mango': 0.016}
```

"girl" comes second: it matches on gender but not on royalty.

### The Embedding Matrix

All the embeddings together form the **embedding matrix** $E$, with one column per word. With 300-dimensional embeddings and a vocabulary of 10,000 words, $E$ is $300 \times 10{,}000$. Multiplying it by the one-hot vector $o_j$ of word $j$ picks out column $j$:

$$
E \, o_j = e_j.
$$

```{figure} ../_static/img/sequence_models/embedding_matrix.svg
:alt: A wide embedding matrix E of size 300 by 10,000 times a one-hot vector o j that has a single 1 in row j gives the column j of E, the 300-dimensional embedding e j.
:width: 80%

Multiplying the embedding matrix by a one-hot vector selects one column: that word's embedding.
```

The multiplication is useful for writing down the math, but wasteful in practice, since almost all its products are with zeros. Frameworks look up the column directly instead, with a layer such as PyTorch's `nn.Embedding`.

**Example: Lookup Equals Multiplication.**

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
vocab_size, dim = 10_000, 300
embedding = nn.Embedding(vocab_size, dim)
E = embedding.weight.T

j = 6257
o_j = torch.zeros(vocab_size)
o_j[j] = 1.0

print(torch.allclose(E @ o_j, embedding(torch.tensor(j))))
```

```text
True
```

### Learning Word Embeddings with a Language Model

How are embeddings learned? One of the first successful methods trains a neural language model and treats $E$ as one of its parameters.[^bengio]

To predict the next word in "I would like a glass of orange ___", look up the embeddings of the previous four words, "a glass of orange". Concatenate them into one vector, 1200 numbers for four 300-dimensional embeddings, feed it to a hidden layer, and then to a softmax over the vocabulary that predicts the next word. Train the whole network, including $E$, with gradient descent on a large corpus.

It works because words that appear in similar contexts get pushed toward similar embeddings. "orange" and "apple" are both followed by "juice", so making that prediction well is easiest if their embeddings are close.

The context does not have to be the previous four words. To learn embeddings, rather than to build a language model, other contexts work well too:

- the four words before the target and the four words after it,
- just the one previous word,
- or a single nearby word, chosen at random from a window around the target. This simple context is the basis of the skip-gram model.

## Word2Vec (CBOW and Skip-Gram)

**Word2Vec** is a family of simpler and much faster ways to learn embeddings.[^word2vec] Its two models are mirror images.

- **Skip-gram** takes one word, the **context** $c$, and predicts a word chosen at random from a window around it, the **target** $t$.
- **Continuous bag of words**, or **CBOW**, takes the surrounding words and predicts the word in the middle.

For skip-gram, take the sentence "I would like a glass of orange juice with my breakfast." With "orange" as the context and a window of a few words on either side, the training pairs might be

| Context | Target |
| --- | --- |
| orange | juice |
| orange | glass |
| orange | my |

Predicting a random nearby word from "orange" is a hard problem, and the model will never be good at it. That is fine: the goal is not the predictions but the embeddings learned along the way.

### The Skip-Gram Model

The model looks up the context word's embedding and passes it to a softmax over the vocabulary:

$$
o_c \;\to\; E \;\to\; e_c \;\to\; \text{softmax} \;\to\; \hat{y}, \qquad p(t \mid c) = \frac{e^{\theta_t^{\top} e_c}}{\sum_{j=1}^{V} e^{\theta_j^{\top} e_c}},
$$

where $V$ is the size of the vocabulary and $\theta_t$ is the softmax parameter vector for output word $t$. The loss is the usual softmax loss, $\L(\hat{y}, y) = -\sum_{i=1}^{V} y_i \log \hat{y}_i$, with $y$ the one-hot vector of the target word.

The problem is speed. Every step computes the sum in the denominator over the whole vocabulary, 10,000 terms or far more.

One fix is the **hierarchical softmax**. It arranges the vocabulary as a binary tree, and predicts a word by a sequence of yes-or-no decisions, left or right at each node, so the cost grows like $\log V$ instead of $V$. The tree is not balanced: common words sit near the top, where they are reached in a few decisions, and rare words deeper down. The next section describes a simpler fix that is used more often.

### Choosing the Context Words

If context words were chosen uniformly at random from the corpus, most of them would be "the", "of", "a", "and", and other very frequent words, and the model would spend most of its effort on them. In practice, the sampling is adjusted so that less frequent words, such as "orange" or "durian", are chosen more often than their raw frequency would suggest.

## Negative Sampling (Skip-Gram Revised)

**Negative sampling** learns embeddings as well as skip-gram, but much faster.[^negsampling] It replaces the hard problem, predicting the target word out of the whole vocabulary, with an easy one: given a pair of words, decide whether they are neighbors.

To build the training data, pick a context word and a real nearby target word, and label the pair 1. Then pick $k$ words at random from the vocabulary and pair each with the same context word, labeled 0:

| Context | Word | Neighbor? |
| --- | --- | ---: |
| orange | juice | 1 |
| orange | king | 0 |
| orange | book | 0 |
| orange | the | 0 |
| orange | of | 0 |

It does not matter that a random word sometimes happens to be a real neighbor, like "of" in the sentence above. A good choice is $k$ between 5 and 20 for small datasets and between 2 and 5 for large ones.

The model is now a logistic regression on the pair:

$$
P(y = 1 \mid c, t) = \sigma\left(\theta_t^{\top} e_c\right).
$$

Think of it as $V$ separate binary classifiers, one per word in the vocabulary, each answering "is this word a neighbor of $c$?". But each step only trains $k + 1$ of them, the true target and the $k$ negatives, instead of computing a softmax over all $V$. That is where the speed comes from.

### Choosing the Negative Words

How should the negative words be sampled? Sampling them by their frequency in the corpus would give mostly "the", "of", and "and". Sampling them uniformly, each with probability $\frac{1}{V}$, would give mostly rare words, unrepresentative of real text. The Word2Vec authors found that something in between works best: sampling word $w_i$ with probability proportional to its frequency raised to the power $\frac{3}{4}$,

$$
P(w_i) = \frac{f(w_i)^{3/4}}{\sum_{j=1}^{V} f(w_j)^{3/4}},
$$

where $f(w_i)$ is how often $w_i$ appears in the corpus.

```{figure} ../_static/img/sequence_models/negative_sampling.svg
:alt: Bar chart on a log scale of the probability of sampling seven words as negatives under three schemes. By frequency, the, of, and and dominate and durian is almost never chosen. Uniformly, every word has the same probability. With frequency to the power three quarters, the common words still lead, but rarer words like glass, orange, juice, and durian get several times more probability than by raw frequency.
:width: 90%

Three ways to sample negative words. Raising the frequencies to the power $3/4$ keeps common words common but gives rarer words a larger share.
```

**Example: The Three Sampling Distributions.**

```python
import torch

words = ["the", "of", "and", "glass", "orange", "juice", "durian"]
counts = torch.tensor([60000.0, 35000.0, 28000.0, 900.0, 400.0, 350.0, 5.0])

for name, weights in [("by frequency", counts), ("uniform", torch.ones_like(counts)), ("frequency^(3/4)", counts ** 0.75)]:
    p = weights / weights.sum()
    print(f"{name:<16}", " ".join(f"{w}={v:.3f}" for w, v in zip(words, p.tolist())))
```

```text
by frequency     the=0.481 of=0.281 and=0.225 glass=0.007 orange=0.003 juice=0.003 durian=0.000
uniform          the=0.143 of=0.143 and=0.143 glass=0.143 orange=0.143 juice=0.143 durian=0.143
frequency^(3/4)  the=0.431 of=0.288 and=0.243 glass=0.018 orange=0.010 juice=0.009 durian=0.000
```

**Example: One Step of Negative Sampling.** The loss rewards a high score for the real target and low scores for the negatives. Only the context word's row of $E$ receives a gradient:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
vocab_size, dim, k = 10_000, 300, 4
E = nn.Embedding(vocab_size, dim)
theta = nn.Embedding(vocab_size, dim)

context = torch.tensor([6257])
target = torch.tensor([4834])
negatives = torch.randint(0, vocab_size, (1, k))

e_c = E(context)
positive_score = (theta(target) * e_c).sum(dim=1)
negative_scores = (theta(negatives) * e_c.unsqueeze(1)).sum(dim=2)
loss = -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_scores).sum(dim=1)).mean()
loss.backward()
print(f"loss {loss:.3f}; rows of E updated: {int((E.weight.grad.abs().sum(dim=1) > 0).sum())}")
```

```text
loss 10.072; rows of E updated: 1
```

### What the Embeddings Learn

To see negative sampling at work, we trained 16-dimensional embeddings on a small made-up corpus of 30,000 sentences built from templates such as "a glass of mango juice please", "the lion slept near the door", and "we spent three days in Cairo". The model was never told which words belong together; it only saw which words appear near each other.

```{figure} ../_static/img/sequence_models/word_embeddings.svg
:alt: A two-dimensional projection of learned word embeddings. The words fall into five tight, well-separated clusters: fruits, drinks, animals, cities, and numbers.
:width: 80%

Embeddings learned by negative sampling on a toy corpus, projected to two dimensions. Words used in similar contexts end up together: fruits, drinks, animals, cities, and numbers each form their own cluster.
```

## GloVe (Global Vectors for Word Representation)

**GloVe** learns embeddings from a different starting point: a table of how often words appear together.[^glove] Let $X_{ij}$ be the number of times word $i$ appears in the context of word $j$, for example within a window of 10 words. If the context is a symmetric window, then $X_{ij} = X_{ji}$.

GloVe learns vectors whose inner products predict the logarithm of these counts, by minimizing

$$
\sum_{i=1}^{V} \sum_{j=1}^{V} f\left(X_{ij}\right) \left(\theta_i^{\top} e_j + b_i + b_j' - \log X_{ij}\right)^2,
$$

where $\theta_i$ and $e_j$ are vectors to learn, $b_i$ and $b_j'$ are bias terms, and $f$ is a weighting function. The weighting does two jobs.

- $f(0) = 0$, so pairs that never appear together are skipped, and $\log 0$ is never computed.
- It limits the influence of very frequent pairs, such as "of the", while still giving reasonable weight to rare words, such as "durian". The original paper uses $f(x) = (x / x_{\max})^{3/4}$ for counts below $x_{\max} = 100$, and 1 above.

In this objective, $\theta_i$ and $e_j$ play symmetric roles, so both are initialized randomly and learned, and the final embedding of a word averages the two:

$$
e_w^{\text{final}} = \frac{e_w + \theta_w}{2}.
$$

GloVe is used less than Word2Vec, but its simplicity has made it popular, and pre-trained GloVe embeddings are widely available.

**Example: The GloVe Weighting Function.**

```python
def glove_weight(x, x_max=100, alpha=0.75):
    return (x / x_max) ** alpha if x < x_max else 1.0


for x in [0, 1, 10, 50, 100, 10_000]:
    print(x, round(glove_weight(x), 3))
```

```text
0 0.0
1 0.032
10 0.178
50 0.595
100 1.0
10000 1.0
```

## Interpretability of Word Embeddings

The featurized table at the start of this part had dimensions with clear meanings: gender, royalty, age, food. Learned embeddings do not. Nothing in Skip-gram or GloVe forces each dimension to mean one thing, or the dimensions to be independent of each other.

The reason is that the objectives only depend on inner products such as $\theta_i^{\top} e_j$. For any invertible matrix $A$,

$$
\left(A \theta_i\right)^{\top} \left(A^{-\top} e_j\right) = \theta_i^{\top} A^{\top} A^{-\top} e_j = \theta_i^{\top} e_j,
$$

so transforming all the vectors by $A$ and $A^{-\top}$ gives exactly the same objective. Training can land on any of these transformed solutions, and in most of them a meaningful direction such as gender is spread across many dimensions at once. The analogy property still holds, because it depends on differences between vectors rather than on individual dimensions.

## Sentiment Classification

**Sentiment classification** reads a piece of text and decides how positive or negative it is, for example predicting the star rating of a restaurant review. Labeled data is often scarce, so word embeddings learned from a large corpus make a big difference.

```{figure} ../_static/img/sequence_models/sentiment_models.svg
:alt: Top: each word of the review the service was awful is looked up as an embedding, the embeddings are averaged, and a softmax predicts the rating. Bottom: the embeddings feed an RNN one word at a time, and a softmax on the last activation predicts the rating.
:width: 100%

Two sentiment classifiers. The averaging model ignores word order. The RNN reads the words in order.
```

The simplest model looks up the embedding of every word in the review, averages them into a single vector of the same size, and passes it to a softmax with one output per class, such as 5 for a five-star scale. Averaging handles reviews of any length, and thanks to the embeddings, the model knows that "awful" and "dreadful" mean similar things even if only one of them appears in the labeled data.

Its weakness is that it ignores word order. "The ads promised good food and good service, but we got neither" contains "good" twice, and the average leans positive, though the review is clearly negative. An RNN fixes this: feed the embeddings into an RNN one word at a time, a many-to-one architecture, and make the prediction from the last activation. The RNN can learn that "but we got neither" reverses what came before.

**Example: The Averaging Model.** PyTorch's `EmbeddingBag` looks up and averages in one step:

```python
import torch
import torch.nn as nn

vocab_size, dim, n_classes = 10_000, 50, 5
model = nn.Sequential(
    nn.EmbeddingBag(vocab_size, dim, mode="mean"),
    nn.Linear(dim, n_classes),
)
review = torch.tensor([[12, 845, 3, 9021, 77]])
print(torch.softmax(model(review), dim=1).shape)
```

```text
torch.Size([1, 5])
```

## Debiasing Word Embeddings

Word embeddings learn from text written by people, and they absorb the biases in it: gender, ethnicity, age, and others.[^bolukbasi] Embeddings that correctly solve "man is to woman as king is to queen" have also been found to answer "man is to computer programmer as woman is to homemaker", or "father is to doctor as mother is to nurse". When such embeddings are used to screen job applications or rank search results, these biases can do real harm.

Some words are **definitional** for a bias: "grandmother" and "grandfather", "girl" and "boy", "she" and "he" differ in gender by definition. Others, such as "doctor" or "babysitter", should be neutral. One method reduces the bias in three steps.

1. **Identify the bias direction.** For gender, take the differences between definitional pairs, $e_{\text{he}} - e_{\text{she}}$, $e_{\text{male}} - e_{\text{female}}$, $e_{\text{grandfather}} - e_{\text{grandmother}}$, and so on, and average them. In practice, a technique called **singular value decomposition** is used instead of a simple average, because the bias can span more than one direction. Every direction perpendicular to it is a non-bias direction.
2. **Neutralize.** For every word that is not definitional, remove its component along the bias direction, projecting it onto the non-bias directions.
3. **Equalize pairs.** Move each definitional pair, such as "grandmother" and "grandfather", so that the two words differ only along the bias direction and sit at the same distance from the non-bias axis. Then every neutral word, such as "babysitter", is exactly as close to one as to the other.

```{figure} ../_static/img/sequence_models/debias.svg
:alt: Two plots with a horizontal bias direction and a vertical non-bias direction. Left: babysitter, which starts off to one side, is projected onto the non-bias axis. Right: grandmother and grandfather, shown in gray before, are moved to symmetric positions on either side of the non-bias axis, so that dashed lines to babysitter have equal length.
:width: 100%

Neutralizing removes a neutral word's component along the bias direction (left). Equalizing makes a definitional pair symmetric, so neutral words are equally close to both (right). Gray dots show positions before.
```

A remaining challenge is deciding which words are definitional. Most words are not, so the list can be kept short, and a classifier can be trained to recognize the definitional ones.

**Example: Neutralize and Equalize.** On the two-dimensional vectors of the figure, with the bias direction along the first axis:

```python
import torch


def neutralize(e, g):
    return e - (e @ g) / (g @ g) * g


def equalize(e_1, e_2, g):
    mu = (e_1 + e_2) / 2
    mu_orth = neutralize(mu, g)
    g_hat = g / g.norm()
    scale = (1 - mu_orth.norm() ** 2).sqrt()
    e_1_bias = (e_1 @ g_hat) * g_hat
    e_2_bias = (e_2 @ g_hat) * g_hat
    mu_bias = (mu @ g_hat) * g_hat
    new_1 = scale * (e_1_bias - mu_bias) / (e_1_bias - mu_bias).norm() + mu_orth
    new_2 = scale * (e_2_bias - mu_bias) / (e_2_bias - mu_bias).norm() + mu_orth
    return new_1, new_2


g = torch.tensor([1.0, 0.0])
babysitter = torch.tensor([0.45, 0.80])
print("babysitter after neutralize:", neutralize(babysitter, g).tolist())

grandmother, grandfather = torch.tensor([0.75, 0.40]), torch.tensor([-0.55, 0.62])
gm, gf = equalize(grandmother, grandfather, g)
nb = neutralize(babysitter, g)
print("distances to babysitter:", round(float((gm - nb).norm()), 4), round(float((gf - nb).norm()), 4))
```

```text
babysitter after neutralize: [0.0, 0.800000011920929]
distances to babysitter: 0.9077 0.9077
```

## Basic Sequence Models

### Sequence-to-Sequence Models

Machine translation maps a sentence in one language to a sentence in another, possibly of a different length. A **sequence-to-sequence** model does this with two RNNs.[^seq2seq] An **encoder** reads the source sentence and summarizes it in its final activation. A **decoder** starts from that summary and writes the translation one word at a time, feeding each word back in as the next input, until it outputs `<EOS>`.

```{figure} ../_static/img/sequence_models/seq2seq.svg
:alt: Top: an encoder RNN reads the French sentence le chat dort and passes its final activation to a decoder RNN that writes the cat sleeps end of sentence. Bottom: a ConvNet encodes an image and a decoder RNN writes the caption a cat on grass.
:width: 100%

Two encoder-decoder models. The encoder can be an RNN for a sentence or a ConvNet for an image.
```

The same pattern works for **image captioning**: a ConvNet, pre-trained on image recognition with its softmax removed, encodes the image as a vector, and a decoder RNN writes a sentence describing it.

### Translation as a Conditional Language Model

The decoder looks just like the language model from earlier in this part, except that instead of starting from $a^{\langle 0 \rangle} = 0$, it starts from the encoder's summary of the source sentence. So a translation model is a **conditional language model**: it gives the probability of an English sentence conditioned on, for example, a French sentence $x$:

$$
P\left(y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle} \mid x^{\langle 1 \rangle}, \dots, x^{\langle T_x \rangle}\right).
$$

For translation, we do not want to sample a sentence at random, which would give a different and sometimes poor translation each time. We want the single most likely translation:

$$
\argmax_{y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle}} P\left(y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle} \mid x\right).
$$

### Why Not Greedy Search?

The simplest approach, **greedy search**, picks the single most likely first word, then the most likely second word given the first, and so on. But the sequence of individually most likely words is not necessarily the most likely sentence. A common word early on can lead to a sentence that is worse overall: "Karim is going to visit Beirut" may beat "Karim will visit Beirut" at the third word, simply because "going" is a common word after "is", even though the second sentence is the better translation.

Checking every possible sentence is impossible: with 10,000 words and sentences of 10 words, there are $10{,}000^{10}$ of them. So we use an approximate search algorithm.

## Beam Search Algorithm

**Beam search** keeps track of the $B$ most likely partial sentences at each step instead of just one, where the **beam width** $B$ is a hyperparameter.

1. Run the encoder on the source sentence $x$. For the first word, keep the $B$ words with the highest $P\left(y^{\langle 1 \rangle} \mid x\right)$.
2. For each of the $B$ first words, run the decoder one more step to get the probability of every second word. Out of all $B \times V$ two-word combinations, keep the $B$ with the highest probability
$$
P\left(y^{\langle 1 \rangle}, y^{\langle 2 \rangle} \mid x\right) = P\left(y^{\langle 1 \rangle} \mid x\right) P\left(y^{\langle 2 \rangle} \mid x, y^{\langle 1 \rangle}\right).
$$
3. Extend each of the $B$ two-word sequences by one more word, and again keep the $B$ best three-word sequences.
4. Repeat. When a sequence ends with `<EOS>`, set it aside as a finished candidate. At the end, output the most likely finished candidate.

At every step, beam search runs $B$ copies of the decoder, one per partial sentence. With $B = 1$, it is greedy search.

```{figure} ../_static/img/sequence_models/beam_search.svg
:alt: A search tree. From the start, the first word is the with probability 0.5, a with 0.4, or end of sentence with 0.1. After the, the sentence can continue with cat, dog, or end. After a, it continues with cat with probability 0.9. The greedy path, start then the then end, has probability 0.20. Beam search with width 2 keeps both the and a and finds a cat end, with probability 0.36.
:width: 90%

A toy example where greedy search fails. Greedy search commits to "the", the likelier first word, and can do no better than 0.20. Beam search with $B = 2$ keeps "a" as well and finds "a cat", with probability 0.36.
```

**Example: Greedy Search Versus Beam Search.** The toy model of the figure, as a table of next-word probabilities:

```python
import math

model = {
    (): {"the": 0.5, "a": 0.4, "<EOS>": 0.1},
    ("the",): {"cat": 0.3, "dog": 0.3, "<EOS>": 0.4},
    ("a",): {"cat": 0.9, "<EOS>": 0.1},
    ("the", "cat"): {"<EOS>": 1.0},
    ("the", "dog"): {"<EOS>": 1.0},
    ("a", "cat"): {"<EOS>": 1.0},
}


def greedy():
    seq = ()
    while not seq or seq[-1] != "<EOS>":
        probs = model[seq]
        seq += (max(probs, key=probs.get),)
    return seq


def beam_search(B):
    beams = [((), 0.0)]
    finished = []
    while beams:
        candidates = [(seq + (w,), score + math.log(p)) for seq, score in beams for w, p in model[seq].items()]
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for seq, score in candidates[:B]:
            (finished if seq[-1] == "<EOS>" else beams).append((seq, score))
    return max(finished, key=lambda c: c[1])


g = greedy()
print("greedy:", " ".join(g), round(math.exp(sum(math.log(model[g[:i]][g[i]]) for i in range(len(g)))), 3))
for B in (1, 2):
    seq, score = beam_search(B)
    print(f"beam B={B}:", " ".join(seq), round(math.exp(score), 3))
```

```text
greedy: the <EOS> 0.2
beam B=1: the <EOS> 0.2
beam B=2: a cat <EOS> 0.36
```

### Refinements to Beam Search

The objective is a product of many probabilities,

$$
\argmax_{y} \prod_{t=1}^{T_y} P\left(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle}\right),
$$

and a product of many numbers smaller than 1 quickly becomes too small for a computer to store. Taking the logarithm gives a sum instead, with the same maximum and none of the numerical trouble:

$$
\argmax_{y} \sum_{t=1}^{T_y} \log P\left(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle}\right).
$$

Both forms have a bias toward short sentences. Every extra word multiplies in another probability below 1, or adds another negative logarithm, so longer sentences always score lower. **Length normalization** fixes this by dividing by the length of the sentence, raised to a power $\alpha$:

$$
\argmax_{y} \frac{1}{T_y^{\alpha}} \sum_{t=1}^{T_y} \log P\left(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle}\right),
$$

where $\alpha = 1$ is a full normalization, $\alpha = 0$ none at all, and a value in between, such as $\alpha = 0.7$, often works best. In practice, run beam search, keep the finished candidates of all lengths, and pick the one with the highest normalized score.

### Choosing the Beam Width

A larger $B$ considers more possibilities and usually finds better sentences, but it is slower and uses more memory. A smaller $B$ is faster but finds worse results. Production systems often use $B$ around 10. Research systems trying to squeeze out the best possible score sometimes use 1000 or 3000. The gains shrink as $B$ grows: going from 1 to 10 often helps a lot, while going from 1000 to 3000 helps little.

Unlike exact search algorithms such as breadth-first search or depth-first search, beam search is not guaranteed to find the most likely sentence. That is the price of its speed.

### Error Analysis on Beam Search

When a translation is wrong, there are two suspects: the RNN model, which gives the probabilities, and beam search, which looks for the sentence that maximizes them. [Error analysis](structuring_projects.md#error-analysis) can tell which is to blame.

Take an example where the model's translation $\hat{y}$ is worse than a good human translation $y^*$. Use the RNN to compute both $P(y^* \mid x)$ and $P(\hat{y} \mid x)$, with length normalization if you use it.

- **If $P(y^* \mid x) > P(\hat{y} \mid x)$**, the model rates the human translation higher, but beam search returned $\hat{y}$. Beam search is at fault: it failed to find the better sentence. A larger $B$ may help.
- **If $P(y^* \mid x) \leq P(\hat{y} \mid x)$**, the model itself rates the worse translation higher. The RNN is at fault, and more data, regularization, or a different architecture may help.

Repeat this for many mistranslated examples and count how often each case occurs. That tells you which part deserves your effort.

## BLEU Score

A sentence can often be translated correctly in several ways. How do you score a machine translation when there is no single right answer? The **BLEU** score, for "bilingual evaluation understudy", compares the machine translation with one or more human **reference** translations, and measures how many of its words and phrases appear in them.[^bleu] It is an "understudy" in the theater sense: a stand-in that can take the place of human judges, giving a score quickly and cheaply.

### Modified Precision

An **$n$-gram** is a sequence of $n$ consecutive words: a unigram is a single word, a bigram is two, and so on. The naive measure would count what fraction of the machine translation's $n$-grams appear in a reference. But then a translation like "the the the the" would score perfectly against any reference containing "the". BLEU **clips** each count: an $n$-gram is credited at most as many times as it appears in the single reference that contains it most often. The **modified precision** for $n$-grams is

$$
p_n = \frac{\sum_{n\text{-gram} \in \hat{y}} \text{Count}_{\text{clip}}(n\text{-gram})}{\sum_{n\text{-gram} \in \hat{y}} \text{Count}(n\text{-gram})},
$$

where $\text{Count}$ is how often the $n$-gram appears in the machine translation $\hat{y}$, and $\text{Count}_{\text{clip}}$ is that count, capped at the most times it appears in any one reference.

Take two references and a machine translation:

- Reference 1: "the children are playing in the garden"
- Reference 2: "the kids play in the garden"
- Machine translation: "the the children play in the garden"

| Bigram in the machine translation | Count | $\text{Count}_{\text{clip}}$ |
| --- | ---: | ---: |
| the the | 1 | 0 |
| the children | 1 | 1 |
| children play | 1 | 0 |
| play in | 1 | 1 |
| in the | 1 | 1 |
| the garden | 1 | 1 |

So $p_2 = \frac{4}{6} \approx 0.67$. For unigrams, "the" appears 3 times in the translation but at most twice in either reference, so it is clipped to 2, and $p_1 = \frac{6}{7} \approx 0.86$.

### The Combined Score

BLEU combines the modified precisions for $n = 1$ to 4 by taking their geometric mean, and multiplies by a **brevity penalty** $\text{BP}$:

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\frac{1}{4} \sum_{n=1}^{4} \log p_n\right).
$$

Precision alone rewards translations that are too short: a one-word translation whose word appears in a reference has perfect precision. The brevity penalty fixes this. With $c$ the length of the machine translation and $r$ the length of the reference closest in length,

$$
\text{BP} = \begin{cases} 1 & \text{if } c > r, \\ e^{1 - r/c} & \text{if } c \leq r. \end{cases}
$$

Translations longer than the reference are not penalized, since their extra words already lower the precision. Shorter ones are penalized exponentially.

BLEU transformed machine translation research by giving it a single, cheap, automatic metric, and it is also used for other tasks that generate text, such as image captioning. It is a rough measure: a translation can score poorly by choosing good words that the references happen not to use.

**Example: Computing BLEU from Scratch.**

```python
import math
from collections import Counter


def ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def modified_precision(candidate, references, n):
    counts = ngrams(candidate, n)
    max_ref = Counter()
    for ref in references:
        for gram, c in ngrams(ref, n).items():
            max_ref[gram] = max(max_ref[gram], c)
    clipped = sum(min(c, max_ref[gram]) for gram, c in counts.items())
    return clipped / max(1, sum(counts.values()))


def bleu(candidate, references, max_n=4):
    precisions = [modified_precision(candidate, references, n) for n in range(1, max_n + 1)]
    if min(precisions) == 0:
        return 0.0, precisions
    c = len(candidate)
    r = min((abs(len(ref) - c), len(ref)) for ref in references)[1]
    bp = 1.0 if c > r else math.exp(1 - r / c)
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n), precisions


refs = ["the children are playing in the garden".split(), "the kids play in the garden".split()]
mt = "the the children play in the garden".split()
score, p = bleu(mt, refs)
print("p1..p4:", [round(x, 3) for x in p], "BLEU:", round(score, 3))
```

```text
p1..p4: [0.857, 0.667, 0.4, 0.25] BLEU: 0.489
```

The translation has 7 words and Reference 1 also has 7, so there is no brevity penalty.

## Attention Model

The encoder-decoder model has a bottleneck: it must squeeze the entire source sentence into one vector before the decoder writes a single word. That works for short sentences, but for sentences longer than about 30 or 40 words, translation quality, measured by BLEU, drops sharply.

A human translator does not work that way. They do not memorize a whole long sentence and then write its translation from memory. They read a phrase, translate it, read the next phrase, translate that, and so on, looking back at the relevant part of the source as they go. The **attention model** gives the decoder the same ability: at every step, it looks back at the whole source sentence and chooses which parts to focus on.[^bahdanau] With attention, translation quality stays high even for long sentences.

### How Attention Works

The encoder is usually a bidirectional RNN. We index its time steps by $t'$ and the decoder's by $t$. At each encoder step, the forward and backward activations are stacked into one vector of features,

$$
a^{\langle t' \rangle} = \left(\overrightarrow{a}^{\langle t' \rangle}, \overleftarrow{a}^{\langle t' \rangle}\right).
$$

To produce output word $t$, the decoder computes an **attention weight** $\alpha^{\langle t, t' \rangle}$ for every input position $t'$: how much attention to pay to input word $t'$ when generating output word $t$. The weights are positive and add up to 1 over $t'$. The decoder's input at step $t$ is the **context** $c^{\langle t \rangle}$, the average of the encoder's features weighted by attention:

$$
c^{\langle t \rangle} = \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}, \qquad \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} = 1.
$$

```{figure} ../_static/img/sequence_models/attention_model.svg
:alt: A bidirectional RNN encoder reads x 1 to x 4 and produces activations a 1 to a 4. Attention weights alpha t comma t prime combine them into a context c t, which feeds the decoder at step t together with the previous decoder state s t minus 1. The decoder outputs y hat t.
:width: 100%

The attention model. At each decoder step, the attention weights decide how much of each input position flows into the context.
```

### Computing the Attention Weights

The weights come from a softmax over **scores** $e^{\langle t, t' \rangle}$, which guarantees that they are positive and add up to 1:

$$
\alpha^{\langle t, t' \rangle} = \frac{\exp\left(e^{\langle t, t' \rangle}\right)}{\sum_{t''=1}^{T_x} \exp\left(e^{\langle t, t'' \rangle}\right)}.
$$

How much to attend to input $t'$ should depend on what the decoder has written so far, summarized in its previous state $s^{\langle t-1 \rangle}$, and on the input features $a^{\langle t' \rangle}$. We do not know the right function of these two, so we let a small neural network learn it: one hidden layer, with $s^{\langle t-1 \rangle}$ and $a^{\langle t' \rangle}$ as inputs and $e^{\langle t, t' \rangle}$ as output. It is trained together with the rest of the model by backpropagation.

To see attention at work, we trained a small attention model to reverse sequences of 8 digits. It learned the task perfectly, and its attention weights show how:

```{figure} ../_static/img/sequence_models/attention_heatmap.svg
:alt: A heatmap of attention weights with the input digits 3 1 4 1 5 9 2 6 along the bottom and the output digits 6 2 9 5 1 4 1 3 along the side. The large weights form an anti-diagonal: each output digit attends mostly to the input digit it copies.
:width: 65%

Real attention weights from a model trained to reverse a sequence. To write each output digit, the model attends to the input position it copies from, which forms an anti-diagonal.
```

Nobody told the model which input to look at; it learned the alignment from the input-output pairs alone. In translation, the same kind of plot shows which source words the model looks at to produce each target word, and it often reveals sensible alignments between the two languages.

The downside of attention is cost. It computes $T_x \times T_y$ attention weights, so its cost grows with the product of the two lengths, quadratically for sentences of similar length. For sentences of normal length this is acceptable. The same idea has been applied far beyond translation, for example to image captioning, where the decoder learns to attend to different parts of the image as it writes each word.[^showattend]

**Example: One Step of Attention.** Six encoder positions, a decoder state, and a small scoring network:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
T_x, n_a, n_s = 6, 8, 10
a = torch.randn(T_x, 2 * n_a)
s_prev = torch.randn(n_s)

score = nn.Sequential(nn.Linear(n_s + 2 * n_a, 10), nn.Tanh(), nn.Linear(10, 1))
e = score(torch.cat([s_prev.expand(T_x, n_s), a], dim=1)).squeeze(1)
alpha = torch.softmax(e, dim=0)
context = alpha @ a

print("alpha:", [round(v, 3) for v in alpha.tolist()], "sum:", round(float(alpha.sum()), 3))
print("context:", tuple(context.shape))
```

```text
alpha: [0.215, 0.142, 0.137, 0.177, 0.157, 0.173] sum: 1.0
context: (16,)
```

Before training, the weights are nearly uniform. Training sharpens them into a focus on the relevant positions, as in the heatmap.

## Speech Recognition

In **speech recognition**, the input $x$ is an audio clip and the output $y$ is its transcript. Audio is a sequence of tiny changes in air pressure, thousands per second. It is usually preprocessed into a **spectrogram**, which shows how much energy each frequency has at each moment, and the network reads that sequence of frequency features.

Speech recognition used to rely on hand-designed representations such as **phonemes**, the basic units of sound. With enough data, end-to-end deep learning no longer needs them. Commercial systems are trained on 10,000 to 100,000 hours of audio, or more. An attention model can be used, with the audio as input and the characters of the transcript as output.

### CTC Cost

Another approach uses the **connectionist temporal classification**, or **CTC**, cost.[^ctc] The difficulty in speech is that the input is much longer than the output. With features computed 100 times per second, a 10-second clip gives 1000 input time steps, but its transcript might have only 50 characters.

CTC lets an RNN, often a bidirectional one, output one character per input time step, including a special **blank** character, written here as an underscore, which is different from a space. The output is then collapsed by a simple rule: merge repeated characters that are not separated by a blank, then remove the blanks.

```{figure} ../_static/img/sequence_models/ctc.svg
:alt: A strip of 19 frame outputs, h h h blank e e e blank l l blank l l l blank blank o o o, with the first character of each run outlined. Collapsing gives the word hello.
:width: 100%

CTC collapsing. Repeated characters merge, and blanks separate genuine repeats, such as the two l's in "hello".
```

The blank is what makes double letters possible: "ll" in the output needs a blank between the two l's, or they would merge into one. During training, the CTC cost adds up the probabilities of all the frame-by-frame outputs that collapse to the correct transcript, so the network never needs to be told which frames belong to which character.

**Example: Collapsing and the CTC Loss.** PyTorch's `CTCLoss` sums over all valid alignments for you:

```python
import torch
import torch.nn as nn


def ctc_collapse(frames, blank="_"):
    out = []
    prev = None
    for ch in frames:
        if ch != prev and ch != blank:
            out.append(ch)
        prev = ch
    return "".join(out)


print(ctc_collapse("hhh_eee_ll_lll__ooo"))

torch.manual_seed(0)
T, C, batch = 50, 28, 1
log_probs = torch.randn(T, batch, C).log_softmax(dim=2)
target = torch.tensor([[8, 5, 12, 12, 15]])
loss = nn.CTCLoss(blank=0)(log_probs, target, torch.tensor([T]), torch.tensor([5]))
print(f"CTC loss {loss:.3f}")
```

```text
hello
CTC loss 31.821
```

### Trigger Word Detection

A **trigger word** system listens continuously for a single word or phrase that wakes up a device, the way smart speakers and phones listen for their name. It is a much simpler problem than full speech recognition.

One approach is an RNN that reads the spectrogram and outputs a label at every time step: 0 most of the time, and 1 right after the trigger word has been said.

```{figure} ../_static/img/sequence_models/trigger_word.svg
:alt: A synthetic spectrogram with three bursts of sound, labeled another word, trigger word, and trigger word. Below it, the label is 0 everywhere except for a short run of 1s just after each trigger word ends.
:width: 100%

Training labels for trigger word detection. The target is 1 for a short stretch just after each trigger word ends, and 0 everywhere else, including during other words.
```

This creates a very imbalanced training set, with far more 0s than 1s. A model that always outputs 0 would be over 90% accurate and completely useless, so accuracy is the wrong metric; [precision, recall, and the $F_1$ score](structuring_projects.md#single-number-evaluation-metric) are better choices. A common trick is to output a run of several 1s after each trigger word instead of a single one, which makes the labels a little less imbalanced, though it does not solve the problem entirely.

## Recap

In this part, you learned how recurrent neural networks process sequences by reusing the same weights at every time step, and how they are trained by backpropagation through time. You saw why basic RNNs forget over long stretches and how the GRU and the LSTM fix this with gates that protect a memory cell. You met bidirectional and deep RNNs. You learned how word embeddings capture meaning in vectors, how Word2Vec, negative sampling, and GloVe learn them from raw text, and how to use them for tasks like sentiment classification, along with the biases they can carry. You saw how encoder-decoder models translate, how beam search finds good translations and BLEU scores them, and how attention removes the bottleneck of a single summary vector. Finally, you saw how these ideas come together in speech recognition. This completes the Introduction to Deep Learning.

[^sefidian]: The topics and their order follow Amir Masoud Sefidian, [Coursera Deep Learning Specialization Notes: Sequence Models](https://www.sefidian.com/wp-content/uploads/2023/02/Sequence-Models.pdf), 2023, which summarizes the fifth course of Andrew Ng's Deep Learning Specialization.

[^karpathy]: Karpathy, [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), 2015.

[^gru]: Cho, van Merriënboer, Bahdanau, and Bengio, [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259), 2014; and Chung, Gulcehre, Cho, and Bengio, [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555), 2014.

[^lstm]: Hochreiter and Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), 1997.

[^mikolov]: Mikolov, Yih, and Zweig, [Linguistic Regularities in Continuous Space Word Representations](https://aclanthology.org/N13-1090/), 2013.

[^bengio]: Bengio, Ducharme, Vincent, and Jauvin, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html), 2003.

[^word2vec]: Mikolov, Chen, Corrado, and Dean, [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), 2013.

[^negsampling]: Mikolov, Sutskever, Chen, Corrado, and Dean, [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546), 2013.

[^glove]: Pennington, Socher, and Manning, [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/), 2014.

[^bolukbasi]: Bolukbasi, Chang, Zou, Saligrama, and Kalai, [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520), 2016.

[^seq2seq]: Sutskever, Vinyals, and Le, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), 2014; and Cho and others, [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), 2014.

[^bleu]: Papineni, Roukos, Ward, and Zhu, [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/), 2002.

[^bahdanau]: Bahdanau, Cho, and Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2014.

[^showattend]: Xu and others, [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), 2015.

[^ctc]: Graves, Fernández, Gomez, and Schmidhuber, [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf), 2006.
