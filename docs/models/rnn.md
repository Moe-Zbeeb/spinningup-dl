# Recurrent Neural Network

```{contents} Table of Contents
:depth: 2
```

## Background

A **recurrent neural network** reads a sequence one element at a time, carrying an activation from each step to the next, as described in [Part 5](../intro/sequence_models.md#recurrent-neural-networks-rnn). The companion code trains it as a **character-level language model**: given the characters of a text so far, predict the next one. A model that does this well has learned spelling, punctuation, and some grammar, and it can [generate new text](../intro/sequence_models.md#sampling-novel-sequences) one character at a time.

The same code supports three kinds of recurrent unit, so that you can compare them directly:

- the plain **RNN**, with a tanh activation,
- the **GRU**, with an update gate and a reset gate,
- and the **LSTM**, with a separate memory cell and update, forget, and output gates.

### Quick Facts

- Input: a sequence of character indices. Output: a probability distribution over the next character at every position.
- It can read sequences of any length, and its memory is carried in its activations.
- GRUs and LSTMs learn [long-range dependencies](../intro/sequence_models.md#vanishing-gradients-with-rnns) far better than plain RNNs. The LSTM is the default here.
- Use it for text, audio, and other sequential data, especially when it arrives one step at a time.

### Key Equations

Each character is mapped to a learned embedding $e^{\langle t \rangle}$, which feeds a stack of recurrent layers. For the plain RNN,

$$
a^{\langle t \rangle} = \tanh\left(W_{aa} a^{\langle t-1 \rangle} + W_{ae} e^{\langle t \rangle} + b_a\right),
$$

and for the LSTM, with $h = \begin{bmatrix} a^{\langle t-1 \rangle} \\ e^{\langle t \rangle} \end{bmatrix}$,

$$
\begin{aligned}
c^{\langle t \rangle} &= \sigma\left(W_u h + b_u\right) \ast \tanh\left(W_c h + b_c\right) + \sigma\left(W_f h + b_f\right) \ast c^{\langle t-1 \rangle}, \\
a^{\langle t \rangle} &= \sigma\left(W_o h + b_o\right) \ast \tanh\left(c^{\langle t \rangle}\right),
\end{aligned}
$$

as derived in [Part 5](../intro/sequence_models.md#lstm). A linear layer turns the top layer's activation into scores for every character, and training minimizes the cross-entropy between the predicted distribution and the true next character, averaged over all positions:

$$
J = -\frac{1}{T} \sum_{t=1}^{T} \log P\left(y^{\langle t \rangle} \mid y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle}\right).
$$ (eq-rnn-loss)

The logged **perplexity** is $e^{J}$ on the test text: roughly, how many characters the model is choosing between at each step. A model that guessed uniformly among 65 characters would have a perplexity of 65.

### Pseudocode

```text
Algorithm 1  Training a character-level language model

Input: a text, split into training and test parts
for epoch = 1, 2, ..., epochs do
    for step = 1, 2, ..., steps_per_epoch do
        Sample a batch of random windows of seq_len characters, x, and the same windows shifted by one, y
        Compute the scores for every position with the RNN
        Compute the loss J, equation (1)
        Backpropagate through time, and clip the gradient norm
        Update the parameters with Adam
    end for
    Log the training loss, and the loss and perplexity on the test text
    Generate a sample of text and save the model
end for
```

## Documentation

```python
spinup_dl.rnn(dataset="tiny_shakespeare", cell="lstm", embed_dim=64, hidden_size=256, num_layers=2, dropout=0.0,
              seq_len=128, batch_size=64, epochs=10, steps_per_epoch=200, lr=2e-3, weight_decay=0.0, optimizer="adam",
              grad_clip=1.0, sample_length=300, prompt="\n", seed=0, device="auto", logger_kwargs=dict())
```

Trains a character-level recurrent language model and returns it.

| Argument | Meaning |
| --- | --- |
| `dataset` | `tiny_shakespeare`, about a million characters of Shakespeare's plays, downloaded on first use; `fake`, a short repeated text for testing; or a path to any text file |
| `cell` | `rnn`, `gru`, or `lstm` |
| `embed_dim` | Size of the character embeddings |
| `hidden_size` | Number of units in each recurrent layer |
| `num_layers` | Number of stacked recurrent layers, as in a [deep RNN](../intro/sequence_models.md#deep-rnns) |
| `dropout` | Dropout between recurrent layers, used when `num_layers` is more than 1 |
| `seq_len` | Length of the training windows. Gradients flow back through at most this many steps |
| `steps_per_epoch` | Training steps between evaluations; an "epoch" here is a fixed number of steps, not a full pass through the text |
| `grad_clip` | Maximum gradient norm, against [exploding gradients](../intro/sequence_models.md#vanishing-gradients-with-rnns). 0 turns clipping off |
| `sample_length` | Characters of text to generate after each epoch, into `samples.txt`. 0 turns sampling off |
| `prompt` | Text the samples start from |
| `batch_size`, `epochs`, `lr`, `weight_decay`, `optimizer`, `seed`, `device`, `logger_kwargs` | As for the [MLP](mlp.md#documentation) |

The model class is `spinup_dl.models.rnn.CharRNN(vocab_size, embed_dim, hidden_size, num_layers, cell, dropout)`. Its `generate(ids, length, temperature)` method continues a sequence; see [Experiment Outputs](../user/saving_and_loading.md#generating-text).

### What to Expect

```bash
python -m spinup_dl.run rnn --cell lstm --epochs 20
```

This run took 2 minutes and 23 seconds. The LSTM's test perplexity was lowest, 4.58, after 9 epochs, and rose to 4.93 by epoch 20 while its training loss kept falling: the model began to memorize the training text. This is the overfitting of [Part 2](../intro/improving_networks.md#bias-and-variance), and `--dropout 0.2` with `--num_layers 2` is the first thing to try against it. Here is the start of its sample after 20 epochs:

```text
Tybalt, that myscureless fully of our year:
How far she hath your words I'll render his own day.

DUCHESS OF YORK:
I use their old is lesser than my son.
She hath now blowers to behold that warp,
And he was not foul departure of them,
And I am sour inch and flatterers may
You tell her, and he
```

The trained model from this run is published, so you can use it without training:

```python
from spinup_dl import load_pretrained

model = load_pretrained("lstm_shakespeare")
```

### Things to Try

- Train `--cell rnn gru lstm` with three seeds each and plot `TestLoss`. The gated units should pull ahead.
- Vary `--seq_len 16 64 256`. Short windows limit how far back the model can learn dependencies.
- Train on your own text with `--dataset path/to/file.txt`: song lyrics, code, or your notes for a course.

## References

### Relevant Papers

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), Hochreiter and Schmidhuber, 1997.
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), Cho and others, 2014, which introduced the GRU.
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850), Graves, 2013.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), Karpathy, 2015.

### Other Public Implementations

- [char-rnn](https://github.com/karpathy/char-rnn), the character-level LSTM language model that made this experiment famous, in Lua and Torch.
- [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086), a character-level RNN in about 100 lines of NumPy, with backpropagation through time written out by hand.
- [PyTorch examples: word language model](https://github.com/pytorch/examples/tree/main/word_language_model), word-level RNN, LSTM, GRU, and Transformer language models.
