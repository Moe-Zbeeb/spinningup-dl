# Transformer

```{contents} Table of Contents
:depth: 2
```

## Background

The **Transformer** processes a sequence with [attention](../intro/sequence_models.md#attention-model) alone, without recurrence. Where an RNN passes information along one step at a time, every position in a Transformer can look directly at every earlier position in a single layer. This makes long-range dependencies easier to learn, and lets the whole sequence be processed in parallel, which is why Transformers scale to the very large models behind today's language systems.

The companion code implements a small **decoder-only** Transformer, the design used by GPT-style language models, and trains it on the same character-level task as the [RNN](rnn.md), so the two can be compared directly.

### Quick Facts

- Input: a sequence of up to `context_length` character indices. Output: a distribution over the next character at every position.
- Every position attends to all earlier positions; a **causal mask** stops it from looking ahead.
- It has no built-in notion of order, so a learned **position embedding** is added to each character's embedding.
- Use it for language and, with changes to the input, for images, audio, and almost any other data.

### Key Equations

The core operation is **scaled dot-product attention**. Each position's vector is projected into a **query** $q$, a **key** $k$, and a **value** $v$. Stacking them into matrices $Q$, $K$, and $V$, one row per position,

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V,
$$ (eq-transformer-attention)

where $d_k$ is the size of each key, and the mask $M$ is 0 where position $i$ may attend to position $j \leq i$, and $-\infty$ for $j > i$, so that those weights become 0 after the softmax. Each row of the softmax is a set of [attention weights](../intro/sequence_models.md#computing-the-attention-weights), like those of Part 5. Dividing by $\sqrt{d_k}$ keeps the scores from growing with the vector size, which would saturate the softmax.

**Multi-head attention** runs `n_heads` attentions in parallel on smaller projections and concatenates the results, so that different heads can attend to different things. Each Transformer block combines attention with a small MLP, using [layer normalization](https://arxiv.org/abs/1607.06450) and residual connections:

$$
\begin{aligned}
h &= x + \mathrm{MultiHead}\left(\mathrm{LayerNorm}(x)\right), \\
x' &= h + \mathrm{MLP}\left(\mathrm{LayerNorm}(h)\right).
\end{aligned}
$$

The model is trained with the same next-character cross-entropy loss as the [RNN](rnn.md#key-equations).

### Pseudocode

```text
Algorithm 1  Forward pass of the decoder-only Transformer

Input: character indices x of length t ≤ context_length
H ← token embedding of x + position embedding of 1, ..., t
for each of the n_layers blocks do
    H ← H + causal multi-head attention of LayerNorm(H), equation (1)
    H ← H + MLP of LayerNorm(H)
end for
return the scores for every position: a linear layer applied to LayerNorm(H)
```

## Documentation

```python
spinup_dl.transformer(dataset="tiny_shakespeare", context_length=128, d_model=128, n_heads=4, n_layers=4, dropout=0.1,
                      batch_size=64, epochs=10, steps_per_epoch=200, lr=1e-3, weight_decay=0.01, optimizer="adamw",
                      grad_clip=1.0, sample_length=300, prompt="\n", seed=0, device="auto", logger_kwargs=dict())
```

Trains a decoder-only Transformer language model and returns it.

| Argument | Meaning |
| --- | --- |
| `dataset` | As for the [RNN](rnn.md#documentation) |
| `context_length` | The longest sequence the model can read, and the length of the training windows |
| `d_model` | Size of the vectors that flow through the model. Must be divisible by `n_heads` |
| `n_heads` | Number of attention heads in each block |
| `n_layers` | Number of Transformer blocks |
| `dropout` | Dropout on the embeddings, the attention weights, and the MLP outputs |
| `optimizer` | Defaults to `adamw`, Adam with decoupled weight decay, the usual choice for Transformers |
| `batch_size`, `epochs`, `steps_per_epoch`, `lr`, `weight_decay`, `grad_clip`, `sample_length`, `prompt`, `seed`, `device`, `logger_kwargs` | As for the [RNN](rnn.md#documentation) |

The model class is `spinup_dl.models.transformer.Transformer(vocab_size, context_length, d_model, n_heads, n_layers, dropout)`, with the same `generate` method as the RNN.

### What to Expect

```bash
python -m spinup_dl.run transformer --epochs 3
```

After 3 epochs of 200 steps, which took 40 seconds, the Transformer reached a test perplexity of 6.90, against 5.09 for the LSTM with the same number of steps. With this little training, the LSTM is ahead; train both for longer and compare again. Here is the start of the Transformer's last sample:

```text
For batt uple like mer:
That upintose ind shere yought did bere be ge,
Shall no shick?
Thy have candess. Me, thou me of the in misin.

QUEEN YTIDWBUS:
Hen shall ou.

ANDUSH:
```

### Things to Try

- Plot the Transformer and the LSTM together, against `Time` as well as `Epoch`. Which learns more per step, and which per second?
- Scale the model: `--d_model 64 128 256` with `--n_layers 2 4 6`. Bigger models need more steps before they pay off.
- Load a trained model and plot the attention weights of one head for a line of text, to see what it looks at.

## References

### Relevant Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani and others, 2017.
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), Radford, Narasimhan, Salimans, and Sutskever, 2018.
- [Layer Normalization](https://arxiv.org/abs/1607.06450), Ba, Kiros, and Hinton, 2016.

### Other Public Implementations

- [nanoGPT](https://github.com/karpathy/nanoGPT), a compact and fast GPT implementation that can reproduce GPT-2.
- [minGPT](https://github.com/karpathy/minGPT), a smaller, more educational GPT implementation.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), the original encoder-decoder Transformer for translation, explained line by line.
