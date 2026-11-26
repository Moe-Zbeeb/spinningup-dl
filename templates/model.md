# Name of the Model

```{contents} Table of Contents
:depth: 2
```

## Background

Say what the model is, what problem it was built for, and what the reader should already know, linking to the page that teaches it.

### Quick Facts

- What goes in and what comes out.
- What the model assumes about the data.
- When to reach for it, and when not to.

### Key Equations

Give the forward computation and the loss. Each equation is introduced by the sentence before it and followed by a sentence that defines its symbols.

### Pseudocode

```text
Algorithm 1  Name of the Procedure

Input: what the procedure starts from
for step = 1, 2, ... do
    one line per operation
end for
```

## Documentation

A minimal PyTorch implementation and a usage example that runs as written.

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
print(model(torch.randn(8, 4)).shape)
```

## References

### Relevant Papers

- [Title of the Paper](https://arxiv.org/abs/0000.00000), Authors, Year.

### Other Public Implementations

- [Name of the Implementation](https://github.com/), with a few words on what it is.
