# Part 4: Convolutional Neural Networks

```{contents} Table of Contents
:depth: 2
```

Welcome to Part 4 of our introduction to deep learning! The networks in Parts 1 to 3 connect every input to every unit. That works for a handful of features, but an ordinary color photo has millions of pixels, and a fully connected layer on top of it would need billions of weights. **Convolutional neural networks**, or **ConvNets**, solve this by building in what we know about images: nearby pixels belong together, and a pattern means the same thing wherever it appears.[^sefidian] In this part, we aim to acquaint you with

- the convolution operation and the layers built from it,
- pooling and fully connected layers, and how they combine into a full network,
- the classic architectures and the ideas behind modern ones: residual connections, 1 × 1 convolutions, and inception modules,
- practical advice for computer vision: transfer learning and data augmentation,
- object localization and detection, up to the YOLO algorithm,
- face recognition with Siamese networks and the triplet loss,
- and neural style transfer.

## Fundamentals

### The Convolution Operation

A **convolution** slides a small grid of numbers, called a **filter** or **kernel**, across an image. At every position it multiplies the filter element-wise with the patch of image underneath and adds up the products, giving one output number. We write it with the operator $\ast$.

```{figure} ../_static/img/convnets/conv_example.svg
:alt: A 6 by 6 image whose left half is 10 and right half is 0, convolved with a 3 by 3 filter whose columns are 1, 0, and minus 1, gives a 4 by 4 output whose two middle columns are 30 and outer columns are 0. The top-left 3 by 3 patch of the image and the top-left output entry are outlined.
:width: 100%

A vertical edge detector. The outlined patch times the filter, summed, gives the outlined output: $10 + 10 + 10 + 0 + 0 + 0 - 10 - 10 - 10 = 0$.
```

The filter here, with a column of 1s, a column of 0s, and a column of $-1$s, is a **vertical edge detector**. Over a flat region the positive and negative columns cancel and the output is 0. Where bright pixels on the left meet dark pixels on the right, they no longer cancel, and the output is large. The two columns of 30s in the output mark exactly where the image changes from bright to dark.

**Example: Edge Detection in PyTorch.**

```python
import torch
import torch.nn.functional as F

image = torch.tensor([[10.0] * 3 + [0.0] * 3] * 6).view(1, 1, 6, 6)
vertical_edge = torch.tensor([[1.0, 0.0, -1.0]] * 3).view(1, 1, 3, 3)

print(F.conv2d(image, vertical_edge).squeeze())
```

```text
tensor([[ 0., 30., 30.,  0.],
        [ 0., 30., 30.,  0.],
        [ 0., 30., 30.,  0.],
        [ 0., 30., 30.,  0.]])
```

On a real image, the same idea finds outlines. Turning the filter on its side detects horizontal edges instead:

```{figure} ../_static/img/convnets/edge_detection.svg
:alt: A simple grayscale drawing of a house with a roof, a door, and a sun, next to the output of a vertical edge filter, which highlights the walls and the door's sides, and the output of a horizontal edge filter, which highlights the roof's base, the top of the door, and the ground line.
:width: 100%

A drawing convolved with a vertical and a horizontal edge filter. Each filter responds to edges in its own direction.
```

Classic computer vision used hand-designed filters like these, such as the **Sobel** filter, which weights the middle row more heavily, $\begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$. The key idea of ConvNets is to treat the nine numbers in the filter as parameters and let backpropagation learn them. A network can then discover edge detectors at any angle, and many other patterns no one would think to design.

### Output Size

Convolving an $n \times n$ image with an $f \times f$ filter gives an $(n - f + 1) \times (n - f + 1)$ output, because the filter fits in $n - f + 1$ positions along each direction. In the example, $6 - 3 + 1 = 4$.

### Padding

Plain convolution has two drawbacks. The image shrinks with every layer, so a deep network would quickly reduce it to nothing. And pixels at the border are covered by far fewer filter positions than pixels in the middle, so information near the edges is underused.

**Padding** fixes both by adding a border of $p$ pixels, set to 0, around the image before convolving. The output becomes $(n + 2p - f + 1) \times (n + 2p - f + 1)$.

```{figure} ../_static/img/convnets/padding_stride.svg
:alt: Left, a 5 by 5 image surrounded by a one-pixel border of zeros, with a 3 by 3 filter outlined in the top-left corner. Right, a 7 by 7 image with three 3 by 3 filter positions in one row, each 2 pixels apart.
:width: 100%

Padding adds a border of zeros (left). A stride makes the filter skip positions (right).
```

Two choices of padding have names.

- **Valid** convolution uses no padding, $p = 0$.
- **Same** convolution pads just enough for the output to be the same size as the input. Setting $n + 2p - f + 1 = n$ gives $p = \frac{f - 1}{2}$, so a $3 \times 3$ filter needs $p = 1$ and a $5 \times 5$ filter needs $p = 2$.

By convention, $f$ is almost always odd. That makes "same" padding symmetric, and it gives the filter a central pixel to refer to its position by.

### Strided Convolution

A **stride** $s$ moves the filter $s$ pixels at a time instead of 1. Combining padding and stride, the output size along each direction is

$$
\left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor,
$$ (eq-convnets-output-size)

where $\lfloor \cdot \rfloor$ rounds down, because a filter position that would hang off the edge of the padded image is skipped. To get an output of a particular size $t$, solve $\frac{n + 2p - f}{s} + 1 = t$ for $p$.

**Example: Checking the Output Size Formula.**

```python
import torch
import torch.nn as nn


def conv_output_size(n, f, p=0, s=1):
    return (n + 2 * p - f) // s + 1


x = torch.randn(1, 3, 39, 39)
for f, p, s in [(3, 0, 1), (3, 1, 1), (5, 2, 2), (7, 0, 3)]:
    out = nn.Conv2d(3, 8, kernel_size=f, padding=p, stride=s)(x)
    print(f"f={f} p={p} s={s}: formula {conv_output_size(39, f, p, s)}, PyTorch {out.shape[-1]}")
```

```text
f=3 p=0 s=1: formula 37, PyTorch 37
f=3 p=1 s=1: formula 39, PyTorch 39
f=5 p=2 s=2: formula 20, PyTorch 20
f=7 p=0 s=3: formula 11, PyTorch 11
```

```{admonition} You Should Know
In mathematics and signal processing, convolution first flips the filter horizontally and vertically, which makes it associative: $(A \ast B) \ast C = A \ast (B \ast C)$. What deep learning calls convolution skips the flip, and is technically called **cross-correlation**. Since the filter is learned anyway, the flip makes no difference, and the deep learning community simply calls it convolution.
```

### Convolutions over Volumes

A color image has three **channels**, red, green, and blue, so it is a volume of size $n \times n \times 3$. To convolve it, the filter gets the same number of channels: a $3 \times 3 \times 3$ cube. At each position, all $3 \times 3 \times 3 = 27$ products are added into one number, so the output is still a flat $4 \times 4$ grid for a $6 \times 6 \times 3$ input. The channels do not enter the output size formula; they only have to match between the input and the filter.

To detect several features at once, apply several filters and stack their outputs as the channels of a new volume.

```{figure} ../_static/img/convnets/conv_volume.svg
:alt: A 6 by 6 by 3 input volume convolved with two 3 by 3 by 3 filters gives two 4 by 4 outputs, which are stacked into a 4 by 4 by 2 output volume.
:width: 100%

Two filters on a color image give an output volume with two channels, one per filter.
```

With $n_C$ input channels and $n_C'$ filters, an $n \times n \times n_C$ input gives an $(n - f + 1) \times (n - f + 1) \times n_C'$ output. The number of channels is sometimes called the **depth** of the volume, but that word is easily confused with the depth of the network.

## One Layer of a Convolutional Network

A convolutional layer does to its input what a fully connected layer does, with convolution in place of the matrix product. Each filter is convolved with the input volume, a bias is added to its output, and the result goes through an activation function such as ReLU. The filter outputs, stacked as channels, form the layer's output volume. In the language of Part 1, the convolution plus the bias is $z^{[l]}$, the filters are $W^{[l]}$, and the stacked activations are $a^{[l]}$.

For layer $l$ we use the following notation.

| Symbol | Meaning |
| --- | --- |
| $f^{[l]}$ | Filter size |
| $p^{[l]}$ | Padding |
| $s^{[l]}$ | Stride |
| $n_C^{[l]}$ | Number of filters, which is the number of output channels |
| $n_H^{[l-1]} \times n_W^{[l-1]} \times n_C^{[l-1]}$ | Size of the input volume: height, width, channels |
| $n_H^{[l]} \times n_W^{[l]} \times n_C^{[l]}$ | Size of the output volume, $a^{[l]}$ |
| $f^{[l]} \times f^{[l]} \times n_C^{[l-1]}$ | Size of each filter |
| $f^{[l]} \times f^{[l]} \times n_C^{[l-1]} \times n_C^{[l]}$ | Size of all the weights together |
| $n_C^{[l]}$ | Number of biases, one per filter |

The output height and width follow {eq}`eq-convnets-output-size`:

$$
n_H^{[l]} = \left\lfloor \frac{n_H^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \right\rfloor, \qquad n_W^{[l]} = \left\lfloor \frac{n_W^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \right\rfloor.
$$

With $m$ examples, the activations form a four-dimensional array $A^{[l]}$ of size $m \times n_H^{[l]} \times n_W^{[l]} \times n_C^{[l]}$.

```{admonition} You Should Know
The layer's parameter count does not depend on the size of the image. Ten filters of size $3 \times 3 \times 3$ have $10 \times (27 + 1) = 280$ parameters, counting one bias per filter, whether the image is $64 \times 64$ or $1000 \times 1000$. This is a large part of why ConvNets scale to big images.

PyTorch stores the channels before the height and width: its volumes are $m \times n_C \times n_H \times n_W$.
```

**Example: Counting a Layer's Parameters.**

```python
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
print("conv weights:", tuple(conv.weight.shape), "parameters:", sum(p.numel() for p in conv.parameters()))
```

```text
conv weights: (10, 3, 3, 3) parameters: 280
```

## Convolutional Network

A ConvNet stacks convolutional layers. As the volume passes through them, its height and width usually shrink while its number of channels grows: the network trades spatial detail for a richer description of what is there. At the end, the volume is unrolled into a vector and passed to a logistic or softmax output, depending on the task.

Besides convolutional layers, written **CONV**, most ConvNets use two other kinds: **pooling** layers, **POOL**, and **fully connected** layers, **FC**. The next two sections cover them. Here is a small network that uses all three, in the style of the classic LeNet-5:

```{figure} ../_static/img/convnets/lenet_like.svg
:alt: A sequence of volumes: input 32 by 32 by 3, conv 5 by 5 to 28 by 28 by 6, max pool to 14 by 14 by 6, conv 5 by 5 to 10 by 10 by 16, max pool to 5 by 5 by 16, then fully connected vectors of 400, 120, 84, and a softmax of 10.
:width: 100%

A small ConvNet. Height and width shrink from 32 to 5 while the channels grow from 3 to 16, before the fully connected layers.
```

**Example: Tracing Shapes through a ConvNet.**

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(400, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10),
)

x = torch.randn(1, 3, 32, 32)
for layer in model:
    x = layer(x)
    if not isinstance(layer, nn.ReLU):
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"{type(layer).__name__:<10}{str(tuple(x.shape[1:])):<16}{n_params:>8} parameters")
```

```text
Conv2d    (6, 28, 28)          456 parameters
MaxPool2d (6, 14, 14)            0 parameters
Conv2d    (16, 10, 10)        2416 parameters
MaxPool2d (16, 5, 5)             0 parameters
Flatten   (400,)                 0 parameters
Linear    (120,)             48120 parameters
Linear    (84,)              10164 parameters
Linear    (10,)                850 parameters
```

Pooling layers have no parameters, so some authors do not count them as layers. In that convention, a CONV layer followed by a POOL layer counts as one layer.

## Pooling Layers

A **pooling layer** shrinks the height and width of a volume, which speeds up the computation and makes the detected features a little more robust to small shifts in position.

**Max pooling** divides the input into regions of size $f \times f$, stepping with stride $s$, and keeps only the largest value in each region:

```{figure} ../_static/img/convnets/pooling.svg
:alt: A 4 by 4 input divided into four colored 2 by 2 regions. Max pooling keeps 9, 2, 6, and 3. Average pooling gives 3.75, 1.25, 3.75, and 2.
:width: 90%

Max pooling and average pooling with $f = 2$ and $s = 2$. Each output is computed from the input region of the same color.
```

The intuition is that a large value means a feature was detected somewhere in that region. Max pooling keeps the strong signal and drops the information about exactly where in the region it was. Its main justification, though, is simply that it works well in practice.

A few properties are worth knowing.

- Pooling works on each channel independently, so it does not change the number of channels.
- It has no parameters to learn. Its hyperparameters are $f$, $s$, and the type of pooling. $f = 2$, $s = 2$, which halves the height and width, is the most common choice.
- Padding is rarely used with pooling.
- The output size follows the same formula as convolution: an $n_H \times n_W \times n_C$ input gives an output of size $\left\lfloor \frac{n_H - f}{s} + 1 \right\rfloor \times \left\lfloor \frac{n_W - f}{s} + 1 \right\rfloor \times n_C$.

**Average pooling** keeps the average of each region instead of the maximum. It is used less, except deep in a network, where it can collapse a whole volume, say $7 \times 7 \times 1000$, into a single vector of 1000 numbers.

**Example: Pooling in PyTorch.**

```python
import torch
import torch.nn as nn

x = torch.tensor([[1.0, 3.0, 2.0, 1.0],
                  [2.0, 9.0, 1.0, 1.0],
                  [1.0, 3.0, 2.0, 3.0],
                  [5.0, 6.0, 1.0, 2.0]]).view(1, 1, 4, 4)

print(nn.MaxPool2d(kernel_size=2, stride=2)(x).squeeze())
print(nn.AvgPool2d(kernel_size=2, stride=2)(x).squeeze())
```

```text
tensor([[9., 2.],
        [6., 3.]])
tensor([[3.7500, 1.2500],
        [3.7500, 2.0000]])
```

## Fully Connected Layers

After the convolutional and pooling layers, the volume is **flattened**: its $n_H \times n_W \times n_C$ numbers are unrolled into one long vector. In the example network, a $5 \times 5 \times 16$ volume becomes a vector of 400 numbers. From there on, the network uses ordinary **fully connected** layers, exactly like the ones in Part 1, usually with ReLU activations, ending in the output layer.

Fully connected layers have far more parameters than convolutional layers. In the example network, the two convolutional layers have 2,872 parameters together, while the three fully connected layers have 59,134.

## Why Convolutions

Consider the first layer of the example network, which turns a $32 \times 32 \times 3$ image, 3,072 numbers, into a $28 \times 28 \times 6$ volume, 4,704 numbers. A fully connected layer between them would need $3{,}072 \times 4{,}704 \approx 14.5$ million weights. The convolutional layer needs 456. Two properties make the difference.

- **Parameter sharing.** A feature detector, such as a vertical edge detector, that is useful in one part of the image is probably useful in other parts too. A convolutional layer uses the same filter at every position, so it learns each detector once instead of once per position.
- **Sparsity of connections.** Each output value depends only on a small patch of the input, the $f \times f$ region under the filter, and not on any of the other pixels. The edge detector's output at the top left of the image has nothing to do with the pixels at the bottom right.

Together, these give a network with far fewer parameters, which needs less data and overfits less. They also build in a useful assumption: a cat shifted a few pixels to the right is still a cat, and because the same filters run everywhere, the network's features shift along with it.

## Historical and Classic Architectures

Studying a few landmark networks is one of the best ways to learn how to put the pieces together. Many of their design choices are still in use.

| Network | Year | Input | Parameters | Key ideas |
| --- | --- | --- | ---: | --- |
| LeNet-5 | 1998 | $32 \times 32 \times 1$ | about 60 thousand | CONV → POOL → CONV → POOL → FC → FC → output |
| AlexNet | 2012 | $227 \times 227 \times 3$ | about 60 million | Much bigger, ReLU, max pooling, trained on GPUs |
| VGG-16 | 2014 | $224 \times 224 \times 3$ | about 138 million | Only $3 \times 3$ convolutions and $2 \times 2$ pooling, in a uniform pattern |

**LeNet-5** read handwritten digits.[^lenet] It already had the shape most ConvNets still follow: height and width shrink as the channels grow, and a few fully connected layers come at the end. It used sigmoid and tanh activations rather than ReLU, average pooling rather than max pooling, and applied a nonlinearity after pooling. It also had a complicated scheme of connecting each filter to only some of the input channels, which modern networks no longer use.

**AlexNet** won the ImageNet image recognition competition in 2012 by a wide margin and convinced much of the computer vision community that deep learning works.[^alexnet] It follows the same pattern as LeNet but is about a thousand times bigger: CONV → POOL → CONV → POOL → CONV → CONV → CONV → POOL → FC → FC → softmax. It used ReLU activations, max pooling, and mostly "same" convolutions. It also used **local response normalization**, which normalizes each position across all the channels; later work found it does not help much, and it is rarely used today.

**VGG-16** simplified the design.[^vgg] Every convolution is $3 \times 3$ with stride 1 and "same" padding, every max pool is $2 \times 2$ with stride 2, and the network alternates blocks of convolutions with pooling: 64 filters, then 128, 256, 512, and 512, followed by two fully connected layers of 4096 units and a softmax over 1000 classes. The 16 refers to its 16 layers with weights. The uniformity makes it easy to understand: the height and width halve after every block while the number of filters doubles, until it reaches 512. Its 138 million parameters make it large even by today's standards.

## ResNet

Very deep networks are hard to train. In theory, adding layers can only help, since the extra layers could learn to do nothing. In practice, the training error of a plain deep network often gets worse as it gets deeper, partly because of the [vanishing and exploding gradients](improving_networks.md#vanishing-and-exploding-gradients) of Part 2. **Residual networks**, or **ResNets**, fixed this and made networks with over 100 layers trainable.[^resnet]

### Residual Blocks

A ResNet is built from **residual blocks**. In a plain network, two layers compute

$$
\begin{aligned}
z^{[l+1]} &= W^{[l+1]} a^{[l]} + b^{[l+1]}, & a^{[l+1]} &= g\left(z^{[l+1]}\right), \\
z^{[l+2]} &= W^{[l+2]} a^{[l+1]} + b^{[l+2]}, & a^{[l+2]} &= g\left(z^{[l+2]}\right).
\end{aligned}
$$

A residual block adds a **shortcut**, also called a **skip connection**: it takes $a^{[l]}$ and adds it back in just before the last activation function,

$$
a^{[l+2]} = g\left(z^{[l+2]} + a^{[l]}\right).
$$ (eq-convnets-residual)

```{figure} ../_static/img/convnets/resnet_block.svg
:alt: The activation a of layer l passes through a linear layer and ReLU, then another linear layer, and the result is added to a of layer l carried along a shortcut, before a final ReLU produces a of layer l plus 2.
:width: 100%

A residual block. The shortcut carries $a^{[l]}$ past two layers.
```

A ResNet is a long sequence of these blocks. The difference in practice is dramatic:

```{figure} ../_static/img/convnets/resnet_depth.svg
:alt: Training loss after training against the number of layers, from 2 to 100 on a log scale. The plain network's loss is near zero up to 8 layers, rises to 0.33 at 20 layers, and sits at chance level, 0.69, from 32 layers on. The residual network's loss stays near zero at every depth.
:width: 80%

A real experiment: plain and residual networks of increasing depth, each trained the same way for 1500 steps on the same two-class problem, averaged over 3 runs. Beyond 20 layers, the plain networks fail to learn at all, while the residual networks reach almost zero training loss at every depth up to 100.
```

### Why Residual Networks Work

The key is that a residual block finds it easy to learn the identity function. Suppose the extra layers turn out to be useless, and regularization drives $W^{[l+2]}$ and $b^{[l+2]}$ to zero. Then $z^{[l+2]} = 0$, and {eq}`eq-convnets-residual` gives

$$
a^{[l+2]} = g\left(a^{[l]}\right) = a^{[l]},
$$

where the last step uses ReLU and the fact that $a^{[l]}$, itself the output of a ReLU, has no negative entries. The block passes its input through unchanged. So adding residual blocks to a network is very unlikely to make its training error worse: the blocks can always fall back to doing nothing, and they only change the output when that helps. A plain layer has no such fallback; learning the identity with weights and a ReLU is surprisingly hard.

The shortcuts also help the gradients. During backpropagation, the gradient flows through the shortcut directly to earlier layers, without being multiplied by the weights of the skipped layers. Early layers keep receiving a useful signal, which counters vanishing gradients.

### Matching Dimensions

For the sum $z^{[l+2]} + a^{[l]}$ to make sense, the two must have the same shape. ResNets therefore use "same" convolutions inside a block. When a block does change the shape, for example by halving the height and width and doubling the channels, the shortcut multiplies $a^{[l]}$ by a matrix $W_s$ first, computing $z^{[l+2]} + W_s a^{[l]}$. $W_s$ can be a learned projection, in practice a $1 \times 1$ convolution, or a fixed operation that pads with zeros. Deeper ResNets also use a variant whose shortcut skips three layers instead of two.

**Example: A Residual Block in PyTorch.** This block uses batch normalization, as real ResNets do, and a $1 \times 1$ convolution as $W_s$ when the shape changes:

```python
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, a):
        z = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(a)))))
        return torch.relu(z + self.shortcut(a))


x = torch.randn(8, 64, 32, 32)
print(ResidualBlock(64, 64)(x).shape)
print(ResidualBlock(64, 128, stride=2)(x).shape)
```

```text
torch.Size([8, 64, 32, 32])
torch.Size([8, 128, 16, 16])
```

## Networks in Networks and 1x1 Convolutions

A $1 \times 1$ convolution sounds pointless: on a single-channel image, it just multiplies every pixel by a number. On a volume with many channels, it is useful. At each pixel, a $1 \times 1 \times n_C$ filter takes the $n_C$ numbers along the channels, computes a weighted sum, adds a bias, and applies a ReLU. That is one unit of a fully connected layer, applied separately to every pixel. With several filters, a $1 \times 1$ convolution is a small fully connected network running at every position, which is why the idea was called **network in network**.

Its most common use is to change the number of channels. Pooling shrinks the height and width; a $1 \times 1$ convolution shrinks, or grows, the channels while leaving the height and width alone. A $28 \times 28 \times 192$ volume through 32 filters of size $1 \times 1$ becomes $28 \times 28 \times 32$.

**Example: Shrinking Channels.** A $1 \times 1$ convolution, and the saving it brings when placed before a larger convolution, as the next section explains:

```python
import torch
import torch.nn as nn

x = torch.randn(1, 192, 28, 28)
print(nn.Conv2d(192, 32, kernel_size=1)(x).shape)

direct = 28 * 28 * 32 * (5 * 5 * 192)
bottleneck = 28 * 28 * 16 * 192 + 28 * 28 * 32 * (5 * 5 * 16)
print(f"direct 5x5: {direct / 1e6:.1f}M multiplications, with a 1x1 bottleneck: {bottleneck / 1e6:.1f}M")
```

```text
torch.Size([1, 32, 28, 28])
direct 5x5: 120.4M multiplications, with a 1x1 bottleneck: 12.4M
```

## Inception Network (GoogLeNet)

Designing a ConvNet means choosing, for every layer, between a $1 \times 1$, a $3 \times 3$, or a $5 \times 5$ convolution, or a pooling layer. The **inception network**, also known as **GoogLeNet**, avoids the choice: it uses all of them in parallel and lets the network learn which combination works.[^inception]

### The Computational Cost

The obvious problem is cost. Take a $28 \times 28 \times 192$ input and a "same" $5 \times 5$ convolution with 32 filters. Each of the $28 \times 28 \times 32$ output values needs $5 \times 5 \times 192$ multiplications, about 120 million in total. A $1 \times 1$ convolution fixes this. First shrink the input to 16 channels, then apply the $5 \times 5$ convolution to the smaller volume:

```{figure} ../_static/img/convnets/bottleneck.svg
:alt: Top, a 28 by 28 by 192 volume goes directly through a 5 by 5 convolution with 32 filters to 28 by 28 by 32, costing 120.4 million multiplications. Bottom, the same volume first goes through a 1 by 1 convolution with 16 filters to 28 by 28 by 16, then through the 5 by 5 convolution to 28 by 28 by 32, costing 2.4 plus 10.0, which is 12.4 million.
:width: 100%

A $1 \times 1$ bottleneck cuts the cost of the same output by a factor of about 10.
```

The narrow middle volume is called a **bottleneck layer**. It cuts the cost by a factor of about 10, and as long as the bottleneck is not too narrow, it does not seem to hurt performance.

### The Inception Module

An **inception module** runs several branches in parallel on the same input and concatenates their outputs along the channels:

```{figure} ../_static/img/convnets/inception_module.svg
:alt: The previous activation feeds four parallel branches: a 1 by 1 convolution with 64 filters; a 1 by 1 convolution with 96 filters followed by a 3 by 3 convolution with 128; a 1 by 1 convolution with 16 followed by a 5 by 5 convolution with 32; and a 3 by 3 max pool followed by a 1 by 1 convolution with 32. The outputs are concatenated into 28 by 28 by 256.
:width: 100%

An inception module. The $1 \times 1$ convolutions in red keep the cost down.
```

- A $1 \times 1$ convolution on its own.
- A $1 \times 1$ bottleneck followed by a $3 \times 3$ convolution.
- A $1 \times 1$ bottleneck followed by a $5 \times 5$ convolution.
- A $3 \times 3$ max pool followed by a $1 \times 1$ convolution. The max pool uses "same" padding and stride 1, an unusual choice for pooling, so that its output has the same height and width as the other branches. The $1 \times 1$ convolution then cuts its 192 channels down, since otherwise this one branch would dominate the output.

Every branch keeps the height and width at $28 \times 28$, so their outputs can be stacked: $64 + 128 + 32 + 32 = 256$ channels.

The full inception network stacks many of these modules, with occasional max pooling layers to shrink the height and width, and ends with a fully connected layer and a softmax. It also has **side branches**: a few hidden layers feed their own small fully connected and softmax layers that try to predict the label. Training them alongside the main output pushes even the middle layers to compute useful features, which has a regularizing effect.

**Example: An Inception Module in PyTorch.**

```python
import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.b1 = nn.Conv2d(c_in, 64, 1)
        self.b3 = nn.Sequential(nn.Conv2d(c_in, 96, 1), nn.ReLU(), nn.Conv2d(96, 128, 3, padding=1))
        self.b5 = nn.Sequential(nn.Conv2d(c_in, 16, 1), nn.ReLU(), nn.Conv2d(16, 32, 5, padding=2))
        self.pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(c_in, 32, 1))

    def forward(self, a):
        branches = [self.b1(a), self.b3(a), self.b5(a), self.pool(a)]
        return torch.relu(torch.cat(branches, dim=1))


print(InceptionModule(192)(torch.randn(1, 192, 28, 28)).shape)
```

```text
torch.Size([1, 256, 28, 28])
```

## Using Transfer Learning

Training a large ConvNet from scratch takes a lot of data and weeks of computation. Fortunately, many researchers release networks already trained on huge datasets such as ImageNet, with a million images in 1000 classes. For almost any computer vision task, starting from one of these through [transfer learning](structuring_projects.md#transfer-learning) beats starting from random weights. How much of the network you retrain depends on how much data you have.

- **Very little data.** Replace the 1000-class softmax with your own output layer, and freeze all the other layers. Since the frozen layers never change, you can run every training image through them once, save the outputs to disk, and then train just the small output layer on those saved features, which is very fast.
- **More data.** Freeze fewer layers: keep the early ones frozen and train the last few along with your new output layer. You can start the unfrozen layers from their pre-trained weights, or replace them with new layers of your own design.
- **A lot of data.** Use the pre-trained weights only as the starting point and train the whole network.

The more data you have, the fewer layers you freeze. The early layers detect generic features such as edges and textures, which transfer well to almost any task, while the later layers are more specific to the original classes.

**Example: Freezing a Pre-Trained Network.** In practice you would load trained weights with `weights="DEFAULT"`; `weights=None` here skips the download:

```python
import torch.nn as nn
import torchvision

model = torchvision.models.resnet18(weights=None)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 3)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable parameters:", trainable)
```

```text
trainable parameters: 1539
```

Of the network's 11 million parameters, only the new output layer's $512 \times 3 + 3 = 1539$ are trained.

## Data Augmentation

For most computer vision tasks, more data helps, which is not true of every area of machine learning. [Data augmentation](improving_networks.md#data-augmentation) is therefore used almost universally in vision. The common methods change an image in ways that keep its label:

```{figure} ../_static/img/convnets/augmentation.svg
:alt: Six versions of a simple drawing of a cat on grass under a sun: the original, mirrored left to right, a random crop zoomed in on the cat, rotated by 15 degrees, sheared sideways, and with its colors shifted toward blue.
:width: 100%

One image and five augmented versions of it. The label is still "cat" in every one.
```

- **Mirroring** left to right.
- **Random cropping**, which works as long as the crops are big enough to still contain the object.
- **Rotation**, **shearing**, and **local warping**, used less often.
- **Color shifting**: adding small random amounts to the red, green, and blue channels. A photo taken under yellow light is still a photo of the same cat. AlexNet used a version called **PCA color augmentation**, which shifts the colors along the directions in which they naturally vary the most.

Augmentation is usually done on the fly during training. One or more CPU threads load and distort the images while the GPU trains on the previous batch, so the augmentation costs almost no extra time. In PyTorch, setting `num_workers` on a `DataLoader` does exactly this.

**Example: An Augmentation Pipeline.** Each call applies a fresh random combination of transformations:

```python
import torch
from torchvision.transforms import v2

augment = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(size=224, scale=(0.6, 1.0), antialias=True),
    v2.RandomRotation(degrees=15),
    v2.RandomAffine(degrees=0, shear=10),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
])

image = torch.rand(3, 256, 256)
print(augment(image).shape)
```

```text
torch.Size([3, 224, 224])
```

## State of Computer Vision

A learning system draws on two sources of knowledge: **labeled data**, pairs of $x$ and $y$, and **hand-engineering**: hand-designed features, carefully crafted architectures, and other specialized components. When there is little data, hand-engineering has to make up for it. When there is a lot, simpler algorithms with less hand-engineering tend to win.

```{figure} ../_static/img/convnets/data_spectrum.svg
:alt: A horizontal axis from little data on the left to lots of data on the right, with object detection, image recognition, and speech recognition placed from left to right. Little data goes with more hand-engineering and specialized architectures; lots of data goes with simpler algorithms and less hand-engineering.
:width: 90%

Tasks placed by how much labeled data is typically available for them.
```

Computer vision has grown enormously in data, but it still has less data than its problems need. Object detection, which needs a box drawn around every object, has much less labeled data than image recognition, which needs one label per image. This is why computer vision still relies on complex architectures, many hyperparameters, and transfer learning, which is a way of borrowing knowledge from a larger dataset.

## Tips for Doing Well on Benchmarks and Competitions

Two techniques reliably add a little accuracy on benchmarks and in competitions. They are rarely worth their cost in production systems, where every prediction has to be fast and cheap.

- **Ensembling.** Train several networks independently, typically 3 to 15, and average their outputs. Each network's random mistakes partly cancel out, often gaining 1% or 2%. But every prediction now costs several networks' worth of computation.
- **Multi-crop at test time.** Run the classifier on several versions of each test image, such as the center crop, the four corner crops, and their mirror images, and average the results. This is data augmentation applied at test time.

## Classification with Localization

Image classification says what is in an image. **Classification with localization** also says where: the network outputs a **bounding box** around the object.

```{figure} ../_static/img/convnets/localization.svg
:alt: Left, a drawing of a cat with a bounding box around it, marked with its center coordinates b x and b y, its width b w, and its height b h, all as fractions of the image size. Right, a schematic face with ten numbered landmark points on the eyebrows, eyes, nose, and mouth.
:width: 100%

Classification with localization outputs a bounding box (left). Landmark detection outputs a set of points (right).
```

To do this, the output layer gets four extra numbers: $b_x$ and $b_y$, the coordinates of the box's center, and $b_w$ and $b_h$, its width and height. All four are measured as fractions of the image size, with $(0, 0)$ at the top left and $(1, 1)$ at the bottom right. One more output, $p_c$, says whether there is an object at all. With three possible classes, the label for each image is a vector of 8 numbers:

$$
y = \begin{bmatrix} p_c & b_x & b_y & b_h & b_w & c_1 & c_2 & c_3 \end{bmatrix}^{\top},
$$

where $c_1$, $c_2$, and $c_3$ say which class the object belongs to.

When there is no object, $p_c = 0$ and the other seven entries are meaningless, so the loss should ignore them. With squared error, for example,

$$
\L(\hat{y}, y) = \begin{cases} \sum_{j=1}^{8} \left(\hat{y}_j - y_j\right)^2 & \text{if } y_1 = 1, \\ \left(\hat{y}_1 - y_1\right)^2 & \text{if } y_1 = 0. \end{cases}
$$

In practice, the different parts often get different losses: logistic loss for $p_c$, squared error for the box, and softmax loss for the classes.

**Example: A Localization Loss.** The second example has no object, so only its $p_c$ counts:

```python
import torch


def localization_loss(y_hat, y):
    has_object = y[:, 0] == 1
    all_terms = ((y_hat - y) ** 2).sum(dim=1)
    pc_term = (y_hat[:, 0] - y[:, 0]) ** 2
    per_example = torch.where(has_object, all_terms, pc_term)
    return per_example.mean()


y = torch.tensor([[1.0, 0.5, 0.7, 0.3, 0.4, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
y_hat = torch.tensor([[0.9, 0.4, 0.7, 0.3, 0.5, 0.1, 0.8, 0.1],
                      [0.2, 0.9, 0.1, 0.5, 0.5, 0.3, 0.3, 0.3]])
print(f"{localization_loss(y_hat, y):.4f}")
```

```text
0.0650
```

## Landmark Detection

The same idea extends from boxes to any points of interest. In **landmark detection**, the network outputs the coordinates of specific points, such as the corners of the eyes, the tip of the nose, and the edges of the mouth, as in the right panel of the figure above. With 64 landmarks on a face, the output has 128 numbers for their coordinates, plus one saying whether there is a face. Landmarks are the basis for applications such as recognizing facial expressions, adding effects to faces in photos, and estimating a person's pose from points on their joints.

```{admonition} You Should Know
The landmark labels must be consistent across all the training and test images. If landmark 1 is the outer corner of the left eye in one image, it must be the outer corner of the left eye in every image.
```

## Object Detection

**Object detection** finds every object in an image, possibly several of different classes, each with its own bounding box.

A simple approach builds on classification. First train a ConvNet to recognize closely cropped images: for example, images that are either filled by a car or contain no car at all. Then detect cars in a full image with **sliding windows**: cut out a small window at the top left, run the ConvNet on it, slide the window along by some stride, run it again, and so on across the whole image. Repeat with larger windows, so that cars of different sizes are caught.

This works, but it is extremely expensive. A small stride means thousands of windows, each needing its own run of the ConvNet. A large stride is cheaper but locates the objects poorly. And it is hard to know in advance which window sizes and strides to use.

## Convolutional Implementation of Sliding Windows

Most of that computation is redundant. Neighboring windows overlap, so the ConvNet computes the same features over and over. The fix is to run the ConvNet once over the whole image, in a way that shares the computation between all the windows.

### Turning Fully Connected Layers into Convolutions

The first step is to notice that a fully connected layer can be written as a convolution. Suppose the last pooling layer outputs a $5 \times 5 \times 16$ volume and the next layer is fully connected with 400 units. Instead, use 400 filters of size $5 \times 5 \times 16$. Each filter covers the whole volume and outputs a single number, so the output is $1 \times 1 \times 400$, which is mathematically the same as the fully connected layer. Further fully connected layers become $1 \times 1$ convolutions in the same way.

### All Windows in One Pass

Now run this fully convolutional network on a larger image. Suppose it was designed for $14 \times 14 \times 3$ windows and you give it a $16 \times 16 \times 3$ image. Every layer simply produces a larger output, and the final output is $2 \times 2$ instead of $1 \times 1$:

```{figure} ../_static/img/convnets/sliding_windows.svg
:alt: A 16 by 16 grid with four overlapping 14 by 14 windows outlined in different colors and styles, offset by 2 pixels. One pass of the ConvNet maps it to a 2 by 2 output whose four cells are colored to match the four windows.
:width: 90%

A ConvNet designed for $14 \times 14$ windows, run once on a $16 \times 16$ image, gives the predictions for all four windows at stride 2 in a single pass.
```

Each of the four outputs is exactly the prediction that the network would have made on one of the four $14 \times 14$ windows, at stride 2. On a $28 \times 28$ image, the same network gives an $8 \times 8$ output, the predictions for 64 windows. The stride between windows is set by the network's pooling, here 2.

**Example: Checking the Shared Computation.** The output for the top-left and bottom-right windows matches running the network on those windows separately:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 400, kernel_size=5), nn.ReLU(),
    nn.Conv2d(400, 400, kernel_size=1), nn.ReLU(),
    nn.Conv2d(400, 4, kernel_size=1),
)

image = torch.randn(1, 3, 16, 16)
all_windows = net(image)
print("one pass over the 16x16 image:", tuple(all_windows.shape))

top_left = net(image[:, :, 0:14, 0:14])
bottom_right = net(image[:, :, 2:16, 2:16])
print(torch.allclose(all_windows[..., 0, 0], top_left[..., 0, 0], atol=1e-5),
      torch.allclose(all_windows[..., 1, 1], bottom_right[..., 0, 0], atol=1e-5))
```

```text
one pass over the 16x16 image: (1, 4, 2, 2)
True True
```

The remaining weakness is that the bounding boxes can only be as accurate as the window positions, and an object rarely lines up exactly with one window.

## YOLO Algorithm

The **YOLO** algorithm, for "You Only Look Once", gets accurate bounding boxes while keeping the single-pass efficiency of the convolutional implementation.[^yolo] It runs fast enough for real-time video.

### The Grid

YOLO places a grid over the image, say $19 \times 19$, and combines the convolutional sliding window idea with classification and localization: for every grid cell, the network outputs a localization vector $y = (p_c, b_x, b_y, b_h, b_w, c_1, c_2, c_3)$. With 3 classes, the output is a $19 \times 19 \times 8$ volume, produced in a single pass by a ConvNet whose filter sizes and strides are chosen to give that output shape.

Each object is assigned to exactly one cell: the cell that contains the center of its bounding box. That cell is responsible for detecting it, even if the object extends over many other cells.

```{figure} ../_static/img/convnets/yolo_grid.svg
:alt: A drawing of a cat with a 3 by 3 grid over it and a bounding box around the cat. The center cell, which contains the box's center point, is highlighted. A caption lists the cell-relative coordinates b x 0.42, b y 0.62, b w 1.99, and b h 1.45.
:width: 60%

The cell that contains the center of the box is responsible for the object. The box is encoded relative to that cell, so $b_w$ and $b_h$ can be larger than 1.
```

The box is encoded relative to its grid cell, with $(0, 0)$ at the cell's top left and $(1, 1)$ at its bottom right. Then $b_x$ and $b_y$, the box's center, are between 0 and 1, since the center lies inside the cell. $b_w$ and $b_h$ are measured as fractions of the cell size, so they can be greater than 1 for a box that is bigger than the cell.

### Intersection over Union

To judge whether a predicted box is good, compare it with the true box using **intersection over union**, or **IoU**:

$$
\text{IoU} = \frac{\text{area of the intersection}}{\text{area of the union}}.
$$

```{figure} ../_static/img/convnets/iou.svg
:alt: A predicted box of area 12 and a ground truth box of area 16 overlapping in a region of area 4. IoU equals 4 divided by 12 plus 16 minus 4, which is 0.17.
:width: 60%

The union is the total area covered by either box. Its area is the sum of the two areas minus the intersection, which would otherwise be counted twice.
```

An IoU of 1 means the boxes match perfectly, and 0 means they do not overlap. By convention, a prediction counts as correct if its IoU with the true box is at least 0.5. That threshold is just a convention: a stricter application can require 0.6 or 0.7, but going lower than 0.5 is rare.

**Example: Computing IoU.** Boxes are given by their corners $(x_1, y_1, x_2, y_2)$. The result matches torchvision's built-in function:

```python
import torch
from torchvision.ops import box_iou


def iou(box_a, box_b):
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


a, b = [1, 1, 5, 4], [3, 2, 7, 6]
print(f"{iou(a, b):.4f}", f"{box_iou(torch.tensor([a], dtype=torch.float), torch.tensor([b], dtype=torch.float)).item():.4f}")
```

```text
0.1667 0.1667
```

### Non-Max Suppression

In practice, several neighboring cells may each think they contain the center of the same object, and each outputs a box for it. **Non-max suppression** keeps only the best box for each object:

1. Discard every box whose $p_c$ is at most a threshold, such as 0.6.
2. Of the remaining boxes, pick the one with the largest $p_c$ and keep it as a prediction.
3. Discard every remaining box whose IoU with the kept box is at least 0.5: it is probably a duplicate of the same object.
4. Repeat from step 2 until no boxes are left.

```{figure} ../_static/img/convnets/nms.svg
:alt: Left, six predicted boxes with scores: three overlapping boxes around one object scoring 0.90, 0.75, and 0.60, two overlapping boxes around another scoring 0.80 and 0.70, and a small box scoring 0.30. Right, after non-max suppression, only the 0.90 and 0.80 boxes remain.
:width: 100%

Non-max suppression keeps the highest-scoring box for each object and removes the duplicates that overlap it.
```

With several classes, run non-max suppression separately for each class.

**Example: Non-Max Suppression.** The result matches torchvision's `nms`:

```python
import torch
from torchvision.ops import box_iou, nms


def non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.6):
    keep_mask = scores > score_threshold
    boxes, scores = boxes[keep_mask], scores[keep_mask]
    order = scores.argsort(descending=True)
    kept = []
    while order.numel() > 0:
        best = order[0]
        kept.append(int(best))
        overlaps = box_iou(boxes[best].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][overlaps < iou_threshold]
    return boxes[kept], scores[kept]


boxes = torch.tensor([[10, 10, 50, 60], [12, 14, 52, 58], [8, 8, 46, 62],
                      [70, 20, 110, 70], [72, 18, 108, 66], [30, 70, 40, 80]], dtype=torch.float)
scores = torch.tensor([0.9, 0.75, 0.6, 0.8, 0.7, 0.3])

print(non_max_suppression(boxes, scores)[1])
confident = scores > 0.6
print(scores[confident][nms(boxes[confident], scores[confident], iou_threshold=0.5)])
```

```text
tensor([0.9000, 0.8000])
tensor([0.9000, 0.8000])
```

### Anchor Boxes

So far, each grid cell can detect only one object. **Anchor boxes** let a cell detect several, by giving each cell a few predefined box shapes and one output vector per shape.

```{figure} ../_static/img/convnets/anchor_boxes.svg
:alt: Left, two anchor shapes: a tall narrow box and a short wide box. Right, a person and a car whose centers fall in the same grid cell. The tall person box matches the first anchor and the wide car box matches the second.
:width: 90%

A tall anchor for objects like people and a wide anchor for objects like cars. Two objects centered in the same cell can each be assigned to a different anchor.
```

With two anchor boxes, the output for each cell doubles: one vector of $(p_c, b_x, b_y, b_h, b_w, c_1, c_2, c_3)$ per anchor, 16 numbers in all. Each object in a training image is now assigned to the cell containing its center, as before, and to the anchor box in that cell whose shape has the highest IoU with the object's box.

The method has limits. With two anchors, a cell still cannot handle three objects, and two objects with the same shape in the same cell compete for one anchor. With a fine grid such as $19 \times 19$, both cases are rare. The anchor shapes can be chosen by hand, or by running k-means clustering on the shapes of the boxes in the training set, so that the anchors match the objects you actually see.

### Putting YOLO Together

With a $3 \times 3$ grid, 3 classes, and 2 anchor boxes, the network outputs a $3 \times 3 \times 2 \times 8$ volume, where 8 is 5 numbers for $p_c$ and the box plus 3 class scores.

1. Run the ConvNet once on the image. Each cell outputs two predicted boxes.
2. Discard the boxes with a low $p_c$.
3. For each class, run non-max suppression on the remaining boxes to get the final predictions.

**Example: Shaping YOLO's Output.** A ConvNet produces $2 \times 8 = 16$ channels per cell; reshaping gives one 8-number vector per anchor box per cell:

```python
import torch

grid, anchors, classes = 19, 2, 3
output = torch.randn(1, anchors * (5 + classes), grid, grid)
y = output.view(1, anchors, 5 + classes, grid, grid).permute(0, 3, 4, 1, 2)
print(tuple(y.shape))
```

```text
(1, 19, 19, 2, 8)
```

### Region Proposals

A different family of detectors first proposes a few regions likely to contain objects, and only classifies those, instead of considering every position.

- **R-CNN**, for "regions with ConvNets", runs a segmentation algorithm to find blobs that might be objects, draws a box around each one, and classifies each proposed region separately, outputting a label and a refined bounding box. It is slow, because each region needs its own run of the ConvNet.
- **Fast R-CNN** classifies all the proposed regions at once with a convolutional implementation of sliding windows.
- **Faster R-CNN** replaces the segmentation algorithm with a ConvNet that proposes the regions too.

Even Faster R-CNN is usually slower than YOLO.

## Face Recognition

Face recognition systems solve one of two problems.

- **Face verification** is a one-to-one problem. Given an image and a claimed identity, such as an employee's badge, decide whether the image shows that person.
- **Face recognition** is a one-to-many problem. Given an image and a database of $K$ people, decide which of them it is, if any. It is much harder: a verification system that is 99% accurate makes 1% mistakes per comparison, which adds up quickly when every image is compared with 100 people.

### One-Shot Learning

Face recognition has to work from very little data: often a single photo of each person. This is called **one-shot learning**. A ConvNet classifier with one softmax output per person would not work: one example per class is far too little to train it, and the network would have to be retrained every time someone joins.

Instead, learn a **similarity function** $d(\text{img}_1, \text{img}_2)$, the degree of difference between two images. If $d$ is at most a threshold $\tau$, the images show the same person; otherwise, different people. To recognize someone, compare their image with every photo in the database. Adding a new person only means adding their photo.

### Siamese Networks

A **Siamese network** learns the similarity function.[^deepface] It runs each image through the same ConvNet, and takes the output of the last hidden layer, say 128 numbers, as an **encoding** of the image, $f\left(x^{(i)}\right)$. The distance between two images is the squared distance between their encodings:

$$
d\left(x^{(i)}, x^{(j)}\right) = \left\| f\left(x^{(i)}\right) - f\left(x^{(j)}\right) \right\|_2^2.
$$

The goal of training is to learn parameters such that the distance is small when the two images show the same person and large when they show different people.

### The Triplet Loss

The **triplet loss** turns that goal into an objective.[^facenet] Each training example is a triplet of images: an **anchor** $A$, a **positive** $P$, another image of the same person, and a **negative** $N$, an image of someone else.

```{figure} ../_static/img/convnets/triplet.svg
:alt: An anchor image, a positive image of the same person, and a negative image of another person go through the same ConvNet. In the space of encodings, the positive's encoding is close to the anchor's and the negative's is far away, outside a dashed circle.
:width: 100%

The three images of a triplet go through the same network. Training pulls the positive's encoding toward the anchor's and pushes the negative's away.
```

We want the anchor to be closer to the positive than to the negative, by at least a margin $\alpha$:

$$
d(A, P) - d(A, N) + \alpha \leq 0.
$$

Without the margin, the network could satisfy $d(A, P) \leq d(A, N)$ trivially by outputting the same encoding for every image, making both distances zero. The margin forces a real gap between the two. The loss for one triplet is

$$
\L(A, P, N) = \max\left(d(A, P) - d(A, N) + \alpha, \; 0\right),
$$

which is zero once the triplet satisfies the margin, and positive otherwise. The cost sums the loss over all the triplets.

Training needs several pictures of each person, on the order of 10 on average, to form anchor-positive pairs. It also needs well-chosen triplets. With triplets chosen at random, the negative is usually so different from the anchor that the margin is satisfied automatically and the network learns nothing. Instead, choose **hard triplets**, where the negative looks similar to the anchor, so that $d(A, N)$ is close to $d(A, P)$ and the network has to work to separate them.

**Example: The Triplet Loss.** An easy triplet already satisfies the margin and contributes nothing; a hard one, with a negative close to the anchor, contributes to the loss:

```python
import torch


def triplet_loss(f_a, f_p, f_n, alpha=0.2):
    d_pos = ((f_a - f_p) ** 2).sum(dim=1)
    d_neg = ((f_a - f_n) ** 2).sum(dim=1)
    return torch.clamp(d_pos - d_neg + alpha, min=0).mean()


f_a = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
f_p = torch.tensor([[0.1, 0.9], [0.1, 0.9]])
f_n = torch.tensor([[1.0, 0.0], [0.1, 0.8]])
print(f"easy triplet {triplet_loss(f_a[:1], f_p[:1], f_n[:1]):.3f}, hard triplet {triplet_loss(f_a[1:], f_p[1:], f_n[1:]):.3f}")
```

```text
easy triplet 0.000, hard triplet 0.170
```

## Face Verification and Binary Classification

The triplet loss is not the only way to train a Siamese network. Face verification can also be posed as binary classification: given two images, output 1 if they show the same person and 0 otherwise.

Run both images through the Siamese network, and feed the element-wise differences between their encodings into a logistic regression unit:

$$
\hat{y} = \sigma\left( \sum_{k=1}^{N_f} w_k \left| f\left(x^{(i)}\right)_k - f\left(x^{(j)}\right)_k \right| + b \right),
$$

where $N_f$ is the number of numbers in each encoding, $k$ indexes them, and $w_k$ and $b$ are the logistic unit's parameters, trained along with the ConvNet on pairs of matching and mismatching images. A variant replaces the absolute difference with the **chi-squared similarity**, $\frac{\left(f(x^{(i)})_k - f(x^{(j)})_k\right)^2}{f(x^{(i)})_k + f(x^{(j)})_k}$.

```{admonition} You Should Know
In a deployed system, compute the encoding of every employee's photo once and store it. When someone walks up to the camera, only their new image needs to go through the network, and it is compared with the stored encodings. The system does not even need to keep the original photos.
```

## Neural Style Transfer

**Neural style transfer** takes a **content image** $C$, such as a photo, and a **style image** $S$, such as a painting, and generates a new image $G$ that shows the content of $C$ painted in the style of $S$.

### What Deep ConvNets Learn

Style transfer relies on what the layers of a trained ConvNet detect. One way to find out is to pick a unit in some layer, run many images through the network, and look at the image patches that activate that unit most strongly.[^visualizing] In the first layer, units respond to simple things like edges at a particular angle or a particular color. Units in deeper layers see larger regions of the image and respond to increasingly complex features: textures, then parts of objects such as wheels or eyes, then whole objects such as dogs or faces.

### The Cost Function

Style transfer defines a cost function on the generated image itself:[^gatys]

$$
J(G) = \alpha \, J_{\text{content}}(C, G) + \beta \, J_{\text{style}}(S, G),
$$

where $J_{\text{content}}$ measures how different the content of $G$ is from that of $C$, $J_{\text{style}}$ measures how different the style of $G$ is from that of $S$, and $\alpha$ and $\beta$ set their relative weights. To generate the image:

1. Initialize $G$ randomly, for example as a $100 \times 100 \times 3$ image of random noise.
2. Use gradient descent to minimize $J(G)$, updating the pixels of $G$ as $G \leftarrow G - \frac{\partial J}{\partial G}$, scaled by a learning rate.

The network's weights stay fixed throughout. What gradient descent changes is the image.

### The Content Cost

Run $C$ and $G$ through a pre-trained ConvNet, such as VGG, and take the activations of a layer $l$ somewhere in the middle: not so shallow that it only compares pixels, not so deep that it only checks whether both images contain, say, a dog. Call them $a^{[l] (C)}$ and $a^{[l] (G)}$. If they are similar, the two images have similar content. The content cost is

$$
J_{\text{content}}(C, G) = \frac{1}{4 \, n_H \, n_W \, n_C} \sum_{\text{all entries}} \left(a^{[l] (C)} - a^{[l] (G)}\right)^2,
$$

the sum of squared differences over all the entries of the two $n_H \times n_W \times n_C$ volumes, scaled by a normalization constant.

### The Style Cost

Style is captured by **correlations between channels**. Each channel of layer $l$ detects some feature, such as vertical stripes or a patch of orange. In a painting with a strong style, certain features tend to occur together: wherever there are brushstrokes of one kind, there is a particular color. How often two channels are active at the same places describes the texture of the image, regardless of what objects are where.

These correlations are collected in the **style matrix**, also called the **Gram matrix**. For layer $l$, with $a^{[l]}_{ijk}$ the activation at height $i$, width $j$, and channel $k$, it is the $n_C \times n_C$ matrix

$$
G^{[l]}_{kk'} = \sum_{i=1}^{n_H} \sum_{j=1}^{n_W} a^{[l]}_{ijk} \, a^{[l]}_{ijk'},
$$

where entry $(k, k')$ is large when channels $k$ and $k'$ are active at the same positions.

```{figure} ../_static/img/convnets/gram.svg
:alt: Two textures with their Gram matrices over four channels named vertical, horizontal, diagonal, and blob. For stripes, the vertical channel's entry is 1.00 and the horizontal channel's entries are all 0. For the checkerboard, both the vertical and the horizontal channels have large entries.
:width: 80%

Gram matrices of two textures, computed from four simple filters. Stripes activate the vertical edge channel strongly and the horizontal one not at all. A checkerboard activates both. The Gram matrix captures this difference without recording where any edge is.
```

Compute the style matrix for both the style image, $G^{[l] (S)}$, and the generated image, $G^{[l] (G)}$. The style cost for layer $l$ is the squared Frobenius norm of their difference, with a normalization constant:

$$
J^{[l]}_{\text{style}}(S, G) = \frac{1}{\left(2 \, n_H^{[l]} n_W^{[l]} n_C^{[l]}\right)^2} \sum_{k} \sum_{k'} \left(G^{[l] (S)}_{kk'} - G^{[l] (G)}_{kk'}\right)^2.
$$

Style transfer works better when it matches the style at several layers, from shallow ones that capture fine textures to deeper ones that capture larger patterns. The full style cost combines them, with a weight $\lambda^{[l]}$ for each layer:

$$
J_{\text{style}}(S, G) = \sum_{l} \lambda^{[l]} J^{[l]}_{\text{style}}(S, G).
$$

**Example: The Gram Matrix and the Two Costs.** PyTorch volumes put the channels first, so a layer's activations are reshaped into an $n_C \times (n_H n_W)$ matrix, and the Gram matrix is that matrix times its transpose:

```python
import torch


def gram_matrix(a):
    n_c, n_h, n_w = a.shape
    a = a.reshape(n_c, n_h * n_w)
    return a @ a.T


def content_cost(a_c, a_g):
    n_c, n_h, n_w = a_c.shape
    return ((a_c - a_g) ** 2).sum() / (4 * n_h * n_w * n_c)


def style_cost(a_s, a_g):
    n_c, n_h, n_w = a_s.shape
    return ((gram_matrix(a_s) - gram_matrix(a_g)) ** 2).sum() / (2 * n_h * n_w * n_c) ** 2


torch.manual_seed(0)
a_s, a_g = torch.rand(64, 14, 14), torch.rand(64, 14, 14)
print(tuple(gram_matrix(a_s).shape), f"style cost {style_cost(a_s, a_g):.6f}", f"content cost {content_cost(a_s, a_g):.6f}")
```

```text
(64, 64) style cost 0.000097 content cost 0.041533
```

## Recap

In this part, you learned how convolutions detect local patterns with a small number of shared parameters, and how convolutional, pooling, and fully connected layers combine into a ConvNet. You traced the history of ConvNet design from LeNet-5 through AlexNet and VGG-16 to the ideas behind modern networks: residual connections that make very deep networks trainable, $1 \times 1$ convolutions that control the number of channels cheaply, and inception modules that let the network choose its own filter sizes. You saw why transfer learning and data augmentation matter so much in computer vision. You followed object detection from sliding windows to YOLO, with IoU, non-max suppression, and anchor boxes. Finally, you saw how Siamese networks and the triplet loss solve face recognition from a single photo, and how neural style transfer turns a ConvNet's features into art. Part 5 turns from images to sequences: text, speech, and other data that unfolds over time.

[^sefidian]: The topics and their order follow Amir Masoud Sefidian, [Coursera Deep Learning Specialization Notes: Convolutional Neural Networks](https://www.sefidian.com/wp-content/uploads/2022/11/Convolutional-Neural-Networks-Notes.pdf), 2022, which summarizes the fourth course of Andrew Ng's Deep Learning Specialization.

[^lenet]: LeCun, Bottou, Bengio, and Haffner, [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791), 1998.

[^alexnet]: Krizhevsky, Sutskever, and Hinton, [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), 2012.

[^vgg]: Simonyan and Zisserman, [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556), 2014.

[^resnet]: He, Zhang, Ren, and Sun, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015.

[^inception]: Szegedy and others, [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), 2014.

[^yolo]: Redmon, Divvala, Girshick, and Farhadi, [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640), 2015.

[^deepface]: Taigman, Yang, Ranzato, and Wolf, [DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://doi.org/10.1109/CVPR.2014.220), 2014.

[^facenet]: Schroff, Kalenichenko, and Philbin, [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832), 2015.

[^visualizing]: Zeiler and Fergus, [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901), 2013.

[^gatys]: Gatys, Ecker, and Bethge, [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), 2015.
