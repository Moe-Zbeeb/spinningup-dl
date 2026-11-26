# Convolutional Neural Network

```{contents} Table of Contents
:depth: 2
```

## Background

A **convolutional neural network**, or **ConvNet**, replaces the fully connected layers of an MLP with convolutions: small filters that slide across the image, detecting the same pattern wherever it appears. [Part 4](../intro/convnets.md) explains why this works so well on images: [parameter sharing and sparse connections](../intro/convnets.md#why-convolutions) give far fewer parameters, and the network's features move along with the objects in the image.

The companion code includes two ConvNets.

- **LeNet** follows the classic pattern of [LeNet-5](../intro/convnets.md#historical-and-classic-architectures): two blocks of convolution, ReLU, and max pooling, then three fully connected layers.
- **ResNet** is a small [residual network](../intro/convnets.md#resnet): a convolutional stem, three stages of residual blocks with batch normalization that halve the height and width and widen the channels, global average pooling, and a single fully connected layer.

### Quick Facts

- Input: an image of size $n_H \times n_W \times n_C$. Output: one score per class.
- It assumes that nearby pixels are related and that a pattern means the same thing anywhere in the image.
- LeNet trains in seconds and is a good first ConvNet. The ResNet is slower but far stronger on harder datasets such as CIFAR-10.
- Use it for images and other grid-shaped data, such as spectrograms.

### Key Equations

A convolutional layer with filters $W^{[l]}$ computes, for each filter $k$,

$$
z^{[l]}_{ijk} = \sum_{u=1}^{f} \sum_{v=1}^{f} \sum_{c=1}^{n_C^{[l-1]}} W^{[l]}_{uvck} \, a^{[l-1]}_{s(i-1)+u, \; s(j-1)+v, \; c} + b^{[l]}_k,
$$ (eq-cnn-conv)

where $f$ is the filter size and $s$ the stride, followed by an activation, $a^{[l]} = g\left(z^{[l]}\right)$. With padding $p$, the output height is $\left\lfloor \frac{n_H + 2p - f}{s} + 1 \right\rfloor$, and likewise for the width.

A residual block adds its input back before the last activation:

$$
a^{[l+2]} = g\left(z^{[l+2]} + W_s a^{[l]}\right),
$$ (eq-cnn-residual)

where $W_s$ is the identity when the shapes match and a $1 \times 1$ convolution with batch normalization when the block changes the number of channels or the resolution. Both networks are trained with the cross-entropy loss, as for the [MLP](mlp.md).

### Pseudocode

```text
Algorithm 1  Forward pass of the ResNet

Input: a batch of images X
A ← ReLU(BatchNorm(Conv3x3(X)))
for each stage with width w do
    for each residual block in the stage do
        Z ← BatchNorm(Conv3x3(ReLU(BatchNorm(Conv3x3(A)))))
        A ← ReLU(Z + shortcut(A)), equation (2)
    end for
end for
A ← global average pool of A over height and width
return the class scores, a fully connected layer applied to A
```

## Documentation

```python
spinup_dl.cnn(arch="lenet", dataset="mnist", widths=(32, 64, 128), blocks_per_stage=2, augment=False, epochs=10,
              batch_size=128, lr=1e-3, weight_decay=0.0, optimizer="adam", seed=0, device="auto", num_workers=0,
              logger_kwargs=dict())
```

Trains a ConvNet classifier and returns it.

| Argument | Meaning |
| --- | --- |
| `arch` | `lenet` or `resnet` |
| `dataset` | `mnist`, `fashion_mnist`, `cifar10`, or `fake` |
| `widths` | ResNet only: the number of channels in each stage. Each stage after the first halves the height and width |
| `blocks_per_stage` | ResNet only: residual blocks per stage |
| `augment` | Train on randomly shifted and, except for MNIST, mirrored images, as in [data augmentation](../intro/convnets.md#data-augmentation) |
| `epochs`, `batch_size`, `lr`, `weight_decay`, `optimizer`, `seed`, `device`, `num_workers`, `logger_kwargs` | As for the [MLP](mlp.md#documentation) |

The model classes are `spinup_dl.models.cnn.LeNet(in_channels, image_size, num_classes)` and `spinup_dl.models.cnn.ResNet(in_channels, num_classes, widths, blocks_per_stage)`.

### What to Expect

```bash
python -m spinup_dl.run cnn --arch lenet --epochs 3
python -m spinup_dl.run cnn --arch resnet --dataset cifar10 --augment True --epochs 1
```

LeNet reached 98.2% test accuracy on MNIST after 3 epochs, in 8 seconds. The ResNet reached 55.8% on CIFAR-10 after a single epoch, in 30 seconds, and keeps improving with more epochs; CIFAR-10 is a much harder dataset than MNIST, so it needs far longer training.

### Things to Try

- Train the MLP and LeNet on Fashion-MNIST with the same number of epochs, and compare their accuracy and their number of parameters.
- Train the ResNet on CIFAR-10 with and without `--augment True` for 10 epochs each, and plot `TrainAccuracy` against `TestAccuracy` for both.
- Try `--blocks_per_stage 1 2 4` to see how depth affects a residual network.

## References

### Relevant Papers

- [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791), LeCun, Bottou, Bengio, and Haffner, 1998.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He, Zhang, Ren, and Sun, 2015.
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), Ioffe and Szegedy, 2015.

### Other Public Implementations

- [PyTorch examples: MNIST](https://github.com/pytorch/examples/tree/main/mnist), a small ConvNet trained on MNIST.
- [torchvision's ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py), the full-size ResNets, with pre-trained weights for [transfer learning](../intro/convnets.md#using-transfer-learning).
