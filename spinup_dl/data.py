import math
import pathlib
import urllib.request

import torch

from spinup_dl.user_config import DEFAULT_DATASET_DIR

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

IMAGE_INFO = {
    "mnist": {"shape": (1, 28, 28), "classes": 10},
    "fashion_mnist": {"shape": (1, 28, 28), "classes": 10},
    "cifar10": {"shape": (3, 32, 32), "classes": 10},
    "fake": {"shape": (1, 28, 28), "classes": 10},
}


def image_datasets(name, root=None, augment=False):
    import torchvision
    from torchvision import transforms

    if name not in IMAGE_INFO:
        raise ValueError(f"unknown image dataset '{name}'; choose from {sorted(IMAGE_INFO)}")
    root = pathlib.Path(root or DEFAULT_DATASET_DIR)
    info = IMAGE_INFO[name]
    to_tensor = transforms.ToTensor()
    if name == "fake":
        train = torchvision.datasets.FakeData(size=256, image_size=info["shape"], num_classes=10, transform=to_tensor, random_offset=0)
        test = torchvision.datasets.FakeData(size=64, image_size=info["shape"], num_classes=10, transform=to_tensor, random_offset=1000)
        return train, test, info
    train_transform = to_tensor
    if augment:
        size = info["shape"][1]
        train_transform = transforms.Compose([
            transforms.RandomCrop(size, padding=size // 8),
            transforms.RandomHorizontalFlip() if name != "mnist" else transforms.Lambda(lambda x: x),
            to_tensor,
        ])
    cls = {"mnist": torchvision.datasets.MNIST, "fashion_mnist": torchvision.datasets.FashionMNIST, "cifar10": torchvision.datasets.CIFAR10}[name]
    train = cls(root, train=True, download=True, transform=train_transform)
    test = cls(root, train=False, download=True, transform=to_tensor)
    return train, test, info


class TextData:
    def __init__(self, text, test_fraction=0.1):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split = int(len(data) * (1 - test_fraction))
        self.train, self.test = data[:split], data[split:]

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def batch(self, split, batch_size, seq_len, generator=None):
        data = self.train if split == "train" else self.test
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[s:s + seq_len] for s in starts])
        y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
        return x, y


def text_dataset(name, root=None):
    root = pathlib.Path(root or DEFAULT_DATASET_DIR)
    if name == "fake":
        words = ["the cat sat on the mat. ", "a dog ran in the park. ", "we like deep learning. "]
        text = "".join(words[i % 3] for i in range(400))
        return TextData(text)
    if name == "tiny_shakespeare":
        path = root / "tiny_shakespeare.txt"
        if not path.exists():
            root.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {TINY_SHAKESPEARE_URL}", flush=True)
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
        return TextData(path.read_text(encoding="utf-8"))
    path = pathlib.Path(name)
    if path.is_file():
        return TextData(path.read_text(encoding="utf-8"))
    raise ValueError(f"unknown text dataset '{name}'; use 'tiny_shakespeare', 'fake', or a path to a text file")


def flat_dim(shape):
    return math.prod(shape)
