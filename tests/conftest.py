"""Pytest fixtures for synthetic and real-cached MNIST tests."""

from PIL import Image
import pytest
from torchvision import datasets


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Run checks requiring a real local MNIST cache")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="Needs a prepared local MNIST cache; pass --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class TinyMNIST:
    """In-memory MNIST-shaped fixture dataset."""

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        self.size = 18 if train else 7

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        value = int((index * 17 + (0 if self.train else 5)) % 256)
        image = Image.new("L", (28, 28), color=value)
        if self.transform is not None:
            image = self.transform(image)
        return image, index % 10


@pytest.fixture
def patch_mnist(monkeypatch):
    from models.MNIST import data

    monkeypatch.setattr(datasets, "MNIST", TinyMNIST)
    monkeypatch.setattr(data.MNISTDataModule, "train_size", 12)
    monkeypatch.setattr(data.MNISTDataModule, "val_size", 6)
    return TinyMNIST
