"""Offline dataset fixtures: no downloads are performed by routine tests."""

import pytest
from PIL import Image
from torch.utils.data import Dataset

from models.MNIST import data


class TinyMNIST(Dataset):
    def __init__(self, root, train, transform, download):
        self.transform = transform
        self.size = 23 if train else 11

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = Image.new("L", (28, 28), color=index % 256)
        return self.transform(image), index % 10


@pytest.fixture
def tiny_mnist(monkeypatch):
    monkeypatch.setattr(data.datasets, "MNIST", TinyMNIST)
    monkeypatch.setattr(data.MNISTDataModule, "train_size", 18)
    monkeypatch.setattr(data.MNISTDataModule, "val_size", 5)


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Check the existing real MNIST cache")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="Use --run-integration with an existing MNIST cache")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
