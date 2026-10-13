"""Models package for neural network architectures, training, and evaluation."""

from models import MNIST
from models.MNIST import (
    MNISTCNN,
    MNISTCNNConfig,
    MNISTCNNForImageClassification,
    MNISTDataCollator,
    compute_metrics,
    load_mnist_datasets,
    predict,
    test_mnist,
    train_mnist,
)


def main() -> None:
    """Entry point for models package."""
    print("Models MNIST CNN package ready.")


__all__ = [
    "MNIST",
    "MNISTCNNConfig",
    "MNISTCNN",
    "MNISTCNNForImageClassification",
    "load_mnist_datasets",
    "MNISTDataCollator",
    "train_mnist",
    "compute_metrics",
    "test_mnist",
    "predict",
    "main",
]
