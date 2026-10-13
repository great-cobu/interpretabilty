"""MNIST subpackage for CNN modeling, data processing, training, and testing with PyTorch Lightning."""

from typing import Any, Dict, List, Optional, Tuple, Union

from models.MNIST.config import MNISTCNNConfig, MNISTTrainingConfig
from models.MNIST.data import (
    MNIST_MEAN,
    MNIST_STD,
    MNISTDataCollator,
    MNISTDataModule,
    MNISTDataset,
    create_mnist_transform,
    load_mnist_datasets,
    preprocess_image,
)
from models.MNIST.model import (
    MNISTCNN,
    MNISTCNNForImageClassification,
    ModelOutput,
)


def train_mnist(*args: Any, **kwargs: Any) -> Any:
    """Train MNIST CNN using PyTorch Lightning."""
    from models.MNIST.train import train_mnist as _train_mnist
    return _train_mnist(*args, **kwargs)


def compute_metrics(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """Compute classification evaluation metrics."""
    from models.MNIST.train import compute_metrics as _compute_metrics
    return _compute_metrics(*args, **kwargs)


def test_mnist(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Evaluate a trained MNIST CNN on the MNIST test set."""
    from models.MNIST.test import test_mnist as _test_mnist
    return _test_mnist(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run inference on one or more MNIST images."""
    from models.MNIST.test import predict as _predict
    return _predict(*args, **kwargs)


__all__ = [
    # Config
    "MNISTCNNConfig",
    "MNISTTrainingConfig",
    # Model
    "MNISTCNN",
    "MNISTCNNForImageClassification",
    "ModelOutput",
    # Data
    "MNIST_MEAN",
    "MNIST_STD",
    "preprocess_image",
    "create_mnist_transform",
    "load_mnist_datasets",
    "MNISTDataset",
    "MNISTDataCollator",
    "MNISTDataModule",
    # Training & Testing
    "compute_metrics",
    "train_mnist",
    "test_mnist",
    "predict",
]
