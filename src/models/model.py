"""Model architectures for models package."""

from models.MNIST.config import MNISTCNNConfig
from models.MNIST.model import (
    MNISTCNN,
    MNISTCNNForImageClassification,
    ModelOutput,
)

__all__ = [
    "MNISTCNNConfig",
    "MNISTCNN",
    "MNISTCNNForImageClassification",
    "ModelOutput",
]
