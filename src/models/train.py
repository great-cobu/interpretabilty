"""Training module for models package."""

from models.MNIST.train import (
    compute_metrics,
    main,
    train_mnist,
)

__all__ = [
    "train_mnist",
    "compute_metrics",
    "main",
]

if __name__ == "__main__":
    main()
