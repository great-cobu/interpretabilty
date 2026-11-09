"""MNIST CNN architecture, PyTorch Lightning module, interpretability hooks, and synthetic inspection."""

from collections import OrderedDict
import math
from pathlib import Path

import torch
from torch import nn

from models.MNIST.config import VALID_ACTIVATIONS, Config, load_config
from models.MNIST.data import resolve_accelerator
from models.base import BaseMNISTClassifier, inspect_mnist_network, validate_mnist_images


class SMU(nn.Module):
    """Smooth Maximum Unit (Biswas et al., 2022) for C^infinity smooth activation maps."""

    def __init__(self, alpha: float = 0.0, mu: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.mu = float(mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            (1.0 + self.alpha) * x
            + (1.0 - self.alpha) * x * torch.erf(self.mu * (1.0 - self.alpha) * x)
        )


def _make_activation(activation: str) -> nn.Module:
    if not isinstance(activation, str) or activation not in VALID_ACTIVATIONS:
        raise ValueError("activation must be relu, softplus, or SMU")
    key = activation.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "softplus":
        return nn.Softplus()
    return SMU()


class MNISTCNN(nn.Module):
    """Map floating-point images [N, 1, 28, 28] to ten unnormalized logits.

    Includes `get_last_conv_layer()` and `get_feature_maps()` for interpretability (e.g. Grad-CAM).
    """

    def __init__(self, dropout: float = 0.2, activation: str = "relu"):
        super().__init__()
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
        _make_activation(activation)
        self.activation = activation
        self.layers = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 32, kernel_size=3, padding=1)),
            ("relu1", _make_activation(activation)),
            ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("conv2", nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ("relu2", _make_activation(activation)),
            ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("flatten", nn.Flatten(start_dim=1)),
            ("hidden", nn.Linear(64 * 7 * 7, 128)),
            ("relu3", _make_activation(activation)),
            ("dropout", nn.Dropout(p=dropout)),
            ("logits", nn.Linear(128, 10)),
        ]))

    def _validate_images(self, images: torch.Tensor) -> torch.Tensor:
        return validate_mnist_images(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self._validate_images(images)
        return self.layers(images)

    def get_last_conv_layer(self) -> nn.Module:
        """Return the final convolutional layer (`conv2`) for Grad-CAM / saliency attribution."""
        return self.layers.conv2

    def get_feature_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract intermediate activations across layers for interpretability analysis."""
        x = self._validate_images(images)
        features: dict[str, torch.Tensor] = {}
        for name, layer in self.layers.named_children():
            x = layer(x)
            features[name] = x
        return features


class MNISTClassifier(BaseMNISTClassifier):
    """LightningModule wrapping MNISTCNN, phase accuracy metrics, and Adam optimization."""

    def __init__(self, config: dict | Config | None = None):
        resolved = (
            config
            if isinstance(config, Config)
            else (load_config(base=config, use_environment=False) if config is not None else Config())
        )
        network = MNISTCNN(
            dropout=resolved.model.dropout,
            activation=resolved.model.activation,
        )
        super().__init__(network=network, config=resolved)

    def get_last_conv_layer(self) -> nn.Module:
        return self.network.get_last_conv_layer()


def inspect_model(config: Config, output: Path) -> dict:
    """Trace layer outputs and check gradient flow on synthetic images."""
    device = resolve_accelerator(config.trainer.accelerator)
    torch.manual_seed(config.trainer.seed)
    model = MNISTCNN(
        dropout=config.model.dropout,
        activation=config.model.activation,
    )
    return inspect_mnist_network(
        model,
        config,
        output,
        device=device,
        extra_fields={
            "dropout": config.model.dropout,
            "activation": config.model.activation,
        },
    )
