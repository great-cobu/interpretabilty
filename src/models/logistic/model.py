"""Multinomial Logistic Regression for MNIST with glass-box interpretability hooks."""

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from models.MNIST.config import Config
from models.MNIST.data import resolve_accelerator
from models.base import BaseMNISTClassifier, inspect_mnist_network, validate_mnist_images


class LogisticRegression(nn.Module):
    """Map floating-point images [N, 1, 28, 28] to ten unnormalized logits via Flatten -> Linear(784, 10).

    Provides exact glass-box interpretability hooks:
    - `get_feature_maps(images)`: intermediate activations (`flatten`, `logits`)
    - `get_class_templates()`: learned spatial weight templates [10, 28, 28]
    - `get_pixel_attribution(images, target_class)`: exact pixel-wise contributions x * W_c [N, 1, 28, 28]
    """

    def __init__(self, in_features: int = 28 * 28, num_classes: int = 10, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.layers = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten(start_dim=1)),
            ("logits", nn.Linear(in_features, num_classes, bias=bias)),
        ]))

    def _validate_images(self, images: torch.Tensor) -> torch.Tensor:
        return validate_mnist_images(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self._validate_images(images)
        return self.layers(images)

    def get_feature_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract intermediate representations (`flatten`, `logits`)."""
        x = self._validate_images(images)
        features: dict[str, torch.Tensor] = {}
        for name, layer in self.layers.named_children():
            x = layer(x)
            features[name] = x
        return features

    def get_class_templates(self) -> torch.Tensor:
        """Return the learned linear weight matrix reshaped to spatial templates [10, 28, 28]."""
        weight = self.layers.logits.weight.detach()
        return weight.view(self.num_classes, 28, 28)

    def get_pixel_attribution(
        self,
        images: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute exact linear pixel attribution A_i = x_i * W_{c_i} shaped [N, 1, 28, 28].

        Summing A_i over (1, 28, 28) plus bias b_{c_i} equals the exact logit for class c_i.
        """
        x = self._validate_images(images)
        batch_size = x.shape[0]
        if target_class is None:
            with torch.no_grad():
                target_idx = self.forward(x).argmax(dim=1)
        elif isinstance(target_class, int):
            target_idx = torch.full((batch_size,), target_class, dtype=torch.long, device=x.device)
        else:
            target_idx = target_class.to(device=x.device, dtype=torch.long)

        templates = self.layers.logits.weight.detach()[target_idx.clone()].view(batch_size, 1, 28, 28)
        return x.detach() * templates


class LogisticClassifier(BaseMNISTClassifier):
    """LightningModule wrapping LogisticRegression, phase accuracy metrics, and Adam optimization."""

    def __init__(self, config: dict | Config | None = None):
        super().__init__(network=LogisticRegression(), config=config)

    def get_class_templates(self) -> torch.Tensor:
        return self.network.get_class_templates()

    def get_pixel_attribution(
        self,
        images: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.network.get_pixel_attribution(images, target_class=target_class)


def inspect_model(config: Config, output: Path) -> dict:
    """Trace layer outputs and check gradient flow on synthetic images for LogisticRegression."""
    device = resolve_accelerator(config.trainer.accelerator)
    torch.manual_seed(config.trainer.seed)
    model = LogisticRegression()
    return inspect_mnist_network(
        model,
        config,
        output,
        device=device,
        extra_fields={"model": "LogisticRegression"},
    )
