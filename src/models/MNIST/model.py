"""MNIST CNN architecture, PyTorch Lightning module, interpretability hooks, and synthetic inspection."""

from collections import OrderedDict
import json
import math
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from models.MNIST.config import Config, config_dict, load_config
from models.MNIST.data import resolve_accelerator


class MNISTCNN(nn.Module):
    """Map floating-point images [N, 1, 28, 28] to ten unnormalized logits.

    Includes `get_last_conv_layer()` and `get_feature_maps()` for interpretability (e.g. Grad-CAM).
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
        self.layers = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 32, kernel_size=3, padding=1)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("conv2", nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("flatten", nn.Flatten(start_dim=1)),
            ("hidden", nn.Linear(64 * 7 * 7, 128)),
            ("relu3", nn.ReLU()),
            ("dropout", nn.Dropout(p=dropout)),
            ("logits", nn.Linear(128, 10)),
        ]))

    def _validate_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3 and tuple(images.shape[1:]) == (28, 28):
            images = images.unsqueeze(1)
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28) or images.shape[0] == 0:
            raise ValueError("Expected a nonempty image batch shaped [N, 1, 28, 28]")
        if not images.is_floating_point():
            raise ValueError("Images must be floating-point tensors normalized by the data pipeline")
        return images

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


class MNISTClassifier(pl.LightningModule):
    """LightningModule wrapping MNISTCNN, phase accuracy metrics, and Adam optimization."""

    def __init__(self, config: dict | Config | None = None):
        super().__init__()
        if isinstance(config, Config):
            self.config = config
        elif config is not None:
            self.config = load_config(base=config, use_environment=False)
        else:
            self.config = Config()
        self.save_hyperparameters({"config": config_dict(self.config)})
        self.network = MNISTCNN(dropout=self.config.model.dropout)
        self.accuracy = nn.ModuleDict({
            f"{phase}_accuracy": MulticlassAccuracy(num_classes=10, average="micro")
            for phase in ("train", "val", "test")
        })

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def get_last_conv_layer(self) -> nn.Module:
        return self.network.get_last_conv_layer()

    def get_feature_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.network.get_feature_maps(images)

    def _step(self, batch, phase: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {phase} loss; check inputs and optimization settings")
        metric = self.accuracy[f"{phase}_accuracy"]
        metric.update(logits, labels)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, batch_size=len(labels))
        self.log(f"{phase}/accuracy", metric, on_step=False, on_epoch=True)
        if phase == "test":
            self.log("test/count", float(len(labels)), on_step=False, on_epoch=True, reduce_fx="sum")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        return self(images)

    def on_train_epoch_start(self):
        self.accuracy["train_accuracy"].reset()

    def on_validation_epoch_start(self):
        self.accuracy["val_accuracy"].reset()

    def on_test_epoch_start(self):
        self.accuracy["test_accuracy"].reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )


def inspect_model(config: Config, output: Path) -> dict:
    """Trace layer outputs and check gradient flow on synthetic images."""
    device = resolve_accelerator(config.trainer.accelerator)
    torch.manual_seed(config.trainer.seed)
    model = MNISTCNN(dropout=config.model.dropout).to(device)
    images = torch.rand(config.data.batch_size, 1, 28, 28)
    images = ((images - config.data.mean) / config.data.std).to(device)
    labels = (torch.arange(config.data.batch_size) % 10).to(device)

    model.eval()
    with torch.no_grad():
        feature_maps = model.get_feature_maps(images)
        logits = feature_maps["logits"]

    shapes = [
        {"layer": name, "type": type(getattr(model.layers, name)).__name__, "output_shape": list(tensor.shape)}
        for name, tensor in feature_maps.items()
    ]

    model.train()
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    gradients = {
        name: {
            "finite": parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
            "norm": float(parameter.grad.norm()) if parameter.grad is not None else None,
        }
        for name, parameter in model.named_parameters()
    }
    if not torch.isfinite(logits).all() or not torch.isfinite(loss) or not all(g["finite"] for g in gradients.values()):
        raise RuntimeError("Synthetic model inspection produced nonfinite outputs, loss, or missing/nonfinite gradients")

    report = {
        "purpose": "Synthetic shape and gradient inspection; not a model-quality measurement",
        "accelerator": device,
        "seed": config.trainer.seed,
        "dropout": config.model.dropout,
        "input_shape": list(images.shape),
        "layers": shapes,
        "output_shape": list(logits.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "synthetic_cross_entropy": float(loss.detach()),
        "gradients": gradients,
        "optimizer_steps": 0,
    }
    (output / "model_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
