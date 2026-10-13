"""Multinomial Logistic Regression for MNIST with glass-box interpretability hooks."""

from collections import OrderedDict
import json
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from models.MNIST.config import Config, config_dict, load_config
from models.MNIST.data import resolve_accelerator


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


class LogisticClassifier(pl.LightningModule):
    """LightningModule wrapping LogisticRegression, phase accuracy metrics, and Adam optimization."""

    def __init__(self, config: dict | Config | None = None):
        super().__init__()
        if isinstance(config, Config):
            self.config = config
        elif config is not None:
            self.config = load_config(base=config, use_environment=False)
        else:
            self.config = Config()
        self.save_hyperparameters({"config": config_dict(self.config)})
        self.network = LogisticRegression()
        self.accuracy = nn.ModuleDict({
            f"{phase}_accuracy": MulticlassAccuracy(num_classes=10, average="micro")
            for phase in ("train", "val", "test")
        })

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def get_feature_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.network.get_feature_maps(images)

    def get_class_templates(self) -> torch.Tensor:
        return self.network.get_class_templates()

    def get_pixel_attribution(
        self,
        images: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.network.get_pixel_attribution(images, target_class=target_class)

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
    """Trace layer outputs and check gradient flow on synthetic images for LogisticRegression."""
    device = resolve_accelerator(config.trainer.accelerator)
    torch.manual_seed(config.trainer.seed)
    model = LogisticRegression().to(device)
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
        "model": "LogisticRegression",
        "accelerator": device,
        "seed": config.trainer.seed,
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
