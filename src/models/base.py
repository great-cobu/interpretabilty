"""Shared base classes, image validation, synthetic inspection, and CLI helpers for MNIST models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from models.MNIST.config import Config, config_dict, load_config
from models.MNIST.data import resolve_accelerator


def validate_mnist_images(images: torch.Tensor) -> torch.Tensor:
    """Validate and normalize an image batch to shape [N, 1, 28, 28]."""
    if images.ndim == 3 and tuple(images.shape[1:]) == (28, 28):
        images = images.unsqueeze(1)
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28) or images.shape[0] == 0:
        raise ValueError("Expected a nonempty image batch shaped [N, 1, 28, 28]")
    if not images.is_floating_point():
        raise ValueError("Images must be floating-point tensors normalized by the data pipeline")
    return images


class BaseMNISTClassifier(pl.LightningModule):
    """Base LightningModule sharing phase accuracy metrics, step logic, and Adam optimization."""

    network: nn.Module

    def __init__(self, network: nn.Module, config: dict | Config | None = None):
        super().__init__()
        if isinstance(config, Config):
            self.config = config
        elif config is not None:
            self.config = load_config(base=config, use_environment=False)
        else:
            self.config = Config()
        self.save_hyperparameters({"config": config_dict(self.config)})
        self.network = network
        self.accuracy = nn.ModuleDict({
            f"{phase}_accuracy": MulticlassAccuracy(num_classes=10, average="micro")
            for phase in ("train", "val", "test")
        })

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

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


def inspect_mnist_network(
    model: nn.Module,
    config: Config,
    output: Path,
    *,
    device: str | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """Trace layer outputs and check gradient flow on synthetic images."""
    resolved_device = device or resolve_accelerator(config.trainer.accelerator)
    model = model.to(resolved_device)
    images = torch.rand(config.data.batch_size, 1, 28, 28)
    images = ((images - config.data.mean) / config.data.std).to(resolved_device)
    labels = (torch.arange(config.data.batch_size) % 10).to(resolved_device)

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
        **(extra_fields or {}),
        "accelerator": resolved_device,
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


def add_common_cli_arguments(sub: argparse.ArgumentParser) -> None:
    """Add shared data, trainer, optimizer, and tracking CLI flags to a subparser."""
    sub.add_argument("--limit-train-batches", dest="trainer.limit_train_batches", type=int)
    sub.add_argument("--limit-val-batches", dest="trainer.limit_val_batches", type=int)
    sub.add_argument("--config", help="YAML configuration path; otherwise built-in defaults")
    sub.add_argument("--output-dir", help="Output directory; default: unique path in runs/")
    sub.add_argument("--data-dir", dest="data.directory")
    sub.add_argument("--download", dest="data.download", action=argparse.BooleanOptionalAction, default=None)
    sub.add_argument("--batch-size", dest="data.batch_size", type=int)
    sub.add_argument("--num-workers", dest="data.num_workers", type=int)
    sub.add_argument("--split-seed", dest="data.split_seed", type=int)
    sub.add_argument("--split-indices", dest="data.split_indices")
    sub.add_argument("--seed", dest="trainer.seed", type=int)
    sub.add_argument("--max-epochs", dest="trainer.max_epochs", type=int)
    sub.add_argument("--accelerator", dest="trainer.accelerator", choices=("auto", "cpu", "cuda", "mps"))
    sub.add_argument("--learning-rate", dest="optimizer.learning_rate", type=float)
    sub.add_argument("--tracking-enabled", dest="tracking.enabled", action=argparse.BooleanOptionalAction, default=None)
    sub.add_argument("--tracking-uri", dest="tracking.uri")


def load_checkpoint_base_config(ckpt_file: str | Path) -> dict | None:
    """Extract the saved config dictionary from a Lightning checkpoint for resume/test."""
    saved = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    base_cfg = saved.get("hyper_parameters", {}).get("config")
    if isinstance(base_cfg, dict):
        return dict(base_cfg, output_dir=None)
    return None
