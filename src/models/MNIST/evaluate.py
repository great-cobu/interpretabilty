"""Testing, evaluation, confusion matrix, and inference helpers for MNIST."""

import json
from pathlib import Path
import time

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from models.MNIST.config import Config, config_dict
from models.MNIST.data import MNISTDataModule
from models.MNIST.model import MNISTClassifier
from models.MNIST.train import environment, sha256, trainer_options


def load_classifier(checkpoint_path: str | Path, device: str = "cpu") -> MNISTClassifier:
    """Load an MNISTClassifier from a Lightning checkpoint file."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return MNISTClassifier.load_from_checkpoint(
        str(path),
        map_location=device,
        weights_only=True,
    ).eval()


def test_checkpoint(config: Config, output: Path, checkpoint_path: str | Path) -> dict:
    """Measure a fixed checkpoint on the entire ordered test split."""
    pl.seed_everything(config.trainer.seed, workers=True)
    options = trainer_options(config)
    model = load_classifier(checkpoint_path, device="cpu")
    data = MNISTDataModule(config.data, accelerator=options["accelerator"])
    trainer = pl.Trainer(**options, logger=False, enable_checkpointing=False, default_root_dir=str(output))

    started = time.perf_counter()
    results = trainer.test(model, datamodule=data, verbose=False)[0]
    elapsed = time.perf_counter() - started
    count = int(results["test/count"])
    report = {
        "test/loss": float(results["test/loss"]),
        "test/accuracy": float(results["test/accuracy"]),
        "sample_count": count,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "config": config_dict(config),
        "accelerator": options["accelerator"],
        "environment": environment(),
        "elapsed_seconds": elapsed,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def confusion_matrix(model: torch.nn.Module, dataset, batch_size: int = 256) -> torch.Tensor:
    """Compute a 10x10 confusion matrix (rows: true digits, columns: predicted digits)."""
    if model.training:
        raise ValueError("Confusion-matrix inference requires an evaluation-mode model")
    counts = torch.zeros((10, 10), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            predictions = model(images).argmax(dim=1)
            counts += torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    return counts


def predict(model: torch.nn.Module | str | Path, images: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run inference on normalized image batch [N, 1, 28, 28] or [N, 28, 28]."""
    if isinstance(model, (str, Path)):
        model = load_classifier(model)
    model.eval()
    if images.ndim == 3:
        images = images.unsqueeze(1)
    with torch.inference_mode():
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "confidence": confidence,
        "logits": logits,
    }
