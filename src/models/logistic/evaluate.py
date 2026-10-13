"""Testing, evaluation, confusion matrix, and inference helpers for LogisticClassifier."""

import json
from pathlib import Path
import time

import lightning.pytorch as pl
import torch

from models.MNIST.config import Config, config_dict
from models.MNIST.data import MNISTDataModule
from models.MNIST.evaluate import confusion_matrix
from models.MNIST.train import environment, sha256, trainer_options
from models.logistic.model import LogisticClassifier


def load_classifier(checkpoint_path: str | Path, device: str = "cpu") -> LogisticClassifier:
    """Load a LogisticClassifier from a Lightning checkpoint file."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return LogisticClassifier.load_from_checkpoint(
        str(path),
        map_location=device,
        weights_only=True,
    ).eval()


def test_checkpoint(config: Config, output: Path, checkpoint_path: str | Path) -> dict:
    """Measure a fixed LogisticClassifier checkpoint on the entire ordered test split."""
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
        "model": "LogisticRegression",
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


__all__ = [
    "load_classifier",
    "test_checkpoint",
    "confusion_matrix",
    "predict",
]
