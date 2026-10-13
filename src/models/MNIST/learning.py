"""Training, resume, and standalone testing workflows for stages 5–6."""

import csv
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import torch

from models.MNIST.checkpoints import RunCheckpoint, read_checkpoint, sha256
from models.MNIST.config import Config, config_dict, sanitize
from models.MNIST.data import MNISTDataModule, resolve_accelerator
from models.MNIST.module import MNISTClassifier


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "lightning", "torchmetrics", "numpy")
    }}


def trainer_options(config: Config) -> dict:
    return dict(
        accelerator=resolve_accelerator(config.trainer.accelerator), devices=1,
        precision=config.trainer.precision, deterministic=config.trainer.deterministic,
        enable_progress_bar=False, enable_model_summary=False,
    )


def learning_curves(metrics_path: Path, destination: Path) -> None:
    """Render epoch aggregates, joining Lightning's separate train/val CSV rows."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    with metrics_path.open() as stream:
        rows = list(csv.DictReader(stream))
    fig = Figure(figsize=(9, 3.5), layout="constrained")
    FigureCanvasAgg(fig)
    for axis, metric in zip(fig.subplots(1, 2), ("loss", "accuracy")):
        for phase in ("train", "val"):
            points = [(int(float(row["epoch"])) + 1, float(row[f"{phase}/{metric}"]))
                      for row in rows if row.get(f"{phase}/{metric}") and row.get("epoch")]
            if points:
                axis.plot(*zip(*points), marker="o", label=phase)
        axis.set(xlabel="Epoch (1-based)", ylabel=metric, title=f"Training and validation {metric}")
        axis.legend()
    fig.savefig(destination, dpi=150)


def train(config: Config, output: Path, *, resume: str | None = None, saved: dict | None = None, overrides: dict | None = None) -> dict:
    pl.seed_everything(config.trainer.seed, workers=True)
    options = trainer_options(config)
    data = MNISTDataModule(config.data, accelerator=options["accelerator"])
    if saved is not None:
        # Load exact membership before setup; the checkpoint remains self-contained.
        data.config = replace(data.config, split_indices=None)
        data.load_state_dict(saved["MNISTDataModule"])
    data.prepare_data()
    data.setup("fit")
    data.save_split_indices(output / "split_indices.json")
    model = MNISTClassifier(config=config_dict(config))
    checkpoint = RunCheckpoint(output / "checkpoints")
    if resume:
        state = saved.get("callbacks", {}).get(checkpoint.state_key)
        if not state or not state.get("best_model_path"):
            raise ValueError("Resume checkpoint lacks validation checkpoint history")
        historical_best = Path(resume).parent / Path(state["best_model_path"]).name
        if not historical_best.is_file():
            raise ValueError("Resume requires the historical best.ckpt alongside the source checkpoint")
        read_checkpoint(historical_best)
        Path(checkpoint.dirpath).mkdir(parents=True, exist_ok=True)
        shutil.copy2(historical_best, Path(checkpoint.dirpath) / "best.ckpt")
    logger = CSVLogger(save_dir=str(output), name="logs", version="")
    trainer = pl.Trainer(
        **options, default_root_dir=str(output), max_epochs=config.trainer.max_epochs,
        logger=logger, callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=config.trainer.patience,
                          min_delta=0, check_finite=True, check_on_train_epoch_end=True),
            checkpoint,
        ],
        check_val_every_n_epoch=1, num_sanity_val_steps=0, log_every_n_steps=1,
        limit_train_batches=config.trainer.limit_train_batches or 1.0,
        limit_val_batches=config.trainer.limit_val_batches or 1.0,
    )
    started = time.perf_counter()
    trainer.fit(model, datamodule=data, ckpt_path=resume, weights_only=True)
    elapsed = time.perf_counter() - started
    logger.save()
    if not checkpoint.best_model_path or not Path(checkpoint.last_model_path).is_file():
        raise RuntimeError("Training did not produce best and last checkpoints")
    best = read_checkpoint(checkpoint.best_model_path)
    last = read_checkpoint(checkpoint.last_model_path)
    learning_curves(Path(logger.log_dir) / "metrics.csv", output / "learning_curves.png")
    report = {
        "elapsed_seconds": elapsed, "completed_epochs": last["epoch"] + 1,
        "global_step": last["global_step"], "best_epoch": best["epoch"] + 1,
        "best_val_loss": float(checkpoint.best_model_score),
        "best_checkpoint": checkpoint.best_model_path, "last_checkpoint": checkpoint.last_model_path,
        "accelerator": options["accelerator"], "environment": environment(),
        "smoke_run": config.trainer.limit_train_batches is not None or config.trainer.limit_val_batches is not None,
        "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
        "source_checkpoint": str(Path(resume).resolve()) if resume else None,
        "source_checkpoint_sha256": sha256(resume) if resume else None,
        "resumed_from_epoch": saved["epoch"] + 1 if saved else None,
        "resumed_from_global_step": saved["global_step"] if saved else None,
        "overrides": sanitize(overrides or {}),
    }
    (output / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_checkpoint(config: Config, output: Path, checkpoint_path: str, saved: dict) -> dict:
    """Measure a fixed checkpoint using the entire ordered test loader."""
    pl.seed_everything(config.trainer.seed, workers=True)
    options = trainer_options(config)
    # Instantiate from the saved model hyperparameters, not operational overrides.
    model = MNISTClassifier(config=saved["hyper_parameters"]["config"])
    try:
        model.load_state_dict(saved["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint weights do not match the saved CNN architecture") from exc
    data = MNISTDataModule(config.data, accelerator=options["accelerator"])
    trainer = pl.Trainer(**options, logger=False, enable_checkpointing=False, default_root_dir=str(output))
    started = time.perf_counter()
    results = trainer.test(model, datamodule=data, verbose=False)[0]
    elapsed = time.perf_counter() - started
    count = int(results["test/count"])
    if count != len(data.test_dataset):
        raise RuntimeError("Test measurement did not include every test example")
    report = {
        "test/loss": results["test/loss"], "test/accuracy": results["test/accuracy"],
        "sample_count": count, "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint": str(Path(checkpoint_path).resolve()), "config": config_dict(config),
        "accelerator": options["accelerator"], "environment": environment(), "elapsed_seconds": elapsed,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
