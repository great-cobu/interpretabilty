"""Training and resume workflows for LogisticClassifier."""

from dataclasses import replace
import json
from pathlib import Path
import time
from uuid import uuid4

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from models.MNIST.config import Config, load_config, prepare_run_dir
from models.MNIST.data import MNISTDataModule
from models.MNIST.train import environment, learning_curves, sha256, trainer_options
from models.logistic.model import LogisticClassifier


def train(
    config: Config,
    output: Path,
    *,
    resume: str | Path | None = None,
) -> dict:
    """Train or resume LogisticClassifier using PyTorch Lightning."""
    pl.seed_everything(config.trainer.seed, workers=True)
    options = trainer_options(config)
    data = MNISTDataModule(config.data, accelerator=options["accelerator"])

    saved_ckpt = None
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_file():
            raise ValueError(f"Resume checkpoint does not exist: {resume_path}")
        saved_ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        if config.trainer.max_epochs <= saved_ckpt.get("epoch", 0) + 1:
            raise ValueError("Resume --max-epochs must exceed the completed epoch count")
        if "MNISTDataModule" in saved_ckpt:
            data.config = replace(data.config, split_indices=None)
            data.load_state_dict(saved_ckpt["MNISTDataModule"])

    data.prepare_data()
    data.setup("fit")
    data.save_split_indices(output / "split_indices.json")

    model = LogisticClassifier(config=config)
    ckpt_dir = output / "checkpoints"

    class _EpochCheckpoint(ModelCheckpoint):
        def on_train_epoch_end(self, trainer, pl_module):
            super().on_train_epoch_end(trainer, pl_module)
            self.last_model_path = str(Path(self.dirpath) / "last.ckpt")
            trainer.save_checkpoint(self.last_model_path, weights_only=False)
            if not self.best_model_path or not Path(self.best_model_path).is_file():
                self.best_model_path = str(Path(self.dirpath) / "best.ckpt")
                trainer.save_checkpoint(self.best_model_path, weights_only=False)

    checkpoint = _EpochCheckpoint(
        dirpath=str(ckpt_dir.resolve()),
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
        auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=str(output), name="logs", version="")
    trainer = pl.Trainer(
        **options,
        default_root_dir=str(output),
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=config.trainer.patience,
                min_delta=0,
                check_finite=True,
                check_on_train_epoch_end=True,
            ),
            checkpoint,
        ],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        limit_train_batches=config.trainer.limit_train_batches or 1.0,
        limit_val_batches=config.trainer.limit_val_batches or 1.0,
    )

    started = time.perf_counter()
    trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None, weights_only=True)
    elapsed = time.perf_counter() - started
    logger.save()

    best_path = Path(checkpoint.best_model_path) if checkpoint.best_model_path else ckpt_dir / "best.ckpt"
    last_path = Path(checkpoint.last_model_path) if checkpoint.last_model_path else ckpt_dir / "last.ckpt"
    best_data = torch.load(best_path, map_location="cpu", weights_only=True)
    last_data = torch.load(last_path, map_location="cpu", weights_only=True)

    (ckpt_dir / "manifest.json").write_text(
        json.dumps({p.name: sha256(p) for p in (best_path, last_path) if p.is_file()}, indent=2) + "\n"
    )
    learning_curves(Path(logger.log_dir) / "metrics.csv", output / "learning_curves.png")

    report = {
        "model": "LogisticRegression",
        "elapsed_seconds": elapsed,
        "completed_epochs": int(last_data.get("epoch", 0)) + 1,
        "global_step": int(last_data.get("global_step", trainer.global_step)),
        "best_epoch": int(best_data.get("epoch", 0)) + 1,
        "best_val_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else float(trainer.callback_metrics.get("val/loss", 0.0)),
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "accelerator": options["accelerator"],
        "environment": environment(),
        "smoke_run": config.trainer.limit_train_batches is not None or config.trainer.limit_val_batches is not None,
        "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
        "source_checkpoint": str(Path(resume).resolve()) if resume else None,
        "source_checkpoint_sha256": sha256(resume) if resume else None,
        "resumed_from_epoch": int(saved_ckpt["epoch"]) + 1 if saved_ckpt else None,
        "resumed_from_global_step": int(saved_ckpt["global_step"]) if saved_ckpt else None,
    }
    (output / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_demo(root: Path, epochs: int = 2, threads: int = 4) -> Path:
    """Train a bounded CPU smoke run of LogisticRegression."""
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise ValueError("Choose an integer epoch count from 1 to 100")
    if type(threads) is not int or threads < 1:
        raise ValueError("Choose a positive integer thread count")
    root = Path(root).resolve()
    config = load_config(
        overrides={
            "data.directory": str(root / "data"),
            "data.download": False,
            "trainer.accelerator": "cpu",
            "trainer.max_epochs": epochs,
            "trainer.patience": epochs,
            "trainer.limit_train_batches": 20,
            "trainer.limit_val_batches": 5,
            "tracking.enabled": False,
            "output_dir": str(root / "runs" / f"logistic-demo-{uuid4().hex[:12]}"),
        },
        use_environment=False,
    )
    MNISTDataModule(config.data, accelerator="cpu").setup("fit")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        config, output = prepare_run_dir(config, "train-logistic")
        train(config, output)
        return output
    finally:
        torch.set_num_threads(previous_threads)
