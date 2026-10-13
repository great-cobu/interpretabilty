"""Training pipeline for MNIST CNN using PyTorch Lightning."""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset

from models.MNIST.config import MNISTCNNConfig, MNISTTrainingConfig
from models.MNIST.data import (
    MNISTDataCollator,
    MNISTDataModule,
    load_mnist_datasets,
)
from models.MNIST.model import MNISTCNN


def compute_metrics(
    eval_pred_or_y_true: Any,
    y_pred: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute classification evaluation metrics (accuracy, precision, recall, f1).

    Supports:
        - `compute_metrics(eval_pred)` where eval_pred has .predictions and .label_ids,
          or is a tuple of (predictions, label_ids).
        - `compute_metrics(y_true, y_pred)` directly with true labels and predictions/logits.

    Returns:
        Dictionary with 'accuracy', 'f1', 'precision', and 'recall'.
    """
    if y_pred is not None:
        labels = np.asarray(eval_pred_or_y_true)
        preds = np.asarray(y_pred)
    else:
        eval_pred = eval_pred_or_y_true
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            preds = np.asarray(eval_pred.predictions)
            labels = np.asarray(eval_pred.label_ids)
        elif isinstance(eval_pred, (tuple, list)) and len(eval_pred) == 2:
            preds = np.asarray(eval_pred[0])
            labels = np.asarray(eval_pred[1])
        elif isinstance(eval_pred, dict):
            preds = np.asarray(eval_pred.get("predictions", eval_pred.get("logits")))
            labels = np.asarray(eval_pred.get("labels", eval_pred.get("label_ids")))
        else:
            raise ValueError(f"Unrecognized metric input format: {type(eval_pred)}")

    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def train_mnist(
    model: Optional[MNISTCNN] = None,
    config: Optional[MNISTCNNConfig] = None,
    training_config: Optional[MNISTTrainingConfig] = None,
    output_dir: str = "./results/mnist_cnn",
    epochs: int = 5,
    batch_size: int = 64,
    eval_batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    train_val_split: float = 0.1,
    seed: int = 42,
    report_to: str = "none",
    datasets: Optional[Any] = None,
    accelerator: Optional[str] = None,
    devices: Optional[Union[int, str, List[int]]] = None,
    data_dir: str = "./data",
    callbacks: Optional[List[Any]] = None,
    logger: Optional[Any] = None,
    fast_dev_run: bool = False,
    **kwargs: Any,
) -> Tuple[MNISTCNN, pl.Trainer, Dict[str, Any]]:
    """Train the MNIST CNN using PyTorch Lightning.

    Args:
        model: Optional pre-initialized MNISTCNN instance.
        config: Optional MNISTCNNConfig for model initialization.
        training_config: Optional MNISTTrainingConfig helper.
        output_dir: Output path for checkpoints and the saved model.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        eval_batch_size: Batch size for evaluation.
        learning_rate: Peak learning rate for AdamW optimizer.
        weight_decay: Weight decay penalty for AdamW.
        train_val_split: Fraction of training samples used for validation.
        seed: Random seed for reproducibility.
        report_to: Tracking integration ("none", "mlflow", etc.).
        datasets: Optional pre-loaded Dataset mapping or LightningDataModule.
        accelerator: Hardware accelerator ('auto', 'cpu', 'cuda', 'mps').
        devices: Number or list of devices.
        data_dir: Directory where MNIST dataset is cached.
        callbacks: Optional list of additional PyTorch Lightning callbacks.
        logger: Optional PyTorch Lightning logger instance.
        fast_dev_run: If True, run a single batch for quick sanity testing.
        **kwargs: Additional arguments ignored for compatibility.

    Returns:
        Tuple of (trained_model, trainer_instance, train_metrics).
    """
    # 1. Sync configuration parameters
    if training_config is not None:
        output_dir = getattr(training_config, "output_dir", output_dir)
        epochs = getattr(training_config, "max_epochs", epochs)
        batch_size = getattr(training_config, "batch_size", batch_size)
        eval_batch_size = getattr(training_config, "eval_batch_size", eval_batch_size)
        learning_rate = getattr(training_config, "learning_rate", learning_rate)
        weight_decay = getattr(training_config, "weight_decay", weight_decay)
        train_val_split = getattr(training_config, "train_val_split", train_val_split)
        seed = getattr(training_config, "seed", seed)
        report_to = getattr(training_config, "report_to", report_to)
        accelerator = accelerator or getattr(training_config, "accelerator", "auto")
        devices = devices or getattr(training_config, "devices", "auto")
        fast_dev_run = getattr(training_config, "fast_dev_run", fast_dev_run)

    pl.seed_everything(seed, workers=True)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Initialize Model
    if model is None:
        if config is None:
            config = MNISTCNNConfig(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            config.learning_rate = learning_rate
            config.weight_decay = weight_decay
        model = MNISTCNN(config)

    # 3. Setup DataLoaders / DataModule
    train_dataloader: Optional[DataLoader] = None
    val_dataloader: Optional[DataLoader] = None
    datamodule: Optional[pl.LightningDataModule] = None
    collator = MNISTDataCollator()

    if isinstance(datasets, pl.LightningDataModule):
        datamodule = datasets
    elif isinstance(datasets, dict) or hasattr(datasets, "__getitem__"):
        train_ds = datasets["train"] if isinstance(datasets, dict) or hasattr(datasets, "keys") else datasets[0]
        val_ds = (
            datasets.get("validation", datasets.get("test"))
            if isinstance(datasets, dict)
            else (datasets[1] if len(datasets) > 1 else None)
        )

        train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        if val_ds is not None and len(val_ds) > 0:
            val_dataloader = DataLoader(
                val_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
    else:
        datamodule = MNISTDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            train_val_split=train_val_split,
            seed=seed,
        )

    # 4. Configure Callbacks and Loggers
    active_callbacks = list(callbacks or [])

    monitor_metric = "val_loss" if (val_dataloader is not None or (datamodule and train_val_split > 0.0)) else "train_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="mnist-cnn-{epoch:02d}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    active_callbacks.append(checkpoint_callback)

    if logger is None and report_to == "mlflow":
        try:
            from pytorch_lightning.loggers import MLFlowLogger
            logger = MLFlowLogger(
                experiment_name="mnist_cnn",
                save_dir=output_dir,
            )
        except Exception:
            logger = None

    # 5. Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=epochs,
        accelerator=accelerator or "auto",
        devices=devices or "auto",
        callbacks=active_callbacks,
        logger=logger if logger is not None else False,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
    )

    # 6. Run Training
    if datamodule is not None:
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 7. Collect Training Metrics
    metrics: Dict[str, Any] = {
        k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
        for k, v in trainer.callback_metrics.items()
    }

    # 8. Save Model
    model.save_pretrained(output_dir)

    return model, trainer, metrics


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train CNN on MNIST using PyTorch Lightning.")
    parser.add_argument("--output_dir", type=str, default="./results/mnist_cnn", help="Save directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--train_val_split", type=float, default=0.1, help="Train/Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (auto, cpu, cuda, mps)")
    parser.add_argument("--report_to", type=str, default="none", help="Reporter (none, mlflow)")

    args = parser.parse_args()

    train_mnist(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        train_val_split=args.train_val_split,
        seed=args.seed,
        accelerator=args.accelerator,
        report_to=args.report_to,
    )


if __name__ == "__main__":
    main()
