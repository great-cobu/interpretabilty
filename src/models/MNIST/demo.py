"""Explicit, bounded training actions for the optional interactive demo."""

from pathlib import Path
from uuid import uuid4

import torch

from models.MNIST.config import load_config
from models.MNIST.data import MNISTDataModule
from models.MNIST.learning import train
from models.MNIST.runs import create_run


def train_demo(root: Path, epochs: int, threads: int) -> Path:
    """Train a fresh CPU smoke run only when explicitly called by the UI."""
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise ValueError("Choose an integer epoch count from 1 to 100")
    if type(threads) is not int or threads < 1:
        raise ValueError("Choose a positive integer thread count")
    root = Path(root).resolve()
    config = load_config(overrides={
        "data.directory": str(root / "data"), "data.download": False,
        "trainer.accelerator": "cpu", "trainer.max_epochs": epochs,
        "trainer.patience": epochs, "trainer.limit_train_batches": 20,
        "trainer.limit_val_batches": 5, "tracking.enabled": False,
        "output_dir": str(root / "runs" / f"marimo-demo-{uuid4().hex[:12]}"),
    }, use_environment=False)
    # Report cache errors before claiming a run directory.
    MNISTDataModule(config.data, accelerator="cpu").setup("fit")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        config, output = create_run(config, "train")
        train(config, output)
        return output
    finally:
        torch.set_num_threads(previous_threads)
