"""MNIST caching, reproducible subsets, and Lightning data loaders."""

import json
from pathlib import Path
import random

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from models.MNIST.config import DataConfig


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy from the worker seed assigned by PyTorch."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_accelerator(requested: str) -> str:
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available or not available[requested]:
        raise ValueError(f"Requested accelerator {requested!r} is unavailable; use --accelerator cpu")
    return requested


class MNISTDataModule(pl.LightningDataModule):
    """Keep acquisition separate from stage-specific, repeatable dataset setup."""

    train_size = 55_000
    val_size = 5_000

    def __init__(self, config: DataConfig | None = None, accelerator: str = "cpu"):
        super().__init__()
        self.config = config or DataConfig()
        self.accelerator = resolve_accelerator(accelerator)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.config.mean,), (self.config.std,)),
        ])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split_indices = None
        self._train_generator = torch.Generator().manual_seed(self.config.split_seed)

    def _dataset(self, train: bool, download: bool = False):
        try:
            return datasets.MNIST(
                root=self.config.directory, train=train,
                transform=self.transform, download=download,
            )
        except (RuntimeError, OSError) as exc:
            if not download:
                raise RuntimeError(
                    "MNIST cache is missing or unreadable; run mnist-cnn prepare-data "
                    "--download with the same --data-dir before using --no-download"
                ) from exc
            raise RuntimeError(
                "MNIST download failed; check network access and data-directory permissions"
            ) from exc

    def prepare_data(self) -> None:
        # Lightning may run this on a different process: do not assign split state.
        self._dataset(train=True, download=self.config.download)
        self._dataset(train=False, download=self.config.download)

    def _load_indices(self, size: int) -> dict:
        try:
            indices = json.loads(Path(self.config.split_indices).read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot read split indices JSON") from exc
        return self._validate_indices(indices, size)

    def _validate_indices(self, indices: dict, size: int) -> dict:
        if not isinstance(indices, dict) or indices.get("dataset") != "MNIST" or indices.get("version") != 1:
            raise ValueError("Unsupported split indices format")
        if indices.get("seed") != self.config.split_seed:
            raise ValueError("Saved split seed does not match data.split_seed")
        train, val = indices.get("train"), indices.get("validation")
        if not isinstance(train, list) or not isinstance(val, list):
            raise ValueError("Split indices must contain train and validation lists")
        if len(train) != self.train_size or len(val) != self.val_size:
            raise ValueError("Saved split sizes do not match the MNIST split contract")
        combined = train + val
        if any(type(i) is not int for i in combined) or set(combined) != set(range(size)):
            raise ValueError("Split indices must uniquely partition the official training set")
        return indices

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit", "validate", "test", "predict"}:
            raise ValueError(f"Unsupported data stage: {stage}")
        if stage in {None, "fit", "validate"} and self.train_dataset is None:
            full = self._dataset(train=True)
            if len(full) != self.train_size + self.val_size:
                raise ValueError("Unexpected MNIST training dataset size")
            if self.split_indices is not None:
                indices = self._validate_indices(self.split_indices, len(full))
            elif self.config.split_indices:
                indices = self._load_indices(len(full))
            else:
                train, val = random_split(
                    full, [self.train_size, self.val_size],
                    generator=torch.Generator().manual_seed(self.config.split_seed),
                )
                indices = {"version": 1, "dataset": "MNIST", "seed": self.config.split_seed,
                           "train": train.indices, "validation": val.indices}
            self.split_indices = indices
            self.train_dataset = Subset(full, indices["train"])
            self.val_dataset = Subset(full, indices["validation"])
        if stage in {None, "test", "predict"} and self.test_dataset is None:
            self.test_dataset = self._dataset(train=False)

    def save_split_indices(self, path: str | Path) -> None:
        if self.split_indices is None:
            raise RuntimeError("Call setup('fit') before saving split indices")
        Path(path).write_text(json.dumps(self.split_indices) + "\n")

    def _loader(self, dataset, *, shuffle: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup for the requested stage before requesting a loader")
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle,
            num_workers=self.config.num_workers, drop_last=False,
            pin_memory=self.accelerator == "cuda",
            persistent_workers=self.config.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=self._train_generator if shuffle else torch.Generator().manual_seed(self.config.split_seed),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def state_dict(self) -> dict:
        """Lightning stores exact membership and loader randomness in checkpoints."""
        return {"split_indices": self.split_indices, "train_generator": self._train_generator.get_state()}

    def load_state_dict(self, state_dict: dict) -> None:
        indices = state_dict.get("split_indices")
        if indices is not None:
            self.split_indices = self._validate_indices(indices, self.train_size + self.val_size)
            if self.train_dataset is not None:
                full = self.train_dataset.dataset
                self.train_dataset = Subset(full, indices["train"])
                self.val_dataset = Subset(full, indices["validation"])
        if "train_generator" in state_dict:
            self._train_generator.set_state(state_dict["train_generator"].cpu())
