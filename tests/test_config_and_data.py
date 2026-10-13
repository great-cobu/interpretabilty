"""Tests for configuration loading, validation, and MNISTDataModule splits."""

import json
import pytest
import torch

from models.MNIST.config import ConfigError, DataConfig, load_config, prepare_run_dir
from models.MNIST.data import MNISTDataModule, inspect_data, prepare_data


def test_load_config_defaults_and_overrides(tmp_path, monkeypatch):
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text("data:\n  batch_size: 32\ntrainer:\n  max_epochs: 4\n")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    cfg = load_config(yaml_file, overrides={"data.batch_size": 16, "model.dropout": 0.3})
    assert cfg.data.batch_size == 16
    assert cfg.trainer.max_epochs == 4
    assert cfg.model.dropout == 0.3
    assert cfg.tracking.uri == "http://localhost:5000"


@pytest.mark.parametrize(
    "overrides",
    [
        {"data.batch_size": 0},
        {"trainer.max_epochs": -1},
        {"model.dropout": 1.0},
        {"optimizer.learning_rate": 0.0},
        {"trainer.accelerator": "tpu"},
        {"unknown.key": 1},
        {"tracking.enabled": True},
    ],
)
def test_invalid_configs_rejected(overrides, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ConfigError):
        load_config(overrides=overrides, use_environment=False)


def test_prepare_run_dir_writes_yaml(tmp_path):
    cfg = load_config(overrides={"output_dir": str(tmp_path / "run1")}, use_environment=False)
    resolved, out = prepare_run_dir(cfg, "train")
    assert (out / "config.yaml").is_file()
    reloaded = load_config(out / "config.yaml", use_environment=False)
    assert reloaded == resolved


def test_datamodule_deterministic_splits_and_roundtrip(tmp_path, patch_mnist):
    dm = MNISTDataModule(DataConfig(directory=str(tmp_path / "data"), batch_size=5, split_seed=42))
    dm.prepare_data()
    dm.setup()

    train_idx = dm.split_indices["train"]
    val_idx = dm.split_indices["validation"]
    assert len(train_idx) == 12 and len(val_idx) == 6
    assert set(train_idx).isdisjoint(val_idx)

    idx_path = tmp_path / "split_indices.json"
    dm.save_split_indices(idx_path)

    dm2 = MNISTDataModule(DataConfig(directory=str(tmp_path / "data"), split_indices=str(idx_path)))
    dm2.setup("fit")
    assert dm2.split_indices == dm.split_indices

    images, labels = next(iter(dm.train_dataloader()))
    assert images.shape == (5, 1, 28, 28)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_prepare_and_inspect_data_helpers(tmp_path, patch_mnist):
    cfg = load_config(
        overrides={
            "data.directory": str(tmp_path / "data"),
            "data.batch_size": 5,
            "trainer.accelerator": "cpu",
        },
        use_environment=False,
    )
    prep = prepare_data(cfg)
    assert prep["dataset"] == "MNIST"

    out = tmp_path / "inspect"
    out.mkdir()
    summary = inspect_data(cfg, out)
    assert summary["splits"]["train"]["count"] == 12
    assert summary["splits"]["validation"]["count"] == 6
    assert summary["splits"]["test"]["count"] == 7
    assert (out / "split_indices.json").is_file()
    assert (out / "data_summary.json").is_file()


@pytest.mark.integration
def test_real_cached_mnist_integration():
    dm = MNISTDataModule(DataConfig(directory="data", download=False, batch_size=256))
    dm.setup()
    assert len(dm.train_dataset) == 55_000
    assert len(dm.val_dataset) == 5_000
    assert len(dm.test_dataset) == 10_000
    images, labels = next(iter(dm.val_dataloader()))
    assert images.shape == (256, 1, 28, 28)
    assert labels.shape == (256,)
