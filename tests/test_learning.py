"""Small Lightning integrations: metrics, checkpoint fidelity, and process isolation."""

import json
import subprocess
import sys

import lightning.pytorch as pl
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.MNIST.checkpoints import checkpoint_config, read_checkpoint, sha256
from models.MNIST.cli import main
from models.MNIST.config import ConfigError
from models.MNIST.module import MNISTClassifier


def test_metric_weighting_reset_and_nonfinite_loss():
    model = MNISTClassifier()
    model.network = nn.Identity()
    logits = torch.zeros(5, 10)
    logits[:, 0] = 3
    labels = torch.tensor([0, 0, 0, 1, 1])
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    result = trainer.test(model, dataloaders=loader, verbose=False)[0]
    assert result["test/accuracy"] == pytest.approx(0.6)
    assert result["test/loss"] == pytest.approx(float(F.cross_entropy(logits, labels)))
    assert result["test/count"] == 5
    second = DataLoader(TensorDataset(logits, torch.ones(5, dtype=torch.long)), batch_size=3)
    result = trainer.test(model, dataloaders=second, verbose=False)[0]
    assert result["test/accuracy"] == 0 and result["test/count"] == 5
    bad = DataLoader(TensorDataset(torch.full((1, 10), float("nan")), torch.tensor([0])))
    with pytest.raises(RuntimeError, match="Nonfinite test loss"):
        trainer.test(model, dataloaders=bad, verbose=False)


def test_fit_checkpoint_resume_and_fresh_process_test(tiny_mnist, tmp_path):
    original = tmp_path / "original"
    assert main(["train", "--no-download", "--accelerator", "cpu", "--batch-size", "4",
                 "--max-epochs", "1", "--output-dir", str(original)]) == 0
    last_path = original / "checkpoints/last.ckpt"
    best_path = original / "checkpoints/best.ckpt"
    last = read_checkpoint(last_path)
    summary = json.loads((original / "training_summary.json").read_text())
    assert last["epoch"] == 0 and last["global_step"] == 5
    assert summary["completed_epochs"] == 1
    assert (original / "learning_curves.png").is_file()
    assert (original / "logs/metrics.csv").is_file()
    assert last["MNISTDataModule"]["split_indices"] == json.loads((original / "split_indices.json").read_text())
    checkpoint_model = MNISTClassifier.load_from_checkpoint(best_path, map_location="cpu", weights_only=True).eval()
    expected_model = MNISTClassifier(config=last["hyper_parameters"]["config"])
    expected_model.load_state_dict(last["state_dict"])
    expected_model.eval()
    images = torch.randn(3, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(checkpoint_model(images), expected_model(images))

    resumed = tmp_path / "resumed"
    assert main(["train", "--resume", str(last_path), "--max-epochs", "2", "--output-dir", str(resumed)]) == 0
    continued = read_checkpoint(resumed / "checkpoints/last.ckpt")
    assert continued["epoch"] == 1 and continued["global_step"] == 10
    assert all(int(state["step"]) == 10 for state in continued["optimizer_states"][0]["state"].values())
    assert continued["MNISTDataModule"]["split_indices"] == last["MNISTDataModule"]["split_indices"]
    assert not torch.equal(last["state_dict"]["network.layers.conv1.weight"], continued["state_dict"]["network.layers.conv1.weight"])
    info = json.loads((resumed / "training_summary.json").read_text())
    assert info["resumed_from_global_step"] == 5
    assert info["source_checkpoint_sha256"] == sha256(last_path)

    # Fresh interpreter, fixture data only; no preceding in-memory model or Trainer.
    script = '''import runpy, sys
from models.MNIST import data
from models.MNIST.cli import main
data.datasets.MNIST = runpy.run_path("tests/conftest.py")["TinyMNIST"]
data.MNISTDataModule.train_size = 18
data.MNISTDataModule.val_size = 5
raise SystemExit(main(sys.argv[1:]))
'''
    tested = tmp_path / "tested"
    result = subprocess.run([sys.executable, "-c", script, "test", "--checkpoint", str(best_path),
                             "--batch-size", "3", "--output-dir", str(tested)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((tested / "metrics.json").read_text())
    assert metrics["sample_count"] == 11
    assert metrics["checkpoint_sha256"] == sha256(best_path)
    assert metrics["config"]["model"] == last["hyper_parameters"]["config"]["model"]
    assert 0 <= metrics["test/accuracy"] <= 1

    # Resume also works in a fresh interpreter, restoring the epoch and optimizer.
    fresh_resume = tmp_path / "fresh-resume"
    result = subprocess.run([sys.executable, "-c", script, "train", "--resume", str(last_path),
                             "--max-epochs", "2", "--output-dir", str(fresh_resume)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read_checkpoint(fresh_resume / "checkpoints/last.ckpt")["global_step"] == 10

    # Model/preprocessing are recovered from checkpoint metadata, even without
    # the original config.yaml; incompatible changes fail before run creation.
    with pytest.raises(ValueError, match="optimizer"):
        checkpoint_config(dict(last, optimizer_states=[]), None, {"trainer.max_epochs": 2}, resume=True)
    with pytest.raises(ConfigError, match="dropout"):
        checkpoint_config(last, None, {"model.dropout": 0.8}, resume=True)
    with pytest.raises(ConfigError, match="mean"):
        checkpoint_config(last, None, {"data.mean": 0.5}, resume=False)
    with pytest.raises(ConfigError, match="max-epochs"):
        checkpoint_config(last, None, {}, resume=True)
    with pytest.raises(ConfigError, match="batch_size"):
        checkpoint_config(last, None, {"data.batch_size": 2}, resume=True)
    assert main(["test", "--checkpoint", str(best_path), "--dropout", "0.8", "--output-dir", str(tmp_path / "bad")]) == 2
    assert not (tmp_path / "bad").exists()


def test_best_selection_and_last_saved_without_improvement(tiny_mnist, tmp_path, monkeypatch):
    from models.MNIST import learning

    class IncreasingValidationLoss(MNISTClassifier):
        def validation_step(self, batch, batch_idx):
            self.log("val/loss", float(self.current_epoch + 1), on_epoch=True, batch_size=len(batch[1]))
            self.log("val/accuracy", 0.0, on_epoch=True, batch_size=len(batch[1]))

    monkeypatch.setattr(learning, "MNISTClassifier", IncreasingValidationLoss)
    first = tmp_path / "first"
    assert main(["train", "--accelerator", "cpu", "--max-epochs", "2", "--batch-size", "4", "--output-dir", str(first)]) == 0
    best = read_checkpoint(first / "checkpoints/best.ckpt")
    last = read_checkpoint(first / "checkpoints/last.ckpt")
    assert best["epoch"] == 0 and last["epoch"] == 1
    best_hash = sha256(first / "checkpoints/best.ckpt")
    resumed = tmp_path / "resumed"
    assert main(["train", "--resume", str(first / "checkpoints/last.ckpt"), "--max-epochs", "8", "--output-dir", str(resumed)]) == 0
    info = json.loads((resumed / "training_summary.json").read_text())
    assert info["completed_epochs"] == 4  # Three failed validation improvements: patience=3.
    assert info["best_epoch"] == 1
    assert sha256(resumed / "checkpoints/best.ckpt") == best_hash
    assert read_checkpoint(resumed / "checkpoints/last.ckpt")["epoch"] == 3


def test_invalid_and_corrupt_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        read_checkpoint(tmp_path / "missing.ckpt")
    bad = tmp_path / "bad.ckpt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="Cannot load"):
        read_checkpoint(bad)
    torch.save({"state_dict": {}}, bad)
    with pytest.raises(ValueError, match="Incompatible"):
        read_checkpoint(bad)
    (tmp_path / "manifest.json").write_text(json.dumps({"bad.ckpt": "wrong"}))
    with pytest.raises(ValueError, match="SHA-256"):
        read_checkpoint(bad)


def test_checkpoint_requires_metadata_and_rejects_incompatible_weights(tmp_path):
    model = MNISTClassifier()
    payload = {
        "mnist_format_version": 1, "hyper_parameters": {"config": model.hparams.config},
        "state_dict": {"wrong_key": torch.tensor(1)}, "epoch": 0, "global_step": 1,
    }
    path = tmp_path / "incompatible.ckpt"
    torch.save(payload, path)
    assert main(["test", "--checkpoint", str(path), "--accelerator", "cpu", "--output-dir", str(tmp_path / "run")]) == 2
    assert not (tmp_path / "run/metrics.json").exists()
    payload["hyper_parameters"] = None
    torch.save(payload, path)
    with pytest.raises(ValueError, match="reconstruction"):
        read_checkpoint(path)
