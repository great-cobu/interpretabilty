"""Tests for training, checkpoint resume, evaluation, prediction, and CLI entry points."""

import json
import subprocess
import sys

import torch

from models.MNIST.cli import main
from models.MNIST.evaluate import confusion_matrix, load_classifier, predict
from models.MNIST.train import train_demo


def test_cli_and_module_help():
    console = subprocess.run(["mnist-cnn", "--help"], capture_output=True, text=True)
    mod_alias = subprocess.run([sys.executable, "-m", "mnist_cnn", "--help"], capture_output=True, text=True)
    mod_main = subprocess.run([sys.executable, "-m", "models.MNIST", "--help"], capture_output=True, text=True)
    assert console.returncode == mod_alias.returncode == mod_main.returncode == 0


def test_train_resume_test_and_predict_end_to_end(tmp_path, patch_mnist):
    run1 = tmp_path / "run1"
    rc = main([
        "train",
        "--accelerator", "cpu",
        "--max-epochs", "2",
        "--batch-size", "4",
        "--output-dir", str(run1),
    ])
    assert rc == 0
    assert (run1 / "checkpoints/best.ckpt").is_file()
    assert (run1 / "checkpoints/last.ckpt").is_file()
    assert (run1 / "checkpoints/manifest.json").is_file()
    assert (run1 / "learning_curves.png").is_file()
    summary1 = json.loads((run1 / "training_summary.json").read_text())
    assert summary1["completed_epochs"] == 2

    # Resume to epoch 3
    run2 = tmp_path / "run2"
    rc_resume = main([
        "train",
        "--resume", str(run1 / "checkpoints/last.ckpt"),
        "--accelerator", "cpu",
        "--max-epochs", "3",
        "--output-dir", str(run2),
    ])
    assert rc_resume == 0
    summary2 = json.loads((run2 / "training_summary.json").read_text())
    assert summary2["completed_epochs"] == 3
    assert summary2["resumed_from_epoch"] == 2

    # Test checkpoint
    test_out = tmp_path / "test_out"
    rc_test = main([
        "test",
        "--checkpoint", str(run2 / "checkpoints/best.ckpt"),
        "--accelerator", "cpu",
        "--output-dir", str(test_out),
    ])
    assert rc_test == 0
    metrics = json.loads((test_out / "metrics.json").read_text())
    assert metrics["sample_count"] == 7

    # Load classifier and run predict() + confusion_matrix()
    clf = load_classifier(run2 / "checkpoints/best.ckpt")
    sample_imgs = torch.randn(3, 1, 28, 28)
    preds = predict(clf, sample_imgs)
    assert preds["predictions"].shape == (3,)
    assert preds["probabilities"].shape == (3, 10)
    assert torch.allclose(preds["probabilities"].sum(dim=1), torch.ones(3), atol=1e-5)

    dataset = [(torch.randn(1, 28, 28), i % 10) for i in range(10)]
    cm = confusion_matrix(clf, dataset, batch_size=4)
    assert cm.shape == (10, 10)
    assert int(cm.sum()) == 10


def test_train_demo_helper(tmp_path, patch_mnist):
    out = train_demo(tmp_path, epochs=1, threads=1)
    assert (out / "training_summary.json").is_file()
    assert (out / "checkpoints/best.ckpt").is_file()
