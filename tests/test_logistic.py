"""Unit and integration tests for models.logistic (Multinomial Logistic Regression)."""

import json
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from models.MNIST.config import load_config
from models.logistic.cli import main
from models.logistic.evaluate import confusion_matrix, load_classifier, predict
from models.logistic.model import LogisticClassifier, LogisticRegression, inspect_model
from models.logistic.train import train_demo


@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_logistic_forward_parameter_count_and_gradients(batch_size):
    torch.manual_seed(42)
    model = LogisticRegression()
    assert sum(p.numel() for p in model.parameters()) == 7_850

    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.arange(batch_size, dtype=torch.long) % 10

    logits = model(images)
    assert logits.shape == (batch_size, 10)
    assert torch.isfinite(logits).all()

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_glass_box_interpretability_hooks_and_exact_logit_completeness():
    torch.manual_seed(7)
    clf = LogisticClassifier()
    images = torch.randn(5, 1, 28, 28)

    # 1. Feature maps
    features = clf.get_feature_maps(images)
    assert features["flatten"].shape == (5, 784)
    assert features["logits"].shape == (5, 10)

    # 2. Class templates [10, 28, 28]
    templates = clf.get_class_templates()
    assert templates.shape == (10, 28, 28)

    # 3. Exact linear pixel attribution completeness: sum(x * W_c) + b_c == logit_c
    logits = clf(images)
    bias = clf.network.layers.logits.bias
    for c in range(10):
        attr = clf.get_pixel_attribution(images, target_class=c)
        assert attr.shape == (5, 1, 28, 28)
        reconstructed_logit = attr.sum(dim=(1, 2, 3)) + bias[c]
        torch.testing.assert_close(reconstructed_logit, logits[:, c], rtol=1e-5, atol=1e-5)

    # Default target_class=None uses predicted class
    pred_classes = logits.argmax(dim=1)
    pred_attr = clf.get_pixel_attribution(images)
    reconstructed_pred = pred_attr.sum(dim=(1, 2, 3)) + bias[pred_classes]
    expected_pred_logits = logits[torch.arange(5), pred_classes]
    torch.testing.assert_close(reconstructed_pred, expected_pred_logits, rtol=1e-5, atol=1e-5)


def test_inspect_logistic_model_writes_summary(tmp_path):
    cfg = load_config("configs/logistic.yaml", overrides={"data.batch_size": 5, "trainer.accelerator": "cpu"}, use_environment=False)
    report = inspect_model(cfg, tmp_path)
    assert report["model"] == "LogisticRegression"
    assert report["output_shape"] == [5, 10]
    assert report["parameter_count"] == 7_850
    assert (tmp_path / "model_summary.json").is_file()


def test_logistic_cli_and_module_help():
    console = subprocess.run(["mnist-logistic", "--help"], capture_output=True, text=True)
    mod_main = subprocess.run([sys.executable, "-m", "models.logistic", "--help"], capture_output=True, text=True)
    assert console.returncode == mod_main.returncode == 0


def test_logistic_train_resume_test_and_predict_end_to_end(tmp_path, patch_mnist):
    run1 = tmp_path / "run1"
    rc = main([
        "train",
        "--config", "configs/logistic.yaml",
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
    assert summary1["model"] == "LogisticRegression"
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
    assert metrics["model"] == "LogisticRegression"
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


def test_logistic_train_demo_helper(tmp_path, patch_mnist):
    out = train_demo(tmp_path, epochs=1, threads=1)
    assert (out / "training_summary.json").is_file()
    assert (out / "checkpoints/best.ckpt").is_file()
