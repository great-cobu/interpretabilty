"""Tests for MNISTCNN architecture, interpretability hooks, and synthetic inspection."""

import pytest
import torch
from torch.nn import functional as F

from models.MNIST.cli import main as cli_main
from models.MNIST.config import load_config
from models.MNIST.model import MNISTCNN, MNISTClassifier, SMU, inspect_model


@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_forward_loss_and_gradients(batch_size):
    torch.manual_seed(42)
    model = MNISTCNN(dropout=0.2)
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.arange(batch_size, dtype=torch.long) % 10

    logits = model(images)
    assert logits.shape == (batch_size, 10)
    assert torch.isfinite(logits).all()

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_interpretability_hooks():
    model = MNISTClassifier()
    images = torch.randn(4, 1, 28, 28)

    last_conv = model.get_last_conv_layer()
    assert isinstance(last_conv, torch.nn.Conv2d)
    assert last_conv.out_channels == 64

    features = model.get_feature_maps(images)
    assert features["conv1"].shape == (4, 32, 28, 28)
    assert features["pool1"].shape == (4, 32, 14, 14)
    assert features["conv2"].shape == (4, 64, 14, 14)
    assert features["pool2"].shape == (4, 64, 7, 7)
    assert features["hidden"].shape == (4, 128)
    assert features["logits"].shape == (4, 10)


def test_invalid_inputs_rejected():
    model = MNISTCNN()
    with pytest.raises(ValueError):
        model(torch.zeros((0, 1, 28, 28)))
    with pytest.raises(ValueError):
        model(torch.zeros((2, 3, 28, 28)))
    with pytest.raises(ValueError):
        model(torch.zeros((2, 1, 28, 28), dtype=torch.uint8))


def test_inspect_model_writes_summary(tmp_path):
    cfg = load_config(overrides={"data.batch_size": 5, "trainer.accelerator": "cpu"}, use_environment=False)
    report = inspect_model(cfg, tmp_path)
    assert report["output_shape"] == [5, 10]
    assert report["parameter_count"] == 421_642
    assert report["activation"] == "relu"
    assert (tmp_path / "model_summary.json").is_file()


@pytest.mark.parametrize(
    ("activation", "expected_cls"),
    [
        ("relu", torch.nn.ReLU),
        ("softplus", torch.nn.Softplus),
        ("smu", SMU),
        ("SMU", SMU),
    ],
)
def test_configurable_activations_and_cli_override(tmp_path, activation, expected_cls):
    cfg = load_config(
        overrides={
            "data.batch_size": 4,
            "trainer.accelerator": "cpu",
            "model.activation": activation,
        },
        use_environment=False,
    )
    clf = MNISTClassifier(cfg)
    assert clf.network.activation == activation
    assert isinstance(clf.network.layers.relu1, expected_cls)
    assert isinstance(clf.network.layers.relu2, expected_cls)
    assert isinstance(clf.network.layers.relu3, expected_cls)
    assert sum(p.numel() for p in clf.parameters()) == 421_642

    out_dir = tmp_path / f"inspect_{activation}"
    rc = cli_main([
        "inspect-model",
        "--accelerator", "cpu",
        "--batch-size", "4",
        "--activation", activation,
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    reloaded = load_config(out_dir / "config.yaml", use_environment=False)
    assert reloaded.model.activation == activation


def test_invalid_activation_rejected():
    with pytest.raises(ValueError, match="activation"):
        MNISTCNN(activation="tanh")
