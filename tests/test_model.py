"""Behavior checks for the stage-4 network; no data download or training loop."""

import json

import pytest
import torch
from torch.nn import functional as F

from models.MNIST.cli import main
from models.MNIST.model import MNISTCNN


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_forward_and_backward(batch_size):
    torch.manual_seed(42)
    model = MNISTCNN()
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.arange(batch_size) % 10
    logits = model(images)
    assert logits.shape == (batch_size, 10)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    loss = F.cross_entropy(logits, labels)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name


def test_output_is_logits_not_probabilities():
    model = MNISTCNN(dropout=0).eval()
    # Force known, unrestricted class scores to catch an accidental final softmax.
    with torch.no_grad():
        model.layers.logits.weight.zero_()
        model.layers.logits.bias.copy_(torch.arange(10) - 5)
        logits = model(torch.zeros(2, 1, 28, 28))
    assert torch.equal(logits[0], torch.arange(10) - 5)
    probabilities = logits.softmax(dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(logits.argmax(dim=1), probabilities.argmax(dim=1))
    labels = torch.tensor([0, 9])
    expected = -logits.log_softmax(dim=1)[torch.arange(2), labels].mean()
    assert torch.allclose(F.cross_entropy(logits, labels), expected)


def test_eval_is_repeatable_and_dropout_only_affects_training():
    torch.manual_seed(42)
    model = MNISTCNN(dropout=0.5)
    images = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        model.eval()
        first = model(images)
        assert torch.equal(first, model(images))
        model.train()
        assert not torch.equal(model(images), model(images))


@pytest.mark.parametrize("shape", [(1, 28, 28), (2, 3, 28, 28), (2, 1, 32, 32), (0, 1, 28, 28)])
def test_invalid_image_shapes(shape):
    with pytest.raises(ValueError, match="shaped"):
        MNISTCNN()(torch.zeros(shape))


def test_integer_images_rejected():
    with pytest.raises(ValueError, match="floating-point"):
        MNISTCNN()(torch.zeros(1, 1, 28, 28, dtype=torch.uint8))


@pytest.mark.parametrize("dropout", [-0.1, 1, float("nan"), float("inf"), True, "0.2"])
def test_invalid_dropout(dropout):
    with pytest.raises(ValueError, match="dropout"):
        MNISTCNN(dropout=dropout)


def test_cli_inspection_is_synthetic_and_saves_shape_trace(tmp_path, monkeypatch):
    from models.MNIST.data import MNISTDataModule

    def no_download(self):
        pytest.fail("Model inspection must not acquire a dataset")

    monkeypatch.setattr(MNISTDataModule, "prepare_data", no_download)
    output = tmp_path / "model"
    assert main(["inspect-model", "--batch-size", "3", "--dropout", "0", "--accelerator", "cpu", "--output-dir", str(output)]) == 0
    report = json.loads((output / "model_summary.json").read_text())
    assert report["output_shape"] == [3, 10]
    assert report["optimizer_steps"] == 0
    assert report["dropout"] == 0
    shapes = {layer["layer"]: layer["output_shape"] for layer in report["layers"]}
    assert shapes["pool1"] == [3, 32, 14, 14]
    assert shapes["pool2"] == [3, 64, 7, 7]
    assert shapes["flatten"] == [3, 3136]
    assert shapes["hidden"] == [3, 128]
    assert all(gradient["finite"] for gradient in report["gradients"].values())
    assert (output / "config.yaml").is_file()
