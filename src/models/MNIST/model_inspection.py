"""A synthetic forward/backward demonstration, with no optimizer or dataset."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from models.MNIST.config import Config
from models.MNIST.data import resolve_accelerator
from models.MNIST.model import MNISTCNN


def inspect_model(config: Config, output: Path) -> dict:
    """Trace real layer outputs and check gradient flow on synthetic images."""
    device = resolve_accelerator(config.trainer.accelerator)
    torch.manual_seed(config.trainer.seed)
    model = MNISTCNN(dropout=config.model.dropout).to(device)
    # Generate on CPU for portable, seeded inputs, then move to the target device.
    images = torch.rand(config.data.batch_size, 1, 28, 28)
    images = ((images - config.data.mean) / config.data.std).to(device)
    labels = (torch.arange(config.data.batch_size) % 10).to(device)
    shapes = []

    def record(name):
        def hook(layer, inputs, result):
            shapes.append({"layer": name, "type": type(layer).__name__, "output_shape": list(result.shape)})
        return hook

    handles = [layer.register_forward_hook(record(name)) for name, layer in model.layers.named_children()]
    try:
        model.eval()
        with torch.no_grad():
            logits = model(images)
    finally:
        for handle in handles:
            handle.remove()

    model.train()
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    gradients = {
        name: {"finite": parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
               "norm": float(parameter.grad.norm()) if parameter.grad is not None else None}
        for name, parameter in model.named_parameters()
    }
    if not torch.isfinite(logits).all() or not torch.isfinite(loss) or not all(g["finite"] for g in gradients.values()):
        raise RuntimeError("Synthetic model inspection produced nonfinite outputs, loss, or missing/nonfinite gradients")
    report = {
        "purpose": "Synthetic shape and gradient inspection; not a model-quality measurement",
        "accelerator": device, "seed": config.trainer.seed, "dropout": config.model.dropout,
        "input_shape": list(images.shape), "layers": shapes,
        "output_shape": list(logits.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "synthetic_cross_entropy": float(loss.detach()),
        "gradients": gradients, "optimizer_steps": 0,
    }
    (output / "model_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
