"""Batched validation inference for interactive demonstrations."""

import torch
from torch.utils.data import DataLoader


def confusion_matrix(model, dataset, batch_size: int = 256) -> torch.Tensor:
    """Count every example once; rows are true digits, columns predictions.

    The caller supplies an evaluation-mode CPU model and a validation dataset.
    This ignores training smoke-run batch limits and never loads test data.
    """
    if model.training:
        raise ValueError("Confusion-matrix inference requires an evaluation-mode model")
    counts = torch.zeros((10, 10), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            predictions = model(images).argmax(dim=1)
            counts += torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    return counts
