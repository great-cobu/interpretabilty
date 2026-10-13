"""The MNIST CNN, independent of data loading and Lightning orchestration."""

from collections import OrderedDict
import math

import torch
from torch import nn


class MNISTCNN(nn.Module):
    """Map floating-point images [N, 1, 28, 28] to ten unnormalized logits.

    Inputs use the DataModule's normalization. Pass logits directly to
    torch.nn.functional.cross_entropy with int64 digit labels. Use argmax
    for predicted labels and softmax only when probabilities are needed.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
        self.layers = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 32, kernel_size=3, padding=1)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("conv2", nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("flatten", nn.Flatten(start_dim=1)),
            ("hidden", nn.Linear(64 * 7 * 7, 128)),
            ("relu3", nn.ReLU()),
            ("dropout", nn.Dropout(p=dropout)),
            ("logits", nn.Linear(128, 10)),
        ]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28) or images.shape[0] == 0:
            raise ValueError("Expected a nonempty image batch shaped [N, 1, 28, 28]")
        if not images.is_floating_point():
            raise ValueError("Images must be floating-point tensors normalized by the data pipeline")
        return self.layers(images)
