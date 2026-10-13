import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from models.MNIST.validation import confusion_matrix


def test_confusion_matrix_counts_all_digits_and_partial_batch():
    # Predictions equal input digits, with one deliberately misclassified zero.
    predictions = torch.tensor(list(range(10)) + [1])
    labels = torch.tensor(list(range(10)) + [0])
    dataset = TensorDataset(torch.nn.functional.one_hot(predictions, 10).float(), labels)
    model = nn.Identity().eval()
    actual = confusion_matrix(model, dataset, batch_size=4)
    expected = torch.eye(10, dtype=torch.int64)
    expected[0, 1] = 1
    assert torch.equal(actual, expected)
    assert actual.sum() == len(dataset)
    assert actual.dtype == torch.int64


def test_confusion_matrix_requires_eval_and_handles_empty_dataset():
    model = nn.Identity()
    dataset = TensorDataset(torch.empty((0, 10)), torch.empty(0, dtype=torch.int64))
    with pytest.raises(ValueError, match="evaluation-mode"):
        confusion_matrix(model, dataset)
    assert torch.equal(confusion_matrix(model.eval(), dataset), torch.zeros((10, 10), dtype=torch.int64))
