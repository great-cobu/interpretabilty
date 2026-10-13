"""Multinomial Logistic Regression package for MNIST."""

from models.logistic.evaluate import confusion_matrix, load_classifier, predict, test_checkpoint
from models.logistic.model import LogisticClassifier, LogisticRegression, inspect_model
from models.logistic.train import fit_demo, train, train_demo
from models.MNIST import __version__

__all__ = [
    "__version__",
    "LogisticRegression",
    "LogisticClassifier",
    "inspect_model",
    "train",
    "fit_demo",
    "train_demo",
    "load_classifier",
    "test_checkpoint",
    "confusion_matrix",
    "predict",
]
