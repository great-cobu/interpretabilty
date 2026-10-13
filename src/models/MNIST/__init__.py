"""MNIST CNN package."""

from models.MNIST.config import Config, ConfigError, DataConfig, ModelConfig, OptimizerConfig, TrackingConfig, TrainerConfig, load_config
from models.MNIST.data import MNISTDataModule, inspect_data, prepare_data, resolve_accelerator
from models.MNIST.evaluate import confusion_matrix, load_classifier, predict, test_checkpoint
from models.MNIST.model import MNISTCNN, MNISTClassifier, inspect_model
from models.MNIST.train import learning_curves, train, train_demo

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "ConfigError",
    "DataConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainerConfig",
    "TrackingConfig",
    "load_config",
    "MNISTDataModule",
    "prepare_data",
    "inspect_data",
    "resolve_accelerator",
    "MNISTCNN",
    "MNISTClassifier",
    "inspect_model",
    "train",
    "train_demo",
    "learning_curves",
    "load_classifier",
    "test_checkpoint",
    "confusion_matrix",
    "predict",
]
