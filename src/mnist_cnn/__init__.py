"""Compatibility alias for mnist_cnn -> models.MNIST."""
import importlib
import sys
from models.MNIST import *  # noqa: F401, F403
from models.MNIST import __all__, __version__

for _name in ("cli", "config", "data", "evaluate", "model", "train"):
    _mod = importlib.import_module(f"models.MNIST.{_name}")
    sys.modules[f"mnist_cnn.{_name}"] = _mod
    globals()[_name] = _mod
