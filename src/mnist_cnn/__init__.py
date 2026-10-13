"""Alias mnist_cnn directly to models.MNIST modules so monkeypatching either works identically."""
import importlib
import sys
from models.MNIST import __version__

_SUBMODULES = (
    "checkpoints", "cli", "config", "data", "demo", "learning",
    "model", "model_inspection", "module", "page_export",
    "runs", "validation", "workflows",
)

for _name in _SUBMODULES:
    _mod = importlib.import_module(f"models.MNIST.{_name}")
    sys.modules[f"mnist_cnn.{_name}"] = _mod
    globals()[_name] = _mod

__all__ = ["__version__", *_SUBMODULES]
