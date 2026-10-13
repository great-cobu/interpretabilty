"""Small reusable workflows called by the CLI and, later, demonstrations."""

import json
from pathlib import Path

from models.MNIST.config import Config
from models.MNIST.data import MNISTDataModule, resolve_accelerator


def prepare_data(config: Config) -> dict:
    module = MNISTDataModule(config.data)
    module.prepare_data()
    return {"dataset": "MNIST", "data_directory": str(Path(config.data.directory).resolve()),
            "download_enabled": config.data.download}


def inspect_data(config: Config, output: Path) -> dict:
    """Inspect shapes and membership without fitting a model or computing test scores."""
    accelerator = resolve_accelerator(config.trainer.accelerator)
    module = MNISTDataModule(config.data, accelerator=accelerator)
    module.prepare_data()
    module.setup()
    module.save_split_indices(output / "split_indices.json")
    samples = {}
    for name, loader in (("train", module.train_dataloader()),
                         ("validation", module.val_dataloader()),
                         ("test", module.test_dataloader())):
        images, labels = next(iter(loader))
        samples[name] = {
            "count": len(loader.dataset), "batch_shape": list(images.shape),
            "image_dtype": str(images.dtype), "label_dtype": str(labels.dtype),
            "batch_label_range": [int(labels.min()), int(labels.max())],
            "batch_pixel_range": [float(images.min()), float(images.max())],
            "drop_last": loader.drop_last, "pin_memory": loader.pin_memory,
        }
    report = {"dataset": "MNIST", "accelerator": accelerator,
              "data_directory": str(Path(config.data.directory).resolve()),
              "split_seed": config.data.split_seed, "splits": samples}
    (output / "data_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
