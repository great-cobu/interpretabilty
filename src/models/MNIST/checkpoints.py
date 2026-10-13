"""Checkpoint integrity, reconstruction settings, and stable best/last filenames."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle

from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from models.MNIST.config import ConfigError, load_config


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError("Checkpoint does not exist; supply a local .ckpt file")
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        try:
            expected = json.loads(manifest.read_text())[path.name]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("Checkpoint hash manifest is invalid or has no entry for this file") from exc
        if sha256(path) != expected:
            raise ValueError("Checkpoint SHA-256 does not match its manifest")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("Cannot load checkpoint; expected a compatible MNIST Lightning checkpoint") from exc
    if not isinstance(saved, dict) or saved.get("mnist_format_version") != 1:
        raise ValueError("Incompatible checkpoint format; expected MNIST format version 1")
    hyperparameters = saved.get("hyper_parameters")
    if (not isinstance(saved.get("state_dict"), dict) or not isinstance(hyperparameters, dict)
            or not isinstance(hyperparameters.get("config"), dict)):
        raise ValueError("Checkpoint is missing model weights or reconstruction configuration")
    load_config(base=hyperparameters["config"], use_environment=False)
    if any(type(saved.get(key)) is not int or saved[key] < 0 for key in ("epoch", "global_step")):
        raise ValueError("Checkpoint is missing valid epoch/global-step metadata")
    return saved


def _flatten(values: dict, prefix="") -> dict:
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def checkpoint_config(saved: dict, path, overrides: dict, *, resume: bool):
    """Start from saved settings and permit only explicit operational changes."""
    original = saved["hyper_parameters"]["config"]
    operational = {
        "output_dir", "data.directory", "data.download", "data.num_workers",
        "trainer.accelerator", "tracking.enabled", "tracking.uri", "tracking.experiment_name",
        "tracking.run_name",
    }
    operational |= {"trainer.max_epochs"} if resume else {"data.batch_size"}
    requested = dict(overrides)
    if path is not None:
        # Compare a complete validated profile to the saved configuration.
        profile = _flatten(asdict(load_config(path)))
        source = _flatten(original)
        requested = {k: v for k, v in profile.items() if v != source.get(k)} | requested
    source = _flatten(original)
    for key, value in requested.items():
        if key not in operational and value != source.get(key):
            raise ConfigError(f"Checkpoint fixes {key}; this setting cannot be changed")
    # Each invocation gets a new output directory unless explicitly supplied.
    base = dict(original, output_dir=None)
    config = load_config(base=base, overrides=requested)
    if resume:
        if not saved.get("optimizer_states") or not isinstance(saved.get("MNISTDataModule"), dict):
            raise ValueError("Resume requires optimizer and DataModule state in the checkpoint")
        if config.trainer.max_epochs <= saved["epoch"] + 1:
            raise ConfigError("Resume --max-epochs must exceed the completed epoch count (it is a total, not an increment)")
    return config, {k: v for k, v in requested.items() if v != source.get(k)}


class RunCheckpoint(ModelCheckpoint):
    """Keep best validation loss across resumed runs; always save the latest epoch."""

    def __init__(self, directory: Path):
        super().__init__(
            dirpath=str(directory.resolve()), filename="best", monitor="val/loss", mode="min",
            save_top_k=1, save_last=False, enable_version_counter=False,
            auto_insert_metric_name=False, save_on_train_epoch_end=True, save_weights_only=False,
        )

    def load_state_dict(self, state_dict):
        # Lightning normally discards best-score history when dirpath changes.
        # The workflow copies the historical best into this run before fit.
        state = dict(state_dict, dirpath=self.dirpath)
        for key in ("best_model_path", "kth_best_model_path", "last_model_path"):
            if state.get(key):
                state[key] = str(Path(self.dirpath) / Path(state[key]).name)
        state["best_k_models"] = {
            str(Path(self.dirpath) / Path(path).name): score
            for path, score in state.get("best_k_models", {}).items()
        }
        super().load_state_dict(state)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        # Explicitly write last even when this epoch did not improve val/loss.
        self.last_model_path = str(Path(self.dirpath) / "last.ckpt")
        trainer.save_checkpoint(self.last_model_path, weights_only=False)
        paths = [Path(self.best_model_path), Path(self.last_model_path)]
        (Path(self.dirpath) / "manifest.json").write_text(
            json.dumps({p.name: sha256(p) for p in paths}, indent=2) + "\n"
        )
