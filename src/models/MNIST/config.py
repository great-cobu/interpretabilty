"""Typed configuration with strict validation and safe, reproducible serialization."""

from dataclasses import asdict, dataclass, field, fields
from copy import deepcopy
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import yaml


class ConfigError(ValueError):
    """A configuration cannot be used safely or unambiguously."""


@dataclass(frozen=True)
class DataConfig:
    directory: str = "data"
    download: bool = True
    split_seed: int = 42
    batch_size: int = 64
    num_workers: int = 0
    mean: float = 0.1307
    std: float = 0.3081
    split_indices: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    dropout: float = 0.2


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 0.001
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainerConfig:
    seed: int = 42
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    deterministic: bool = True
    patience: int = 3
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    uri: str | None = None
    experiment_name: str = "mnist-cnn"
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output_dir: str | None = None


# Credentials belong in environment variables, never configuration artifacts.
_SECRET = re.compile(r"password|passwd|token|secret|credential|api[_-]?key|access[_-]?key", re.I)


def sanitize(value):
    """Remove sensitive keys and URL credentials/query strings recursively."""
    if isinstance(value, dict):
        return {k: "<redacted>" if _SECRET.search(str(k)) else sanitize(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
            host = parsed.netloc.rsplit("@", 1)[-1]
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except ValueError:
            return "<redacted-url>"
    return value


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently losing settings."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ConfigError("Configuration keys must be strings")
        if key in result:
            raise ConfigError("Duplicate configuration key")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _section(cls, values):
    if not isinstance(values, dict):
        raise ConfigError(f"{cls.__name__} must be a mapping")
    defaults = cls()
    allowed = {f.name for f in fields(cls)}
    if values.keys() - allowed:
        raise ConfigError(f"Unknown configuration key in {cls.__name__}")
    for name, value in values.items():
        default = getattr(defaults, name)
        expected = type(default)
        if name in {"limit_train_batches", "limit_val_batches"}:
            valid = value is None or (type(value) is int and value > 0)
        elif default is None:
            valid = value is None or (isinstance(value, str) and bool(value.strip()))
        elif expected is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = type(value) is expected
        if not valid:
            raise ConfigError(f"Invalid type or value for {cls.__name__}.{name}")
    return cls(**values)


def load_config(path: str | Path | None = None, overrides: dict | None = None, *, base: dict | None = None, use_environment: bool = True) -> Config:
    """Resolve defaults < YAML < environment (tracking URI only) < CLI."""
    raw = deepcopy(base) if base is not None else {}
    if path is not None:
        try:
            raw = yaml.load(Path(path).read_text(), Loader=_UniqueLoader)
        except FileNotFoundError as exc:
            raise ConfigError("Configuration file not found; --config paths are relative to the current working directory. Run from the project root, use an absolute path, or omit --config to use built-in defaults") from exc
        except PermissionError as exc:
            raise ConfigError("Cannot read configuration; check file access permissions") from exc
        except (OSError, UnicodeError) as exc:
            raise ConfigError("Cannot read configuration; provide a readable UTF-8 YAML file") from exc
        except yaml.YAMLError as exc:
            raise ConfigError("Invalid configuration YAML; check indentation and syntax") from exc
        if raw is None:
            raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    allowed = {f.name for f in fields(Config)}
    if raw.keys() - allowed:
        raise ConfigError("Unknown top-level configuration key")
    if use_environment and os.environ.get("MLFLOW_TRACKING_URI"):
        tracking = raw.setdefault("tracking", {})
        if not isinstance(tracking, dict):
            raise ConfigError("tracking must be a mapping")
        tracking["uri"] = os.environ["MLFLOW_TRACKING_URI"]
    for key, value in (overrides or {}).items():
        parts = key.split(".")
        if len(parts) == 1:
            if key not in allowed:
                raise ConfigError("Unknown configuration override")
            raw[key] = value
        elif len(parts) == 2 and parts[0] in allowed - {"output_dir"}:
            section = raw.setdefault(parts[0], {})
            if not isinstance(section, dict):
                raise ConfigError(f"{parts[0]} must be a mapping")
            section[parts[1]] = value
        else:
            raise ConfigError("Unknown configuration override")
    config = Config(
        data=_section(DataConfig, raw.get("data", {})),
        model=_section(ModelConfig, raw.get("model", {})),
        optimizer=_section(OptimizerConfig, raw.get("optimizer", {})),
        trainer=_section(TrainerConfig, raw.get("trainer", {})),
        tracking=_section(TrackingConfig, raw.get("tracking", {})),
        output_dir=raw.get("output_dir"),
    )
    validate(config)
    return config


def validate(c: Config) -> None:
    if not c.data.directory.strip():
        raise ConfigError("data.directory must not be empty")
    for name, value in (("batch_size", c.data.batch_size), ("max_epochs", c.trainer.max_epochs)):
        if value <= 0:
            raise ConfigError(f"{name} must be positive")
    for name, value in (("num_workers", c.data.num_workers), ("split_seed", c.data.split_seed),
                        ("seed", c.trainer.seed), ("patience", c.trainer.patience)):
        if value < 0:
            raise ConfigError(f"{name} must be nonnegative")
    if max(c.data.split_seed, c.trainer.seed) > 2**32 - 1:
        raise ConfigError("Seeds must fit in an unsigned 32-bit integer")
    if c.data.std <= 0 or not 0 <= c.data.mean <= 1:
        raise ConfigError("Normalization requires positive std and mean in [0, 1]")
    if not 0 <= c.model.dropout < 1:
        raise ConfigError("model.dropout must be in [0, 1)")
    if c.optimizer.learning_rate <= 0 or c.optimizer.weight_decay < 0:
        raise ConfigError("Optimizer requires positive learning_rate and nonnegative weight_decay")
    if c.trainer.accelerator not in {"auto", "cpu", "cuda", "mps"}:
        raise ConfigError("accelerator must be auto, cpu, cuda, or mps")
    if c.trainer.devices != 1 or c.trainer.precision != "32-true":
        raise ConfigError("Version one supports devices=1 and precision=32-true")
    if c.output_dir is not None and (not isinstance(c.output_dir, str) or not c.output_dir.strip()):
        raise ConfigError("output_dir must be a nonempty path or null")
    if not c.tracking.experiment_name.strip():
        raise ConfigError("tracking.experiment_name must not be empty")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in c.tracking.tags.items()):
        raise ConfigError("tracking.tags must map strings to strings")
    if c.tracking.enabled and not c.tracking.uri:
        raise ConfigError("Tracking requires --tracking-uri, MLFLOW_TRACKING_URI, or tracking.uri")


def config_dict(config: Config) -> dict:
    return sanitize(asdict(config))
