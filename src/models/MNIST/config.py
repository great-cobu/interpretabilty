"""Typed configuration with dataclass validation and YAML serialization."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from uuid import uuid4
import yaml


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


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


VALID_ACTIVATIONS = ("relu", "softplus", "smu", "SMU")


@dataclass(frozen=True)
class ModelConfig:
    dropout: float = 0.2
    activation: str = "relu"


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

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build_section(cls, values: dict):
    if not isinstance(values, dict):
        raise ConfigError(f"{cls.__name__} must be a mapping")
    allowed = {f.name for f in fields(cls)}
    if values.keys() - allowed:
        raise ConfigError(f"Unknown key in {cls.__name__}: {sorted(values.keys() - allowed)}")
    return cls(**values)


def validate(c: Config) -> None:
    if not isinstance(c.data.directory, str) or not c.data.directory.strip():
        raise ConfigError("data.directory must not be empty")
    if c.data.batch_size <= 0 or c.trainer.max_epochs <= 0:
        raise ConfigError("batch_size and max_epochs must be positive")
    if min(c.data.num_workers, c.data.split_seed, c.trainer.seed, c.trainer.patience) < 0:
        raise ConfigError("num_workers, seeds, and patience must be nonnegative")
    if c.data.std <= 0 or not 0 <= c.data.mean <= 1:
        raise ConfigError("Normalization requires positive std and mean in [0, 1]")
    if not (isinstance(c.model.dropout, (int, float)) and math.isfinite(c.model.dropout) and 0 <= c.model.dropout < 1):
        raise ConfigError("model.dropout must be in [0, 1)")
    if not isinstance(c.model.activation, str) or c.model.activation not in VALID_ACTIVATIONS:
        raise ConfigError("model.activation must be relu, softplus, or SMU")
    if c.optimizer.learning_rate <= 0 or c.optimizer.weight_decay < 0:
        raise ConfigError("Optimizer requires positive learning_rate and nonnegative weight_decay")
    if c.trainer.accelerator not in {"auto", "cpu", "cuda", "mps"}:
        raise ConfigError("accelerator must be auto, cpu, cuda, or mps")
    for name in ("limit_train_batches", "limit_val_batches"):
        val = getattr(c.trainer, name)
        if val is not None and (type(val) is not int or val <= 0):
            raise ConfigError(f"trainer.{name} must be a positive integer or null")
    if c.tracking.enabled and not c.tracking.uri:
        raise ConfigError("Tracking requires --tracking-uri, MLFLOW_TRACKING_URI, or tracking.uri")


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
    *,
    base: dict | None = None,
    use_environment: bool = True,
) -> Config:
    """Resolve defaults < YAML < environment (MLFLOW_TRACKING_URI) < CLI overrides."""
    raw = deepcopy(base) if base is not None else {}
    if path is not None:
        try:
            loaded = yaml.safe_load(Path(path).read_text())
        except OSError as exc:
            raise ConfigError(f"Cannot read configuration file: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError("Invalid configuration YAML") from exc
        raw = loaded or {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    allowed = {f.name for f in fields(Config)}
    if raw.keys() - allowed:
        raise ConfigError(f"Unknown top-level configuration key: {sorted(raw.keys() - allowed)}")

    if use_environment and os.environ.get("MLFLOW_TRACKING_URI"):
        raw.setdefault("tracking", {})["uri"] = os.environ["MLFLOW_TRACKING_URI"]

    for key, value in (overrides or {}).items():
        parts = key.split(".")
        if len(parts) == 1 and key in allowed:
            raw[key] = value
        elif len(parts) == 2 and parts[0] in allowed - {"output_dir"}:
            raw.setdefault(parts[0], {})[parts[1]] = value
        else:
            raise ConfigError(f"Unknown configuration override: {key}")

    config = Config(
        data=_build_section(DataConfig, raw.get("data", {})),
        model=_build_section(ModelConfig, raw.get("model", {})),
        optimizer=_build_section(OptimizerConfig, raw.get("optimizer", {})),
        trainer=_build_section(TrainerConfig, raw.get("trainer", {})),
        tracking=_build_section(TrackingConfig, raw.get("tracking", {})),
        output_dir=raw.get("output_dir"),
    )
    validate(config)
    return config


def config_dict(config: Config) -> dict:
    return config.to_dict()


def prepare_run_dir(config: Config, command: str) -> tuple[Config, Path]:
    """Create or prepare output run directory and write resolved config.yaml."""
    from dataclasses import replace

    path = Path(config.output_dir) if config.output_dir else Path("runs") / (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{command}-{uuid4().hex[:8]}"
    )
    path.mkdir(parents=True, exist_ok=True)
    resolved = replace(config, output_dir=str(path.resolve()))
    resolved.to_yaml(path / "config.yaml")
    return resolved, path
