"""Configuration classes for MNIST CNN model and PyTorch Lightning training pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class MNISTCNNConfig:
    """Configuration class for MNIST CNN architecture."""

    in_channels: int = 1
    num_classes: int = 10
    conv_channels: List[int] = field(default_factory=lambda: [32, 64])
    fc_dim: int = 128
    dropout_rate: float = 0.25
    image_size: int = 28
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    return_dict: bool = True

    def __post_init__(self) -> None:
        """Validate configuration settings."""
        if self.conv_channels is None:
            self.conv_channels = [32, 64]
        if len(self.conv_channels) < 2:
            raise ValueError("conv_channels must contain at least 2 layer dimensions")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MNISTCNNConfig":
        """Instantiate configuration from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class MNISTTrainingConfig:
    """Configuration class for PyTorch Lightning Trainer and training pipeline."""

    output_dir: str = "./results/mnist_cnn"
    max_epochs: int = 5
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_val_split: float = 0.1
    seed: int = 42
    accelerator: str = "auto"
    devices: Union[int, str, List[int]] = "auto"
    report_to: str = "none"
    fast_dev_run: bool = False
    log_every_n_steps: int = 50
    enable_checkpointing: bool = True

    def __init__(
        self,
        output_dir: str = "./results/mnist_cnn",
        max_epochs: int = 5,
        batch_size: int = 64,
        eval_batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        train_val_split: float = 0.1,
        seed: int = 42,
        accelerator: str = "auto",
        devices: Union[int, str, List[int]] = "auto",
        report_to: str = "none",
        fast_dev_run: bool = False,
        log_every_n_steps: int = 50,
        enable_checkpointing: bool = True,
        # Backward-compatible parameters:
        num_train_epochs: Optional[int] = None,
        per_device_train_batch_size: Optional[int] = None,
        per_device_eval_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.output_dir = output_dir
        self.max_epochs = num_train_epochs if num_train_epochs is not None else max_epochs
        self.batch_size = per_device_train_batch_size if per_device_train_batch_size is not None else batch_size
        self.eval_batch_size = per_device_eval_batch_size if per_device_eval_batch_size is not None else eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_val_split = train_val_split
        self.seed = seed
        self.accelerator = accelerator
        self.devices = devices
        self.report_to = report_to
        self.fast_dev_run = fast_dev_run
        self.log_every_n_steps = log_every_n_steps
        self.enable_checkpointing = enable_checkpointing

    @property
    def num_train_epochs(self) -> int:
        """Alias for max_epochs for backwards compatibility."""
        return self.max_epochs

    @property
    def per_device_train_batch_size(self) -> int:
        """Alias for batch_size for backwards compatibility."""
        return self.batch_size

    @property
    def per_device_eval_batch_size(self) -> int:
        """Alias for eval_batch_size for backwards compatibility."""
        return self.eval_batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "output_dir": self.output_dir,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "train_val_split": self.train_val_split,
            "seed": self.seed,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "report_to": self.report_to,
            "fast_dev_run": self.fast_dev_run,
            "log_every_n_steps": self.log_every_n_steps,
            "enable_checkpointing": self.enable_checkpointing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MNISTTrainingConfig":
        """Instantiate configuration from dictionary."""
        return cls(**data)
