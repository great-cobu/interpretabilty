"""CNN Architecture for MNIST implemented with PyTorch Lightning."""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.MNIST.config import MNISTCNNConfig


@dataclass
class ModelOutput:
    """Lightweight output container for model predictions, loss, and hidden states."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None

    def __iter__(self) -> Iterator[Any]:
        """Support unpacking: (loss, logits) if loss is present, else (logits,)."""
        if self.loss is not None:
            yield self.loss
        yield self.logits
        if self.hidden_states is not None:
            yield self.hidden_states

    def __getitem__(self, index: int) -> Any:
        return list(self)[index]


class MNISTCNN(pl.LightningModule):
    """Convolutional Neural Network for MNIST digit classification using PyTorch Lightning.

    Subclasses `pytorch_lightning.LightningModule` for modular, hardware-agnostic training
    and evaluation. Includes feature map extraction for interpretability (e.g. Grad-CAM).
    """

    def __init__(self, config: Optional[MNISTCNNConfig] = None, **kwargs: Any) -> None:
        super().__init__()

        if config is None:
            config = MNISTCNNConfig(**kwargs)
        elif kwargs:
            cfg_dict = config.to_dict()
            cfg_dict.update(kwargs)
            config = MNISTCNNConfig.from_dict(cfg_dict)

        self.config = config
        self.learning_rate = getattr(config, "learning_rate", 1e-3)
        self.weight_decay = getattr(config, "weight_decay", 1e-4)

        # Save hyperparameters for Lightning checkpointing
        self.save_hyperparameters(config.to_dict())

        c1, c2 = config.conv_channels[0], config.conv_channels[1]

        # Conv Block 1: (B, 1, 28, 28) -> (B, c1, 14, 14)
        self.conv1 = nn.Conv2d(config.in_channels, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=config.dropout_rate)

        # Conv Block 2: (B, c1, 14, 14) -> (B, c2, 7, 7)
        self.conv3 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(c2)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=config.dropout_rate)

        # Fully Connected Head: (B, c2 * 7 * 7) -> (B, num_classes)
        spatial_dim = config.image_size // 4  # 28 -> 14 -> 7
        flat_dim = c2 * spatial_dim * spatial_dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, config.fc_dim)
        self.bn_fc = nn.BatchNorm1d(config.fc_dim)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(config.fc_dim, config.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def get_last_conv_layer(self) -> nn.Module:
        """Return the final convolutional layer (useful for Grad-CAM and interpretability)."""
        return self.conv4

    def get_feature_maps(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate activation feature maps for model interpretability.

        Args:
            pixel_values: Input tensor of shape (B, 1, 28, 28) or (B, 28, 28).

        Returns:
            Dictionary containing intermediate feature map tensors for each layer.
        """
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(1)

        feature_maps: Dict[str, torch.Tensor] = {}
        x = self.relu1(self.bn1(self.conv1(pixel_values)))
        feature_maps["conv1"] = x
        x = self.relu2(self.bn2(self.conv2(x)))
        feature_maps["conv2"] = x
        x = self.dropout1(self.pool1(x))
        feature_maps["pool1"] = x

        x = self.relu3(self.bn3(self.conv3(x)))
        feature_maps["conv3"] = x
        x = self.relu4(self.bn4(self.conv4(x)))
        feature_maps["conv4"] = x
        x = self.dropout2(self.pool2(x))
        feature_maps["pool2"] = x

        return feature_maps

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        x: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[ModelOutput, Tuple[torch.Tensor, ...]]:
        """Forward pass through the CNN.

        Args:
            pixel_values: Image tensor of shape (B, 1, 28, 28) or (B, 28, 28).
            labels: Ground truth class indices of shape (B,).
            output_hidden_states: Whether to return intermediate hidden states.
            return_dict: Whether to return a ModelOutput or a tuple.
            x: Alternative alias for pixel_values.

        Returns:
            ModelOutput containing logits, optional loss, and optional hidden_states.
        """
        input_tensor = pixel_values if pixel_values is not None else x
        if input_tensor is None:
            raise ValueError("Either pixel_values or x must be provided to forward().")

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)

        # Block 1
        h1 = self.relu1(self.bn1(self.conv1(input_tensor)))
        h2 = self.relu2(self.bn2(self.conv2(h1)))
        p1 = self.dropout1(self.pool1(h2))

        # Block 2
        h3 = self.relu3(self.bn3(self.conv3(p1)))
        h4 = self.relu4(self.bn4(self.conv4(h3)))
        p2 = self.dropout2(self.pool2(h4))

        # FC Classification Head
        flat = self.flatten(p2)
        fc_out = self.dropout_fc(self.relu_fc(self.bn_fc(self.fc1(flat))))
        logits = self.classifier(fc_out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        hidden_states = None
        if output_hidden_states:
            hidden_states = (h1, h2, h3, h4, flat, fc_out)

        should_return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

        if not should_return_dict:
            res: Tuple[torch.Tensor, ...] = (logits,)
            if hidden_states is not None:
                res = res + (hidden_states,)
            return ((loss,) + res) if loss is not None else res

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )

    def _extract_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract inputs and targets from various batch formats."""
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            x = batch.get("pixel_values", batch.get("image", batch.get("x")))
            y = batch.get("labels", batch.get("label", batch.get("y")))
            if x is None or y is None:
                raise KeyError(f"Unable to extract x and y from batch dictionary keys: {list(batch.keys())}")
            return x, y
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning training step."""
        x, y = self._extract_batch(batch)
        out = self.forward(pixel_values=x, labels=y)
        loss = out.loss if isinstance(out, ModelOutput) else out[0]
        logits = out.logits if isinstance(out, ModelOutput) else out[1]

        acc = (logits.argmax(dim=-1) == y).float().mean()
        if getattr(self, "_trainer", None) is not None:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning validation step."""
        x, y = self._extract_batch(batch)
        out = self.forward(pixel_values=x, labels=y)
        loss = out.loss if isinstance(out, ModelOutput) else out[0]
        logits = out.logits if isinstance(out, ModelOutput) else out[1]

        acc = (logits.argmax(dim=-1) == y).float().mean()
        if getattr(self, "_trainer", None) is not None:
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning test step."""
        x, y = self._extract_batch(batch)
        out = self.forward(pixel_values=x, labels=y)
        loss = out.loss if isinstance(out, ModelOutput) else out[0]
        logits = out.logits if isinstance(out, ModelOutput) else out[1]

        acc = (logits.argmax(dim=-1) == y).float().mean()
        if getattr(self, "_trainer", None) is not None:
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike]) -> None:
        """Save model weights and configuration to a directory.

        Args:
            save_directory: Directory path to save model artifacts.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save state dict
        weights_path = os.path.join(save_directory, "model.pt")
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(
        cls,
        save_directory_or_file: Union[str, os.PathLike],
        **kwargs: Any,
    ) -> "MNISTCNN":
        """Load model from a checkpoint file or saved directory.

        Args:
            save_directory_or_file: Path to directory or checkpoint file (.ckpt or .pt).

        Returns:
            Loaded MNISTCNN instance.
        """
        path = str(save_directory_or_file)

        # Check if it's a PyTorch Lightning checkpoint file
        if os.path.isfile(path) and path.endswith(".ckpt"):
            return cls.load_from_checkpoint(path, **kwargs)

        if os.path.isdir(path):
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
                cfg_dict.update(kwargs)
                config = MNISTCNNConfig.from_dict(cfg_dict)
            else:
                config = MNISTCNNConfig(**kwargs)

            model = cls(config)

            # Try loading weights
            for candidate in ["model.pt", "pytorch_model.bin", "model.ckpt"]:
                weight_file = os.path.join(path, candidate)
                if os.path.exists(weight_file):
                    checkpoint = torch.load(weight_file, map_location="cpu")
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    # Strip 'model.' or similar prefix if present
                    clean_dict = {
                        (k[6:] if k.startswith("model.") else k): v
                        for k, v in state_dict.items()
                    }
                    model.load_state_dict(clean_dict, strict=False)
                    break
            return model

        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            clean_dict = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }
            hparams = checkpoint.get("hyper_parameters", {})
            hparams.update(kwargs)
            config = MNISTCNNConfig.from_dict(hparams) if hparams else MNISTCNNConfig(**kwargs)
            model = cls(config)
            model.load_state_dict(clean_dict, strict=False)
            return model

        raise FileNotFoundError(f"Cannot load MNISTCNN from: {path}")


# Alias for backwards compatibility
MNISTCNNForImageClassification = MNISTCNN
