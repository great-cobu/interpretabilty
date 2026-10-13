"""Lightning learning behavior around the plain PyTorch CNN."""

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from models.MNIST.config import Config, config_dict, load_config
from models.MNIST.model import MNISTCNN


class MNISTClassifier(pl.LightningModule):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = load_config(base=config, use_environment=False) if config is not None else Config()
        self.save_hyperparameters({"config": config_dict(self.config)})
        self.network = MNISTCNN(dropout=self.config.model.dropout)
        self.accuracy = nn.ModuleDict({
            f"{phase}_accuracy": MulticlassAccuracy(num_classes=10, average="micro")
            for phase in ("train", "val", "test")
        })
        self._resume_rng = None

    def forward(self, images):
        return self.network(images)

    def _step(self, batch, phase):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {phase} loss; check inputs and optimization settings")
        metric = self.accuracy[f"{phase}_accuracy"]
        metric.update(logits, labels)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, batch_size=len(labels))
        self.log(f"{phase}/accuracy", metric, on_step=False, on_epoch=True)
        if phase == "test":
            self.log("test/count", float(len(labels)), on_step=False, on_epoch=True, reduce_fx="sum")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        return self(images)

    def on_train_epoch_start(self):
        self.accuracy["train_accuracy"].reset()

    def on_validation_epoch_start(self):
        self.accuracy["val_accuracy"].reset()

    def on_test_epoch_start(self):
        self.accuracy["test_accuracy"].reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["mnist_format_version"] = 1
        checkpoint["torch_rng_state"] = torch.get_rng_state()

    def on_load_checkpoint(self, checkpoint):
        self._resume_rng = checkpoint.get("torch_rng_state")

    def on_train_start(self):
        if self._resume_rng is not None:
            torch.set_rng_state(self._resume_rng.cpu())
            self._resume_rng = None
