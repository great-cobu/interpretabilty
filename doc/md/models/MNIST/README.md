# MNIST CNN Architecture & PyTorch Lightning Implementation

This document details the reimplementation of the `models.MNIST` package using **PyTorch Lightning** and **Torchvision**, eliminating all dependencies on **Hugging Face** (`transformers`, `datasets`, `huggingface-hub`, `tokenizers`, and `accelerate`).

---

## 1. Overview & Architecture

The `models.MNIST` package provides an end-to-end convolutional neural network (CNN) pipeline tailored for MNIST handwritten digit recognition, complete with intermediate feature map extraction for interpretability analysis (e.g., Grad-CAM).

### Architecture Highlights
- **Backbone**: Two convolutional blocks with Batch Normalization, ReLU activations, 2D Dropout, and Max Pooling.
  - Conv Block 1: `Conv2d(in_channels, c1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(c1, c1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` $\rightarrow$ `Dropout2d`
  - Conv Block 2: `Conv2d(c1, c2)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(c2, c2)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` $\rightarrow$ `Dropout2d`
- **Classifier Head**: Dense linear layers with Batch Normalization, Dropout (0.5), and Cross-Entropy loss.
  - Flatten $\rightarrow$ `Linear(c2 * 7 * 7, fc_dim)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(fc_dim, num_classes)`
- **Interpretability Hooks**:
  - `get_last_conv_layer()`: Returns reference to `conv4` for gradient-based attribution (Grad-CAM).
  - `get_feature_maps(pixel_values)`: Extracts intermediate activations across conv and pool layers.

---

## 2. Package Structure

The package lives under [`src/models/MNIST/`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST) and consists of the following components:

| Module | Description |
| :--- | :--- |
| [`config.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/config.py) | Standalone dataclasses (`MNISTCNNConfig`, `MNISTTrainingConfig`) without Hugging Face dependencies. |
| [`model.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/model.py) | `MNISTCNN` derived from `pytorch_lightning.LightningModule` with `ModelOutput` container and interpretability methods. |
| [`data.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/data.py) | `MNISTDataModule`, `MNISTDataset`, and `MNISTDataCollator` using Torchvision and pure PyTorch. |
| [`train.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/train.py) | Training execution via `pytorch_lightning.Trainer`, metric calculation, and optional MLflow tracking. |
| [`test.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/test.py) | Evaluation routines (`test_mnist`), inference (`predict`), classification reports, and confusion matrices. |
| [`__init__.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/__init__.py) | Unified exports for the subpackage. |

---

## 3. Detailed Component Breakdown

### Configuration ([`config.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/config.py))
Replaced `transformers.PretrainedConfig` and `TrainingArguments` with Python `@dataclass`:
- **`MNISTCNNConfig`**: Defines hyperparameters for channel depth (`conv_channels`), dense dimension (`fc_dim`), dropout rate, and image size. Supports serialization via `to_dict()` and `from_dict()`.
- **`MNISTTrainingConfig`**: Stores training configuration parameters including `output_dir`, `max_epochs`, `batch_size`, `eval_batch_size`, `learning_rate`, `weight_decay`, `accelerator`, and `devices`.

### Model ([`model.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/model.py))
- Inherits from `pytorch_lightning.LightningModule`.
- Implements Lightning lifecycle hooks:
  - `training_step(batch, batch_idx)`: Computes loss and training accuracy; logs metrics to active progress bar/logger.
  - `validation_step(batch, batch_idx)`: Evaluates validation loss and accuracy.
  - `test_step(batch, batch_idx)`: Logs test loss and accuracy.
  - `configure_optimizers()`: Returns `torch.optim.AdamW`.
- **Dual Batch Input Support**: Transparently unpacks either dictionary batches (`{"pixel_values": x, "labels": y}`) or tuple batches (`(x, y)`).
- **`ModelOutput` Dataclass**: Emulates standard model return types with `.logits`, `.loss`, and `.hidden_states`, while also supporting tuple unpacking.
- **Persistence**:
  - `save_pretrained(save_directory)`: Saves `config.json` and `model.pt`.
  - `from_pretrained(save_directory_or_file)`: Automatically detects directory checkpoints, `.pt` state dicts, or Lightning `.ckpt` checkpoints.

### Data & DataModule ([`data.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/data.py))
- Replaces Hugging Face `datasets` with Torchvision:
  - **`MNISTDataModule`**: A `pytorch_lightning.LightningDataModule` handling dataset downloads, train/validation splitting, and DataLoader generation (`train_dataloader()`, `val_dataloader()`, `test_dataloader()`).
  - **`MNISTDataset`**: A custom `torch.utils.data.Dataset` accepting in-memory tensors or numpy arrays for synthetic data testing or offline training.
  - **`preprocess_image(img, normalize=True)`**: Standardizes PIL Images, numpy arrays, or torch tensors to shape `(1, 28, 28)`.
  - **`MNISTDataCollator`**: Batches list of sample dictionaries or tuples into stacked tensors.

### Training ([`train.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/train.py))
- Function **`train_mnist(...)`**:
  - Sets global seed using `pytorch_lightning.seed_everything(seed, workers=True)`.
  - Configures `ModelCheckpoint` to track `val_loss` (or `train_loss`).
  - Configures `MLFlowLogger` when `--report_to mlflow` is selected.
  - Runs training via `pl.Trainer.fit(...)`.
  - Saves the resulting model to `output_dir`.
- Function **`compute_metrics(...)`**: Standalone evaluation computing accuracy, weighted precision, recall, and F1 score without Hugging Face `EvalPrediction`.

### Testing & Inference ([`test.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/MNIST/test.py))
- Function **`test_mnist(...)`**:
  - Accepts a model instance or a path to a checkpoint / directory.
  - Runs evaluation across the MNIST test split.
  - Returns a dictionary with `test_loss`, `test_accuracy`, `classification_report`, `confusion_matrix`, `predictions`, `labels`, and `probabilities`.
- Function **`predict(...)`**:
  - Accepts PIL Images, numpy arrays, or tensors.
  - Returns predicted class labels, full probability distribution, confidence scores, and raw logits.

---

## 4. Dependencies & Environment

The following Hugging Face libraries were removed from [`pyproject.toml`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/pyproject.toml) and [`requirements.txt`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/requirements.txt):
- `transformers`
- `datasets`
- `huggingface-hub`
- `tokenizers`
- `accelerate`

Added dependencies:
- `pytorch-lightning>=2.6.6`
- `torchvision>=0.28.0`

---

## 5. Usage Examples

### CLI Execution

Train the model:
```bash
uv run python -m models.main train \
    --epochs 5 \
    --batch_size 64 \
    --lr 0.001 \
    --accelerator auto \
    --output_dir ./results/mnist_cnn
```

Evaluate a saved checkpoint:
```bash
uv run python -m models.main test \
    --model_dir ./results/mnist_cnn \
    --batch_size 128
```

### Python API Execution

#### Training with PyTorch Lightning
```python
from models.MNIST import MNISTCNN, MNISTCNNConfig, MNISTDataModule, train_mnist

# 1. Initialize configuration and model
config = MNISTCNNConfig(conv_channels=[32, 64], fc_dim=128)
model = MNISTCNN(config)

# 2. Train using PyTorch Lightning
trained_model, trainer, metrics = train_mnist(
    model=model,
    epochs=5,
    batch_size=64,
    accelerator="auto",
    output_dir="./results/mnist_cnn",
)
print("Final metrics:", metrics)
```

#### Inference & Interpretability
```python
from PIL import Image
from models.MNIST import MNISTCNN, predict

# 1. Load model from disk
model = MNISTCNN.from_pretrained("./results/mnist_cnn")

# 2. Run prediction
img = Image.open("sample_digit.png")
result = predict(model, img)
print("Predicted digit:", result["predictions"][0])
print("Confidence:", result["confidence"][0])

# 3. Extract feature maps for interpretability / Grad-CAM
import torch
from models.MNIST import preprocess_image

tensor = preprocess_image(img)  # (1, 28, 28)
feature_maps = model.get_feature_maps(tensor)
print("Conv4 activation shape:", feature_maps["conv4"].shape)
```

---

## 6. Unit Testing

All unit tests in [`tests/test_mnist.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/tests/test_mnist.py) test model architecture, data transformations, PyTorch Lightning training steps, metrics computation, and end-to-end training/testing without any Hugging Face dependencies.

Run the test suite:
```bash
uv run python -m unittest discover tests -v
```
Output:
```text
Ran 13 tests in 0.434s

OK
```

---

## 7. Related Documentation

- [MLflow Integration Plan](file:///Users/frankelliott/Documents/Projects/Python/interpetability/doc/md/models/MNIST/MLFLOW_PLAN.md): End-to-end plan for experiment tracking, model registry, artifact logging, and interpretability visualization using the project's MariaDB & MLflow Docker stack.

