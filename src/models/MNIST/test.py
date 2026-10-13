"""Testing, evaluation, and inference routines for MNIST CNN using PyTorch Lightning and PyTorch."""

import argparse
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.MNIST.data import (
    MNISTDataCollator,
    load_mnist_datasets,
    preprocess_image,
)
from models.MNIST.model import MNISTCNN, ModelOutput


def get_default_device() -> torch.device:
    """Detect and return the best available compute device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_mnist(
    model: Union[str, os.PathLike, MNISTCNN],
    test_dataset: Optional[Any] = None,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained MNIST CNN on the MNIST test set.

    Args:
        model: Either a loaded MNISTCNN instance or a path to a saved model directory/checkpoint.
        test_dataset: Optional pre-loaded test dataset split.
        batch_size: Evaluation batch size.
        device: Device to run evaluation on (defaults to best available: CUDA/MPS/CPU).
        verbose: Whether to print classification report and summary tables.

    Returns:
        Dictionary containing:
            - "test_loss": float
            - "test_accuracy": float
            - "classification_report": dict
            - "confusion_matrix": ndarray
            - "predictions": ndarray
            - "labels": ndarray
            - "probabilities": ndarray
    """
    dev = torch.device(device) if device else get_default_device()

    # 1. Load model if a path was supplied
    if isinstance(model, (str, os.PathLike)):
        model_instance = MNISTCNN.from_pretrained(model)
    else:
        model_instance = model

    model_instance.to(dev)
    model_instance.eval()

    # 2. Load test dataset if not supplied
    if test_dataset is None:
        datasets = load_mnist_datasets(train_val_split=0.0)
        test_dataset = datasets["test"]

    # 3. Create DataLoader
    collator = MNISTDataCollator()
    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    all_losses: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    # 4. Evaluation Loop
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                pixel_values = batch[0].to(dev)
                labels = batch[1].to(dev)
            else:
                pixel_values = batch["pixel_values"].to(dev)
                labels = batch["labels"].to(dev)

            outputs = model_instance(pixel_values=pixel_values)
            logits = outputs.logits if isinstance(outputs, ModelOutput) else (outputs[0] if isinstance(outputs, tuple) else outputs)

            loss = loss_fn(logits, labels)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    total_samples = len(all_labels)
    test_loss = sum(all_losses) / max(total_samples, 1)
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    test_accuracy = float(np.mean(preds_arr == labels_arr)) if total_samples > 0 else 0.0
    labels_to_eval = list(range(10))
    cls_report = classification_report(
        labels_arr,
        preds_arr,
        labels=labels_to_eval,
        digits=4,
        output_dict=True,
        target_names=[str(i) for i in labels_to_eval],
        zero_division=0,
    )
    conf_matrix = confusion_matrix(labels_arr, preds_arr, labels=labels_to_eval)

    if verbose:
        print("=" * 60)
        print("MNIST CNN Test Evaluation Results (PyTorch Lightning)")
        print("=" * 60)
        print(f"Test Loss:     {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}% ({np.sum(preds_arr == labels_arr)}/{total_samples})")
        print("\nClassification Report:")
        print(classification_report(
            labels_arr,
            preds_arr,
            labels=labels_to_eval,
            digits=4,
            target_names=[f"Digit {i}" for i in labels_to_eval],
            zero_division=0,
        ))
        print("Confusion Matrix (rows: true, cols: pred):")
        print(conf_matrix)
        print("=" * 60)

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": cls_report,
        "confusion_matrix": conf_matrix,
        "predictions": preds_arr,
        "labels": labels_arr,
        "probabilities": probs_arr,
    }


def predict(
    model: Union[str, os.PathLike, MNISTCNN],
    images: Union[Image.Image, np.ndarray, torch.Tensor, List[Any]],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Run inference on one or more MNIST images.

    Args:
        model: Either a loaded MNISTCNN instance or a path to a saved model directory/checkpoint.
        images: Single image or list of images (PIL, NumPy array, or Torch Tensor).
        device: Computation device (auto-detected if None).

    Returns:
        Dictionary containing:
            - "predictions": list of predicted digit labels (0-9)
            - "probabilities": ndarray of class probabilities
            - "confidence": list of max class probabilities
            - "logits": ndarray of raw model logits
    """
    dev = torch.device(device) if device else get_default_device()

    if isinstance(model, (str, os.PathLike)):
        model_instance = MNISTCNN.from_pretrained(model)
    else:
        model_instance = model

    model_instance.to(dev)
    model_instance.eval()

    if not isinstance(images, list):
        images_list = [images]
    else:
        images_list = images

    tensors = [preprocess_image(img, normalize=True) for img in images_list]
    batch_tensor = torch.stack(tensors).to(dev)

    with torch.no_grad():
        outputs = model_instance(pixel_values=batch_tensor)
        logits = outputs.logits if isinstance(outputs, ModelOutput) else (outputs[0] if isinstance(outputs, tuple) else outputs)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    preds_list = preds.cpu().numpy().tolist()
    probs_arr = probs.cpu().numpy()
    conf_list = probs_arr[np.arange(len(preds_list)), preds_list].tolist()

    return {
        "predictions": preds_list,
        "probabilities": probs_arr,
        "confidence": conf_list,
        "logits": logits.cpu().numpy(),
    }


def main() -> None:
    """CLI entry point for testing."""
    parser = argparse.ArgumentParser(description="Test CNN on MNIST test dataset.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./results/mnist_cnn",
        help="Path to saved model checkpoint directory or file",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")

    args = parser.parse_args()

    test_mnist(
        model=args.model_dir,
        batch_size=args.batch_size,
        device=args.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()
