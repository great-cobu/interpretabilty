"""Testing, evaluation, interpretability visualizations, and MLflow logging for LogisticClassifier."""

import json
from pathlib import Path
import time

import lightning.pytorch as pl
import torch

from models.MNIST.config import Config, config_dict
from models.MNIST.data import MNISTDataModule
from models.MNIST.evaluate import confusion_matrix
from models.MNIST.train import environment, sha256, trainer_options
from models.logistic.model import LogisticClassifier


def load_classifier(checkpoint_path: str | Path, device: str = "cpu") -> LogisticClassifier:
    """Load a LogisticClassifier from a Lightning checkpoint file."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return LogisticClassifier.load_from_checkpoint(
        str(path),
        map_location=device,
        weights_only=True,
    ).eval()


def save_class_templates_plot(model: LogisticClassifier, destination: str | Path) -> Path:
    """Render the 10 learned 28x28 class weight templates W_0..W_9 with diverging colormap."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dest = Path(destination)
    templates = model.get_class_templates().detach().cpu().numpy()
    biases = model.network.layers.logits.bias.detach().cpu().numpy()
    vmax = float(max(abs(templates.min()), abs(templates.max()), 1e-6))

    fig = Figure(figsize=(11, 4.8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 5)
    im = None
    for digit, ax in enumerate(axes.flat):
        im = ax.imshow(templates[digit], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Class {digit} (b={biases[digit]:+.2f})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Weight W[c, i, j]")
    fig.suptitle("Learned Logistic Regression Class Weight Templates (W_0 .. W_9)", fontsize=11, fontweight="bold")
    fig.savefig(dest, dpi=150)
    return dest


def save_confusion_matrix_plot(matrix: torch.Tensor | list[list[int]], destination: str | Path) -> Path:
    """Render a 10x10 confusion matrix heatmap with cell count annotations."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dest = Path(destination)
    tensor = matrix if isinstance(matrix, torch.Tensor) else torch.tensor(matrix, dtype=torch.int64)
    arr = tensor.detach().cpu().numpy()

    fig = Figure(figsize=(6.5, 5.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Sample Count")

    thresh = arr.max() / 2.0 if arr.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = int(arr[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=7.5,
            )
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_title("MNIST Test Split Confusion Matrix (10x10)", fontweight="bold")
    fig.savefig(dest, dpi=150)
    return dest


def save_sample_attributions_plot(
    model: LogisticClassifier,
    data: MNISTDataModule,
    destination: str | Path,
    *,
    num_samples: int = 5,
) -> Path:
    """Render sample test digits alongside exact per-pixel attributions (x * W_c)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dest = Path(destination)
    data.prepare_data()
    data.setup("test")
    batch_images, batch_labels = next(iter(data.test_dataloader()))
    n = min(num_samples, batch_images.shape[0])
    images = batch_images[:n]
    labels = batch_labels[:n]

    pred_out = predict(model, images)
    preds = pred_out["predictions"]
    confs = pred_out["confidence"]
    attributions = model.get_pixel_attribution(images, target_class=preds).squeeze(1).detach().cpu().numpy()
    imgs_np = images.squeeze(1).detach().cpu().numpy()

    vmax = float(max(abs(attributions.min()), abs(attributions.max()), 1e-6))
    fig = Figure(figsize=(2.3 * n, 4.6), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, n, squeeze=False)
    for idx in range(n):
        ax_img = axes[0, idx]
        ax_attr = axes[1, idx]
        ax_img.imshow(imgs_np[idx], cmap="gray")
        ax_img.set_title(
            f"True: {int(labels[idx])} | Pred: {int(preds[idx])}\nConf: {float(confs[idx]):.1%}",
            fontsize=8.5,
        )
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_attr.imshow(imgs_np[idx], cmap="gray", alpha=0.25)
        ax_attr.imshow(attributions[idx], cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.85)
        ax_attr.set_title(f"x ⊙ W_{int(preds[idx])} Attribution", fontsize=8.5)
        ax_attr.set_xticks([])
        ax_attr.set_yticks([])

    fig.suptitle("Exact Per-Pixel Logit Attribution (x ⊙ W_pred)", fontsize=10.5, fontweight="bold")
    fig.savefig(dest, dpi=150)
    return dest


def test_checkpoint(config: Config, output: Path, checkpoint_path: str | Path) -> dict:
    """Measure a fixed LogisticClassifier checkpoint on the entire ordered test split."""
    pl.seed_everything(config.trainer.seed, workers=True)
    options = trainer_options(config)
    model = load_classifier(checkpoint_path, device="cpu")
    data = MNISTDataModule(config.data, accelerator=options["accelerator"])
    trainer = pl.Trainer(**options, logger=False, enable_checkpointing=False, default_root_dir=str(output))

    started = time.perf_counter()
    results = trainer.test(model, datamodule=data, verbose=False)[0]
    elapsed = time.perf_counter() - started
    count = int(results["test/count"])

    data.setup("test")
    cm = confusion_matrix(model, data.test_dataset, batch_size=config.data.batch_size)
    cm_path = save_confusion_matrix_plot(cm, output / "confusion_matrix.png")
    templates_path = save_class_templates_plot(model, output / "class_templates.png")
    attr_path = save_sample_attributions_plot(model, data, output / "sample_attributions.png")

    report = {
        "model": "LogisticRegression",
        "test/loss": float(results["test/loss"]),
        "test/accuracy": float(results["test/accuracy"]),
        "sample_count": count,
        "confusion_matrix": cm.tolist(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "artifacts": {
            "confusion_matrix_plot": str(cm_path.resolve()),
            "class_templates_plot": str(templates_path.resolve()),
            "sample_attributions_plot": str(attr_path.resolve()),
        },
        "config": config_dict(config),
        "accelerator": options["accelerator"],
        "environment": environment(),
        "elapsed_seconds": elapsed,
    }
    metrics_file = output / "metrics.json"
    metrics_file.write_text(json.dumps(report, indent=2) + "\n")

    if config.tracking.enabled and config.tracking.uri:
        _log_evaluation_to_mlflow(config, Path(checkpoint_path), report, [metrics_file, cm_path, templates_path, attr_path])

    return report


def _log_evaluation_to_mlflow(
    config: Config,
    checkpoint_path: Path,
    report: dict,
    artifact_files: list[Path],
) -> None:
    """Log evaluation metrics and interpretability plots to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(config.tracking.uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    run_id = None
    summary_candidate = checkpoint_path.resolve().parent.parent / "training_summary.json"
    if summary_candidate.is_file():
        try:
            summary_data = json.loads(summary_candidate.read_text())
            run_id = summary_data.get("mlflow_run_id")
        except (OSError, json.JSONDecodeError):
            run_id = None

    with mlflow.start_run(run_id=run_id, run_name=config.tracking.run_name or "logistic-eval") as active_run:
        mlflow.log_metrics(
            {
                "test/loss": report["test/loss"],
                "test/accuracy": report["test/accuracy"],
                "test/sample_count": float(report["sample_count"]),
            }
        )
        for artifact in artifact_files:
            if artifact.is_file():
                mlflow.log_artifact(str(artifact), artifact_path="evaluation")
        report["mlflow_run_id"] = active_run.info.run_id


def predict(model: torch.nn.Module | str | Path, images: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run inference on normalized image batch [N, 1, 28, 28] or [N, 28, 28]."""
    if isinstance(model, (str, Path)):
        model = load_classifier(model)
    model.eval()
    if images.ndim == 3:
        images = images.unsqueeze(1)
    with torch.no_grad():
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "confidence": confidence,
        "logits": logits,
    }


__all__ = [
    "load_classifier",
    "test_checkpoint",
    "confusion_matrix",
    "save_class_templates_plot",
    "save_confusion_matrix_plot",
    "save_sample_attributions_plot",
    "predict",
]
