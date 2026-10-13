"""Convex L-BFGS fitting, interpretability template rendering, and MLflow logging for LogisticClassifier."""

from dataclasses import replace
import json
from pathlib import Path
import time
from uuid import uuid4

import lightning.pytorch as pl
import torch
from torch.nn import functional as F

from models.MNIST.config import Config, config_dict, load_config, prepare_run_dir
from models.MNIST.data import MNISTDataModule
from models.MNIST.train import environment, sha256, trainer_options
from models.logistic.evaluate import load_classifier, save_class_templates_plot
from models.logistic.model import LogisticClassifier


def _collect_split_tensors(dataloader, limit_batches: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack batches from a DataLoader into contiguous (images, labels) tensors."""
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for idx, (images, labels) in enumerate(dataloader):
        if limit_batches is not None and idx >= limit_batches:
            break
        xs.append(images)
        ys.append(labels)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train(
    config: Config,
    output: Path,
    *,
    resume: str | Path | None = None,
) -> dict:
    """Fit LogisticClassifier to convex optimality using full-batch L-BFGS with L2 regularization."""
    pl.seed_everything(config.trainer.seed, workers=True)
    options = trainer_options(config)
    data = MNISTDataModule(config.data, accelerator="cpu")

    saved_ckpt = None
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_file():
            raise ValueError(f"Resume checkpoint does not exist: {resume_path}")
        saved_ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        if "MNISTDataModule" in saved_ckpt:
            data.config = replace(data.config, split_indices=None)
            data.load_state_dict(saved_ckpt["MNISTDataModule"])

    data.prepare_data()
    data.setup("fit")
    data.save_split_indices(output / "split_indices.json")

    model = LogisticClassifier(config=config)
    if saved_ckpt is not None and "state_dict" in saved_ckpt:
        model.load_state_dict(saved_ckpt["state_dict"])

    x_train, y_train = _collect_split_tensors(data.train_dataloader(), config.trainer.limit_train_batches)
    x_val, y_val = _collect_split_tensors(data.val_dataloader(), config.trainer.limit_val_batches)

    weight_decay = float(config.optimizer.weight_decay)
    max_iter = 100
    tolerance_grad = 1e-5
    tolerance_change = 1e-9

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    eval_calls = 0

    def closure() -> torch.Tensor:
        nonlocal eval_calls
        eval_calls += 1
        optimizer.zero_grad()
        logits = model(x_train)
        ce_loss = F.cross_entropy(logits, y_train)
        w = model.network.layers.logits.weight
        l2_penalty = 0.5 * weight_decay * (w * w).sum()
        total_loss = ce_loss + l2_penalty
        total_loss.backward()
        return total_loss

    started = time.perf_counter()
    model.train()
    optimizer.step(closure)
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        train_logits = model(x_train)
        train_loss = float(F.cross_entropy(train_logits, y_train))
        train_acc = float((train_logits.argmax(dim=1) == y_train).float().mean())

        val_logits = model(x_val)
        val_loss = float(F.cross_entropy(val_logits, y_val))
        val_acc = float((val_logits.argmax(dim=1) == y_val).float().mean())

        w = model.network.layers.logits.weight
        grad_norm = float(w.grad.norm()) if w.grad is not None else 0.0

    state = optimizer.state.get(model.network.layers.logits.weight, {})
    solver_iterations = int(state.get("n_iter", eval_calls))

    ckpt_dir = output / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.ckpt"
    last_path = ckpt_dir / "last.ckpt"

    ckpt_payload = {
        "state_dict": model.state_dict(),
        "hyper_parameters": {"config": config_dict(config)},
        "pytorch-lightning_version": pl.__version__,
        "MNISTDataModule": data.state_dict(),
        "solver": "L-BFGS",
        "solver_iterations": solver_iterations,
        "epoch": 0,
        "global_step": solver_iterations,
    }
    torch.save(ckpt_payload, best_path)
    torch.save(ckpt_payload, last_path)

    manifest_path = ckpt_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({p.name: sha256(p) for p in (best_path, last_path) if p.is_file()}, indent=2) + "\n"
    )

    best_model = load_classifier(best_path, device="cpu")
    templates_path = save_class_templates_plot(best_model, output / "class_templates.png")

    mlflow_run_id = None
    mlflow_experiment_id = None
    metrics = {
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "solver/iterations": float(solver_iterations),
        "solver/grad_norm": grad_norm,
    }

    report = {
        "model": "LogisticRegression",
        "solver": "L-BFGS",
        "weight_decay": weight_decay,
        "tolerance_grad": tolerance_grad,
        "solver_iterations": solver_iterations,
        "function_evaluations": eval_calls,
        "grad_norm": grad_norm,
        "elapsed_seconds": elapsed,
        "best_val_loss": val_loss,
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "class_templates_plot": str(templates_path.resolve()),
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment_id": mlflow_experiment_id,
        "accelerator": options["accelerator"],
        "environment": environment(),
        "smoke_run": config.trainer.limit_train_batches is not None or config.trainer.limit_val_batches is not None,
        "metrics": metrics,
        "source_checkpoint": str(Path(resume).resolve()) if resume else None,
        "source_checkpoint_sha256": sha256(resume) if resume else None,
    }
    summary_path = output / "training_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")

    if config.tracking.enabled and config.tracking.uri:
        import mlflow

        mlflow.set_tracking_uri(config.tracking.uri)
        exp = mlflow.set_experiment(config.tracking.experiment_name)
        with mlflow.start_run(
            run_name=config.tracking.run_name or output.name,
            tags=dict(config.tracking.tags) if config.tracking.tags else None,
        ) as active_run:
            mlflow.log_params(
                {
                    "model": "LogisticRegression",
                    "solver": "L-BFGS",
                    "line_search": "strong_wolfe",
                    "weight_decay": weight_decay,
                    "tolerance_grad": tolerance_grad,
                    "train_samples": int(x_train.shape[0]),
                    "val_samples": int(x_val.shape[0]),
                }
            )
            mlflow.log_metrics(metrics)
            report["mlflow_run_id"] = active_run.info.run_id
            report["mlflow_experiment_id"] = exp.experiment_id
            summary_path.write_text(json.dumps(report, indent=2) + "\n")
            for artifact in (output / "config.yaml", templates_path, summary_path, manifest_path):
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact), artifact_path="training")

    return report


def fit_demo(root: Path, weight_decay: float = 0.01, threads: int = 4) -> Path:
    """Fit a convex L-BFGS logistic regression run with user-selected L2 regularization."""
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError("Choose a nonnegative L2 regularization strength (weight_decay)")
    if type(threads) is not int or threads < 1:
        raise ValueError("Choose a positive integer thread count")
    root = Path(root).resolve()
    config = load_config(
        overrides={
            "data.directory": str(root / "data"),
            "data.download": False,
            "optimizer.weight_decay": float(weight_decay),
            "trainer.accelerator": "cpu",
            "trainer.limit_train_batches": 40,
            "trainer.limit_val_batches": 10,
            "tracking.enabled": False,
            "output_dir": str(root / "runs" / f"logistic-lbfgs-l2-{weight_decay:g}-{uuid4().hex[:8]}"),
        },
        use_environment=False,
    )
    MNISTDataModule(config.data, accelerator="cpu").setup("fit")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        config, output = prepare_run_dir(config, "fit-logistic")
        train(config, output)
        return output
    finally:
        torch.set_num_threads(previous_threads)


def train_demo(root: Path, epochs: int = 2, threads: int = 4, *, weight_decay: float = 0.01) -> Path:
    """Backward-compatible alias for fit_demo."""
    return fit_demo(root, weight_decay=weight_decay, threads=threads)
