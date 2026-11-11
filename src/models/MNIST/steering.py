"""Stage 1 Riemannian manifold steering primitives: smooth checkpoint discovery, orthonormal 2D charts, and decision landscapes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

# NumPy >= 2.0 removed ``np.trapz`` in favor of ``np.trapezoid``; shim for geomstats 2.8.0.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from sklearn import config_context
import torch
from torch import nn
import yaml

from models.MNIST.config import load_config, prepare_run_dir
from models.MNIST.train import train

SMOOTH_ACTIVATIONS = frozenset({"smu", "softplus"})


@dataclass(frozen=True)
class RunDescriptor:
    """Metadata for a discovered MNISTCNN checkpoint directory."""

    run_dir: Path
    checkpoint_path: Path
    activation: str
    val_accuracy: float | None
    completed_epochs: int

    @property
    def is_smooth(self) -> bool:
        """Return True when the checkpoint uses a C^infinity activation (SMU or softplus)."""
        return self.activation.lower() in SMOOTH_ACTIVATIONS

    @property
    def label(self) -> str:
        """Human-readable label for UI dropdowns."""
        acc = f"{self.val_accuracy:.2%}" if self.val_accuracy is not None else "n/a"
        smooth_tag = "smooth" if self.is_smooth else "piecewise-linear"
        return f"{self.run_dir.name} [{self.activation} · {smooth_tag} · val acc {acc}]"


def discover_cnn_runs(root: str | Path, *, smooth_only: bool = False) -> list[RunDescriptor]:
    """Scan ``root / 'runs'`` for completed MNISTCNN checkpoints, prioritizing smooth runs."""
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return []

    descriptors: list[RunDescriptor] = []
    for summary_path in runs_dir.glob("*/training_summary.json"):
        run_dir = summary_path.parent
        ckpt_path = run_dir / "checkpoints" / "best.ckpt"
        config_path = run_dir / "config.yaml"
        if not ckpt_path.is_file() or not config_path.is_file():
            continue

        try:
            summary = json.loads(summary_path.read_text())
            cfg = yaml.safe_load(config_path.read_text()) or {}
        except (OSError, ValueError, yaml.YAMLError):
            continue

        experiment_name = str((cfg.get("tracking") or {}).get("experiment_name") or "")
        if "logistic" in experiment_name.lower() or run_dir.name.startswith("logistic"):
            continue

        model_cfg = cfg.get("model") or {}
        activation = str(model_cfg.get("activation") or "relu")
        metrics = summary.get("metrics") or {}
        val_acc_raw = metrics.get("val/accuracy")
        val_acc = float(val_acc_raw) if isinstance(val_acc_raw, (int, float)) else None
        completed_epochs = int(summary.get("completed_epochs") or 0)

        desc = RunDescriptor(
            run_dir=run_dir,
            checkpoint_path=ckpt_path,
            activation=activation,
            val_accuracy=val_acc,
            completed_epochs=completed_epochs,
        )
        if smooth_only and not desc.is_smooth:
            continue
        descriptors.append(desc)

    descriptors.sort(
        key=lambda d: (
            d.is_smooth,
            d.val_accuracy if d.val_accuracy is not None else -1.0,
            d.run_dir.stat().st_mtime,
        ),
        reverse=True,
    )
    return descriptors


def train_smooth_demo(
    root: str | Path,
    *,
    activation: str = "SMU",
    epochs: int = 5,
    threads: int = 4,
    limit_train_batches: int = 25,
    limit_val_batches: int = 5,
) -> Path:
    """Train and save a compact smooth MNISTCNN checkpoint for manifold steering."""
    root_path = Path(root).resolve()
    torch.set_num_threads(max(1, int(threads)))
    run_dir = root_path / "runs" / f"steering-{activation.lower()}-{uuid4().hex[:8]}"
    base_cfg = {
        "data": {
            "directory": str(root_path / "data"),
            "download": True,
            "num_workers": 0,
        },
        "model": {
            "dropout": 0.2,
            "activation": activation,
        },
        "trainer": {
            "max_epochs": int(epochs),
            "patience": int(epochs),
            "accelerator": "cpu",
            "limit_train_batches": int(limit_train_batches),
            "limit_val_batches": int(limit_val_batches),
        },
        "tracking": {
            "enabled": False,
            "experiment_name": "mnist-cnn",
        },
        "output_dir": str(run_dir),
    }
    config = load_config(base=base_cfg, use_environment=False)
    config, output = prepare_run_dir(config, "train-steering")
    train(config, output)
    return output


def ensure_smooth_checkpoint(
    root: str | Path,
    *,
    activation: str = "SMU",
    epochs: int = 5,
    threads: int = 4,
) -> RunDescriptor:
    """Return the best existing smooth MNISTCNN run, or train one automatically if none exists."""
    smooth_runs = discover_cnn_runs(root, smooth_only=True)
    if smooth_runs:
        return smooth_runs[0]
    output = train_smooth_demo(root, activation=activation, epochs=epochs, threads=threads)
    refreshed = [r for r in discover_cnn_runs(root, smooth_only=True) if r.run_dir.resolve() == output.resolve()]
    if refreshed:
        return refreshed[0]
    return RunDescriptor(
        run_dir=output,
        checkpoint_path=output / "checkpoints" / "best.ckpt",
        activation=activation,
        val_accuracy=None,
        completed_epochs=epochs,
    )


@dataclass(frozen=True)
class TangentChart2D:
    """Orthonormal 2D affine chart psi(u^1, u^2) = x0 + u^1 e1 + u^2 e2 in MNIST image space."""

    x0: torch.Tensor  # [1, 28, 28] chart origin
    e1: torch.Tensor  # [1, 28, 28] unit basis vector 1
    e2: torch.Tensor  # [1, 28, 28] unit basis vector 2, <e1, e2> = 0
    u_a: torch.Tensor  # [2] chart coordinates of anchor A
    u_b: torch.Tensor  # [2] chart coordinates of anchor B
    u_c: torch.Tensor  # [2] chart coordinates of anchor C
    mode: str = "gram_schmidt"
    explained_variance_ratio: tuple[float, float] | None = None

    @classmethod
    def from_three_images(
        cls,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_c: torch.Tensor,
    ) -> TangentChart2D:
        """Build an exact 2D plane through (x_a, x_b, x_c) via Gram-Schmidt orthonormalization.

        Places x_a at u_a = (0, 0), x_b on the positive u^1-axis, and x_c in the upper half-plane u^2 > 0.
        """
        xa = x_a.detach().clone().double().reshape(1, 28, 28)
        xb = x_b.detach().clone().double().reshape(1, 28, 28)
        xc = x_c.detach().clone().double().reshape(1, 28, 28)

        diff_b = xb - xa
        norm_b = float(diff_b.norm())
        if norm_b < 1e-6:
            raise ValueError("Anchor images x_a and x_b must be distinct")
        e1 = diff_b / norm_b

        diff_c = xc - xa
        proj_c = float((diff_c * e1).sum())
        orth_c = diff_c - proj_c * e1
        norm_c = float(orth_c.norm())
        if norm_c <= 1e-5 * max(float(diff_c.norm()), 1.0):
            raise ValueError("Anchor image x_c must not be collinear with x_a and x_b")
        e2 = orth_c / norm_c

        u_a = torch.zeros(2, dtype=torch.float32)
        u_b = torch.tensor([norm_b, 0.0], dtype=torch.float32)
        u_c = torch.tensor([proj_c, norm_c], dtype=torch.float32)
        return cls(
            x0=xa.float(),
            e1=e1.float(),
            e2=e2.float(),
            u_a=u_a,
            u_b=u_b,
            u_c=u_c,
            mode="gram_schmidt",
            explained_variance_ratio=None,
        )

    @classmethod
    def from_tangent_pca(
        cls,
        samples: torch.Tensor,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_c: torch.Tensor,
    ) -> TangentChart2D:
        """Build a data-driven 2D orthonormal chart via Geomstats FrechetMean + TangentPCA."""
        flat_samples = samples.detach().cpu().float().reshape(samples.shape[0], -1).numpy().astype(np.float64)
        if flat_samples.shape[0] < 3 or flat_samples.shape[1] != 784:
            raise ValueError("TangentPCA requires at least 3 MNIST samples of 784 pixels")

        space = Euclidean(dim=784)
        gs_samples = gs.from_numpy(flat_samples)
        use_array_api = gs.__name__.endswith("pytorch")
        with config_context(array_api_dispatch=use_array_api):
            mean_est = FrechetMean(space=space)
            mean_est.fit(gs_samples)
            base_point = mean_est.estimate_

            tpca = TangentPCA(space=space, n_components=2)
            tpca.fit(gs_samples, base_point=base_point)

        x0 = torch.as_tensor(base_point, dtype=torch.float32).cpu().reshape(1, 28, 28)
        e1 = torch.as_tensor(tpca.components_[0], dtype=torch.float32).cpu().reshape(1, 28, 28)
        e2 = torch.as_tensor(tpca.components_[1], dtype=torch.float32).cpu().reshape(1, 28, 28)

        # Re-orthonormalize in float32 for machine-precision unit norm and orthogonality
        e1 = e1 / e1.norm()
        e2 = e2 - (e2 * e1).sum() * e1
        e2 = e2 / e2.norm()

        xa = x_a.detach().clone().float().reshape(1, 28, 28)
        xb = x_b.detach().clone().float().reshape(1, 28, 28)
        xc = x_c.detach().clone().float().reshape(1, 28, 28)

        # Orient e1 and e2 consistently so x_b lies toward +u^1 from x_a and x_c lies toward +u^2
        if float(((xb - xa) * e1).sum()) < 0.0:
            e1 = -e1
        if float(((xc - xa) * e2).sum()) < 0.0:
            e2 = -e2

        evr = tuple(float(v) for v in torch.as_tensor(tpca.explained_variance_ratio_[:2]).cpu().tolist())
        temp_chart = cls(
            x0=x0,
            e1=e1,
            e2=e2,
            u_a=torch.zeros(2),
            u_b=torch.zeros(2),
            u_c=torch.zeros(2),
            mode="tangent_pca",
            explained_variance_ratio=(evr[0], evr[1]),
        )
        return cls(
            x0=x0,
            e1=e1,
            e2=e2,
            u_a=temp_chart.encode(xa),
            u_b=temp_chart.encode(xb),
            u_c=temp_chart.encode(xc),
            mode="tangent_pca",
            explained_variance_ratio=(evr[0], evr[1]),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Project image(s) ([1, 28, 28], [28, 28], or [N, 1, 28, 28]) onto 2D coordinates ([2] or [N, 2])."""
        x = images.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3 and x.shape[0] == 1 and images.ndim <= 3:
            delta = x - self.x0.to(x.device)
            u1 = (delta * self.e1.to(x.device)).sum()
            u2 = (delta * self.e2.to(x.device)).sum()
            return torch.stack([u1, u2], dim=0)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        delta = x - self.x0.to(x.device).unsqueeze(0)
        u1 = (delta * self.e1.to(x.device).unsqueeze(0)).sum(dim=(1, 2, 3))
        u2 = (delta * self.e2.to(x.device).unsqueeze(0)).sum(dim=(1, 2, 3))
        return torch.stack([u1, u2], dim=-1)

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        """Map 2D chart coordinates ([2] or [N, 2]) to MNIST tensors ([1, 28, 28] or [N, 1, 28, 28])."""
        x0 = self.x0.to(dtype=u.dtype, device=u.device)
        e1 = self.e1.to(dtype=u.dtype, device=u.device)
        e2 = self.e2.to(dtype=u.dtype, device=u.device)
        if u.ndim == 1:
            if u.shape[0] != 2:
                raise ValueError(f"Expected 2D chart coordinate of shape (2,), got {tuple(u.shape)}")
            return x0 + u[0] * e1 + u[1] * e2
        if u.ndim == 2 and u.shape[1] == 2:
            u1 = u[:, 0].reshape(-1, 1, 1, 1)
            u2 = u[:, 1].reshape(-1, 1, 1, 1)
            return x0.unsqueeze(0) + u1 * e1.unsqueeze(0) + u2 * e2.unsqueeze(0)
        raise ValueError(f"Expected 2D coordinates of shape (2,) or (N, 2), got {tuple(u.shape)}")

    def grid_bounds(self, padding: float = 0.28) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute padded (u1_min, u1_max), (u2_min, u2_max) bounds enclosing all three anchors."""
        anchors = torch.stack([self.u_a, self.u_b, self.u_c], dim=0)
        mins = anchors.min(dim=0).values
        maxs = anchors.max(dim=0).values
        span = max(float((maxs - mins).max()), 1.0)
        margin = padding * span
        return (
            (float(mins[0]) - margin, float(maxs[0]) + margin),
            (float(mins[1]) - margin, float(maxs[1]) + margin),
        )


@dataclass(frozen=True)
class DecisionLandscape:
    """Evaluated classifier decision, confidence, and entropy fields across a 2D TangentChart2D grid."""

    u1: torch.Tensor  # [G]
    u2: torch.Tensor  # [G]
    predictions: torch.Tensor  # [G, G] int64
    confidence: torch.Tensor  # [G, G] float32
    entropy: torch.Tensor  # [G, G] float32 in nats
    probabilities: torch.Tensor  # [G, G, 10] float32


def evaluate_decision_landscape(
    model: nn.Module,
    chart: TangentChart2D,
    *,
    resolution: int = 45,
    padding: float = 0.28,
    batch_size: int = 512,
) -> DecisionLandscape:
    """Evaluate predicted digit classes, max confidence, and Shannon entropy across the 2D chart."""
    if resolution < 5:
        raise ValueError("resolution must be at least 5")

    net = model.network if hasattr(model, "network") else model
    net.eval()
    device = next(net.parameters()).device

    (u1_min, u1_max), (u2_min, u2_max) = chart.grid_bounds(padding=padding)
    u1 = torch.linspace(u1_min, u1_max, resolution, dtype=torch.float32)
    u2 = torch.linspace(u2_min, u2_max, resolution, dtype=torch.float32)
    grid_u1, grid_u2 = torch.meshgrid(u1, u2, indexing="xy")
    coords = torch.stack([grid_u1.reshape(-1), grid_u2.reshape(-1)], dim=-1)

    probs_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, coords.shape[0], batch_size):
            batch_coords = coords[start : start + batch_size].to(device)
            images = chart.decode(batch_coords)
            logits = net(images)
            probs = logits.softmax(dim=-1).cpu()
            probs_chunks.append(probs)

    all_probs = torch.cat(probs_chunks, dim=0)  # [G*G, 10]
    log_probs = torch.log(all_probs.clamp_min(1e-12))
    entropy = -(all_probs * log_probs).sum(dim=-1)
    confidence, predictions = all_probs.max(dim=-1)

    return DecisionLandscape(
        u1=u1,
        u2=u2,
        predictions=predictions.reshape(resolution, resolution),
        confidence=confidence.reshape(resolution, resolution),
        entropy=entropy.reshape(resolution, resolution),
        probabilities=all_probs.reshape(resolution, resolution, 10),
    )
