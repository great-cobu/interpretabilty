"""Riemannian activation manifold steering, pullback metrics, Christoffel symbols, and geodesic solvers for MNIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

import torch
from torch import nn
from torch.func import jacrev

from models.MNIST.config import load_config, prepare_run_dir
from models.MNIST.data import MNISTDataModule
from models.MNIST.train import train


@dataclass(frozen=True)
class TangentChart2D:
    """Orthonormal 2D affine chart phi^{-1}(u) = x0 + u^1 e1 + u^2 e2 anchored at three images (x_a, x_b, x_c)."""

    x0: torch.Tensor  # [1, 28, 28]
    e1: torch.Tensor  # [1, 28, 28], unit norm
    e2: torch.Tensor  # [1, 28, 28], unit norm, orthogonal to e1
    u_a: torch.Tensor  # [2]
    u_b: torch.Tensor  # [2]
    u_c: torch.Tensor  # [2]

    @classmethod
    def from_three_images(
        cls,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_c: torch.Tensor,
    ) -> TangentChart2D:
        """Construct an orthonormal 2D chart with x_a at the origin and x_b along +u^1."""
        xa = x_a.detach().clone().float().reshape(1, 28, 28)
        xb = x_b.detach().clone().float().reshape(1, 28, 28)
        xc = x_c.detach().clone().float().reshape(1, 28, 28)

        diff_b = xb - xa
        norm_b = float(diff_b.norm())
        if norm_b < 1e-6:
            raise ValueError("Anchor images x_a and x_b must be distinct")
        e1 = diff_b / norm_b

        diff_c = xc - xa
        proj_c = (diff_c * e1).sum()
        orth_c = diff_c - proj_c * e1
        norm_c = float(orth_c.norm())
        if norm_c < 1e-6:
            raise ValueError("Anchor image x_c must not be collinear with x_a and x_b")
        e2 = orth_c / norm_c

        u_a = torch.zeros(2, dtype=torch.float32)
        u_b = torch.tensor([norm_b, 0.0], dtype=torch.float32)
        u_c = torch.tensor([float(proj_c), norm_c], dtype=torch.float32)
        return cls(x0=xa, e1=e1, e2=e2, u_a=u_a, u_b=u_b, u_c=u_c)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Project image(s) [1, 28, 28] or [N, 1, 28, 28] into 2D chart coordinates u = (u^1, u^2)."""
        if image.ndim == 3:
            delta = image.float() - self.x0
            return torch.stack([(delta * self.e1).sum(), (delta * self.e2).sum()])
        delta = image.float() - self.x0.unsqueeze(0)
        u1 = (delta * self.e1.unsqueeze(0)).sum(dim=(1, 2, 3))
        u2 = (delta * self.e2.unsqueeze(0)).sum(dim=(1, 2, 3))
        return torch.stack([u1, u2], dim=-1)

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        """Map 2D chart coordinates u ([2] or [N, 2]) to normalized image tensor(s) ([1, 28, 28] or [N, 1, 28, 28])."""
        if u.ndim == 1:
            return self.x0 + u[0] * self.e1 + u[1] * self.e2
        u1 = u[:, 0].view(-1, 1, 1, 1)
        u2 = u[:, 1].view(-1, 1, 1, 1)
        return self.x0.unsqueeze(0) + u1 * self.e1.unsqueeze(0) + u2 * self.e2.unsqueeze(0)


def make_immersion_fn(
    model: nn.Module,
    chart: TangentChart2D,
    layer: str = "relu3",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a smooth map f: R^2 -> R^D mapping chart coordinates u to hidden activations at `layer`."""
    net = model.network if hasattr(model, "network") else model
    net.eval()

    def immersion(u: torch.Tensor) -> torch.Tensor:
        img = chart.decode(u).unsqueeze(0)
        features = net.get_feature_maps(img)
        return features[layer].reshape(-1)

    return immersion


def pullback_activation_metric(
    model: nn.Module,
    chart: TangentChart2D,
    u: torch.Tensor,
    *,
    layer: str = "relu3",
    damping: float = 1e-2,
) -> torch.Tensor:
    """Compute the 2x2 pullback Riemannian metric g_{ij}(u) = (J_h(u)^T J_h(u))_{ij} + damping * delta_{ij}."""
    immersion = make_immersion_fn(model, chart, layer=layer)
    jac = jacrev(immersion)(u)  # [D, 2]
    eye = torch.eye(u.shape[0], dtype=u.dtype, device=u.device)
    return jac.T @ jac + damping * eye


def pullback_fisher_metric(
    model: nn.Module,
    chart: TangentChart2D,
    u: torch.Tensor,
    *,
    damping: float = 1e-2,
) -> torch.Tensor:
    """Compute the 2x2 pullback Fisher Information Metric g^F_{ij}(u) from output log-probabilities."""
    net = model.network if hasattr(model, "network") else model
    net.eval()

    def log_probs_fn(coords: torch.Tensor) -> torch.Tensor:
        img = chart.decode(coords).unsqueeze(0)
        logits = net(img).squeeze(0)
        return torch.log_softmax(logits, dim=-1)

    log_p = log_probs_fn(u)
    probs = log_p.exp()  # [10]
    d_log_p = jacrev(log_probs_fn)(u)  # [10, 2]
    fisher = torch.einsum("c,ci,cj->ij", probs, d_log_p, d_log_p)
    eye = torch.eye(u.shape[0], dtype=u.dtype, device=u.device)
    return fisher + damping * eye


def christoffel_symbols(
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute Levi-Civita Christoffel symbols Gamma^k_{ij}(u) of shape (2, 2, 2) via torch.func.jacrev."""
    g = metric_fn(u)
    g_inv = torch.linalg.inv(g)
    dg = jacrev(metric_fn)(u)  # dg[j, l, i] = d(g_{jl}) / d(u^i)
    bracket = dg.permute(1, 2, 0) + dg.permute(1, 0, 2) - dg.permute(2, 0, 1)
    return 0.5 * torch.einsum("kl,lij->kij", g_inv, bracket)


def shoot_geodesic(
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    u0: torch.Tensor,
    v0: torch.Tensor,
    *,
    num_steps: int = 25,
    t_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Integrate the geodesic ODE d^2 u^k / dt^2 + Gamma^k_{ij}(u) du^i/dt du^j/dt = 0 via 4th-order Runge-Kutta."""
    dt = t_max / float(num_steps)
    t_vals = torch.linspace(0.0, t_max, num_steps + 1, dtype=u0.dtype)
    us = [u0.detach().clone()]
    vs = [v0.detach().clone()]

    def accel(u_pt: torch.Tensor, v_pt: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            gamma = christoffel_symbols(metric_fn, u_pt)
            return -torch.einsum("kij,i,j->k", gamma, v_pt, v_pt)

    u_curr = u0.detach().clone()
    v_curr = v0.detach().clone()
    for _ in range(num_steps):
        k1_u = v_curr
        k1_v = accel(u_curr, v_curr)

        k2_u = v_curr + 0.5 * dt * k1_v
        k2_v = accel(u_curr + 0.5 * dt * k1_u, k2_u)

        k3_u = v_curr + 0.5 * dt * k2_v
        k3_v = accel(u_curr + 0.5 * dt * k2_u, k3_u)

        k4_u = v_curr + dt * k3_v
        k4_v = accel(u_curr + dt * k3_u, k4_u)

        u_curr = u_curr + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        v_curr = v_curr + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        us.append(u_curr.clone())
        vs.append(v_curr.clone())

    return {
        "t": t_vals,
        "u": torch.stack(us, dim=0),
        "v": torch.stack(vs, dim=0),
    }


def curve_energy_and_length(
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    curve_u: torch.Tensor,
) -> dict[str, float | torch.Tensor]:
    """Compute Riemannian path energy E[gamma], arc length L[gamma], and local segment speeds ||du/dt||_g."""
    num_segments = curve_u.shape[0] - 1
    dt = 1.0 / float(num_segments)
    du = curve_u[1:] - curve_u[:-1]  # [N, 2]
    midpoints = 0.5 * (curve_u[1:] + curve_u[:-1])  # [N, 2]

    speeds_sq = []
    for s in range(num_segments):
        g_mid = metric_fn(midpoints[s]).detach()
        vel = du[s] / dt
        speeds_sq.append(torch.einsum("i,ij,j->", vel, g_mid, vel))

    speeds_sq_t = torch.stack(speeds_sq)
    speeds_t = torch.sqrt(torch.clamp(speeds_sq_t, min=0.0))
    energy = float(0.5 * (speeds_sq_t * dt).sum())
    length = float((speeds_t * dt).sum())
    return {
        "energy": energy,
        "length": length,
        "segment_speeds": speeds_t,
    }


def solve_bvp_geodesic(
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    u_start: torch.Tensor,
    u_end: torch.Tensor,
    *,
    num_nodes: int = 18,
    opt_steps: int = 40,
    lr: float = 0.08,
) -> dict[str, torch.Tensor | float]:
    """Solve the boundary-value geodesic between u_start and u_end by minimizing discrete Riemannian energy."""
    t_vals = torch.linspace(0.0, 1.0, num_nodes, dtype=u_start.dtype)
    linear_curve = (1.0 - t_vals.unsqueeze(1)) * u_start.unsqueeze(0) + t_vals.unsqueeze(1) * u_end.unsqueeze(0)

    interior = nn.Parameter(linear_curve[1:-1].clone())
    optimizer = torch.optim.Adam([interior], lr=lr)
    dt = 1.0 / float(num_nodes - 1)

    for _ in range(opt_steps):
        optimizer.zero_grad()
        full_curve = torch.cat([u_start.unsqueeze(0), interior, u_end.unsqueeze(0)], dim=0)
        du = full_curve[1:] - full_curve[:-1]
        midpoints = 0.5 * (full_curve[1:] + full_curve[:-1])
        total_energy = torch.tensor(0.0, dtype=u_start.dtype)
        for s in range(num_nodes - 1):
            g_mid = metric_fn(midpoints[s])
            vel = du[s] / dt
            total_energy = total_energy + 0.5 * torch.einsum("i,ij,j->", vel, g_mid, vel) * dt
        total_energy.backward()
        optimizer.step()

    final_curve = torch.cat([u_start.unsqueeze(0), interior.detach(), u_end.unsqueeze(0)], dim=0)
    lin_stats = curve_energy_and_length(metric_fn, linear_curve)
    geo_stats = curve_energy_and_length(metric_fn, final_curve)
    return {
        "t": t_vals,
        "linear_u": linear_curve.detach(),
        "geodesic_u": final_curve,
        "linear_energy": float(lin_stats["energy"]),
        "geodesic_energy": float(geo_stats["energy"]),
        "linear_length": float(lin_stats["length"]),
        "geodesic_length": float(geo_stats["length"]),
        "linear_speeds": lin_stats["segment_speeds"],
        "geodesic_speeds": geo_stats["segment_speeds"],
    }


def train_smooth_demo(
    root: Path,
    *,
    activation: str = "SMU",
    epochs: int = 2,
    threads: int = 4,
) -> Path:
    """Train a bounded CPU demo run of MNISTCNN with a smooth C^infinity activation (SMU or softplus)."""
    if activation not in ("SMU", "smu", "softplus", "relu"):
        raise ValueError("activation must be SMU, softplus, or relu")
    if type(epochs) is not int or not 1 <= epochs <= 50:
        raise ValueError("Choose an integer epoch count from 1 to 50")
    if type(threads) is not int or threads < 1:
        raise ValueError("Choose a positive integer thread count")
    root = Path(root).resolve()
    config = load_config(
        overrides={
            "data.directory": str(root / "data"),
            "data.download": False,
            "model.activation": activation,
            "trainer.accelerator": "cpu",
            "trainer.max_epochs": epochs,
            "trainer.patience": epochs,
            "trainer.limit_train_batches": 25,
            "trainer.limit_val_batches": 5,
            "tracking.enabled": False,
            "output_dir": str(root / "runs" / f"steering-{activation.lower()}-{uuid4().hex[:8]}"),
        },
        use_environment=False,
    )
    MNISTDataModule(config.data, accelerator="cpu").setup("fit")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        config, output = prepare_run_dir(config, "train-steering")
        train(config, output)
        return output
    finally:
        torch.set_num_threads(previous_threads)
