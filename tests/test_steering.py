"""Tests for Riemannian activation manifold steering on MNIST (Option A)."""

import pytest
import torch

from models.MNIST.model import MNISTCNN
from models.MNIST.steering import (
    TangentChart2D,
    christoffel_symbols,
    pullback_activation_metric,
    pullback_fisher_metric,
    shoot_geodesic,
    solve_bvp_geodesic,
    train_smooth_demo,
)


def _sample_chart() -> TangentChart2D:
    torch.manual_seed(11)
    xa = torch.randn(1, 28, 28)
    xb = torch.randn(1, 28, 28)
    xc = torch.randn(1, 28, 28)
    return TangentChart2D.from_three_images(xa, xb, xc)


def test_tangent_chart_2d_orthonormal_roundtrip():
    chart = _sample_chart()
    assert torch.allclose(chart.e1.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(chart.e2.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose((chart.e1 * chart.e2).sum(), torch.tensor(0.0), atol=1e-6)

    u_pts = torch.tensor([[0.0, 0.0], [1.5, -0.7], [-0.4, 2.1]])
    imgs = chart.decode(u_pts)
    assert imgs.shape == (3, 1, 28, 28)
    recovered = chart.encode(imgs)
    torch.testing.assert_close(recovered, u_pts, rtol=1e-5, atol=1e-5)


def test_pullback_metrics_and_christoffel_symbols_with_smu():
    torch.manual_seed(42)
    chart = _sample_chart()
    model = MNISTCNN(activation="SMU").eval()
    u = torch.tensor([0.6, 0.4])

    g_act = pullback_activation_metric(model, chart, u, damping=1e-2)
    g_fish = pullback_fisher_metric(model, chart, u, damping=1e-2)
    assert g_act.shape == (2, 2)
    assert g_fish.shape == (2, 2)
    assert torch.allclose(g_act, g_act.T, atol=1e-6)
    assert torch.allclose(g_fish, g_fish.T, atol=1e-6)
    assert torch.linalg.eigvalsh(g_act).min() > 0
    assert torch.linalg.eigvalsh(g_fish).min() > 0

    metric_fn = lambda coords: pullback_activation_metric(model, chart, coords, damping=1e-2)
    gamma = christoffel_symbols(metric_fn, u)
    assert gamma.shape == (2, 2, 2)
    assert torch.isfinite(gamma).all()
    # Torsion-free symmetry: Gamma^k_{ij} == Gamma^k_{ji}
    torch.testing.assert_close(gamma, gamma.transpose(1, 2), rtol=1e-5, atol=1e-5)
    assert float(gamma.detach().norm()) > 0.0


def test_shoot_and_solve_bvp_geodesic():
    torch.manual_seed(7)
    chart = _sample_chart()
    model = MNISTCNN(activation="SMU").eval()
    metric_fn = lambda coords: pullback_activation_metric(model, chart, coords, damping=1e-2)

    ivp = shoot_geodesic(metric_fn, chart.u_a, torch.tensor([1.0, 0.5]), num_steps=8)
    assert ivp["u"].shape == (9, 2)
    assert torch.isfinite(ivp["u"]).all()

    bvp = solve_bvp_geodesic(
        metric_fn,
        chart.u_a,
        chart.u_b,
        num_nodes=8,
        opt_steps=10,
        lr=0.05,
    )
    assert bvp["geodesic_u"].shape == (8, 2)
    torch.testing.assert_close(bvp["geodesic_u"][0], chart.u_a)
    torch.testing.assert_close(bvp["geodesic_u"][-1], chart.u_b)
    assert bvp["geodesic_energy"] <= bvp["linear_energy"] + 1e-6


def test_train_smooth_demo_helper(tmp_path, patch_mnist):
    out = train_smooth_demo(tmp_path, activation="SMU", epochs=1, threads=1)
    assert (out / "checkpoints" / "best.ckpt").is_file()
    assert (out / "training_summary.json").is_file()
