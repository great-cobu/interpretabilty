"""Tests for Stage 1 Riemannian manifold steering primitives."""

import json
import math

import pytest
import torch

from models.MNIST.model import MNISTCNN
from models.MNIST.steering import (
    TangentChart2D,
    discover_cnn_runs,
    evaluate_decision_landscape,
)


def test_gram_schmidt_chart_orthonormality_and_exact_anchors():
    torch.manual_seed(7)
    x_a = torch.randn(1, 28, 28)
    x_b = torch.randn(1, 28, 28)
    x_c = torch.randn(1, 28, 28)

    chart = TangentChart2D.from_three_images(x_a, x_b, x_c)

    # Orthonormality of basis vectors e1, e2
    assert torch.isclose(chart.e1.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(chart.e2.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose((chart.e1 * chart.e2).sum(), torch.tensor(0.0), atol=1e-6)

    # Anchor coordinate geometry: u_a = (0, 0), u_b = (||x_b - x_a||, 0), u_c^2 > 0
    assert torch.allclose(chart.u_a, torch.zeros(2), atol=1e-6)
    assert chart.u_b[0] > 0.0 and abs(float(chart.u_b[1])) < 1e-6
    assert chart.u_c[1] > 0.0

    # Exact reconstruction of all three anchors
    assert torch.allclose(chart.decode(chart.u_a), x_a, atol=1e-5)
    assert torch.allclose(chart.decode(chart.u_b), x_b, atol=1e-5)
    assert torch.allclose(chart.decode(chart.u_c), x_c, atol=1e-5)

    # Isometric property: chart Euclidean distance equals exact L2 image distance
    u_p = torch.tensor([1.5, -2.25])
    u_q = torch.tensor([-0.75, 3.1])
    chart_dist = (u_p - u_q).norm()
    pixel_dist = (chart.decode(u_p) - chart.decode(u_q)).norm()
    assert torch.isclose(chart_dist, pixel_dist, atol=1e-5)


def test_gram_schmidt_rejects_degenerate_anchors():
    x = torch.ones(1, 28, 28)
    y = torch.zeros(1, 28, 28)
    with pytest.raises(ValueError, match="distinct"):
        TangentChart2D.from_three_images(x, x, y)
    with pytest.raises(ValueError, match="collinear"):
        TangentChart2D.from_three_images(x, y, 0.5 * (x + y))


def test_tangent_pca_chart_with_geomstats():
    torch.manual_seed(11)
    samples = torch.randn(24, 1, 28, 28)
    chart = TangentChart2D.from_tangent_pca(samples, samples[0], samples[1], samples[2])

    assert chart.mode == "tangent_pca"
    assert chart.explained_variance_ratio is not None
    assert len(chart.explained_variance_ratio) == 2
    assert 0.0 < chart.explained_variance_ratio[0] <= 1.0
    assert chart.explained_variance_ratio[0] >= chart.explained_variance_ratio[1] > 0.0

    assert torch.isclose(chart.e1.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(chart.e2.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose((chart.e1 * chart.e2).sum(), torch.tensor(0.0), atol=1e-6)

    # Batch decode -> encode round-trip on in-plane points is exact
    coords = torch.tensor([[0.0, 0.0], [2.0, -1.5], [-3.2, 4.1]])
    decoded = chart.decode(coords)
    assert decoded.shape == (3, 1, 28, 28)
    reencoded = chart.encode(decoded)
    assert torch.allclose(reencoded, coords, atol=1e-5)


def test_evaluate_decision_landscape_on_smooth_cnn():
    torch.manual_seed(19)
    model = MNISTCNN(dropout=0.0, activation="SMU").eval()
    x_a, x_b, x_c = torch.randn(1, 28, 28), torch.randn(1, 28, 28), torch.randn(1, 28, 28)
    chart = TangentChart2D.from_three_images(x_a, x_b, x_c)

    landscape = evaluate_decision_landscape(model, chart, resolution=15)
    assert landscape.u1.shape == (15,)
    assert landscape.u2.shape == (15,)
    assert landscape.predictions.shape == (15, 15)
    assert landscape.confidence.shape == (15, 15)
    assert landscape.entropy.shape == (15, 15)
    assert landscape.probabilities.shape == (15, 15, 10)

    prob_sums = landscape.probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    assert (landscape.confidence >= 0.1 - 1e-5).all() and (landscape.confidence <= 1.0 + 1e-5).all()
    assert (landscape.entropy >= -1e-6).all() and (landscape.entropy <= math.log(10.0) + 1e-5).all()


def test_discover_cnn_runs_prioritizes_smooth_and_skips_logistic(tmp_path):
    def _make_run(name: str, activation: str | None, experiment: str, val_acc: float):
        run_dir = tmp_path / "runs" / name
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "best.ckpt").write_bytes(b"stub")
        model_block = f"  dropout: 0.2\n  activation: {activation}\n" if activation else "  dropout: 0.0\n"
        (run_dir / "config.yaml").write_text(
            f"model:\n{model_block}tracking:\n  experiment_name: {experiment}\n"
        )
        (run_dir / "training_summary.json").write_text(
            json.dumps({"completed_epochs": 5, "metrics": {"val/accuracy": val_acc}})
        )

    _make_run("relu-cnn", "relu", "mnist-cnn", 0.99)
    _make_run("smu-cnn", "SMU", "mnist-cnn", 0.98)
    _make_run("logistic-demo-1", None, "mnist-logistic", 0.92)

    all_cnn = discover_cnn_runs(tmp_path, smooth_only=False)
    assert [r.run_dir.name for r in all_cnn] == ["smu-cnn", "relu-cnn"]
    assert all_cnn[0].is_smooth is True
    assert all_cnn[1].is_smooth is False

    smooth_only = discover_cnn_runs(tmp_path, smooth_only=True)
    assert [r.run_dir.name for r in smooth_only] == ["smu-cnn"]
