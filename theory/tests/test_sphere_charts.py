"""Unit tests for local tangent-plane charts, overlapping atlas, geodesics, and PyTorch autodiff on S^2."""

from __future__ import annotations

import numpy as np
import torch

from riemannian_manifolds.atlas import (
    SphereAtlas,
    TangentChart,
    christoffel_symbols_autodiff,
    pullback_metric_autodiff,
)


def test_tangent_frame_is_orthonormal() -> None:
    for chart in SphereAtlas.default_voyage_charts():
        frame = np.stack([chart.e1, chart.e2, chart.center], axis=0)
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)


def test_tangent_chart_roundtrip() -> None:
    chart = TangentChart(theta_0=1.0, phi_0=-0.3, angular_radius=0.45)
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    pts = chart.patch_point_from_polar(0.75, angles)
    assert np.all(chart.contains(pts))

    uv = chart.phi(pts)
    reconstructed = chart.phi_inv(uv)
    np.testing.assert_allclose(reconstructed, pts, atol=1e-12)


def test_voyage_covered_by_overlapping_charts() -> None:
    charts = SphereAtlas.default_voyage_charts(angular_radius=0.44)
    t_vals = np.linspace(0.0, 1.0, 100)
    pts = SphereAtlas.geodesic_voyage(t_vals)

    in_any = np.zeros(len(t_vals), dtype=bool)
    for chart in charts:
        in_any |= chart.contains(pts)
    assert np.all(in_any), "Every point along the geodesic voyage must lie in at least one chart"

    overlap_12 = charts[0].contains(pts) & charts[1].contains(pts)
    overlap_23 = charts[1].contains(pts) & charts[2].contains(pts)
    assert np.any(overlap_12), "Charts U_1 and U_2 must overlap along the voyage"
    assert np.any(overlap_23), "Charts U_2 and U_3 must overlap along the voyage"


def test_transition_diffeomorphism_and_jacobian() -> None:
    charts = SphereAtlas.default_voyage_charts(angular_radius=0.44)
    t_vals = np.linspace(0.0, 1.0, 120)
    pts = SphereAtlas.geodesic_voyage(t_vals)

    for c_a, c_b in [(charts[0], charts[1]), (charts[1], charts[2])]:
        mask = c_a.contains(pts) & c_b.contains(pts)
        overlap_pts = pts[mask]
        uv_a = c_a.phi(overlap_pts)
        uv_b_direct = c_b.phi(overlap_pts)
        uv_b_via_tau = c_a.transition_to(c_b, uv_a)
        np.testing.assert_allclose(uv_b_via_tau, uv_b_direct, atol=1e-12)

        jac_ab = c_a.transition_jacobian(c_b, uv_a)
        jac_ba = c_b.transition_jacobian(c_a, uv_b_direct)
        dets = np.linalg.det(jac_ab)
        assert np.all(dets > 0.5), "Transition Jacobian must be nonsingular on chart overlaps"
        prod = jac_ba @ jac_ab
        expected = np.broadcast_to(np.eye(2), prod.shape)
        np.testing.assert_allclose(prod, expected, atol=1e-11)


def test_metric_tensor_matches_jacobian_pullback() -> None:
    chart = TangentChart(theta_0=0.95, phi_0=0.4, angular_radius=0.45)
    np.testing.assert_allclose(chart.metric_tensor(np.array([0.0, 0.0])), np.eye(2), atol=1e-12)

    uv = np.array([0.18, -0.14])
    eps = 1e-7
    d_du = (chart.phi_inv(uv + np.array([eps, 0.0])) - chart.phi_inv(uv - np.array([eps, 0.0]))) / (
        2.0 * eps
    )
    d_dv = (chart.phi_inv(uv + np.array([0.0, eps])) - chart.phi_inv(uv - np.array([0.0, eps]))) / (
        2.0 * eps
    )
    jac = np.stack([d_du, d_dv], axis=-1)
    g_numeric = jac.T @ jac
    np.testing.assert_allclose(chart.metric_tensor(uv), g_numeric, atol=1e-8)


def test_geodesic_minimizes_length_and_has_zero_covariant_acceleration() -> None:
    geo_dist = SphereAtlas.geodesic_distance(SphereAtlas.start_point(), SphereAtlas.end_point())
    geo_arc_len = SphereAtlas.curve_arc_length(SphereAtlas.geodesic_voyage, num_steps=1000)
    detour_arc_len = SphereAtlas.curve_arc_length(SphereAtlas.voyage_curve, num_steps=1000)

    np.testing.assert_allclose(geo_arc_len, geo_dist, atol=1e-6)
    assert detour_arc_len > geo_arc_len + 1e-2, "Geodesic (great circle) must be strictly shorter than detour"

    charts = SphereAtlas.default_voyage_charts()
    chart = charts[1]
    t0 = 0.46
    eps = 1e-4

    uv_geo_0 = chart.phi(SphereAtlas.geodesic_voyage(t0))
    uv_geo_m = chart.phi(SphereAtlas.geodesic_voyage(t0 - eps))
    uv_geo_p = chart.phi(SphereAtlas.geodesic_voyage(t0 + eps))
    v_geo = (uv_geo_p - uv_geo_m) / (2.0 * eps)
    a_geo = (uv_geo_p - 2.0 * uv_geo_0 + uv_geo_m) / (eps**2)
    cov_acc_geo = chart.covariant_acceleration(uv_geo_0, v_geo, a_geo)
    np.testing.assert_allclose(cov_acc_geo, np.zeros(2), atol=1e-6)

    uv_det_0 = chart.phi(SphereAtlas.voyage_curve(t0))
    uv_det_m = chart.phi(SphereAtlas.voyage_curve(t0 - eps))
    uv_det_p = chart.phi(SphereAtlas.voyage_curve(t0 + eps))
    v_det = (uv_det_p - uv_det_m) / (2.0 * eps)
    a_det = (uv_det_p - 2.0 * uv_det_0 + uv_det_m) / (eps**2)
    cov_acc_det = chart.covariant_acceleration(uv_det_0, v_det, a_det)
    assert np.linalg.norm(cov_acc_det) > 0.5, "Non-geodesic detour must have non-zero covariant acceleration"


def test_autodiff_metric_and_christoffel_match_analytical() -> None:
    chart = TangentChart(theta_0=0.95, phi_0=0.4, angular_radius=0.45)
    p_t = torch.tensor(chart.center, dtype=torch.float64)
    e1_t = torch.tensor(chart.e1, dtype=torch.float64)
    e2_t = torch.tensor(chart.e2, dtype=torch.float64)

    def phi_inv_torch(uv_t: torch.Tensor) -> torch.Tensor:
        u, v = uv_t[0], uv_t[1]
        z = torch.sqrt(1.0 - u**2 - v**2)
        return z * p_t + u * e1_t + v * e2_t

    uv_np = np.array([0.16, -0.11], dtype=np.float64)
    uv_torch = torch.tensor(uv_np, dtype=torch.float64)

    g_auto = pullback_metric_autodiff(phi_inv_torch, uv_torch).numpy()
    gamma_auto = christoffel_symbols_autodiff(phi_inv_torch, uv_torch).numpy()

    np.testing.assert_allclose(g_auto, chart.metric_tensor(uv_np), atol=1e-13)
    np.testing.assert_allclose(gamma_auto, chart.christoffel_symbols(uv_np), atol=1e-13)
