"""Local tangent-plane coordinate charts, overlapping atlas, and geodesics on the 2-sphere S^2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TangentChart:
    """A local coordinate chart (U_alpha, phi_alpha) on S^2 via orthogonal projection onto T_p S^2.

    Given a chart center (waypoint) p in S^2 with spherical angles (theta_0, phi_0)
    and angular radius rho in (0, pi/2):
        - Domain: U = {x in S^2 | <x, p> > cos(rho)}
        - Tangent frame at p:
            e_1 = (-sin(phi_0), cos(phi_0), 0)                                        (local East)
            e_2 = (-cos(theta_0)*cos(phi_0), -cos(theta_0)*sin(phi_0), sin(theta_0))  (local North)
            n   = p                                                                   (outward unit normal)
        - Forward chart map phi: U -> B_{sin(rho)}(0) subset R^2:
            phi(x) = (<x, e_1>, <x, e_2>) = (u, v)
        - Inverse parameterization phi^{-1}: B_{sin(rho)}(0) -> U subset S^2:
            phi^{-1}(u, v) = sqrt(1 - u^2 - v^2) * p + u * e_1 + v * e_2
    """

    theta_0: float
    phi_0: float
    angular_radius: float = 0.45
    name: str = "U"

    @property
    def center(self) -> NDArray[np.float64]:
        """Unit vector p in S^2 at the center of the chart."""
        return SphereAtlas.spherical_to_cartesian(self.theta_0, self.phi_0)

    @property
    def e1(self) -> NDArray[np.float64]:
        """Unit tangent vector pointing East at the chart center."""
        return np.array(
            [-np.sin(self.phi_0), np.cos(self.phi_0), 0.0],
            dtype=np.float64,
        )

    @property
    def e2(self) -> NDArray[np.float64]:
        """Unit tangent vector pointing North at the chart center (p x e_1)."""
        return np.array(
            [
                -np.cos(self.theta_0) * np.cos(self.phi_0),
                -np.cos(self.theta_0) * np.sin(self.phi_0),
                np.sin(self.theta_0),
            ],
            dtype=np.float64,
        )

    @property
    def euclidean_radius(self) -> float:
        """Radius sin(rho) of the projected disk phi(U) in R^2."""
        return float(np.sin(self.angular_radius))

    def contains(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return boolean mask indicating whether points on S^2 lie inside U."""
        pts = np.asarray(points, dtype=np.float64)
        dots = np.sum(pts * self.center, axis=-1)
        return dots > np.cos(self.angular_radius)

    def phi(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Orthogonal projection phi: U -> R^2 onto local tangent basis (e_1, e_2)."""
        pts = np.asarray(points, dtype=np.float64)
        u = np.sum(pts * self.e1, axis=-1)
        v = np.sum(pts * self.e2, axis=-1)
        return np.stack([u, v], axis=-1)

    def phi_inv(self, uv: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse parameterization phi^{-1}: B_{sin(rho)}(0) -> U subset S^2."""
        uv_arr = np.asarray(uv, dtype=np.float64)
        u = uv_arr[..., 0:1]
        v = uv_arr[..., 1:2]
        r2 = np.clip(u**2 + v**2, 0.0, 1.0)
        normal_comp = np.sqrt(1.0 - r2)
        return normal_comp * self.center + u * self.e1 + v * self.e2

    def tangent_plane_point(self, uv: NDArray[np.float64]) -> NDArray[np.float64]:
        """Embed 2D chart coordinates (u, v) into the 3D affine tangent plane p + T_p S^2."""
        uv_arr = np.asarray(uv, dtype=np.float64)
        u = uv_arr[..., 0:1]
        v = uv_arr[..., 1:2]
        return self.center + u * self.e1 + v * self.e2

    def project_to_tangent_plane_3d(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project 3D point(s) x in S^2 orthogonally onto the 3D affine tangent plane p + T_p S^2."""
        return self.tangent_plane_point(self.phi(points))

    def transition_to(
        self,
        other: TangentChart,
        uv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transition map tau_{alpha,beta} = phi_beta o phi_alpha^{-1}: R^2 -> R^2."""
        sphere_pts = self.phi_inv(uv)
        return other.phi(sphere_pts)

    def transition_jacobian(
        self,
        other: TangentChart,
        uv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Jacobian matrix d(tau_{alpha,beta})(u, v) in R^{2x2} of the transition diffeomorphism."""
        uv_arr = np.asarray(uv, dtype=np.float64)
        u = uv_arr[..., 0:1]
        v = uv_arr[..., 1:2]
        r2 = np.clip(u**2 + v**2, 0.0, 1.0 - 1e-12)
        z_norm = np.sqrt(1.0 - r2)

        dphi_inv_du = self.e1 - (u / z_norm) * self.center
        dphi_inv_dv = self.e2 - (v / z_norm) * self.center

        j11 = np.sum(dphi_inv_du * other.e1, axis=-1)
        j12 = np.sum(dphi_inv_dv * other.e1, axis=-1)
        j21 = np.sum(dphi_inv_du * other.e2, axis=-1)
        j22 = np.sum(dphi_inv_dv * other.e2, axis=-1)

        row1 = np.stack([j11, j12], axis=-1)
        row2 = np.stack([j21, j22], axis=-1)
        return np.stack([row1, row2], axis=-2)

    def metric_tensor(self, uv: NDArray[np.float64]) -> NDArray[np.float64]:
        """Pullback Riemannian metric g_{ij}(u, v) = delta_{ij} + (u_i u_j) / (1 - u^2 - v^2)."""
        uv_arr = np.asarray(uv, dtype=np.float64)
        u = uv_arr[..., 0]
        v = uv_arr[..., 1]
        r2 = u**2 + v**2
        denom = np.maximum(1.0 - r2, 1e-12)
        g11 = 1.0 + (u**2) / denom
        g12 = (u * v) / denom
        g22 = 1.0 + (v**2) / denom
        row1 = np.stack([g11, g12], axis=-1)
        row2 = np.stack([g12, g22], axis=-1)
        return np.stack([row1, row2], axis=-2)

    def christoffel_symbols(self, uv: NDArray[np.float64]) -> NDArray[np.float64]:
        """Christoffel symbols Gamma^k_{ij}(u, v) = u^k * g_{ij}(u, v) of the Levi-Civita connection.

        Returns:
            Array of shape (..., 2, 2, 2) indexed by (..., k, i, j).
        """
        uv_arr = np.asarray(uv, dtype=np.float64)
        g = self.metric_tensor(uv_arr)
        return uv_arr[..., :, None, None] * g[..., None, :, :]

    def covariant_acceleration(
        self,
        uv: NDArray[np.float64],
        duv_dt: NDArray[np.float64],
        d2uv_dt2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute covariant acceleration nabla_{dot{gamma}} dot{gamma} in local coordinates:

            a^k = d^2 u^k / dt^2 + Gamma^k_{ij}(u, v) * (du^i / dt) * (du^j / dt)

        A constant-speed curve on S^2 is a geodesic iff its covariant acceleration vanishes (a^k = 0).
        """
        gamma = self.christoffel_symbols(uv)
        v_arr = np.asarray(duv_dt, dtype=np.float64)
        acc_arr = np.asarray(d2uv_dt2, dtype=np.float64)
        christoffel_term = np.einsum("...kij,...i,...j->...k", gamma, v_arr, v_arr)
        return acc_arr + christoffel_term

    def patch_point_from_polar(
        self,
        r_frac: NDArray[np.float64] | float,
        angle: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        """Sample a point in U_alpha using polar coordinates (r_frac in [0, 1], angle in [0, 2pi])."""
        r = np.asarray(r_frac, dtype=np.float64) * self.euclidean_radius
        ang = np.asarray(angle, dtype=np.float64)
        uv = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1)
        return self.phi_inv(uv)


class SphereAtlas:
    """Atlas of overlapping local tangent-plane charts and geodesics along a ship's voyage on S^2."""

    NORTH_POLE: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])
    SOUTH_POLE: NDArray[np.float64] = np.array([0.0, 0.0, -1.0])

    # Start and end spherical coordinates of the ship's ocean voyage
    START_THETA: float = 1.18
    START_PHI: float = -0.65
    END_THETA: float = 0.66
    END_PHI: float = 0.65

    @staticmethod
    def spherical_to_cartesian(
        theta: NDArray[np.float64] | float,
        phi: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        """Convert spherical angles (colatitude theta in [0, pi], longitude phi in [0, 2pi)) to S^2 in R^3."""
        theta_arr = np.asarray(theta, dtype=np.float64)
        phi_arr = np.asarray(phi, dtype=np.float64)
        x = np.sin(theta_arr) * np.cos(phi_arr)
        y = np.sin(theta_arr) * np.sin(phi_arr)
        z = np.cos(theta_arr)
        return np.stack([x, y, z], axis=-1)

    @classmethod
    def start_point(cls) -> NDArray[np.float64]:
        """Starting point p_0 in S^2 of the ship's voyage."""
        return cls.spherical_to_cartesian(cls.START_THETA, cls.START_PHI)

    @classmethod
    def end_point(cls) -> NDArray[np.float64]:
        """Destination point p_1 in S^2 of the ship's voyage."""
        return cls.spherical_to_cartesian(cls.END_THETA, cls.END_PHI)

    @staticmethod
    def geodesic_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
        """Great-circle Riemannian distance Omega = arccos(<p, q>) between two points on S^2."""
        dot = float(np.clip(np.dot(p, q), -1.0, 1.0))
        return float(np.arccos(dot))

    @classmethod
    def geodesic_voyage(cls, t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Great-circle geodesic gamma_{geo}(t) on S^2 from start_point to end_point for t in [0, 1].

        Parameterized at constant speed Omega = arccos(<p_0, p_1>) via spherical linear interpolation (Slerp):
            gamma_{geo}(t) = (sin((1 - t)*Omega) / sin(Omega)) * p_0 + (sin(t*Omega) / sin(Omega)) * p_1
        """
        t_arr = np.asarray(t, dtype=np.float64)
        p0 = cls.start_point()
        p1 = cls.end_point()
        omega = cls.geodesic_distance(p0, p1)
        sin_omega = np.sin(omega)
        w0 = np.sin((1.0 - t_arr) * omega) / sin_omega
        w1 = np.sin(t_arr * omega) / sin_omega
        return w0[..., None] * p0 + w1[..., None] * p1

    @classmethod
    def voyage_curve(cls, t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Non-geodesic curved detour trajectory gamma_{detour}(t) sharing the same endpoints."""
        t_arr = np.asarray(t, dtype=np.float64)
        theta = cls.START_THETA + (cls.END_THETA - cls.START_THETA) * t_arr + 0.14 * np.sin(np.pi * t_arr)
        phi = cls.START_PHI + (cls.END_PHI - cls.START_PHI) * t_arr
        return cls.spherical_to_cartesian(theta, phi)

    @staticmethod
    def curve_arc_length(
        curve_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        num_steps: int = 500,
    ) -> float:
        """Numerically integrate the Riemannian arc length L[gamma] = int_0^1 ||dot{gamma}(t)|| dt on S^2."""
        t_vals = np.linspace(0.0, 1.0, num_steps)
        pts = curve_fn(t_vals)
        diffs = np.diff(pts, axis=0)
        chord_lengths = np.linalg.norm(diffs, axis=-1)
        # Convert small chords to exact spherical arcs
        arc_lengths = 2.0 * np.arcsin(np.clip(chord_lengths / 2.0, 0.0, 1.0))
        return float(np.sum(arc_lengths))

    @classmethod
    def default_voyage_charts(cls, angular_radius: float = 0.44) -> list[TangentChart]:
        """Return three overlapping local tangent-plane charts centered along the geodesic voyage."""
        waypoints_t = [0.15, 0.50, 0.85]
        names = ["U_1", "U_2", "U_3"]
        charts: list[TangentChart] = []
        for t_val, name in zip(waypoints_t, names, strict=True):
            pt = cls.geodesic_voyage(t_val)
            theta = float(np.arccos(np.clip(pt[2], -1.0, 1.0)))
            phi = float(np.arctan2(pt[1], pt[0]))
            charts.append(
                TangentChart(
                    theta_0=theta,
                    phi_0=phi,
                    angular_radius=angular_radius,
                    name=name,
                )
            )
        return charts
