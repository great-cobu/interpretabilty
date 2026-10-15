"""Coordinate charts, atlases, and automatic differentiation for Riemannian manifolds."""

from riemannian_manifolds.atlas.autodiff import (
    christoffel_symbols_autodiff,
    pullback_metric_autodiff,
)
from riemannian_manifolds.atlas.sphere import SphereAtlas, TangentChart
from riemannian_manifolds.atlas.sphere_scene import (
    SphereChartsScene,
    render_sphere_charts_animation,
)

__all__ = [
    "SphereAtlas",
    "SphereChartsScene",
    "TangentChart",
    "christoffel_symbols_autodiff",
    "pullback_metric_autodiff",
    "render_sphere_charts_animation",
]
