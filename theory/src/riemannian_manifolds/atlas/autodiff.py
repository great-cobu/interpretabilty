"""Automatic differentiation utilities for Riemannian geometry of activation manifolds using PyTorch."""

from __future__ import annotations

from typing import Callable

import torch
from torch.func import jacrev


def pullback_metric_autodiff(
    immersion_fn: Callable[[torch.Tensor], torch.Tensor],
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute the pullback Riemannian metric g_{ij}(u) = (J_f(u)^T J_f(u))_{ij} via torch.func.jacrev.

    Args:
        immersion_fn: Smooth map f: R^m -> R^D (e.g., inverse chart phi^{-1} or neural activation map f_theta).
        u: 1D tensor of shape (m,) representing local chart or latent coordinates.

    Returns:
        Metric tensor g_{ij}(u) of shape (m, m).
    """
    jacobian = jacrev(immersion_fn)(u)  # shape (D, m)
    return jacobian.T @ jacobian


def christoffel_symbols_autodiff(
    immersion_fn: Callable[[torch.Tensor], torch.Tensor],
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute Levi-Civita Christoffel symbols Gamma^k_{ij}(u) via second-order automatic differentiation.

    Uses torch.func.jacrev to differentiate g_{ij}(u) = (J_f(u)^T J_f(u))_{ij} with respect to u^m:
        Gamma^k_{ij}(u) = 1/2 * g^{kl}(u) * (d_i g_{jl} + d_j g_{il} - d_l g_{ij})

    Args:
        immersion_fn: Smooth map f: R^m -> R^D with C^2 smoothness (e.g., Softplus/SMU network or chart map).
        u: 1D tensor of shape (m,) representing local coordinates.

    Returns:
        Christoffel symbols tensor Gamma^k_{ij}(u) of shape (m, m, m) indexed by (k, i, j).
    """
    metric_fn = lambda coords: pullback_metric_autodiff(immersion_fn, coords)
    g = metric_fn(u)  # shape (m, m)
    g_inv = torch.linalg.inv(g)  # shape (m, m), indexed by (k, l)
    # dg[j, l, i] = d(g_{jl}) / d(u^i)
    dg = jacrev(metric_fn)(u)  # shape (m, m, m)

    # d_i g_{jl} + d_j g_{il} - d_l g_{ij} indexed by (l, i, j)
    bracket = dg.permute(1, 2, 0) + dg.permute(1, 0, 2) - dg.permute(2, 0, 1)
    return 0.5 * torch.einsum("kl,lij->kij", g_inv, bracket)
