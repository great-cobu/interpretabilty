# `riemannian-manifolds`: Differential Geometry of Activation Manifolds

Executable differential geometry and automatic differentiation library accompanying the paper [*Mechanistic Interpretability via Riemannian Information Geometry on Activation Space*](mechanistic_interpretability_geometry.tex).

---

## 1. Overview & Pedagogical Structure

Every mathematical concept in `riemannian-manifolds` is presented through the unified triad defined in [`spec/pedagogy.md`](spec/pedagogy.md):

1. **Natural Language**: Intuitive geometric motivation before formalism (e.g., why a curved manifold $M$ requires an overlapping atlas of local flat charts $\{(U_\alpha, \varphi_\alpha)\}$).
2. **Animation (`manim` & `marimo`)**: Dynamic 3D/2D Manim scenes (`SphereChartsScene`) and interactive Marimo notebooks (`notebooks/manifold_atlas.py`, `notebooks/sphere_charts.py`).
3. **Equations & Executable Code**: Analytical differential geometry on $S^2$ paired with exact PyTorch `torch.func.jacrev` second-order automatic differentiation.

---

## 2. Core Modules (`src/riemannian_manifolds/atlas/`)

* **[`sphere.py`](src/riemannian_manifolds/atlas/sphere.py)**:
  * `TangentChart`: Local orthogonal tangent-plane coordinate chart $(U_\alpha, \varphi_\alpha)$ on $S^2$, inverse parameterization $\varphi_\alpha^{-1}(u, v)$, transition diffeomorphisms $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$, analytical transition Jacobians $d\tau_{\alpha\beta}$, closed-form pullback metric $g_{ij}(u, v) = \delta_{ij} + \frac{u_i u_j}{1 - u^2 - v^2}$, Christoffel symbols $\Gamma^k_{ij}(u, v) = u_k g_{ij}(u, v)$, and covariant acceleration $\nabla_{\dot{\gamma}}\dot{\gamma}$.
  * `SphereAtlas`: Great-circle geodesic trajectories, non-geodesic detours, Riemannian arc-length integration, and overlapping three-chart voyage coverage.
* **[`autodiff.py`](src/riemannian_manifolds/atlas/autodiff.py)**:
  * `pullback_metric_autodiff(immersion_fn, u)`: Computes $g_{ij}(u) = (J_f(u)^\top J_f(u))_{ij}$ for any smooth immersion $f : \mathbb{R}^m \to \mathbb{R}^D$ via `torch.func.jacrev`.
  * `christoffel_symbols_autodiff(immersion_fn, u)`: Computes the Levi-Civita Christoffel symbols $\Gamma^k_{ij}(u) = \frac{1}{2} g^{kl}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$ via nested `jacrev` calls (requiring $C^2$ smooth activations such as `Softplus` or `SMU` in neural networks).
* **[`sphere_scene.py`](src/riemannian_manifolds/atlas/sphere_scene.py)**:
  * Manim 3D scene rendering the sphere $S^2$, overlapping tangent-plane charts, and great-circle geodesic navigation.

---

## 3. Running Tests & Notebooks

```bash
# Run unit tests (verifying analytical vs. jacrev autodiff parity to 1e-13)
uv run pytest

# Launch the 7-slide interactive Marimo presentation
uv run marimo edit notebooks/manifold_atlas.py

# Launch the interactive 3-chart voyage explorer
uv run marimo edit notebooks/sphere_charts.py
```
