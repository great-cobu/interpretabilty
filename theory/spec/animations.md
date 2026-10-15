# Animation Specification: `riemannian-manifolds`

## 1. Purpose & Scope

This specification defines the architecture, mathematical conventions, visual standards, and roadmap for animations in the `riemannian-manifolds` package (`theory/`). Every animation pairs:

1. **Exact mathematical primitives** in Python (`numpy` / `scipy` / `torch`) tested by `pytest`.
2. **Declarative 3D/2D animations** rendered with **Manim** (`ThreeDScene` / `Scene`).
3. **Interactive exploration notebooks** built with **Marimo** (`notebooks/`), embedding the rendered Manim video alongside interactive sliders and live coordinate/tensor readouts.

---

## 2. Module & File Conventions

Each geometric topic lives in its corresponding subpackage under `src/riemannian_manifolds/<subpackage>/`:

| Layer | Naming Convention | Responsibility | Example |
| :--- | :--- | :--- | :--- |
| **Mathematical Model** | `<topic>.py` | Pure coordinate maps, inverse maps, transition functions, metric tensors, connections, and ODE integrators. | `atlas/sphere.py` (`SphereAtlas`) |
| **Manim Scene** | `<topic>_scene.py` | `Scene` / `ThreeDScene` class + `render_<topic>_animation()` programmatic entry point. | `atlas/sphere_scene.py` (`SphereChartsScene`) |
| **Interactive Notebook** | `notebooks/<topic>.py` | Marimo notebook displaying the Manim `.mp4` (`mo.video`) + interactive `mo.ui.slider` views. | `notebooks/sphere_charts.py` |
| **Unit Tests** | `tests/test_<topic>.py` | Property and round-trip tests verifying mathematical identities used in the animation. | `tests/test_sphere_charts.py` |

---

## 3. Visual & Mathematical Standards

### 3.1 Scene Layout & Camera Choreography
- **Background**: Dark slate (`#0e1117`) for high contrast with 3D surfaces, coordinate grids, and LaTeX overlays.
- **Fixed-in-Frame HUD (`add_fixed_in_frame_mobjects`)**:
  - Top-left: Scene title and manifold definition (e.g., $S^2 = \{x \in \mathbb{R}^3 \mid \|x\| = 1\}$).
  - Top-right: Active chart maps $(\varphi_\alpha, U_\alpha)$ color-coded to match their geometric rays and planar projections.
  - Bottom edge: Transition maps $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$ and Riemannian metric tensor expressions $g_{ij}$.
- **3D Orientation**: Default initial camera at $\phi \approx 65^\circ\text{–}70^\circ$, $\theta \approx -45^\circ$, with smooth orbital rotation (`begin_ambient_camera_rotation` or `move_camera`) during dynamic trajectories.

### 3.2 Color Semantics
- **Manifold surface ($M$)**: Semi-transparent blue checkerboard (`BLUE_D` / `BLUE_B`, opacity `0.20–0.25`) so interior projection rays remain visible.
- **Singularities / Poles / Basepoints**: High-visibility markers (`RED_C` for North Pole $N$, `GOLD_B` for South Pole $S$, `WHITE` for moving point $p(t) \in M$).
- **Chart 1 ($\varphi_N$)**: Teal (`TEAL_C` / `BLUE_C`) for spherical patch, projection rays, and planar coordinates $(u_N, v_N)$.
- **Chart 2 ($\varphi_S$)**: Orange (`ORANGE` / `GOLD_B`) for spherical patch, projection rays, and planar coordinates $(u_S, v_S)$.
- **Overlap / Invariant Locus**: Yellow (`YELLOW`) for the equator $S^1 = \{z = 0\}$ where $\|\varphi_N(p)\| = \|\varphi_S(p)\| = 1$.

### 3.3 Interactive Notebook Presentation (`marimo`)
- **Code Hidden by Default (`@app.cell(hide_code=True)`)**: All cells in Marimo notebooks (`notebooks/<topic>.py`) must default to having their Python code hidden (`hide_code=True`). When opened in `marimo edit`, the user sees a clean presentation of natural language, animations, interactive controls, and equations without visual code clutter, while retaining the ability to reveal and inspect or edit any cell's code whenever they wish.

---

## 4. Animation Catalog & Roadmap

### 4.1 Coordinate Charts & Atlases (`riemannian_manifolds.atlas`)
- [x] **`SphereChartsScene`** (`atlas/sphere_scene.py`)
  - **Concepts**: Navigating a curved ocean $S^2$ with a sequence of small overlapping local charts $(U_1, \varphi_1), (U_2, \varphi_2), (U_3, \varphi_3)$ constructed via orthogonal projection onto local tangent planes $T_{p_\alpha} S^2$; inverse graph parameterizations $\varphi_\alpha^{-1}(u, v) = \sqrt{1 - u^2 - v^2}\,p_\alpha + u\,e_1^{(\alpha)} + v\,e_2^{(\alpha)}$; transition hand-offs $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$ in the overlap regions $U_\alpha \cap U_\beta$; and pullback metric $g_{ij}(u, v) = \delta_{ij} + \frac{u_i u_j}{1 - u^2 - v^2}$.

### 4.2 Tangent Spaces & Differentials (`riemannian_manifolds.tangent`) — *Planned*
- [ ] **`TangentSpaceScene`**
  - **Concepts**: Curves $\gamma : (-\varepsilon, \varepsilon) \to M$ with $\gamma(0) = p$ and velocity vector $\dot{\gamma}(0) \in T_p M$; coordinate basis vectors $\left.\frac{\partial}{\partial u}\right|_p, \left.\frac{\partial}{\partial v}\right|_p$; Jacobian of the transition map $d(\tau_{NS})_{\varphi_N(p)}$ transforming tangent vectors between charts.

### 4.3 Riemannian Metric & Distortion (`riemannian_manifolds.metric`) — *Planned*
- [ ] **`PullbackMetricScene`**
  - **Concepts**: First fundamental form $g_p : T_p M \times T_p M \to \mathbb{R}$; pullback metric $g_{ij}(u, v) = \lambda(u, v)^2 \delta_{ij}$ with conformal factor $\lambda(u, v) = \frac{2}{1 + u^2 + v^2}$; Tissot indicatrices (unit tangent circles in $T_p S^2$ mapped to Euclidean circles of radius $1/\lambda(u, v)$ in the chart); area and arc-length integration.

### 4.4 Levi-Civita Connection, Parallel Transport & Geodesics (`riemannian_manifolds.geodesics`) — *Planned*
- [ ] **`ParallelTransportHolonomyScene`**
  - **Concepts**: Covariant derivative $\nabla_{\dot{\gamma}} V = 0$; parallel transport of a tangent vector around a spherical triangle on $S^2$ demonstrating holonomy angle $\Delta \theta = \iint_{\Omega} K \, dA$; geodesic equation $\ddot{x}^k + \Gamma^k_{ij} \dot{x}^i \dot{x}^j = 0$ and the exponential map $\exp_p : T_p M \to M$.

### 4.5 Curvature & Jacobi Fields (`riemannian_manifolds.curvature`) — *Planned*
- [ ] **`GeodesicDeviationScene`**
  - **Concepts**: Riemann curvature tensor $R(X, Y)Z$; Gaussian/sectional curvature $K$ comparing positive ($S^2$), zero ($\mathbb{R}^2$), and negative ($\mathbb{H}^2$) curvature; Jacobi equation $\frac{D^2 J}{dt^2} + R(J, \dot{\gamma})\dot{\gamma} = 0$ governing the convergence/divergence of nearby geodesics.

---

## 5. Execution & Rendering Commands

From the repository root (`/Users/frankelliott/Documents/Projects/Python/interpetability`):

```bash
# Render the sphere charts Manim animation (MP4)
PATH="/Library/TeX/texbin:$PATH" uv run --project theory python -m riemannian_manifolds.atlas.sphere_scene

# Launch the interactive Marimo viewer notebook
uv run --project theory marimo edit theory/notebooks/sphere_charts.py

# Run mathematical verification tests
uv run --project theory pytest
```
