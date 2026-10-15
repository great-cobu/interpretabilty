# Pedagogy Specification: `riemannian-manifolds`

## 1. Core Exposition Rule: Natural Language, Animation, and Equations

Every mathematical concept in `riemannian-manifolds` must be presented through the unified triad of **Natural Language**, **Animation**, and **Equations**:

1. **Natural Language**:
   - Introduce every concept in clear, plain natural language explaining *what* the geometric object is, *why* it is needed, and *how* it behaves intuitively before introducing formalism.
   - Connect every visual motion and every algebraic symbol back to a concrete natural-language explanation.
2. **Animation (`manim` & `marimo`)**:
   - Follow the natural-language intuition with a dynamic Manim animation (and interactive Marimo visualization) that brings the geometric construction to life in 2D/3D space.
   - Every stage of the animation must visually enact the ideas described in the natural-language narrative.
3. **Equations**:
   - Anchor the natural-language explanation and visual animation in precise mathematical equations (coordinate maps, transition functions, pullback metrics, differential equations), both in the written exposition (`paper/`, `notebooks/`) and synchronized directly on-screen (`MathTex` HUD overlays in Manim).
   - Realize these equations directly in executable Python (`src/riemannian_manifolds/`).

---

## 2. Structure of a Pedagogical Unit

Every topic in the project follows the **Natural Language $\rightarrow$ Animation $\rightarrow$ Equations** progression:

| Step | Modality | Pedagogical Role |
| :--- | :--- | :--- |
| **1. Natural Language** | Prose in `paper/` and `notebooks/` | Motivate the geometric problem and explain the core idea in clear, accessible language (e.g., why a flat map of the entire sphere must tear or distort, requiring overlapping local charts). |
| **2. Animation** | Manim Scene (`*_scene.py`) & Marimo (`notebooks/*.py`) | Visually demonstrate the geometric process (e.g., projecting rays from the North and South poles onto the equatorial plane and morphing curved surface patches into flat coordinate grids). |
| **3. Equations** | LaTeX in `paper/`, Manim `MathTex`, & Python in `src/` | State the exact symbolic formulas governing what was just explained and animated (e.g., $\varphi_N(x,y,z) = \left(\frac{x}{1-z}, \frac{y}{1-z}\right)$, $\tau_{NS}(u,v) = \frac{(u,v)}{u^2+v^2}$, and $g_{ij} = \frac{4}{(1+u^2+v^2)^2}\delta_{ij}$). |

---

## 3. Guidelines for Each Modality

### 3.1 Natural Language Guidelines
- Lead with intuition and motivation before notation.
- Explain the meaning of each term in an equation in plain English so the reader never encounters an unmotivated symbol.
- Summarize the geometric takeaway after presenting the animation and equations.

### 3.2 Animation Guidelines
- Pace animations so the viewer can follow one geometric idea at a time.
- Synchronize visual transformations (such as unfolding a patch or moving a point across an overlap) with the corresponding equations on-screen.
- Embed rendered Manim animations inside Marimo notebooks alongside interactive controls so learners can explore the geometry at their own pace.

### 3.3 Equation Guidelines
- State domains, codomains, and coordinate variables explicitly.
- Keep notation consistent across `paper/`, Manim `MathTex` overlays, Marimo markdown cells, and Python variable names in `src/riemannian_manifolds/`.
- Verify all equations computationally via unit tests in `tests/`.

---

## 4. Curriculum Progression

Each module applies the **Natural Language, Animation, and Equations** exposition rule to build Riemannian geometry step by step:

1. **Module 1 — Atlases & Coordinate Charts (`riemannian_manifolds.atlas`)**
   - *Natural Language*: Why a ship sailing on a curved ocean navigates using a sequence of small overlapping flat charts created by projecting onto local tangent planes.
   - *Animation*: Three overlapping ocean patches $U_1, U_2, U_3$ on $S^2$ lifting to their local tangent planes $T_{p_\alpha} S^2$, and a sailing ship tracked across chart overlaps (`SphereChartsScene`).
   - *Equations*: Orthogonal tangent-plane projections $\varphi_\alpha(x)$, inverse graph parameterizations $\varphi_\alpha^{-1}(u, v)$, transition maps $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$, and pullback metric $g_{ij}(u, v) = \delta_{ij} + \frac{u_i u_j}{1 - u^2 - v^2}$.
2. **Module 2 — Tangent Spaces & Differentials (`riemannian_manifolds.tangent`)**
   - *Natural Language*: How velocity vectors of curves on a surface form a local flat plane at each point, and how velocities translate between charts.
   - *Animation*: Curves passing through $p \in S^2$, their velocity vectors in $T_p S^2$, and pushforwards into chart coordinates.
   - *Equations*: Coordinate basis $\left.\frac{\partial}{\partial u}\right|_p, \left.\frac{\partial}{\partial v}\right|_p$ and the Jacobian transition matrix $d(\tau_{NS})$.
3. **Module 3 — Riemannian Metrics & Pullbacks (`riemannian_manifolds.metric`)**
   - *Natural Language*: How to measure true surface lengths, angles, and areas using distorted flat map coordinates.
   - *Animation*: Unit circles on $S^2$ projecting to circles of varying size (Tissot indicatrices) in the stereographic plane.
   - *Equations*: First fundamental form, conformal factor $\lambda(u, v) = \frac{2}{1+u^2+v^2}$, and pullback metric tensor $g_{ij}(u, v) = \lambda(u, v)^2 \delta_{ij}$.
4. **Module 4 — Connections, Parallel Transport & Geodesics (`riemannian_manifolds.geodesics`)**
   - *Natural Language*: What it means to move "straight ahead" or keep a vector "pointing the same way" on a curved manifold.
   - *Animation*: Parallel transport of a tangent vector around a spherical triangle showing holonomy, and great-circle geodesics.
   - *Equations*: Christoffel symbols $\Gamma^k_{ij}$, covariant derivative $\nabla_{\dot{\gamma}} V = 0$, and the geodesic ODE $\ddot{x}^k + \Gamma^k_{ij}\dot{x}^i\dot{x}^j = 0$.
5. **Module 5 — Curvature (`riemannian_manifolds.curvature`)**
   - *Natural Language*: How curvature measures the failure of parallel transport around small loops and causes nearby straight paths (geodesics) to converge or diverge.
   - *Animation*: Geodesic deviation and holonomy comparison on positively curved ($S^2$), flat ($\mathbb{R}^2$), and negatively curved ($\mathbb{H}^2$) surfaces.
   - *Equations*: Riemann curvature tensor $R(X,Y)Z$, Gaussian/sectional curvature $K$, and the Jacobi equation.
