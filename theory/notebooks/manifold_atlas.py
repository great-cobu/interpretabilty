import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import importlib
    from pathlib import Path
    import sys

    _src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import riemannian_manifolds.atlas.sphere as _sphere_mod
    import riemannian_manifolds.atlas as _atlas_mod

    importlib.reload(_sphere_mod)
    importlib.reload(_atlas_mod)

    from riemannian_manifolds.atlas import (
        SphereAtlas,
        render_sphere_charts_animation,
    )

    return Path, SphereAtlas, mo, np, plt, render_sphere_charts_animation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Manifolds, Coordinate Charts, Atlases, and Geodesics on $S^2$

    ## 1. Fundamental Definitions: Manifold, Chart, Atlas, and Geodesic

    ### Definition 1: Manifold ($M$)
    - **Natural Language**: A **manifold** is a space that may be curved or closed on a global scale—like the ocean surface covering a spherical Earth ($S^2$)—but that looks locally like flat Euclidean space ($\mathbb{R}^n$) in a small neighborhood around every point. To a sailor looking out at a small patch of calm sea, the ocean appears flat even though the Earth is round.
    - **Mathematical Definition**: An $n$-dimensional **topological manifold** is a Hausdorff, second-countable topological space $M$ such that every point $p \in M$ has an open neighborhood $U \subset M$ homeomorphic to an open subset of $\mathbb{R}^n$. For the unit 2-sphere:
      $$S^2 = \left\{ x = (x^1, x^2, x^3) \in \mathbb{R}^3 \;\middle|\; \|x\|^2 = (x^1)^2 + (x^2)^2 + (x^3)^2 = 1 \right\}, \qquad n = 2$$

    ---

    ### Definition 2: Coordinate Chart ($(U_\alpha, \varphi_\alpha)$)
    - **Natural Language**: A **coordinate chart** (or simply **chart**) is a single local flat map covering a small open patch $U_\alpha \subset M$ of the manifold. On an ocean voyage, we construct a chart at a waypoint $p_\alpha \in S^2$ by resting a flat **tangent plane** $T_{p_\alpha} S^2$ against the sphere at $p_\alpha$ and projecting the nearby curved ocean patch $U_\alpha$ orthogonally onto that flat plane.
    - **Mathematical Definition**: A **chart** on $M$ is a pair $(U_\alpha, \varphi_\alpha)$ where $U_\alpha \subset M$ is an open set and
      $$\varphi_\alpha : U_\alpha \xrightarrow{\;\sim\;} \varphi_\alpha(U_\alpha) \subset \mathbb{R}^n$$
      is a homeomorphism onto an open subset of $\mathbb{R}^n$, assigning local Euclidean coordinates $(u^1, \dots, u^n) = \varphi_\alpha(x)$ to every point $x \in U_\alpha$.

    ---

    ### Definition 3: Atlas ($\mathcal{A}$) and Diffeomorphic Transition Maps ($\tau_{\alpha\beta}$)
    - **Natural Language**: Because a closed sphere cannot be flattened onto a single plane without tearing or severe horizon distortion, a single chart is not enough. An **atlas** is a collection of small, overlapping charts that together cover the entire manifold.
      **Why Diffeomorphism Enables Consistent Navigation**: As a ship sails from one chart $U_\alpha$ into an overlapping chart $U_\beta$, it lies in the overlap region $U_\alpha \cap U_\beta$ and is tracked on **both** flat tangent planes at once. The **transition map** $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$ translates the ship's coordinates from Chart $\alpha$ to Chart $\beta$. Requiring $\tau_{\alpha\beta}$ to be a **diffeomorphism**—a bijective map that is infinitely differentiable ($C^\infty$) in both directions with an invertible Jacobian matrix $d\tau_{\alpha\beta}$—guarantees that positions, velocities, compass headings, and differential equations of motion translate uniquely and consistently from one chart to the next without jumps or kinks.
    - **Mathematical Definition**: A **$C^\infty$ (smooth) atlas** on $M$ is a family of charts $\mathcal{A} = \{(U_\alpha, \varphi_\alpha)\}_{\alpha \in I}$ covering $M = \bigcup_{\alpha \in I} U_\alpha$ such that for every pair of overlapping charts with $U_\alpha \cap U_\beta \neq \varnothing$, the **transition map**
      $$\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1} : \varphi_\alpha(U_\alpha \cap U_\beta) \xrightarrow{\;\sim\;} \varphi_\beta(U_\alpha \cap U_\beta)$$
      is a **$C^\infty$ diffeomorphism** between open subsets of $\mathbb{R}^n$:
      $$\tau_{\alpha\beta} \in C^\infty, \qquad \tau_{\alpha\beta}^{-1} = \tau_{\beta\alpha} \in C^\infty, \qquad \det\bigl(d\tau_{\alpha\beta}(u)\bigr) \neq 0 \quad \forall u \in \varphi_\alpha(U_\alpha \cap U_\beta)$$

    ---

    ### Definition 4: Riemannian Metric ($g$) and Geodesic ($\gamma$)
    - **Natural Language**: A **Riemannian metric** $g$ equips each tangent plane $T_p M$ with an inner product so we can measure true curved lengths, speeds, and angles directly inside our flat charts. A **geodesic** is the generalization of a straight line to a curved manifold: it is both the **locally shortest path** between two points and the path followed by a ship that **sails with its rudder locked straight ahead** (experiencing **zero covariant/tangential acceleration** $\nabla_{\dot{\gamma}}\dot{\gamma} = 0$ along the water). On the sphere $S^2$, geodesics are **great circles**—the intersections of $S^2$ with planes through the center of the Earth.
    - **Mathematical Definition**: A smooth curve $\gamma : [0, 1] \to (M, g)$ with chart coordinates $u(t) = \varphi_\alpha(\gamma(t))$ is a **geodesic** if its covariant acceleration vanishes:
      $$\left(\nabla_{\dot{\gamma}}\dot{\gamma}\right)^k = \ddot{u}^k(t) + \Gamma^k_{ij}\bigl(u(t)\bigr)\,\dot{u}^i(t)\,\dot{u}^j(t) = 0$$
      where $\Gamma^k_{ij}$ are the **Christoffel symbols** of the Levi-Civita connection determined by the metric tensor $g_{ij}$.
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo, render_sphere_charts_animation):
    video_candidates = list(
        (Path(__file__).resolve().parents[1] / "media").rglob("sphere_charts.mp4")
    )
    if video_candidates:
        video_path = video_candidates[0]
    else:
        video_path = render_sphere_charts_animation()

    mo.vstack(
        [
            mo.md(
                r"""
                ## 2. Animation: Overlapping Tangent-Plane Charts & Geodesic (Great Circle) Navigation

                The Manim 3D animation below illustrates:
                1. Three overlapping ocean patches $U_1$ (teal), $U_2$ (yellow), and $U_3$ (orange) on the manifold $S^2$ projecting orthogonally onto their flat tangent planes $T_{p_1}S^2, T_{p_2}S^2, T_{p_3}S^2$.
                2. The **great-circle geodesic** $\gamma_{\text{geo}}(t)$ (solid green) compared against a curved non-geodesic detour (dashed grey).
                3. The ship (red dot) sailing along the geodesic across the chart overlaps $U_1 \cap U_2$ and $U_2 \cap U_3$, tracked simultaneously on both adjacent tangent planes via the transition diffeomorphism $\tau_{\alpha\beta}$.
                """
            ),
            mo.video(src=str(video_path), autoplay=True, loop=True, controls=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    voyage_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.02,
        value=0.32,
        label=r"Ship Voyage Progress $t \in [0, 1]$",
    )
    route_selector = mo.ui.radio(
        options={
            "Geodesic (Great Circle — Shortest & Zero Rudder Turn)": "geodesic",
            "Non-Geodesic Detour (Requires Sideways Steering)": "detour",
        },
        value="Geodesic (Great Circle — Shortest & Zero Rudder Turn)",
        label="Active Ship Route",
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### Interactive Atlas & Geodesic Explorer

                Select the **Geodesic (Great Circle)** or **Non-Geodesic Detour** route and drag $t \in [0, 1]$ to sail the ship across the three overlapping tangent-plane charts $(U_1, \varphi_1)$, $(U_2, \varphi_2)$, and $(U_3, \varphi_3)$:
                """
            ),
            mo.hstack([voyage_slider, route_selector], justify="start", gap=2.5),
        ]
    )
    return route_selector, voyage_slider


@app.cell(hide_code=True)
def _(SphereAtlas, np, plt, route_selector, voyage_slider):
    t_val = float(voyage_slider.value)
    route_mode = route_selector.value
    charts = SphereAtlas.default_voyage_charts(angular_radius=0.44)
    colors = ["teal", "goldenrod", "darkorange"]

    active_curve_fn = (
        SphereAtlas.geodesic_voyage
        if route_mode == "geodesic"
        else SphereAtlas.voyage_curve
    )
    ship_pt = active_curve_fn(t_val)

    eps_t = 1e-4
    t_minus = max(0.0, t_val - eps_t)
    t_plus = min(1.0, t_val + eps_t)
    dt_span = t_plus - t_minus

    t_samples = np.linspace(0.0, 1.0, 160)
    geodesic_pts = SphereAtlas.geodesic_voyage(t_samples)
    detour_pts = SphereAtlas.voyage_curve(t_samples)

    fig = plt.figure(figsize=(14, 4.1), dpi=120)

    # Panel 1: 3D Ocean Globe + Great Circle Geodesic + Detour + Tangent Planes
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    u_grid, v_grid = np.meshgrid(
        np.linspace(0, np.pi, 26),
        np.linspace(0, 2 * np.pi, 52),
    )
    ax3d.plot_surface(
        np.sin(u_grid) * np.cos(v_grid),
        np.sin(u_grid) * np.sin(v_grid),
        np.cos(u_grid),
        alpha=0.15,
        color="royalblue",
        edgecolor="none",
    )
    ax3d.plot(
        detour_pts[:, 0],
        detour_pts[:, 1],
        detour_pts[:, 2],
        "--",
        color="dimgray",
        lw=1.5,
        label="Detour γ_detour",
    )
    ax3d.plot(
        geodesic_pts[:, 0],
        geodesic_pts[:, 1],
        geodesic_pts[:, 2],
        color="forestgreen",
        lw=2.4,
        label="Geodesic (Great Circle)",
    )
    ax3d.scatter(*ship_pt, color="crimson", s=60, zorder=10, label="Ship γ(t)")

    ang_circle = np.linspace(0.0, 2.0 * np.pi, 60)
    active_records = []

    for idx, (chart, col) in enumerate(zip(charts, colors, strict=True), start=1):
        patch_bdry = chart.patch_point_from_polar(1.0, ang_circle)
        ax3d.plot(
            patch_bdry[:, 0],
            patch_bdry[:, 1],
            patch_bdry[:, 2],
            color=col,
            lw=1.3,
        )
        lim = chart.euclidean_radius
        uv_circle = np.stack(
            [lim * np.cos(ang_circle), lim * np.sin(ang_circle)], axis=-1
        )
        plane_bdry = chart.tangent_plane_point(uv_circle) + 0.12 * chart.center
        ax3d.plot(
            plane_bdry[:, 0],
            plane_bdry[:, 1],
            plane_bdry[:, 2],
            "--",
            color=col,
            lw=1.1,
        )

        in_chart = bool(chart.contains(ship_pt))
        uv_ship = chart.phi(ship_pt)
        g_ship = chart.metric_tensor(uv_ship)

        uv_minus = chart.phi(active_curve_fn(t_minus))
        uv_plus = chart.phi(active_curve_fn(t_plus))
        uv_mid = chart.phi(active_curve_fn(0.5 * (t_minus + t_plus)))
        duv_dt = (uv_plus - uv_minus) / dt_span
        d2uv_dt2 = (uv_plus - 2.0 * uv_mid + uv_minus) / ((0.5 * dt_span) ** 2)
        cov_acc = chart.covariant_acceleration(uv_ship, duv_dt, d2uv_dt2)

        active_records.append((idx, chart, col, in_chart, uv_ship, g_ship, cov_acc))

        if in_chart:
            plane_pt = (
                chart.project_to_tangent_plane_3d(ship_pt) + 0.12 * chart.center
            )
            ax3d.plot(
                [ship_pt[0], plane_pt[0]],
                [ship_pt[1], plane_pt[1]],
                [ship_pt[2], plane_pt[2]],
                color=col,
                lw=2.0,
            )
            ax3d.scatter(*plane_pt, color=col, s=40)

    ax3d.view_init(elev=22, azim=-10)
    ax3d.set_xlim(-1.2, 1.2)
    ax3d.set_ylim(-1.2, 1.2)
    ax3d.set_zlim(-1.1, 1.1)
    ax3d.set_title("Ocean S² & Great Circle")
    ax3d.legend(loc="upper left", fontsize=6.5)

    # Panels 2, 3, 4: Flat 2D Tangent-Plane Nautical Charts phi_1(U_1), phi_2(U_2), phi_3(U_3)
    for idx, chart, col, in_chart, uv_ship, _, _ in active_records:
        ax = fig.add_subplot(1, 4, idx + 1)
        lim = chart.euclidean_radius

        ax.plot(
            lim * np.cos(ang_circle),
            lim * np.sin(ang_circle),
            color=col,
            lw=2.0,
        )
        ax.fill(
            lim * np.cos(ang_circle),
            lim * np.sin(ang_circle),
            color=col,
            alpha=0.10 if not in_chart else 0.22,
        )

        geo_mask = chart.contains(geodesic_pts)
        if np.any(geo_mask):
            uv_geo = chart.phi(geodesic_pts[geo_mask])
            ax.plot(
                uv_geo[:, 0],
                uv_geo[:, 1],
                color="forestgreen",
                lw=2.2,
                label="Geodesic",
            )

        det_mask = chart.contains(detour_pts)
        if np.any(det_mask):
            uv_det = chart.phi(detour_pts[det_mask])
            ax.plot(
                uv_det[:, 0],
                uv_det[:, 1],
                "--",
                color="dimgray",
                lw=1.5,
                label="Detour",
            )

        for other_idx, other_chart, other_col, *_ in active_records:
            if other_idx != idx:
                other_bdry_3d = other_chart.patch_point_from_polar(1.0, ang_circle)
                in_this = chart.contains(other_bdry_3d)
                if np.any(in_this):
                    uv_ov = chart.phi(other_bdry_3d[in_this])
                    ax.plot(
                        uv_ov[:, 0],
                        uv_ov[:, 1],
                        ":",
                        color=other_col,
                        lw=1.4,
                        label=rf"$U_{idx} \cap U_{other_idx}$",
                    )

        if in_chart:
            ax.scatter(
                uv_ship[0],
                uv_ship[1],
                color="crimson",
                s=65,
                zorder=6,
                label=rf"$\varphi_{idx}(\gamma(t))$",
            )

        ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
        ax.axvline(0, color="grey", lw=0.6, alpha=0.5)
        ax.set_xlim(-lim * 1.18, lim * 1.18)
        ax.set_ylim(-lim * 1.18, lim * 1.18)
        ax.set_aspect("equal")
        status = "ACTIVE" if in_chart else "outside"
        ax.set_title(rf"Chart {idx}: $T_{{p_{idx}}}S^2$ ({status})")
        ax.set_xlabel(rf"$u_{idx}$")
        ax.set_ylabel(rf"$v_{idx}$")
        ax.legend(loc="upper left", fontsize=6.5)

    fig.tight_layout()
    fig
    return active_records, route_mode, ship_pt, t_val


@app.cell(hide_code=True)
def _(SphereAtlas, active_records, mo, np, route_mode, ship_pt, t_val):
    geo_length = SphereAtlas.geodesic_distance(
        SphereAtlas.start_point(), SphereAtlas.end_point()
    )
    detour_length = SphereAtlas.curve_arc_length(SphereAtlas.voyage_curve)

    lines = []
    active_list = [rec for rec in active_records if rec[3]]
    for _idx, _chart, _, _in_chart, _uv_ship, _g_ship, _cov_acc in active_records:
        if _in_chart:
            _cov_norm = float(np.sqrt(_cov_acc @ _g_ship @ _cov_acc))
            lines.append(
                rf"- **Chart $U_{_idx}$ (Active)**: $\varphi_{_idx}(\gamma(t)) = ({_uv_ship[0]:+.4f},\; {_uv_ship[1]:+.4f})$, "
                rf"$\quad \|\nabla_{{\dot{{\gamma}}}}\dot{{\gamma}}\|_g = {_cov_norm:.4f}$ "
                rf"*(covariant acceleration: {'0 on geodesic' if route_mode == 'geodesic' else 'non-zero steering on detour'})*"
            )
        else:
            lines.append(
                rf"- **Chart $U_{_idx}$ (Outside Domain)**: Ship is currently outside $U_{_idx}$."
            )

    if len(active_list) >= 2:
        _idx_a, _chart_a, _, _, _uv_a, *_ = active_list[0]
        _idx_b, _chart_b, _, _, _uv_b, *_ = active_list[1]
        _uv_b_trans = _chart_a.transition_to(_chart_b, _uv_a)
        _jac = _chart_a.transition_jacobian(_chart_b, _uv_a)
        _det_jac = float(np.linalg.det(_jac))
        transition_line = (
            rf"- **Overlap Transition Diffeomorphism ($U_{_idx_a} \cap U_{_idx_b}$)**: "
            rf"$\tau_{{{_idx_a}{_idx_b}}}(\varphi_{_idx_a}(\gamma(t))) = ({_uv_b_trans[0]:+.4f},\; {_uv_b_trans[1]:+.4f}) = \varphi_{_idx_b}(\gamma(t))$"
            "\n    "
            rf"- **Transition Jacobian $d\tau_{{{_idx_a}{_idx_b}}}$**: "
            rf"$\begin{{pmatrix}} {_jac[0,0]:+.4f} & {_jac[0,1]:+.4f} \\ {_jac[1,0]:+.4f} & {_jac[1,1]:+.4f} \end{{pmatrix}}$, "
            rf"$\quad \det(d\tau_{{{_idx_a}{_idx_b}}}) = {_det_jac:+.4f} \neq 0$ *(invertible diffeomorphism)*"
        )
    else:
        transition_line = r"- **Single-Chart Region**: Ship is currently inside a single chart; advance $t$ into an overlap region ($t \in [0.26, 0.40]$ or $t \in [0.60, 0.74]$) to inspect the transition diffeomorphism $\tau_{\alpha\beta}$ and its Jacobian $d\tau_{\alpha\beta}$."

    readout_md = "\n    ".join(lines + [transition_line])

    mo.md(rf"""
    ## 3. Equations: Coordinate Charts, Transition Diffeomorphisms, Metric, and Geodesics

    1. **Local Tangent-Plane Chart $(U_\alpha, \varphi_\alpha)$ and Inverse $\varphi_\alpha^{{-1}}$**:
       At each waypoint $p_\alpha \in S^2$ with orthonormal East-North tangent basis $(e_1^{{(\alpha)}}, e_2^{{(\alpha)}})$ spanning $T_{{p_\alpha}}S^2$:
       $$\varphi_\alpha(x) = (u, v) = \bigl(\langle x, e_1^{{(\alpha)}} \rangle,\; \langle x, e_2^{{(\alpha)}} \rangle\bigr), \qquad \varphi_\alpha^{{-1}}(u, v) = \sqrt{{1 - u^2 - v^2}}\,p_\alpha + u\,e_1^{{(\alpha)}} + v\,e_2^{{(\alpha)}}$$

    2. **Transition Diffeomorphism $\tau_{{\alpha\beta}}$ and Velocity Transformation ($d\tau_{{\alpha\beta}}$)**:
       On $U_\alpha \cap U_\beta$, coordinates and velocity vectors $(\dot{{u}}_\alpha, \dot{{v}}_\alpha)$ transform consistently via $\tau_{{\alpha\beta}} = \varphi_\beta \circ \varphi_\alpha^{{-1}}$ and its invertible Jacobian matrix $d\tau_{{\alpha\beta}}$:
       $$\tau_{{\alpha\beta}}(u, v) = \left(\left\langle \varphi_\alpha^{{-1}}(u,v),\; e_1^{{(\beta)}} \right\rangle,\; \left\langle \varphi_\alpha^{{-1}}(u,v),\; e_2^{{(\beta)}} \right\rangle\right), \qquad \begin{{pmatrix}} \dot{{u}}_\beta \\ \dot{{v}}_\beta \end{{pmatrix}} = d\tau_{{\alpha\beta}}(u_\alpha, v_\alpha) \begin{{pmatrix}} \dot{{u}}_\alpha \\ \dot{{v}}_\alpha \end{{pmatrix}}$$

    3. **Pullback Riemannian Metric $g_{{ij}}(u, v)$ and Christoffel Symbols $\Gamma^k_{{ij}}(u, v)$**:
       $$g_{{ij}}(u, v) = \delta_{{ij}} + \frac{{u_i u_j}}{{1 - u^2 - v^2}}, \qquad \Gamma^k_{{ij}}(u, v) = u^k\,g_{{ij}}(u, v)$$

    4. **Geodesic Equation & Great-Circle Parameterization on $S^2$**:
       $$\left(\nabla_{{\dot{{\gamma}}}}\dot{{\gamma}}\right)^k = \ddot{{u}}^k + \Gamma^k_{{ij}}(u, v)\,\dot{{u}}^i \dot{{u}}^j = 0 \quad \Longleftrightarrow \quad \gamma_{{\text{{geo}}}}(t) = \frac{{\sin\bigl((1 - t)\Omega\bigr)}}{{\sin\Omega}}\,p_0 + \frac{{\sin(t\Omega)}}{{\sin\Omega}}\,p_1, \quad \Omega = \arccos(\langle p_0, p_1 \rangle)$$

    ---

    ### Live Numerical Readout at Voyage Time $t = {t_val:.2f}$ ({'Geodesic Great Circle' if route_mode == 'geodesic' else 'Non-Geodesic Detour'})

    - **Total Route Arc Length Comparison**: Geodesic Great Circle $L[\gamma_{{\text{{geo}}}}] = {geo_length:.4f}\text{{ rad}}$ vs. Detour Route $L[\gamma_{{\text{{detour}}}}] = {detour_length:.4f}\text{{ rad}}$ *(Geodesic saves {((detour_length - geo_length) / geo_length) * 100:.2f}% distance)*
    - **Ship Position on Ocean**: $\gamma(t) = ({ship_pt[0]:+.4f},\; {ship_pt[1]:+.4f},\; {ship_pt[2]:+.4f}) \in S^2$
    {readout_md}
    """)
    return


if __name__ == "__main__":
    app.run()
