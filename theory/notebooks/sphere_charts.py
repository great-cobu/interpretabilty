import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", css_file="theme.css")


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
    # Manifolds, Atlases, Tangent-Plane Charts, and Geodesics on $S^2$

    ## 1. Natural Language: Foundations, Consistent Navigation, and Geodesics

    ### Core Definitions: Manifold, Chart, and Atlas

    - **Manifold ($M$)**: A **manifold** is a space that may be curved or closed on a global scale—like the ocean surface covering a spherical Earth ($S^2$)—but that looks locally like flat Euclidean space ($\mathbb{R}^n$) in the neighborhood of every point.
    - **Coordinate Chart ($(U_\alpha, \varphi_\alpha)$)**: A **chart** is a local flat map of a small open patch $U_\alpha \subset M$. It pairs the curved patch $U_\alpha$ with a continuous, invertible coordinate map $\varphi_\alpha : U_\alpha \to \varphi_\alpha(U_\alpha) \subset \mathbb{R}^n$ that assigns flat coordinates $(u, v)$ to each point—here constructed by projecting a local patch of ocean orthogonally onto its flat **tangent plane** $T_{p_\alpha} S^2$ at a waypoint $p_\alpha$.
    - **Atlas ($\mathcal{A} = \{(U_\alpha, \varphi_\alpha)\}$)**: Because a curved sphere cannot be flattened onto a single plane without tearing or severe horizon distortion, we cover the manifold with a collection of **small overlapping charts** called an **atlas**, whose domains together cover the space ($M = \bigcup_\alpha U_\alpha$).

    ### Why Diffeomorphism Enables Consistent Navigation

    When a ship sails from one chart $(U_\alpha, \varphi_\alpha)$ into an overlapping chart $(U_\beta, \varphi_\beta)$, the **transition map** $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$ converts flat coordinates $(u_\alpha, v_\alpha)$ on the first tangent plane into flat coordinates $(u_\beta, v_\beta)$ on the second. Requiring every transition map to be a **diffeomorphism**—bijective and infinitely differentiable ($C^\infty$) in both directions with an everywhere-invertible Jacobian matrix $d\tau_{\alpha\beta}$ ($\det d\tau_{\alpha\beta} \neq 0$)—guarantees that smooth trajectories, velocities, compass bearings, and differential equations of motion translate uniquely and consistently from chart to chart.

    ### Defining a Geodesic: Sailing on a Great Circle

    What does it mean for a ship to sail in a "straight line" across a curved ocean where no Euclidean straight line exists? In Riemannian geometry, this path is called a **geodesic**, defined by two equivalent properties:

    1. **Locally Shortest Path**: A geodesic $\gamma(t)$ minimizes the Riemannian arc length between nearby points on the manifold.
    2. **Zero Covariant Acceleration ($\nabla_{\dot{\gamma}}\dot{\gamma} = 0$)**: A geodesic is the path followed by a ship that **locks its rudder straight ahead and sails at constant speed without turning left or right**. In 3D space, the ship's acceleration vector points purely perpendicular to the ocean surface (straight toward the center of the Earth to stay on the sphere), with **zero tangential acceleration** along the water.

    On the sphere $S^2$, every geodesic is an arc of a **great circle**—the intersection of the ocean surface with a plane passing through the center of the Earth. In each local tangent-plane chart $(U_\alpha, \varphi_\alpha)$, the great circle passes straight through the waypoint center $(0, 0)$ without bending ($\Gamma^k_{ij}(0, 0) = 0$) and satisfies the **geodesic differential equation** $\ddot{u}^k + \Gamma^k_{ij}(u, v)\,\dot{u}^i \dot{u}^j = 0$, which is preserved across chart overlaps by the transition diffeomorphisms $\tau_{\alpha\beta}$.
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
                ## 2. Animation: Geodesic (Great Circle) Navigation Across Overlapping Tangent Charts

                The 3D animation below constructs three overlapping tangent-plane charts $U_1$ (teal), $U_2$ (yellow), and $U_3$ (orange), contrasts a curved non-geodesic detour (dashed grey) with the **geodesic great-circle route** (solid green), and sails the ship (red dot) along the geodesic across the overlapping tangent planes.
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
                ### Interactive Geodesic & Tangent-Chart Explorer

                Choose whether the ship sails along the **Geodesic (Great Circle)** or the **Non-Geodesic Detour**, and slide $t \in [0, 1]$ to compare their paths, arc lengths, and covariant accelerations across the three tangent-plane charts:
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

    # Finite-difference 1st and 2nd time derivatives for covariant acceleration readout
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
        uv_circle = np.stack([lim * np.cos(ang_circle), lim * np.sin(ang_circle)], axis=-1)
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

        # Compute local velocity du/dt, acceleration d^2u/dt^2, and covariant acceleration nabla_{dot{gamma}} dot{gamma}
        uv_minus = chart.phi(active_curve_fn(t_minus))
        uv_plus = chart.phi(active_curve_fn(t_plus))
        uv_mid = chart.phi(active_curve_fn(0.5 * (t_minus + t_plus)))
        duv_dt = (uv_plus - uv_minus) / dt_span
        d2uv_dt2 = (uv_plus - 2.0 * uv_mid + uv_minus) / ((0.5 * dt_span) ** 2)
        cov_acc = chart.covariant_acceleration(uv_ship, duv_dt, d2uv_dt2)

        active_records.append((idx, chart, col, in_chart, uv_ship, g_ship, cov_acc))

        if in_chart:
            plane_pt = chart.project_to_tangent_plane_3d(ship_pt) + 0.12 * chart.center
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

        # Plot geodesic great circle in chart (solid green)
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

        # Plot detour curve in chart (dashed grey)
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

        # Show overlap boundaries with adjacent charts
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
    ## 3. Equations: Manifold, Smooth Atlas, Transition Diffeomorphisms, and Geodesics

    1. **Manifold, Charts, and Smooth Atlas**:
       An $n$-dimensional **smooth manifold** $M$ (here $S^2 \subset \mathbb{{R}}^3$ with $n = 2$) is equipped with a **smooth atlas** $\mathcal{{A}} = \{{(U_\alpha, \varphi_\alpha)\}}$ covering $M = \bigcup_\alpha U_\alpha$ such that on every overlap $U_\alpha \cap U_\beta \neq \varnothing$, the **transition map** is a $C^\infty$ **diffeomorphism**:
       $$\tau_{{\alpha\beta}} = \varphi_\beta \circ \varphi_\alpha^{{-1}} : \varphi_\alpha(U_\alpha \cap U_\beta) \xrightarrow{{\;\sim\;}} \varphi_\beta(U_\alpha \cap U_\beta), \qquad \tau_{{\alpha\beta}}^{{-1}} = \tau_{{\beta\alpha}}, \qquad \det\bigl(d\tau_{{\alpha\beta}}(u,v)\bigr) \neq 0$$

    2. **Local Tangent-Plane Chart $(U_\alpha, \varphi_\alpha)$ and Pullback Metric $g_{{ij}}(u, v)$**:
       At each waypoint $p_\alpha \in S^2$ with orthonormal East-North basis $(e_1^{{(\alpha)}}, e_2^{{(\alpha)}})$ spanning $T_{{p_\alpha}}S^2$:
       $$\varphi_\alpha(x) = (u, v) = \bigl(\langle x, e_1^{{(\alpha)}} \rangle,\; \langle x, e_2^{{(\alpha)}} \rangle\bigr), \qquad \varphi_\alpha^{{-1}}(u, v) = \sqrt{{1 - u^2 - v^2}}\,p_\alpha + u\,e_1^{{(\alpha)}} + v\,e_2^{{(\alpha)}}, \qquad g_{{ij}}(u, v) = \delta_{{ij}} + \frac{{u_i u_j}}{{1 - u^2 - v^2}}$$

    3. **Defining a Geodesic (Shortest Path & Zero Covariant Acceleration)**:
       A smooth curve $\gamma : [0, 1] \to M$ is a **geodesic** if it is a critical point of the Riemannian arc-length functional $L[\gamma] = \int_0^1 \sqrt{{g_{{ij}}\,\dot{{u}}^i \dot{{u}}^j}}\,dt$ and has **zero covariant acceleration** ($\nabla_{{\dot{{\gamma}}}}\dot{{\gamma}} = 0$). In any local tangent-plane chart $(U_\alpha, \varphi_\alpha)$, the geodesic satisfies the **geodesic differential equation**:
       $$\left(\nabla_{{\dot{{\gamma}}}}\dot{{\gamma}}\right)^k = \ddot{{u}}^k + \Gamma^k_{{ij}}(u, v)\,\dot{{u}}^i \dot{{u}}^j = 0, \qquad \text{{where}} \quad \Gamma^k_{{ij}}(u, v) = u^k\,g_{{ij}}(u, v)$$
       Notice that at the chart origin $(u, v) = (0, 0)$, the Christoffel symbols vanish ($\Gamma^k_{{ij}}(0, 0) = 0$), so a geodesic passing through the waypoint center appears completely straight ($\ddot{{u}}^k(0) = 0$) at the point of tangency.

    4. **Great-Circle Geodesic on $S^2$ (Spherical Linear Interpolation)**:
       In ambient 3D coordinates on the unit sphere $S^2$, the unique minimizing geodesic between $p_0, p_1 \in S^2$ separated by central angle $\Omega = \arccos(\langle p_0, p_1 \rangle)$ is the **great-circle arc**:
       $$\gamma_{{\text{{geo}}}}(t) = \frac{{\sin\bigl((1 - t)\Omega\bigr)}}{{\sin\Omega}}\,p_0 + \frac{{\sin(t\Omega)}}{{\sin\Omega}}\,p_1, \qquad L[\gamma_{{\text{{geo}}}}] = \Omega$$

    ---

    ### Live Numerical Readout at Voyage Time $t = {t_val:.2f}$ ({'Geodesic Great Circle' if route_mode == 'geodesic' else 'Non-Geodesic Detour'})

    - **Total Route Arc Length Comparison**: Geodesic Great Circle $L[\gamma_{{\text{{geo}}}}] = {geo_length:.4f}\text{{ rad}}$ vs. Detour Route $L[\gamma_{{\text{{detour}}}}] = {detour_length:.4f}\text{{ rad}}$ *(Geodesic saves {((detour_length - geo_length) / geo_length) * 100:.2f}% distance)*
    - **Ship Position on Ocean**: $\gamma(t) = ({ship_pt[0]:+.4f},\; {ship_pt[1]:+.4f},\; {ship_pt[2]:+.4f}) \in S^2$
    {readout_md}
    """)
    return


if __name__ == "__main__":
    app.run()
