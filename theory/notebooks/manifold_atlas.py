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
    import torch

    import riemannian_manifolds.atlas.autodiff as _autodiff_mod
    import riemannian_manifolds.atlas.sphere as _sphere_mod
    import riemannian_manifolds.atlas as _atlas_mod

    importlib.reload(_autodiff_mod)
    importlib.reload(_sphere_mod)
    importlib.reload(_atlas_mod)

    from riemannian_manifolds.atlas import (
        SphereAtlas,
        christoffel_symbols_autodiff,
        pullback_metric_autodiff,
        render_sphere_charts_animation,
    )

    return (
        Path,
        SphereAtlas,
        christoffel_symbols_autodiff,
        mo,
        np,
        plt,
        pullback_metric_autodiff,
        render_sphere_charts_animation,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Manifolds, Coordinate Charts, Atlases, and Geodesics on $S^2$
    *Slide 1 of 7 — Fundamental Definitions: Manifold & Coordinate Chart*

    ---

    ### Definition 1: Manifold ($M$)
    - **Natural Language**: A **manifold** is a space that may be curved or closed on a global scale—like the ocean surface covering a spherical Earth ($S^2$)—but that looks locally like flat Euclidean space ($\mathbb{R}^n$) in a small neighborhood around every point. To a sailor looking out at a small patch of calm sea, the ocean appears flat even though the Earth is round.
    - **Mathematical Definition**: An $n$-dimensional **topological manifold** is a Hausdorff, second-countable topological space $M$ such that every point $p \in M$ has an open neighborhood $U \subset M$ homeomorphic to an open subset of $\mathbb{R}^n$. For the unit 2-sphere ($n = 2$):
      $$S^2 = \left\{ x = (x^1, x^2, x^3) \in \mathbb{R}^3 \;\middle|\; \|x\|^2 = (x^1)^2 + (x^2)^2 + (x^3)^2 = 1 \right\}$$

    ---

    ### Definition 2: Coordinate Chart ($(U_\alpha, \varphi_\alpha)$)
    - **Natural Language**: A **coordinate chart** (or simply **chart**) is a single local flat map covering a small open patch $U_\alpha \subset M$ of the manifold. On an ocean voyage, we construct a chart at a waypoint $p_\alpha \in S^2$ by resting a flat **tangent plane** $T_{p_\alpha} S^2$ against the sphere at $p_\alpha$ and projecting the nearby curved ocean patch $U_\alpha$ orthogonally onto that flat plane.
    - **Mathematical Definition**: A **chart** on $M$ is a pair $(U_\alpha, \varphi_\alpha)$ where $U_\alpha \subset M$ is an open set and
      $$\varphi_\alpha : U_\alpha \xrightarrow{\;\sim\;} \varphi_\alpha(U_\alpha) \subset \mathbb{R}^n$$
      is a homeomorphism onto an open subset of $\mathbb{R}^n$, assigning local Euclidean coordinates $(u, v) = \varphi_\alpha(x)$ to every point $x \in U_\alpha$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Smooth Atlases & Diffeomorphic Navigation
    *Slide 2 of 7 — Definition 3: Smooth Atlas ($\mathcal{A}$) and Transition Diffeomorphisms ($\tau_{\alpha\beta}$)*

    ---

    ### Definition 3: Smooth Atlas ($\mathcal{A}$)
    - **Natural Language**: Because a closed sphere cannot be flattened onto a single plane without tearing or severe horizon distortion, a single chart is not enough. An **atlas** is a collection of small, overlapping charts $\mathcal{A} = \{(U_\alpha, \varphi_\alpha)\}_{\alpha \in I}$ whose domains together cover the entire manifold ($M = \bigcup_{\alpha \in I} U_\alpha$).

    ---

    ### Why Diffeomorphism Enables Consistent Navigation
    - **Navigating Across Chart Overlaps**: As a ship sails from one chart $U_\alpha$ into an overlapping chart $U_\beta$, it passes through the overlap region $U_\alpha \cap U_\beta$ and is tracked on **both** flat tangent planes simultaneously. The **transition map** $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$ converts the ship's coordinates from Chart $\alpha$ to Chart $\beta$.
    - **Mathematical Guarantee**: For a smooth manifold, every transition map on $U_\alpha \cap U_\beta \neq \varnothing$ is a **$C^\infty$ diffeomorphism**:
      $$\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1} : \varphi_\alpha(U_\alpha \cap U_\beta) \xrightarrow{\;\sim\;} \varphi_\beta(U_\alpha \cap U_\beta)$$
      $$\tau_{\alpha\beta} \in C^\infty, \qquad \tau_{\alpha\beta}^{-1} = \tau_{\beta\alpha} \in C^\infty, \qquad \det\bigl(d\tau_{\alpha\beta}(u, v)\bigr) \neq 0$$
    - Because $\tau_{\alpha\beta}$ is bijective and its Jacobian matrix $d\tau_{\alpha\beta}$ is everywhere invertible, positions, velocities, compass bearings, and differential equations of motion translate uniquely and reversibly from one chart to the next without jumps or kinks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Riemannian Metrics & Geodesics (Great Circles)
    *Slide 3 of 7 — Definition 4: Riemannian Metric ($g$) and Geodesic ($\gamma$)*

    ---

    ### Definition 4: Riemannian Metric ($g$) and Geodesic ($\gamma$)
    - **Riemannian Metric**: A **Riemannian metric** $g$ equips each tangent plane $T_p M$ with a smoothly varying inner product $g_p : T_p M \times T_p M \to \mathbb{R}$, allowing us to measure true surface lengths, speeds, and angles inside flat chart coordinates $(u, v)$.
    - **What Is a Geodesic?**: A **geodesic** is the generalization of a straight line to a curved manifold, characterized by two equivalent properties:
      1. **Locally Shortest Path**: It minimizes the Riemannian arc length $L[\gamma] = \int_0^1 \sqrt{g_{ij}(u)\,\dot{u}^i \dot{u}^j}\,dt$ between nearby points.
      2. **Zero Covariant Acceleration ($\nabla_{\dot{\gamma}}\dot{\gamma} = 0$)**: It is the path followed by a ship that **locks its rudder straight ahead and sails at constant speed without turning left or right**. Its 3D acceleration points purely along the surface normal toward the Earth's center, with zero tangential acceleration along the water.
    - **Great Circles on $S^2$**: On the sphere $S^2$, every geodesic is an arc of a **great circle** (the intersection of $S^2$ with a plane through the origin). In local chart coordinates $(u, v)$, a geodesic satisfies the **geodesic differential equation**:
      $$\left(\nabla_{\dot{\gamma}}\dot{\gamma}\right)^k = \ddot{u}^k(t) + \Gamma^k_{ij}\bigl(u(t)\bigr)\,\dot{u}^i(t)\,\dot{u}^j(t) = 0$$
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
                # 3D Animation: Overlapping Tangent Charts & Geodesic Navigation
                *Slide 4 of 7 — Manim 3D Visualization*

                Three overlapping ocean patches $U_1$ (teal), $U_2$ (yellow), and $U_3$ (orange) project orthogonally onto their flat tangent planes $T_{p_1}S^2, T_{p_2}S^2, T_{p_3}S^2$. The ship (red dot) sails along the **great-circle geodesic** (solid green, contrasted with the dashed grey detour) and projects simultaneously onto both adjacent tangent planes in each overlap region $U_\alpha \cap U_\beta$:
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
        value=0.46,
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
                # Interactive Atlas & Geodesic Explorer
                *Slide 5 of 7 — Live Chart Projections, Overlaps, and Geodesic Verification*

                Toggle between **Geodesic (Great Circle)** and **Non-Geodesic Detour**, and drag $t \in [0, 1]$ to compare the ship's position, velocity vector $\dot{u}$ (blue arrow), and sideways steering / covariant acceleration vector $\nabla_{\dot{\gamma}}\dot{\gamma}$ (red arrow, non-zero only on the detour!):
                """
            ),
            mo.hstack([voyage_slider, route_selector], justify="start", gap=2.5),
        ]
    )
    return route_selector, voyage_slider


@app.cell(hide_code=True)
def _(SphereAtlas, mo, np, plt, route_selector, voyage_slider):
    t_val = float(voyage_slider.value)
    route_mode = route_selector.value
    is_geodesic = route_mode == "geodesic"
    charts = SphereAtlas.default_voyage_charts(angular_radius=0.44)
    colors = ["teal", "goldenrod", "darkorange"]

    active_curve_fn = (
        SphereAtlas.geodesic_voyage if is_geodesic else SphereAtlas.voyage_curve
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
    route_title = (
        "ACTIVE: Geodesic (Great Circle) — Zero Sideways Acceleration (∇_γ̇ γ̇ = 0)"
        if is_geodesic
        else "ACTIVE: Non-Geodesic Detour — Sideways Rudder Acceleration (∇_γ̇ γ̇ ≠ 0, Red Arrow)"
    )
    fig.suptitle(
        route_title,
        fontsize=11,
        fontweight="bold",
        color="forestgreen" if is_geodesic else "crimson",
    )

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
        "-" if not is_geodesic else "--",
        color="crimson" if not is_geodesic else "dimgray",
        lw=2.8 if not is_geodesic else 1.3,
        alpha=1.0 if not is_geodesic else 0.45,
        label="Detour γ_detour",
    )
    ax3d.plot(
        geodesic_pts[:, 0],
        geodesic_pts[:, 1],
        geodesic_pts[:, 2],
        "-" if is_geodesic else "--",
        color="forestgreen",
        lw=2.8 if is_geodesic else 1.4,
        alpha=1.0 if is_geodesic else 0.45,
        label="Geodesic (Great Circle)",
    )
    ax3d.scatter(
        *ship_pt,
        color="forestgreen" if is_geodesic else "crimson",
        edgecolors="black",
        s=75,
        zorder=10,
        label="Ship γ(t)",
    )

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

        active_records.append(
            (idx, chart, col, in_chart, uv_ship, g_ship, duv_dt, cov_acc)
        )

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
    for idx, chart, col, in_chart, uv_ship, _, duv_dt, cov_acc in active_records:
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
                "-" if is_geodesic else "--",
                color="forestgreen",
                lw=2.8 if is_geodesic else 1.4,
                alpha=1.0 if is_geodesic else 0.45,
                label="Geodesic",
            )

        det_mask = chart.contains(detour_pts)
        if np.any(det_mask):
            uv_det = chart.phi(detour_pts[det_mask])
            ax.plot(
                uv_det[:, 0],
                uv_det[:, 1],
                "-" if not is_geodesic else "--",
                color="crimson" if not is_geodesic else "dimgray",
                lw=2.8 if not is_geodesic else 1.4,
                alpha=1.0 if not is_geodesic else 0.45,
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
                color="forestgreen" if is_geodesic else "crimson",
                edgecolors="black",
                s=75,
                zorder=7,
                label=rf"$\varphi_{idx}(\gamma(t))$",
            )
            # Velocity vector arrow (blue)
            vel_scale = 0.14
            ax.quiver(
                uv_ship[0],
                uv_ship[1],
                vel_scale * duv_dt[0],
                vel_scale * duv_dt[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="royalblue",
                width=0.014,
                zorder=8,
                label=r"Velocity $\dot{u}$",
            )
            # Covariant acceleration / steering arrow (red, visible on detour)
            acc_norm = float(np.linalg.norm(cov_acc))
            if acc_norm > 1e-3:
                acc_scale = 0.09
                ax.quiver(
                    uv_ship[0],
                    uv_ship[1],
                    acc_scale * cov_acc[0],
                    acc_scale * cov_acc[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color="crimson",
                    width=0.016,
                    zorder=9,
                    label=r"Steering $\nabla_{\dot{\gamma}}\dot{\gamma}$",
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
        ax.legend(loc="upper left", fontsize=6.0)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    geo_length = SphereAtlas.geodesic_distance(
        SphereAtlas.start_point(), SphereAtlas.end_point()
    )
    detour_length = SphereAtlas.curve_arc_length(SphereAtlas.voyage_curve)

    lines = []
    active_list = [rec for rec in active_records if rec[3]]
    for _idx, _chart, _, _in_chart, _uv_ship, _g_ship, _duv_dt, _cov_acc in active_records:
        if _in_chart:
            _cov_norm = float(np.sqrt(_cov_acc @ _g_ship @ _cov_acc))
            lines.append(
                rf"- **Chart $U_{_idx}$ (Active)**: $\varphi_{_idx}(\gamma(t)) = ({_uv_ship[0]:+.4f},\; {_uv_ship[1]:+.4f})$, "
                rf"$\quad \|\nabla_{{\dot{{\gamma}}}}\dot{{\gamma}}\|_g = {_cov_norm:.4f}$ "
                rf"*(covariant acceleration: {'0 on geodesic — no rudder turn' if is_geodesic else 'non-zero sideways rudder steering on detour'})*"
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
            rf"$\tau_{{{_idx_a}{_idx_b}}}(\varphi_{_idx_a}(\gamma(t))) = ({_uv_b_trans[0]:+.4f},\; {_uv_b_trans[1]:+.4f}) = \varphi_{_idx_b}(\gamma(t))$, "
            rf"$\quad \det(d\tau_{{{_idx_a}{_idx_b}}}) = {_det_jac:+.4f} \neq 0$"
        )
    else:
        transition_line = r"- **Single-Chart Region**: Advance $t$ into an overlap ($t \in [0.26, 0.40]$ or $t \in [0.60, 0.74]$) to inspect the transition diffeomorphism $\tau_{\alpha\beta}$ and its Jacobian $d\tau_{\alpha\beta}$."

    readout_md = "\n    ".join(lines + [transition_line])

    mo.vstack(
        [
            fig,
            mo.md(
                rf"""
                **Live Readout at $t = {t_val:.2f}$ ({'Geodesic Great Circle' if is_geodesic else 'Non-Geodesic Detour'})** — **Total Arc Length**: Geodesic $L[\gamma_{{\text{{geo}}}}] = {geo_length:.4f}\text{{ rad}}$ vs. Detour $L[\gamma_{{\text{{detour}}}}] = {detour_length:.4f}\text{{ rad}}$ *(Geodesic saves {((detour_length - geo_length) / geo_length) * 100:.2f}%)*
                {readout_md}
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Governing Equations: Charts, Transitions, Metric, and Geodesics
    *Slide 6 of 7 — Complete Symbolic Summary*

    ---

    1. **Local Tangent-Plane Chart $(U_\alpha, \varphi_\alpha)$ and Inverse $\varphi_\alpha^{-1}$**:
       At each waypoint $p_\alpha \in S^2$ with orthonormal East-North tangent basis $(e_1^{(\alpha)}, e_2^{(\alpha)})$ spanning $T_{p_\alpha}S^2$:
       $$\varphi_\alpha(x) = (u, v) = \bigl(\langle x, e_1^{(\alpha)} \rangle,\; \langle x, e_2^{(\alpha)} \rangle\bigr), \qquad \varphi_\alpha^{-1}(u, v) = \sqrt{1 - u^2 - v^2}\,p_\alpha + u\,e_1^{(\alpha)} + v\,e_2^{(\alpha)}$$

    2. **Transition Diffeomorphism $\tau_{\alpha\beta}$ and Velocity Transformation ($d\tau_{\alpha\beta}$)**:
       On $U_\alpha \cap U_\beta$, coordinates and velocity vectors $(\dot{u}_\alpha, \dot{v}_\alpha)$ transform consistently via $\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}$ and its invertible Jacobian matrix $d\tau_{\alpha\beta}$:
       $$\tau_{\alpha\beta}(u, v) = \left(\left\langle \varphi_\alpha^{-1}(u,v),\; e_1^{(\beta)} \right\rangle,\; \left\langle \varphi_\alpha^{-1}(u,v),\; e_2^{(\beta)} \right\rangle\right), \qquad \begin{pmatrix} \dot{u}_\beta \\ \dot{v}_\beta \end{pmatrix} = d\tau_{\alpha\beta}(u_\alpha, v_\alpha) \begin{pmatrix} \dot{u}_\alpha \\ \dot{v}_\alpha \end{pmatrix}$$

    3. **Pullback Riemannian Metric $g_{ij}(u, v)$ and Christoffel Symbols $\Gamma^k_{ij}(u, v)$**:
       $$g_{ij}(u, v) = \delta_{ij} + \frac{u_i u_j}{1 - u^2 - v^2}, \qquad \Gamma^k_{ij}(u, v) = u^k\,g_{ij}(u, v)$$

    4. **Geodesic Equation & Great-Circle Parameterization on $S^2$**:
       $$\left(\nabla_{\dot{\gamma}}\dot{\gamma}\right)^k = \ddot{u}^k + \Gamma^k_{ij}(u, v)\,\dot{u}^i \dot{u}^j = 0 \quad \Longleftrightarrow \quad \gamma_{\text{geo}}(t) = \frac{\sin\bigl((1 - t)\Omega\bigr)}{\sin\Omega}\,p_0 + \frac{\sin(t\Omega)}{\sin\Omega}\,p_1, \quad \Omega = \arccos(\langle p_0, p_1 \rangle)$$
    """)
    return


@app.cell(hide_code=True)
def _(
    SphereAtlas,
    christoffel_symbols_autodiff,
    mo,
    np,
    pullback_metric_autodiff,
    torch,
):
    _chart = SphereAtlas.default_voyage_charts()[1]
    _u_np = np.array([0.14, -0.19], dtype=np.float64)
    _u_t = torch.tensor(_u_np, dtype=torch.float64)
    _p_t = torch.tensor(_chart.center, dtype=torch.float64)
    _e1_t = torch.tensor(_chart.e1, dtype=torch.float64)
    _e2_t = torch.tensor(_chart.e2, dtype=torch.float64)

    def _sphere_immersion(u):
        w = torch.sqrt(1.0 - torch.sum(u * u))
        return w * _p_t + u[0] * _e1_t + u[1] * _e2_t

    _g_auto = pullback_metric_autodiff(_sphere_immersion, _u_t).detach().numpy()
    _g_exact = _chart.metric_tensor(_u_np)
    _g_err = float(np.max(np.abs(_g_auto - _g_exact)))

    _gamma_auto = (
        christoffel_symbols_autodiff(_sphere_immersion, _u_t).detach().numpy()
    )
    _gamma_exact = _chart.christoffel_symbols(_u_np)
    _gamma_err = float(np.max(np.abs(_gamma_auto - _gamma_exact)))

    mo.vstack(
        [
            mo.md(
                r"""
                # Modeling Deep Learning Activation Spaces via Riemannian Geometry & Autodiff
                *Slide 7 of 7 — From $S^2$ to Neural Activation Manifolds with Automatic Differentiation*

                ---

                ### 1. The Activation Manifold of a Deep Learning Model
                - **Natural Language**: On $S^2$, we derived the inverse chart $\varphi_\alpha^{-1}(u, v)$, metric $g_{ij}$, and Christoffel symbols $\Gamma^k_{ij}$ by hand because the sphere has a closed-form equation. In a deep neural network, an encoder or hidden layer maps local intrinsic coordinates $u \in U \subset \mathbb{R}^d$ into a high-dimensional activation space $h = f_\theta(u) \in \mathbb{R}^D$ ($D \gg d$), tracing out a curved **activation manifold** $\mathcal{M} = f_\theta(U) \subset \mathbb{R}^D$.
                - **Why Smooth Activations Matter**: Because the Christoffel symbols $\Gamma^k_{ij}(u)$ depend on **second derivatives** of $f_\theta(u)$ (first derivatives of the metric $\partial_m g_{ij}$), piecewise-linear activations like $\text{ReLU}$ have zero second derivative almost everywhere and undefined curvature at hinges. Using smooth $C^\infty$ activations such as **$\text{Softplus}$** or **$\text{SMU}$ (Smooth Maximum Unit)** turns $f_\theta$ into a true smooth immersion with well-defined Riemannian curvature and geodesics everywhere.
                - **Pullback Metric & Levi-Civita Connection from the Network Jacobian $J_{f_\theta}(u) = \frac{\partial f_\theta}{\partial u} \in \mathbb{R}^{D \times d}$**:
                  $$g_{ij}(u) = \Bigl(J_{f_\theta}(u)^\top J_{f_\theta}(u)\Bigr)_{ij} = \sum_{a=1}^D \frac{\partial f_\theta^a}{\partial u^i}\frac{\partial f_\theta^a}{\partial u^j}, \qquad \Gamma^k_{ij}(u) = \frac{1}{2}\,g^{km}(u)\!\left(\frac{\partial g_{mi}}{\partial u^j} + \frac{\partial g_{mj}}{\partial u^i} - \frac{\partial g_{ij}}{\partial u^m}\right)$$

                ---

                ### 2. How Our Project Packages Power This via Automatic Differentiation
                Every differential-geometric object on $\mathcal{M} = f_\theta(U)$ is computed **exactly** (to floating-point precision, with zero finite-difference truncation error) by composing the packages already installed in `pyproject.toml`:

                | Project Package | Role in the Riemannian Activation-Space Pipeline |
                | :--- | :--- |
                | **`torch` & `torch.func`** (`jacrev`, `jacfwd`, `jvp`, `vjp`) | **1st-Order Autodiff**: `jacrev(f_theta)(u)` computes the exact chart/decoder Jacobian $J_{f_\theta}(u)$ and transition Jacobians $d\tau_{\alpha\beta}$, yielding the pullback metric $g(u) = J_{f_\theta}(u)^\top J_{f_\theta}(u)$.<br>**2nd-Order Autodiff**: Nesting `jacrev(lambda u: pullback_metric_autodiff(f_theta, u))(u)` differentiates right through the first-order graph to give the exact metric derivatives $\partial_m g_{ij}(u)$ and Christoffel symbols $\Gamma^k_{ij}(u)$. |
                | **`scipy`** (`scipy.integrate.solve_ivp`, `solve_bvp`, `scipy.optimize`) | Integrates the **geodesic ODE** $\ddot{u}^k + \Gamma^k_{ij}(u)\dot{u}^i\dot{u}^j = 0$ and solves two-point boundary value problems between neural activations $h_0, h_1 \in \mathcal{M}$, calling PyTorch's autodiff $\Gamma^k_{ij}(u)$ at each solver step. |
                | **`numpy`, `matplotlib`, `manim`, & `marimo`** | Bridges exact PyTorch tensors to 3D Manim animations (`SphereChartsScene`) and reactive Marimo notebooks for interactive exploration of charts, curvature, and geodesics. |

                ---
                """
            ),
            mo.md(
                rf"""
                ### 3. Live `torch.func.jacrev` Verification (`riemannian_manifolds.atlas.autodiff`)
                Evaluating `pullback_metric_autodiff` and `christoffel_symbols_autodiff` on $S^2$ at $u = ({_u_np[0]:+.2f}, {_u_np[1]:+.2f})$ against the closed-form equations on Slide 6:
                $$\max_{{i,j}} \bigl|g^{{\text{{autodiff}}}}_{{ij}}(u) - g^{{\text{{exact}}}}_{{ij}}(u)\bigr| = {_g_err:.2e}, \qquad \max_{{k,i,j}} \bigl|\Gamma^{{k,\text{{autodiff}}}}_{{ij}}(u) - \Gamma^{{k,\text{{exact}}}}_{{ij}}(u)\bigr| = {_gamma_err:.2e}$$
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
