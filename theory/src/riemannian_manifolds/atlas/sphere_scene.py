"""Manim 3D animation of overlapping tangent-plane charts and geodesic (great-circle) navigation on S^2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    BLUE_B,
    BLUE_D,
    BLUE_E,
    DEGREES,
    DOWN,
    GREEN_C,
    GREY_B,
    LEFT,
    ORANGE,
    RED_C,
    TEAL_C,
    UL,
    UR,
    WHITE,
    YELLOW,
    Create,
    DashedLine,
    DashedVMobject,
    Dot3D,
    FadeIn,
    FadeOut,
    MathTex,
    ParametricFunction,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    TransformFromCopy,
    ValueTracker,
    VGroup,
    always_redraw,
    tempconfig,
)

from riemannian_manifolds.atlas.sphere import SphereAtlas, TangentChart


class SphereChartsScene(ThreeDScene):
    """Manim ThreeDScene illustrating overlapping tangent-plane charts and sailing on a geodesic (great circle)."""

    SPHERE_RADIUS: float = 2.15
    TANGENT_OFFSET: float = 0.28

    def _build_spherical_patch(self, chart: TangentChart, color: str) -> Surface:
        """Create the curved ocean patch U_alpha on the sphere."""
        R = self.SPHERE_RADIUS
        return Surface(
            lambda r_frac, ang: R * chart.patch_point_from_polar(r_frac, ang),
            u_range=[0.0, 1.0],
            v_range=[0.0, 2.0 * np.pi],
            resolution=(8, 20),
            fill_opacity=0.45,
            checkerboard_colors=[color, color],
            stroke_color=color,
            stroke_width=1.0,
            stroke_opacity=0.75,
        )

    def _build_tangent_plane_chart(self, chart: TangentChart, color: str) -> Surface:
        """Create the flat tangent-plane nautical chart hovering tangent to p_alpha."""
        R = self.SPHERE_RADIUS
        offset = self.TANGENT_OFFSET * chart.center
        lim = chart.euclidean_radius
        return Surface(
            lambda u, v: R * chart.tangent_plane_point(np.array([u, v])) + offset,
            u_range=[-lim, lim],
            v_range=[-lim, lim],
            resolution=(7, 7),
            fill_opacity=0.32,
            checkerboard_colors=[color, color],
            stroke_color=color,
            stroke_width=1.3,
            stroke_opacity=0.9,
        )

    def _build_projection_rays(self, chart: TangentChart, color: str) -> VGroup:
        """Create orthogonal projection rays between the curved patch U_alpha and its tangent plane."""
        R = self.SPHERE_RADIUS
        offset = self.TANGENT_OFFSET * chart.center
        rays = VGroup()
        for ang in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False):
            pt_unit = chart.patch_point_from_polar(0.92, float(ang))
            pt_sphere = R * pt_unit
            pt_plane = R * chart.project_to_tangent_plane_3d(pt_unit) + offset
            rays.add(
                DashedLine(
                    start=pt_sphere,
                    end=pt_plane,
                    color=color,
                    dash_length=0.07,
                    stroke_width=1.8,
                )
            )
        return rays

    def construct(self) -> None:
        self.camera.background_color = "#0e1117"
        R = self.SPHERE_RADIUS

        self.set_camera_orientation(phi=64 * DEGREES, theta=-12 * DEGREES, zoom=0.95)

        axes = ThreeDAxes(
            x_range=[-3.0, 3.0, 1],
            y_range=[-3.0, 3.0, 1],
            z_range=[-2.6, 2.6, 1],
            x_length=6.0,
            y_length=6.0,
            z_length=5.2,
            axis_config={"color": GREY_B, "stroke_width": 1.2, "stroke_opacity": 0.45},
        )

        # Ocean Globe S^2
        ocean_sphere = Surface(
            lambda u, v: R
            * np.array(
                [
                    np.sin(u) * np.cos(v),
                    np.sin(u) * np.sin(v),
                    np.cos(u),
                ]
            ),
            u_range=[0, np.pi],
            v_range=[0, 2 * np.pi],
            resolution=(24, 48),
            fill_opacity=0.25,
            checkerboard_colors=[BLUE_E, BLUE_D],
            stroke_color=BLUE_B,
            stroke_width=0.5,
            stroke_opacity=0.30,
        )

        charts = SphereAtlas.default_voyage_charts(angular_radius=0.44)
        colors = [TEAL_C, YELLOW, ORANGE]

        # Fixed 2D HUD overlays: Title + Natural Language + Equations
        title = Text(
            "Atlas of Tangent Charts & Geodesic Navigation on S²",
            font_size=24,
            color=WHITE,
        ).to_corner(UL, buff=0.30)

        nl_caption = Text(
            "Manifold S² covered along a route by overlapping tangent-plane charts (U_α, φ_α).",
            font_size=17,
            color=GREY_B,
        ).next_to(title, DOWN, aligned_edge=LEFT, buff=0.14)

        eq_chart = MathTex(
            r"\varphi_\alpha : U_\alpha \subset S^2 \to T_{p_\alpha} S^2 \cong \mathbb{R}^2, \quad "
            r"\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1} \in \mathrm{Diff}^\infty",
            font_size=21,
            color=WHITE,
        ).next_to(nl_caption, DOWN, aligned_edge=LEFT, buff=0.14)

        self.add_fixed_in_frame_mobjects(title, nl_caption, eq_chart)
        self.play(Create(axes), Create(ocean_sphere), run_time=1.6)

        patches = []
        planes = []
        rays_list = []
        centers = []

        for chart, col in zip(charts, colors, strict=True):
            patch = self._build_spherical_patch(chart, col)
            plane = self._build_tangent_plane_chart(chart, col)
            rays = self._build_projection_rays(chart, col)
            center_dot = Dot3D(point=R * chart.center, color=col, radius=0.055)

            patches.append(patch)
            planes.append(plane)
            rays_list.append(rays)
            centers.append(center_dot)

        # --- Phase 1: Lift overlapping ocean patches U_1, U_2, U_3 to tangent planes ---
        nl_phase1 = Text(
            "Diffeomorphic transitions τ_{αβ} glue local tangent planes into a consistent atlas.",
            font_size=17,
            color=TEAL_C,
        ).move_to(nl_caption, aligned_edge=LEFT)

        eq_inv = MathTex(
            r"\varphi_\alpha^{-1}(u,v) = \sqrt{1-u^2-v^2}\,p_\alpha + u\,e_1^{(\alpha)} + v\,e_2^{(\alpha)}",
            font_size=21,
            color=TEAL_C,
        ).to_corner(UR, buff=0.30)

        self.add_fixed_in_frame_mobjects(nl_phase1, eq_inv)
        self.play(FadeOut(nl_caption), FadeIn(nl_phase1), FadeIn(eq_inv), run_time=0.8)

        for idx in range(3):
            self.play(
                FadeIn(centers[idx]),
                Create(patches[idx]),
                run_time=0.8,
            )
            self.play(
                Create(rays_list[idx]),
                TransformFromCopy(patches[idx], planes[idx]),
                run_time=1.0,
            )

        # --- Phase 2: Defining & Sailing on a Geodesic (Great Circle) vs. Detour ---
        nl_phase2 = Text(
            "Geodesic (Great Circle, green): shortest path with zero covariant acceleration (∇_γ̇ γ̇ = 0).",
            font_size=17,
            color=GREEN_C,
        ).move_to(nl_phase1, aligned_edge=LEFT)

        eq_geodesic = MathTex(
            r"\nabla_{\dot{\gamma}}\dot{\gamma} = 0 \iff "
            r"\ddot{u}^k + \Gamma^k_{ij}\dot{u}^i\dot{u}^j = 0 \quad "
            r"\bigl(\Gamma^k_{ij} = u^k g_{ij}\bigr)",
            font_size=22,
            color=GREEN_C,
        ).to_edge(DOWN, buff=0.30)

        self.add_fixed_in_frame_mobjects(nl_phase2, eq_geodesic)
        self.play(
            FadeOut(nl_phase1),
            FadeIn(nl_phase2),
            FadeIn(eq_geodesic),
            FadeOut(rays_list[0]),
            FadeOut(rays_list[1]),
            FadeOut(rays_list[2]),
            run_time=1.0,
        )

        # Full great circle (faint green ring around the globe) + geodesic voyage arc + detour arc
        p0 = SphereAtlas.start_point()
        p1 = SphereAtlas.end_point()
        normal_gc = np.cross(p0, p1)
        normal_gc /= np.linalg.norm(normal_gc)
        tangent_gc = np.cross(normal_gc, p0)

        full_great_circle = DashedVMobject(
            ParametricFunction(
                lambda s: R * (np.cos(s) * p0 + np.sin(s) * tangent_gc),
                t_range=[0.0, 2.0 * np.pi],
                color=GREEN_C,
                stroke_width=1.6,
                stroke_opacity=0.45,
            ),
            num_dashes=48,
        )

        detour_3d = DashedVMobject(
            ParametricFunction(
                lambda t: R * SphereAtlas.voyage_curve(t),
                t_range=[0.0, 1.0],
                color=GREY_B,
                stroke_width=2.5,
            ),
            num_dashes=18,
        )

        geodesic_3d = ParametricFunction(
            lambda t: R * SphereAtlas.geodesic_voyage(t),
            t_range=[0.0, 1.0],
            color=GREEN_C,
            stroke_width=4.5,
        )

        t_ranges_per_chart = [(0.0, 0.42), (0.26, 0.74), (0.58, 1.0)]
        plane_geodesic_curves = VGroup()
        for chart, col, (t0, t1) in zip(charts, colors, t_ranges_per_chart, strict=True):
            offset = self.TANGENT_OFFSET * chart.center
            plane_geodesic_curves.add(
                ParametricFunction(
                    lambda t, c=chart, off=offset: (
                        R * c.project_to_tangent_plane_3d(SphereAtlas.geodesic_voyage(t)) + off
                    ),
                    t_range=[t0, t1],
                    color=col,
                    stroke_width=3.5,
                )
            )

        t_tracker = ValueTracker(0.0)

        ship_dot = always_redraw(
            lambda: Dot3D(
                point=R * SphereAtlas.geodesic_voyage(t_tracker.get_value()),
                color=RED_C,
                radius=0.075,
            )
        )

        def _active_ship_projections() -> VGroup:
            group = VGroup()
            t_val = float(t_tracker.get_value())
            pt_unit = SphereAtlas.geodesic_voyage(t_val)
            pt_sphere = R * pt_unit
            for chart, col in zip(charts, colors, strict=True):
                if bool(chart.contains(pt_unit)):
                    offset = self.TANGENT_OFFSET * chart.center
                    pt_plane = R * chart.project_to_tangent_plane_3d(pt_unit) + offset
                    group.add(
                        DashedLine(
                            start=pt_sphere,
                            end=pt_plane,
                            color=col,
                            dash_length=0.06,
                            stroke_width=2.8,
                        ),
                        Dot3D(point=pt_plane, color=col, radius=0.06),
                    )
            return group

        active_projections = always_redraw(_active_ship_projections)

        self.play(
            Create(full_great_circle),
            Create(detour_3d),
            Create(geodesic_3d),
            Create(plane_geodesic_curves),
            FadeIn(ship_dot),
            FadeIn(active_projections),
            run_time=1.8,
        )

        # Sail the ship along the geodesic (great circle) across the overlapping charts
        self.move_camera(
            phi=60 * DEGREES,
            theta=18 * DEGREES,
            added_anims=[t_tracker.animate.set_value(1.0)],
            run_time=6.0,
        )
        self.wait(1.5)


def render_sphere_charts_animation(
    output_dir: str | Path | None = None,
    quality: str = "medium_quality",
) -> Path:
    """Render SphereChartsScene to an MP4 file and return its path."""
    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[3] / "media"
    )
    config_overrides = {
        "media_dir": str(base_dir),
        "quality": quality,
        "disable_caching": True,
        "output_file": "sphere_charts.mp4",
    }
    with tempconfig(config_overrides):
        scene = SphereChartsScene()
        scene.render()
        return Path(scene.renderer.file_writer.movie_file_path)


if __name__ == "__main__":
    video_path = render_sphere_charts_animation()
    print(f"Rendered animation to: {video_path}")
