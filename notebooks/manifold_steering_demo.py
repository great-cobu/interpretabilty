"""Interactive Marimo demonstration of Riemannian Activation Manifold Steering on MNIST."""
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", css_file="theme.css")


@app.cell
def _():
    from dataclasses import replace
    import json
    import os
    from pathlib import Path
    import marimo as mo
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from models.MNIST.config import DataConfig
    from models.MNIST.data import MNISTDataModule
    from models.MNIST.model import MNISTCNN, MNISTClassifier
    from models.MNIST.steering import (
        TangentChart2D,
        christoffel_symbols,
        pullback_activation_metric,
        pullback_fisher_metric,
        shoot_geodesic,
        solve_bvp_geodesic,
        train_smooth_demo,
    )

    def notebook_md(text, *, supporting=False):
        """Render prose with the shared theme; use Markdown heading levels."""
        _classes = "mnist-prose mnist-supporting" if supporting else "mnist-prose"
        return mo.Html(f'<div class="{_classes}">{mo.md(text).text}</div>')

    root = Path(__file__).resolve().parents[1]
    return (
        DataConfig,
        Ellipse,
        MNISTCNN,
        MNISTClassifier,
        MNISTDataModule,
        TangentChart2D,
        christoffel_symbols,
        json,
        mo,
        notebook_md,
        np,
        os,
        plt,
        pullback_activation_metric,
        pullback_fisher_metric,
        replace,
        root,
        shoot_geodesic,
        solve_bvp_geodesic,
        torch,
        train_smooth_demo,
    )


@app.cell(hide_code=True)
def _(notebook_md):
    notebook_md(
        r"""
        # Riemannian Activation Manifold Steering — Interactive Demo

        When steering a neural network representation from a starting concept $x_A$ toward a target concept $x_B$, **Euclidean linear steering** ($u_{\text{lin}}(t) = (1 - t)u_A + t u_B$) moves in a straight line in flat coordinates, cutting blindly through regions of high curvature and rapid representation distortion.

        Equipping the local chart $u = (u^1, u^2) \in \mathbb{R}^2$ with the **pullback Riemannian metric** $g_{ij}(u) = (J_h(u)^\top J_h(u))_{ij}$ (or the **Fisher Information Metric** $g^F_{ij}(u)$) and **Christoffel symbols** $\Gamma^k_{ij}(u)$ via `torch.func.jacrev` allows us to compute **Riemannian geodesic steering trajectories** that minimize path energy:
        $$E[\gamma] = \frac{1}{2}\int_0^1 g_{ij}(\gamma(t))\,\dot{\gamma}^i(t)\,\dot{\gamma}^j(t)\,dt \qquad \Longleftrightarrow \qquad \ddot{\gamma}^k(t) + \Gamma^k_{ij}(\gamma(t))\,\dot{\gamma}^i(t)\,\dot{\gamma}^j(t) = 0$$
        """,
        supporting=True,
    )
    return


@app.cell
def _(mo, notebook_md, os):
    training_form = mo.ui.dictionary({
        "activation": mo.ui.dropdown(
            options={
                "SMU (Smooth Maximum Unit — C^∞ smooth)": "SMU",
                "Softplus (C^∞ smooth)": "softplus",
                "ReLU (piecewise linear — zero 2nd derivative)": "relu",
            },
            value="SMU (Smooth Maximum Unit — C^∞ smooth)",
            label="CNN Activation Function",
        ),
        "epochs": mo.ui.number(1, 20, step=1, value=2, label="Training epochs"),
        "threads": mo.ui.number(
            1, os.cpu_count() or 1, step=1,
            value=min(8, os.cpu_count() or 1), label="CPU threads",
        ),
    }).form(submit_button_label="Train smooth CNN checkpoint")
    mo.vstack([
        notebook_md(
            "## 1. Select or Train a Smooth CNN (`SMU` / `Softplus`)\n"
            "Second-order automatic differentiation (`christoffel_symbols`) requires a $C^2$-smooth activation function "
            "such as **`SMU`** or **`Softplus`**. Train a new checkpoint below or pick an existing run.",
            supporting=True,
        ),
        training_form,
    ])
    return (training_form,)


@app.cell
def _(mo, notebook_md, root, train_smooth_demo, training_form):
    trained_run = None
    training_error = None
    if training_form.value is not None:
        try:
            with mo.status.spinner(title="Training smooth MNISTCNN checkpoint…"):
                trained_run = train_smooth_demo(
                    root,
                    activation=str(training_form.value["activation"]),
                    epochs=int(training_form.value["epochs"]),
                    threads=int(training_form.value["threads"]),
                )
        except (OSError, RuntimeError, ValueError) as exc:
            training_error = str(exc)
    notebook_md(
        f"Training failed: {training_error}" if training_error else
        f"Saved smooth steering run: `{trained_run}`" if trained_run else ""
    )
    return (trained_run,)


@app.cell
def _(json, mo, root, trained_run):
    cnn_runs = []
    for _summary_path in (root / "runs").glob("*/training_summary.json"):
        try:
            _data = json.loads(_summary_path.read_text())
            if _data.get("model") != "LogisticRegression" and (_summary_path.parent / "checkpoints" / "best.ckpt").is_file():
                cnn_runs.append(_summary_path.parent)
        except (OSError, ValueError):
            pass
    cnn_runs.sort(
        key=lambda p: ("steering" in p.name, p.stat().st_mtime),
        reverse=True,
    )
    run_picker = mo.ui.dropdown(
        options={p.name: str(p) for p in cnn_runs},
        value=trained_run.name if trained_run else cnn_runs[0].name if cnn_runs else None,
        label="Saved MNISTCNN run",
    )
    run_picker
    return (run_picker,)


@app.cell
def _(MNISTClassifier, notebook_md, run_picker):
    classifier = None
    checkpoint_error = None
    if run_picker.value:
        from pathlib import Path as _Path
        try:
            classifier = MNISTClassifier.load_from_checkpoint(
                _Path(run_picker.value) / "checkpoints" / "best.ckpt",
                map_location="cpu",
                weights_only=True,
            ).eval()
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            checkpoint_error = str(exc)
    _act = classifier.network.activation if classifier is not None else "unknown"
    notebook_md(
        f"Checkpoint error: {checkpoint_error}" if checkpoint_error else
        f"Loaded **`MNISTClassifier`** with activation **`{_act}`** on CPU." if classifier else
        "Train or select a checkpoint above."
    )
    return (classifier,)


@app.cell
def _(DataConfig, MNISTDataModule, classifier, notebook_md, replace, root):
    _settings = classifier.config.data if classifier is not None else DataConfig(directory=str(root / "data"))
    data = MNISTDataModule(replace(_settings, download=False), accelerator="cpu")
    data_error = None
    try:
        data.setup("fit")
    except (RuntimeError, ValueError, OSError) as exc:
        data_error = str(exc)
    digit_indices = {digit: [] for digit in range(10)}
    if data_error is None:
        _subset = data.val_dataset
        for _pos, _src_idx in enumerate(_subset.indices):
            _d = int(_subset.dataset.targets[_src_idx])
            digit_indices[_d].append(_pos)
    notebook_md(
        f"**Cache unavailable:** {data_error}" if data_error else
        "Loaded 5,000 validation examples for tangent-chart construction."
    )
    return data, data_error, digit_indices


@app.cell
def _(mo, notebook_md):
    digit_a_picker = mo.ui.dropdown(
        options={f"Digit {d}": d for d in range(10)},
        value="Digit 3",
        label="Start digit (x_A)",
    )
    digit_b_picker = mo.ui.dropdown(
        options={f"Digit {d}": d for d in range(10)},
        value="Digit 8",
        label="Target digit (x_B)",
    )
    digit_c_picker = mo.ui.dropdown(
        options={f"Digit {d}": d for d in range(10)},
        value="Digit 5",
        label="Plane reference digit (x_C)",
    )
    metric_picker = mo.ui.dropdown(
        options={
            "Pullback Activation Metric g_ij = (J_h^T J_h)_ij": "activation",
            "Pullback Fisher Information Metric g^F_ij": "fisher",
        },
        value="Pullback Activation Metric g_ij = (J_h^T J_h)_ij",
        label="Riemannian Metric on Chart",
    )
    sample_offset = mo.ui.slider(0, 25, value=0, step=1, label="Validation example index")
    t_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="Steering parameter t ∈ [0, 1]")

    mo.vstack([
        notebook_md(
            "## 2. Configure the 2D Local Chart & Riemannian Metric\n"
            "Three validation digits $(x_A, x_B, x_C)$ span an orthonormal 2D affine chart "
            "$\\varphi^{-1}(u^1, u^2) = x_A + u^1 e_1 + u^2 e_2$ in image space. "
            "Use the sliders to compare **Euclidean Linear Steering** against **Riemannian Geodesic Steering**.",
            supporting=True,
        ),
        mo.hstack([digit_a_picker, digit_b_picker, digit_c_picker, metric_picker], justify="start"),
        mo.hstack([sample_offset, t_slider], justify="start"),
    ])
    return (
        digit_a_picker,
        digit_b_picker,
        digit_c_picker,
        metric_picker,
        sample_offset,
        t_slider,
    )


@app.cell
def _(
    TangentChart2D,
    christoffel_symbols,
    classifier,
    data,
    data_error,
    digit_a_picker,
    digit_b_picker,
    digit_c_picker,
    digit_indices,
    metric_picker,
    np,
    pullback_activation_metric,
    pullback_fisher_metric,
    sample_offset,
    shoot_geodesic,
    solve_bvp_geodesic,
    torch,
):
    steering_state = None
    steering_error = None
    if classifier is not None and data_error is None:
        try:
            _da = int(digit_a_picker.value)
            _db = int(digit_b_picker.value)
            _dc = int(digit_c_picker.value)
            _off = int(sample_offset.value)

            _idx_a = digit_indices[_da][_off % len(digit_indices[_da])]
            _idx_b = digit_indices[_db][(_off + 1) % len(digit_indices[_db])]
            _idx_c = digit_indices[_dc][(_off + 2) % len(digit_indices[_dc])]

            _xa, _ = data.val_dataset[_idx_a]
            _xb, _ = data.val_dataset[_idx_b]
            _xc, _ = data.val_dataset[_idx_c]

            _chart = TangentChart2D.from_three_images(_xa, _xb, _xc)

            if metric_picker.value == "fisher":
                _metric_fn = lambda u: pullback_fisher_metric(classifier, _chart, u, damping=2e-2)
            else:
                _metric_fn = lambda u: pullback_activation_metric(classifier, _chart, u, layer="relu3", damping=2e-2)

            _bvp = solve_bvp_geodesic(
                _metric_fn,
                _chart.u_a,
                _chart.u_b,
                num_nodes=17,
                opt_steps=35,
                lr=0.10,
            )

            # Initial-value geodesic shot from u_A with initial velocity pointing slightly into the chart plane
            _v0 = 0.75 * (_chart.u_b - _chart.u_a) + 0.35 * (_chart.u_c - _chart.u_a)
            _ivp = shoot_geodesic(_metric_fn, _chart.u_a, _v0, num_steps=20, t_max=1.0)

            # Evaluate background grid of metric volume element sqrt(det g(u)) and predicted classes
            _span_x = float(_chart.u_b[0])
            _span_y = float(_chart.u_c[1])
            _u1_grid = np.linspace(-0.20 * _span_x, 1.20 * _span_x, 16)
            _u2_grid = np.linspace(-0.35 * _span_y, 1.15 * _span_y, 14)
            _U1, _U2 = np.meshgrid(_u1_grid, _u2_grid)
            _grid_pts = torch.tensor(np.stack([_U1.ravel(), _U2.ravel()], axis=-1), dtype=torch.float32)

            with torch.no_grad():
                _grid_imgs = _chart.decode(_grid_pts)
                _grid_logits = classifier(_grid_imgs)
                _grid_preds = _grid_logits.argmax(dim=1).numpy().reshape(_U1.shape)

            _vol_elems = np.zeros(_grid_pts.shape[0], dtype=np.float64)
            for _k in range(_grid_pts.shape[0]):
                _g_k = _metric_fn(_grid_pts[_k]).detach()
                _vol_elems[_k] = float(torch.sqrt(torch.clamp(torch.linalg.det(_g_k), min=1e-12)))
            _VOL = np.log10(_vol_elems.reshape(_U1.shape) + 1e-12)

            # Compute class probabilities along linear and geodesic BVP curves
            with torch.no_grad():
                _lin_imgs = _chart.decode(_bvp["linear_u"])
                _geo_imgs = _chart.decode(_bvp["geodesic_u"])
                _lin_probs = classifier(_lin_imgs).softmax(dim=1).numpy()
                _geo_probs = classifier(_geo_imgs).softmax(dim=1).numpy()

            # Compute Christoffel norm at midpoint of linear curve
            _u_mid = 0.5 * (_chart.u_a + _chart.u_b)
            _gamma_mid = christoffel_symbols(_metric_fn, _u_mid).detach()

            steering_state = {
                "chart": _chart,
                "metric_fn": _metric_fn,
                "bvp": _bvp,
                "ivp": _ivp,
                "U1": _U1,
                "U2": _U2,
                "VOL": _VOL,
                "grid_preds": _grid_preds,
                "lin_probs": _lin_probs,
                "geo_probs": _geo_probs,
                "gamma_mid_norm": float(_gamma_mid.norm()),
                "digits": (_da, _db, _dc),
            }
        except (RuntimeError, ValueError) as exc:
            steering_error = str(exc)
    return steering_error, steering_state


@app.cell
def _(
    Ellipse,
    christoffel_symbols,
    data,
    mo,
    notebook_md,
    np,
    plt,
    steering_error,
    steering_state,
    t_slider,
    torch,
):
    dashboard = notebook_md("Select a valid checkpoint and digits to view the manifold steering dashboard.")
    if steering_error:
        dashboard = notebook_md(f"**Steering computation error:** `{steering_error}`")
    elif steering_state is not None:
        _chart = steering_state["chart"]
        _metric_fn = steering_state["metric_fn"]
        _bvp = steering_state["bvp"]
        _ivp = steering_state["ivp"]
        _U1, _U2, _VOL = steering_state["U1"], steering_state["U2"], steering_state["VOL"]
        _lin_u = _bvp["linear_u"].numpy()
        _geo_u = _bvp["geodesic_u"].numpy()
        _ivp_u = _ivp["u"].numpy()
        _t_vals = _bvp["t"].numpy()
        _da, _db, _dc = steering_state["digits"]

        _t_sel = float(t_slider.value)
        _node_idx = int(round(_t_sel * (len(_t_vals) - 1)))
        _u_lin_t = _bvp["linear_u"][_node_idx]
        _u_geo_t = _bvp["geodesic_u"][_node_idx]

        # Compute instantaneous Christoffel acceleration vector at _u_geo_t
        _vel_t = _chart.u_b - _chart.u_a
        _gamma_t = christoffel_symbols(_metric_fn, _u_geo_t).detach()
        _christoffel_force = -torch.einsum("kij,i,j->k", _gamma_t, _vel_t, _vel_t).numpy()

        _fig = plt.figure(figsize=(13.5, 8.2), dpi=115)

        # Panel 1: 2D Chart with Metric Volume Element, Tissot Ellipses, Linear vs Geodesic Paths
        _ax1 = _fig.add_subplot(2, 2, 1)
        _cf = _ax1.contourf(_U1, _U2, _VOL, levels=14, cmap="viridis", alpha=0.82)
        _fig.colorbar(_cf, ax=_ax1, label=r"$\log_{10}\sqrt{\det g(u)}$ (local distortion)")
        _ax1.contour(_U1, _U2, steering_state["grid_preds"], levels=np.arange(-0.5, 10.5, 1.0), colors="white", linewidths=0.6, alpha=0.5)

        # Plot Tissot indicatrix ellipses along geodesic path
        for _step_i in range(2, len(_t_vals) - 2, 4):
            _u_pt = _bvp["geodesic_u"][_step_i]
            _g_pt = _metric_fn(_u_pt).detach().numpy()
            _eigvals, _eigvecs = np.linalg.eigh(_g_pt)
            _eigvals = np.maximum(_eigvals, 1e-8)
            _angle = np.degrees(np.arctan2(_eigvecs[1, 0], _eigvecs[0, 0]))
            _scale = 0.08 * float(_chart.u_b[0])
            _ell = Ellipse(
                xy=(float(_u_pt[0]), float(_u_pt[1])),
                width=_scale / np.sqrt(_eigvals[0] / _eigvals.max()),
                height=_scale / np.sqrt(_eigvals[1] / _eigvals.max()),
                angle=_angle,
                edgecolor="#facc15",
                facecolor="none",
                lw=1.1,
                alpha=0.85,
            )
            _ax1.add_patch(_ell)

        _ax1.plot(_lin_u[:, 0], _lin_u[:, 1], "--", color="#f87171", lw=2.2, label="Euclidean Linear Steering")
        _ax1.plot(_geo_u[:, 0], _geo_u[:, 1], "-o", color="#38bdf8", lw=2.5, markersize=3.5, label="Riemannian Geodesic (BVP)")
        _ax1.plot(_ivp_u[:, 0], _ivp_u[:, 1], "-.", color="#fbbf24", lw=1.8, label="Geodesic Shooting (IVP)")

        _ax1.scatter([float(_chart.u_a[0])], [float(_chart.u_a[1])], color="#22c55e", s=80, zorder=10, label=f"Start x_A (digit {_da})")
        _ax1.scatter([float(_chart.u_b[0])], [float(_chart.u_b[1])], color="#ef4444", s=80, zorder=10, label=f"Target x_B (digit {_db})")
        _ax1.scatter([float(_chart.u_c[0])], [float(_chart.u_c[1])], color="#e879f9", s=70, zorder=10, label=f"Ref x_C (digit {_dc})")
        _ax1.scatter([float(_u_lin_t[0])], [float(_u_lin_t[1])], color="#f87171", edgecolors="black", s=95, zorder=12)
        _ax1.scatter([float(_u_geo_t[0])], [float(_u_geo_t[1])], color="#38bdf8", edgecolors="black", s=95, zorder=12)

        _ax1.set_title("Activation Manifold Chart $(u^1, u^2)$ & Geodesics")
        _ax1.set_xlabel(r"$u^1$ (direction $x_A \to x_B$)")
        _ax1.set_ylabel(r"$u^2$ (orthogonal direction toward $x_C$)")
        _ax1.legend(loc="upper right", fontsize=7.5)

        # Panel 2: Local Riemannian Speed ||du/dt||_g along the trajectory
        _ax2 = _fig.add_subplot(2, 2, 2)
        _seg_t = 0.5 * (_t_vals[:-1] + _t_vals[1:])
        _ax2.plot(_seg_t, _bvp["linear_speeds"].numpy(), "--o", color="#ef4444", lw=2.0, markersize=4, label="Euclidean Linear Path")
        _ax2.plot(_seg_t, _bvp["geodesic_speeds"].numpy(), "-o", color="#0284c7", lw=2.2, markersize=4, label="Riemannian Geodesic Path")
        _ax2.axvline(_t_sel, color="gray", linestyle=":", lw=1.4, label=f"Selected t = {_t_sel:.2f}")
        _ax2.set_title(r"Riemannian Speed $\|\dot{\gamma}(t)\|_g = \sqrt{g_{ij}\dot{u}^i\dot{u}^j}$")
        _ax2.set_xlabel("Trajectory parameter $t$")
        _ax2.set_ylabel(r"Intrinsic speed $\|\dot{\gamma}(t)\|_g$")
        _ax2.grid(alpha=0.25)
        _ax2.legend(fontsize=8)

        # Panel 3: Class Probability Evolution along Linear vs Geodesic Paths
        _ax3 = _fig.add_subplot(2, 2, 3)
        _lin_p = steering_state["lin_probs"]
        _geo_p = steering_state["geo_probs"]
        _ax3.plot(_t_vals, _lin_p[:, _da], "--", color="#16a34a", lw=1.7, label=f"Linear P(y={_da})")
        _ax3.plot(_t_vals, _geo_p[:, _da], "-", color="#16a34a", lw=2.4, label=f"Geodesic P(y={_da})")
        _ax3.plot(_t_vals, _lin_p[:, _db], "--", color="#dc2626", lw=1.7, label=f"Linear P(y={_db})")
        _ax3.plot(_t_vals, _geo_p[:, _db], "-", color="#dc2626", lw=2.4, label=f"Geodesic P(y={_db})")
        if _dc not in (_da, _db):
            _ax3.plot(_t_vals, _geo_p[:, _dc], ":", color="#9333ea", lw=1.8, label=f"Geodesic P(y={_dc})")
        _ax3.axvline(_t_sel, color="gray", linestyle=":", lw=1.4)
        _ax3.set_ylim(-0.02, 1.05)
        _ax3.set_title("Output Class Probabilities Along Steering Paths")
        _ax3.set_xlabel("Trajectory parameter $t$")
        _ax3.set_ylabel(r"Class probability $P(y \mid \gamma(t))$")
        _ax3.grid(alpha=0.25)
        _ax3.legend(fontsize=7.5, ncol=2)

        # Panel 4: Decoded Steered Images at t (Linear vs Geodesic + Difference)
        _ax4 = _fig.add_subplot(2, 2, 4)
        _img_lin = (_chart.decode(_u_lin_t)[0] * data.config.std + data.config.mean).numpy()
        _img_geo = (_chart.decode(_u_geo_t)[0] * data.config.std + data.config.mean).numpy()
        _diff_img = _img_geo - _img_lin
        _strip = np.concatenate([
            np.clip(_img_lin, 0.0, 1.0),
            np.ones((28, 2)),
            np.clip(_img_geo, 0.0, 1.0),
            np.ones((28, 2)),
            0.5 + 2.0 * _diff_img,
        ], axis=1)
        _ax4.imshow(np.clip(_strip, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
        _ax4.set_xticks([14, 44, 74])
        _ax4.set_xticklabels([
            f"Linear u_lin({_t_sel:.2f})",
            f"Geodesic u_geo({_t_sel:.2f})",
            "Difference (Geo - Lin)",
        ])
        _ax4.set_yticks([])
        _ax4.set_title(f"Steered Images at t = {_t_sel:.2f} (node {_node_idx + 1}/{len(_t_vals)})")

        _fig.tight_layout()

        _e_lin = _bvp["linear_energy"]
        _e_geo = _bvp["geodesic_energy"]
        _l_lin = _bvp["linear_length"]
        _l_geo = _bvp["geodesic_length"]
        _reduction = (1.0 - _e_geo / max(_e_lin, 1e-12)) * 100.0

        _summary_md = notebook_md(
            f"### Quantitative Comparison ($x_A = {_da} \\to x_B = {_db}$)\n"
            f"- **Euclidean Linear Steering:** Energy $E[\\gamma_{{\\text{{lin}}}}] = {_e_lin:.4f}$, "
            f"Riemannian Length $L[\\gamma_{{\\text{{lin}}}}] = {_l_lin:.4f}$\n"
            f"- **Riemannian Geodesic Steering:** Energy $E[\\gamma_{{\\text{{geo}}}}] = {_e_geo:.4f}$ "
            f"(**{_reduction:.1f}% lower path energy**), "
            f"Riemannian Length $L[\\gamma_{{\\text{{geo}}}}] = {_l_geo:.4f}$\n"
            f"- **Christoffel Curvature Norm $\\|\\Gamma^k_{{ij}}(u_{{\\text{{mid}}}})\\|_F$:** `{steering_state['gamma_mid_norm']:.4e}` "
            f"· Instantaneous geodesic drift $-\\Gamma^k_{{ij}}\\dot{{u}}^i\\dot{{u}}^j$ at $t={_t_sel:.2f}$: "
            f"`({_christoffel_force[0]:+.3f}, {_christoffel_force[1]:+.3f})`"
        )
        dashboard = mo.vstack([mo.as_html(_fig), _summary_md])
        plt.close(_fig)
    dashboard
    return


if __name__ == "__main__":
    app.run()
