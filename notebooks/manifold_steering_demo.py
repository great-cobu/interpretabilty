"""Interactive Marimo demonstration of Riemannian Manifold Steering on MNIST (Stage 1: 2D Chart & Decision Landscape)."""
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", css_file="theme.css")


@app.cell(hide_code=True)
def _():
    from dataclasses import replace
    from matplotlib.colors import BoundaryNorm
    import os
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from models.MNIST.config import DataConfig
    from models.MNIST.data import MNISTDataModule
    from models.MNIST.model import MNISTClassifier
    from models.MNIST.steering import (
        TangentChart2D,
        discover_cnn_runs,
        ensure_smooth_checkpoint,
        evaluate_decision_landscape,
        train_smooth_demo,
    )

    def notebook_md(text, *, supporting=False):
        """Render prose with the shared MNIST notebook theme."""
        _classes = "mnist-prose mnist-supporting" if supporting else "mnist-prose"
        return mo.Html(f'<div class="{_classes}">{mo.md(text).text}</div>')

    root = Path(__file__).resolve().parents[1]
    return (
        BoundaryNorm,
        DataConfig,
        MNISTClassifier,
        MNISTDataModule,
        TangentChart2D,
        discover_cnn_runs,
        ensure_smooth_checkpoint,
        evaluate_decision_landscape,
        mo,
        notebook_md,
        np,
        os,
        plt,
        replace,
        root,
        torch,
        train_smooth_demo,
    )


@app.cell(hide_code=True)
def _(notebook_md):
    notebook_md(
        r"""
        # Riemannian Manifold Steering on MNIST — Stage 1
        **Orthonormal 2D Chart & Classifier Decision Landscape.**
        An isometric 2D affine chart $\psi(u^1, u^2) = x_0 + u^1 e_1 + u^2 e_2$ ($\langle e_i, e_j \rangle = \delta_{ij}$) embeds a flat Euclidean slice into normalized pixel space $\mathbb{R}^{1 \times 28 \times 28}$, ensuring zero chart distortion ($\|\psi(u) - \psi(v)\|_2 = \|u - v\|_2$) so every curved boundary and entropy ridge comes purely from the smooth neural network.
        """,
        supporting=True,
    )
    return


@app.cell(hide_code=True)
def _(mo, notebook_md, os):
    training_form = mo.ui.dictionary(
        {
            "activation": mo.ui.dropdown(
                options={
                    "SMU (C^∞ Smooth Maximum Unit)": "SMU",
                    "Softplus (C^∞ smooth)": "softplus",
                    "ReLU (piecewise-linear baseline)": "relu",
                },
                value="SMU (C^∞ Smooth Maximum Unit)",
                label="CNN activation",
            ),
            "epochs": mo.ui.number(1, 25, step=1, value=5, label="Epochs"),
            "threads": mo.ui.number(
                1,
                os.cpu_count() or 1,
                step=1,
                value=min(8, os.cpu_count() or 1),
                label="CPU threads",
            ),
        }
    ).form(submit_button_label="Train additional checkpoint")

    mo.accordion(
        {
            "Optional: Train an additional smooth or ReLU checkpoint": mo.vstack(
                [
                    notebook_md(
                        "A cached smooth `SMU` checkpoint loads automatically below. "
                        "Use this form only if you want to train a fresh `SMU`, `softplus`, or `relu` comparison run.",
                        supporting=True,
                    ),
                    training_form,
                ]
            )
        }
    )
    return (training_form,)


@app.cell(hide_code=True)
def _(
    discover_cnn_runs,
    ensure_smooth_checkpoint,
    mo,
    notebook_md,
    root,
    train_smooth_demo,
    training_form,
):
    trained_output = None
    training_error = None
    if training_form.value is not None:
        try:
            with mo.status.spinner(title="Training MNISTCNN checkpoint…"):
                trained_output = train_smooth_demo(
                    root,
                    activation=str(training_form.value["activation"]),
                    epochs=int(training_form.value["epochs"]),
                    threads=int(training_form.value["threads"]),
                )
        except (OSError, RuntimeError, ValueError) as exc:
            training_error = str(exc)

    default_smooth = ensure_smooth_checkpoint(root)
    cnn_runs = discover_cnn_runs(root, smooth_only=False)
    run_options = {desc.label: str(desc.run_dir) for desc in cnn_runs}
    preferred_dir = str(trained_output.resolve()) if trained_output is not None else str(default_smooth.run_dir)
    default_label = next(
        (label for label, path in run_options.items() if path == preferred_dir),
        next(iter(run_options.keys()), None),
    )

    run_picker = mo.ui.dropdown(
        options=run_options,
        value=default_label,
        label="Active MNISTCNN checkpoint",
        full_width=True,
    )

    status_note = (
        notebook_md(f"**Training error:** {training_error}")
        if training_error
        else notebook_md(f"Trained new checkpoint: `{trained_output.name}`", supporting=True)
        if trained_output is not None
        else notebook_md("")
    )
    mo.vstack([run_picker, status_note])
    return cnn_runs, run_picker


@app.cell(hide_code=True)
def _(DataConfig, MNISTClassifier, MNISTDataModule, cnn_runs, notebook_md, replace, root, run_picker):
    selected_desc = next((d for d in cnn_runs if str(d.run_dir) == run_picker.value), None)
    classifier = None
    load_error = None
    if selected_desc is not None:
        try:
            classifier = MNISTClassifier.load_from_checkpoint(
                str(selected_desc.checkpoint_path),
                map_location="cpu",
                weights_only=True,
            ).eval()
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            load_error = str(exc)

    _base_data_cfg = classifier.config.data if classifier is not None else DataConfig(directory=str(root / "data"))
    data = MNISTDataModule(
        replace(_base_data_cfg, directory=str(root / "data"), download=False, num_workers=0),
        accelerator="cpu",
    )
    data_error = None
    try:
        data.setup("fit")
    except (OSError, RuntimeError, ValueError) as exc:
        data_error = str(exc)

    digit_indices = {d: [] for d in range(10)}
    if data_error is None:
        _subset = data.val_dataset
        for _pos, _src_idx in enumerate(_subset.indices):
            _digit = int(_subset.dataset.targets[_src_idx])
            digit_indices[_digit].append(_pos)

    if load_error or data_error:
        _msg = f"**Setup issue:** {load_error or data_error}"
    elif selected_desc is not None:
        _acc_str = f"{selected_desc.val_accuracy:.2%}" if selected_desc.val_accuracy is not None else "n/a"
        _msg = (
            f"Loaded **`{selected_desc.run_dir.name}`** · activation **`{selected_desc.activation}`** "
            f"({'$C^\\infty$ smooth' if selected_desc.is_smooth else 'piecewise-linear'}) · "
            f"validation accuracy **{_acc_str}** · 5,000 cached validation digits ready."
        )
    else:
        _msg = "Select a checkpoint above."
    notebook_md(_msg, supporting=True)
    return classifier, data, data_error, digit_indices, selected_desc


@app.cell(hide_code=True)
def _(mo, notebook_md):
    chart_mode = mo.ui.radio(
        options={
            "Exact 3-Digit Plane (Gram–Schmidt)": "gram_schmidt",
            "Geomstats TangentPCA + Fréchet Mean": "tangent_pca",
        },
        value="Exact 3-Digit Plane (Gram–Schmidt)",
        inline=True,
        label="2D Chart Construction",
    )
    digit_a = mo.ui.dropdown(options={str(d): d for d in range(10)}, value="4", label="Source Digit A (x_A)")
    digit_b = mo.ui.dropdown(options={str(d): d for d in range(10)}, value="9", label="Target Digit B (x_B)")
    digit_c = mo.ui.dropdown(options={str(d): d for d in range(10)}, value="7", label="Context Digit C (x_C)")
    sample_offset = mo.ui.slider(0, 25, step=1, value=0, label="Validation style index")
    grid_res = mo.ui.slider(25, 65, step=10, value=45, label="Grid resolution (G × G)")

    mo.vstack(
        [
            notebook_md("## 1. Configure the 2D Coordinate Chart", supporting=True),
            chart_mode,
            mo.hstack([digit_a, digit_b, digit_c], justify="start", gap=1.5),
            mo.hstack([sample_offset, grid_res], justify="start", gap=2.0),
        ]
    )
    return chart_mode, digit_a, digit_b, digit_c, grid_res, sample_offset


@app.cell(hide_code=True)
def _(
    TangentChart2D,
    chart_mode,
    classifier,
    data,
    data_error,
    digit_a,
    digit_b,
    digit_c,
    digit_indices,
    evaluate_decision_landscape,
    grid_res,
    mo,
    sample_offset,
    torch,
):
    chart = None
    landscape = None
    anchors_meta = None
    chart_error = None

    if classifier is not None and data_error is None:
        try:
            _da, _db, _dc = int(digit_a.value), int(digit_b.value), int(digit_c.value)
            _off = int(sample_offset.value)

            def _pick(digit_val: int, extra_shift: int = 0) -> tuple[torch.Tensor, int]:
                pool = digit_indices[digit_val]
                idx = pool[(_off + extra_shift) % len(pool)]
                img, lbl = data.val_dataset[idx]
                return img, int(lbl)

            _xa, _la = _pick(_da, 0)
            _xb, _lb = _pick(_db, 1 if _db == _da else 0)
            _xc, _lc = _pick(_dc, 2 if _dc in (_da, _db) else 0)

            if chart_mode.value == "tangent_pca":
                _pca_tensors = []
                for _d in dict.fromkeys([_da, _db, _dc]):
                    for _idx in digit_indices[_d][:24]:
                        _img, _ = data.val_dataset[_idx]
                        _pca_tensors.append(_img)
                _pca_stack = torch.stack(_pca_tensors, dim=0)
                chart = TangentChart2D.from_tangent_pca(_pca_stack, _xa, _xb, _xc)
            else:
                chart = TangentChart2D.from_three_images(_xa, _xb, _xc)

            landscape = evaluate_decision_landscape(
                classifier,
                chart,
                resolution=int(grid_res.value),
            )
            anchors_meta = {"x_a": _xa, "x_b": _xb, "x_c": _xc, "labels": (_la, _lb, _lc)}
        except (RuntimeError, ValueError) as exc:
            chart_error = str(exc)

    if chart is not None:
        (_u1_min, _u1_max), (_u2_min, _u2_max) = chart.grid_bounds()
        _mid = 0.5 * (chart.u_a + chart.u_b)
        probe_u1 = mo.ui.slider(
            round(_u1_min, 2),
            round(_u1_max, 2),
            step=0.1,
            value=round(float(_mid[0]), 2),
            label="Probe u¹ (defaults to Euclidean midpoint)",
        )
        probe_u2 = mo.ui.slider(
            round(_u2_min, 2),
            round(_u2_max, 2),
            step=0.1,
            value=round(float(_mid[1]), 2),
            label="Probe u² (defaults to Euclidean midpoint)",
        )
    else:
        probe_u1 = mo.ui.slider(-5.0, 5.0, step=0.1, value=0.0, label="Probe u¹")
        probe_u2 = mo.ui.slider(-5.0, 5.0, step=0.1, value=0.0, label="Probe u²")

    mo.hstack([probe_u1, probe_u2], justify="start", gap=2.0)
    return anchors_meta, chart, chart_error, landscape, probe_u1, probe_u2


@app.cell(hide_code=True)
def _(
    BoundaryNorm,
    anchors_meta,
    chart,
    chart_error,
    classifier,
    data,
    landscape,
    mo,
    notebook_md,
    np,
    plt,
    probe_u1,
    probe_u2,
    torch,
):
    if chart_error:
        stage1_view = notebook_md(f"**Chart construction error:** {chart_error}")
    elif chart is None or landscape is None or anchors_meta is None:
        stage1_view = notebook_md("Waiting for checkpoint and validation dataset…")
    else:
        _std, _mean = float(data.config.std), float(data.config.mean)

        def _denorm(img_t: torch.Tensor) -> np.ndarray:
            return (img_t.detach().cpu()[0] * _std + _mean).clamp(0.0, 1.0).numpy()

        _u_probe = torch.tensor([float(probe_u1.value), float(probe_u2.value)], dtype=torch.float32)
        _x_probe = chart.decode(_u_probe)

        _t_chord = torch.linspace(0.0, 1.0, 65).unsqueeze(-1)
        _u_chord = (1.0 - _t_chord) * chart.u_a.unsqueeze(0) + _t_chord * chart.u_b.unsqueeze(0)

        with torch.inference_mode():
            _anchor_imgs = torch.stack(
                [
                    chart.decode(chart.u_a),
                    chart.decode(chart.u_b),
                    chart.decode(chart.u_c),
                    _x_probe,
                ],
                dim=0,
            )
            _anchor_probs = classifier(_anchor_imgs).softmax(dim=-1).numpy()
            _chord_preds = classifier(chart.decode(_u_chord)).argmax(dim=-1).tolist()

        # --- Figure 1: Anchors, Basis Vectors (e1, e2), and Live Decoded Probe ---
        _fig1 = plt.figure(figsize=(11.2, 5.2), layout="constrained")
        _gs = _fig1.add_gridspec(2, 5, height_ratios=[1.0, 1.05])

        _la, _lb, _lc = anchors_meta["labels"]
        _thumbs = [
            (
                _denorm(chart.decode(chart.u_a)),
                f"Anchor A (true {_la})\nu_A=({chart.u_a[0]:.1f}, {chart.u_a[1]:.1f}) · pred {_anchor_probs[0].argmax()}",
                "gray",
                (0.0, 1.0),
            ),
            (
                _denorm(chart.decode(chart.u_b)),
                f"Anchor B (true {_lb})\nu_B=({chart.u_b[0]:.1f}, {chart.u_b[1]:.1f}) · pred {_anchor_probs[1].argmax()}",
                "gray",
                (0.0, 1.0),
            ),
            (
                _denorm(chart.decode(chart.u_c)),
                f"Anchor C (true {_lc})\nu_C=({chart.u_c[0]:.1f}, {chart.u_c[1]:.1f}) · pred {_anchor_probs[2].argmax()}",
                "gray",
                (0.0, 1.0),
            ),
            (
                chart.e1[0].numpy(),
                "Basis vector e₁\n(unit norm, signed)",
                "RdBu_r",
                (-float(chart.e1.abs().max()), float(chart.e1.abs().max())),
            ),
            (
                chart.e2[0].numpy(),
                "Basis vector e₂\n(⊥ e₁, unit norm)",
                "RdBu_r",
                (-float(chart.e2.abs().max()), float(chart.e2.abs().max())),
            ),
        ]
        for _col, (_arr, _title, _cmap, (_vmin, _vmax)) in enumerate(_thumbs):
            _ax = _fig1.add_subplot(_gs[0, _col])
            _ax.imshow(_arr, cmap=_cmap, vmin=_vmin, vmax=_vmax)
            _ax.set_title(_title, fontsize=8.5)
            _ax.axis("off")

        _ax_probe_img = _fig1.add_subplot(_gs[1, :2])
        _ax_probe_img.imshow(_denorm(_x_probe), cmap="gray", vmin=0.0, vmax=1.0)
        _probe_pred = int(_anchor_probs[3].argmax())
        _probe_conf = float(_anchor_probs[3].max())
        _ax_probe_img.set_title(
            f"Decoded Probe ψ({ _u_probe[0]:.2f}, { _u_probe[1]:.2f})\n"
            f"Predicted: {_probe_pred} ({_probe_conf:.1%})",
            fontsize=9.5,
        )
        _ax_probe_img.axis("off")

        _ax_probe_bar = _fig1.add_subplot(_gs[1, 2:])
        _colors = ["#1f77b4" if i != _probe_pred else "#d62728" for i in range(10)]
        _ax_probe_bar.bar(range(10), _anchor_probs[3], color=_colors)
        _ax_probe_bar.set_xticks(range(10))
        _ax_probe_bar.set_ylim(0.0, 1.0)
        _ax_probe_bar.set_xlabel("Digit class")
        _ax_probe_bar.set_ylabel("p(y | ψ(u))")
        _ax_probe_bar.set_title(
            "Classifier Posterior at Probe Point (notice blended uncertainty on Euclidean midpoints)",
            fontsize=9.5,
        )

        # --- Figure 2: 2D Decision Regions & Predictive Entropy Landscape ---
        _fig2, (_ax_dec, _ax_ent) = plt.subplots(1, 2, figsize=(11.4, 5.0), layout="constrained")
        _u1_np = landscape.u1.numpy()
        _u2_np = landscape.u2.numpy()
        _extent = [float(_u1_np[0]), float(_u1_np[-1]), float(_u2_np[0]), float(_u2_np[-1])]

        _cmap10 = plt.get_cmap("tab10", 10)
        _norm10 = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), _cmap10.N)
        _im_dec = _ax_dec.imshow(
            landscape.predictions.numpy(),
            origin="lower",
            extent=_extent,
            cmap=_cmap10,
            norm=_norm10,
            alpha=np.clip(landscape.confidence.numpy(), 0.35, 0.92),
            aspect="equal",
        )
        _ax_dec.contour(
            _u1_np,
            _u2_np,
            landscape.predictions.numpy(),
            levels=np.arange(-0.5, 10.5, 1.0),
            colors="white",
            linewidths=0.75,
            alpha=0.7,
        )
        _cbar_dec = _fig2.colorbar(_im_dec, ax=_ax_dec, ticks=range(10), fraction=0.046, pad=0.03)
        _cbar_dec.set_label("Predicted digit class")
        _ax_dec.set_title("Classifier Decision Basins (shaded by confidence)")
        _ax_dec.set_xlabel("Chart coordinate u¹")
        _ax_dec.set_ylabel("Chart coordinate u²")

        _im_ent = _ax_ent.imshow(
            landscape.entropy.numpy(),
            origin="lower",
            extent=_extent,
            cmap="magma",
            vmin=0.0,
            vmax=float(np.log(10.0)),
            aspect="equal",
        )
        _ax_ent.contour(
            _u1_np,
            _u2_np,
            landscape.entropy.numpy(),
            levels=6,
            colors="white",
            linewidths=0.5,
            alpha=0.45,
        )
        _cbar_ent = _fig2.colorbar(_im_ent, ax=_ax_ent, fraction=0.046, pad=0.03)
        _cbar_ent.set_label("Shannon entropy H(p) [nats]")
        _ax_ent.set_title("Predictive Entropy H(p(y | ψ(u)))")
        _ax_ent.set_xlabel("Chart coordinate u¹")
        _ax_ent.set_ylabel("Chart coordinate u²")

        for _ax in (_ax_dec, _ax_ent):
            _ax.plot(
                [float(chart.u_a[0]), float(chart.u_b[0])],
                [float(chart.u_a[1]), float(chart.u_b[1])],
                "--",
                color="black" if _ax is _ax_dec else "cyan",
                linewidth=1.8,
                label="Euclidean line A → B",
            )
            for _pt, _tag, _mcol in (
                (chart.u_a, f"A ({_la})", "#ffffff"),
                (chart.u_b, f"B ({_lb})", "#ffffff"),
                (chart.u_c, f"C ({_lc})", "#ffe066"),
            ):
                _ax.scatter(
                    float(_pt[0]),
                    float(_pt[1]),
                    s=85,
                    color=_mcol,
                    edgecolors="black",
                    linewidths=1.4,
                    zorder=5,
                )
                _ax.annotate(
                    _tag,
                    (float(_pt[0]), float(_pt[1])),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="black" if _ax is _ax_dec else "white",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        fc="white" if _ax is _ax_dec else "#1b1b2f",
                        ec="none",
                        alpha=0.78,
                    ),
                )
            _ax.scatter(
                float(_u_probe[0]),
                float(_u_probe[1]),
                marker="*",
                s=190,
                color="#ff2e63",
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
                label="Probe ψ(u)",
            )
            _ax.legend(loc="lower right", fontsize=8)

        _visited_classes = []
        for _c in _chord_preds:
            if not _visited_classes or _visited_classes[-1] != _c:
                _visited_classes.append(_c)
        _chord_seq = " → ".join(str(c) for c in _visited_classes)
        _orthonorm_err = float(
            max(
                abs(float(chart.e1.norm()) - 1.0),
                abs(float(chart.e2.norm()) - 1.0),
                abs(float((chart.e1 * chart.e2).sum())),
            )
        )
        _pca_note = (
            f" · Geomstats `TangentPCA` variance ratio: "
            f"**{chart.explained_variance_ratio[0]:.1%}, {chart.explained_variance_ratio[1]:.1%}**"
            if chart.explained_variance_ratio is not None
            else " · Exact Gram–Schmidt plane through $(x_A, x_B, x_C)$"
        )

        _summary_md = notebook_md(
            f"### Stage 1 Geometric Diagnostics\n"
            f"- **Chart Isometry Check:** $\\max(|\\|e_1\\|-1|, |\\|e_2\\|-1|, |\\langle e_1, e_2\\rangle|) = {_orthonorm_err:.2e}${_pca_note}\n"
            f"- **Euclidean Chord Distance $\\|u_B - u_A\\|_2$:** `{float((chart.u_b - chart.u_a).norm()):.2f}` "
            f"· **Classes crossed along straight line $A \\to B$:** `{_chord_seq}`\n"
            f"- **Next (Stage 2):** Wrap this smooth network and chart in a **StochMan `Manifold`** subclass with vectorized `torch.func` Jacobians to overlay the pullback volume density $\\log\\sqrt{{\\det g(u)}}$ and **Tissot metric ellipses** on these entropy ridges.",
            supporting=True,
        )

        stage1_view = mo.vstack(
            [
                notebook_md("## 2. Anchor Digits, Basis Templates & Decoded Probe", supporting=True),
                mo.as_html(_fig1),
                notebook_md("## 3. 2D Decision Basins & Predictive Entropy Landscape", supporting=True),
                mo.as_html(_fig2),
                _summary_md,
            ]
        )
        plt.close(_fig1)
        plt.close(_fig2)

    stage1_view
    return


if __name__ == "__main__":
    app.run()
