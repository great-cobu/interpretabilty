"""Interactive Marimo demonstration of Multinomial Logistic Regression and glass-box interpretability on MNIST."""
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
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from models.MNIST.config import DataConfig
    from models.MNIST.data import MNISTDataModule
    from models.logistic.evaluate import confusion_matrix
    from models.logistic.model import LogisticClassifier, LogisticRegression
    from models.logistic.train import train_demo

    def notebook_md(text, *, supporting=False):
        """Render prose with the shared theme; use Markdown heading levels."""
        _classes = "mnist-prose mnist-supporting" if supporting else "mnist-prose"
        return mo.Html(f'<div class="{_classes}">{mo.md(text).text}</div>')

    root = Path(__file__).resolve().parents[1]
    return DataConfig, LogisticClassifier, LogisticRegression, MNISTDataModule, confusion_matrix, json, mo, notebook_md, np, os, plt, replace, root, torch, train_demo


@app.cell(hide_code=True)
def _(notebook_md):
    notebook_md(r"""
    # MNIST Logistic Regression — Glass-Box Interpretability Demo
    Explore cached MNIST validation digits, train a fast multinomial logistic regression model ($7,850$ parameters),
    and inspect its **exact linear pixel attributions** ($x \odot W_c$) and **learned spatial class templates** ($W_c \in \mathbb{R}^{28 \times 28}$).
    """, supporting=True)
    return


@app.cell
def _(mo, notebook_md, os):
    training_form = mo.ui.dictionary({
        "epochs": mo.ui.number(1, 100, step=1, value=3, label="Training epochs"),
        "threads": mo.ui.number(1, os.cpu_count() or 1, step=1,
                                value=min(8, os.cpu_count() or 1), label="Training CPU threads"),
    }).form(submit_button_label="Train new logistic demo run")
    mo.vstack([
        notebook_md("## Train a Logistic Regression demo\nChoose the number of epochs, then press **Train new logistic demo run**. "
                    "Each epoch trains on 20 batches and validates on 5 batches on CPU in milliseconds.", supporting=True),
        training_form,
    ])
    return (training_form,)


@app.cell
def _(mo, notebook_md, root, train_demo, training_form):
    trained_run = None
    training_error = None
    if training_form.value is not None:
        try:
            with mo.status.spinner(title="Training logistic regression demo…"):
                trained_run = train_demo(root, int(training_form.value["epochs"]), int(training_form.value["threads"]))
        except (OSError, RuntimeError, ValueError) as exc:
            training_error = str(exc)
    notebook_md(f"Training failed: {training_error}" if training_error else
                f"Saved run: `{trained_run}`" if trained_run else "")
    return (trained_run,)


@app.cell
def _(json, mo, root, trained_run):
    logistic_runs = []
    for _summary_path in (root / "runs").glob("*/training_summary.json"):
        try:
            _data = json.loads(_summary_path.read_text())
            if _data.get("model") == "LogisticRegression":
                logistic_runs.append(_summary_path.parent)
        except (OSError, ValueError):
            pass
    logistic_runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run_picker = mo.ui.dropdown(
        options={p.name: str(p) for p in logistic_runs},
        value=trained_run.name if trained_run else logistic_runs[0].name if logistic_runs else None,
        label="Saved Logistic Regression run",
    )
    run_picker
    return (run_picker,)


@app.cell
def _(mo):
    digit_picker = mo.ui.radio(
        options={str(_digit): _digit for _digit in range(10)},
        value="0", inline=True, label="True validation digit",
    )
    target_class_picker = mo.ui.dropdown(
        options={"Predicted digit": -1, **{f"Digit {d}": d for d in range(10)}},
        value="Predicted digit",
        label="Attribution target class (c)",
    )
    return digit_picker, target_class_picker


@app.cell
def _(mo, os):
    cpu_count = os.cpu_count() or 1
    thread_picker = mo.ui.slider(1, cpu_count, value=min(8, cpu_count), label="CPU inference threads")
    thread_picker
    return (thread_picker,)


@app.cell
def _(notebook_md, thread_picker, torch):
    torch.set_num_threads(thread_picker.value)
    inference_threads = torch.get_num_threads()
    notebook_md(f"Using **{inference_threads} CPU threads** for inference.")
    return (inference_threads,)


@app.cell
def _(json, run_picker):
    selected_run = None
    summary = None
    config = None
    run_error = None
    if run_picker.value:
        from pathlib import Path as _Path
        selected_run = _Path(run_picker.value)
        try:
            summary = json.loads((selected_run / "training_summary.json").read_text())
            config = (selected_run / "config.yaml").read_text()
        except (OSError, ValueError) as exc:
            run_error = str(exc)
    return config, run_error, selected_run, summary


@app.cell
def _(LogisticClassifier, notebook_md, selected_run):
    classifier = None
    checkpoint_error = None
    if selected_run:
        try:
            classifier = LogisticClassifier.load_from_checkpoint(
                selected_run / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=True,
            ).eval()
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            checkpoint_error = str(exc)
    notebook_md(f"Checkpoint unavailable: {checkpoint_error}" if checkpoint_error else
                "Predictions and glass-box attributions use this run's best checkpoint on CPU." if classifier else
                "Select or train a Logistic Regression run to enable predictions.")
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
    notebook_md(f"**Cache unavailable:** {data_error}." if data_error else
                "Cached MNIST: 55,000 training and 5,000 validation examples. The preview uses validation data.")
    return data, data_error


@app.cell
def _(data, data_error):
    digit_indices = {digit: [] for digit in range(10)}
    if data_error is None:
        _subset = data.val_dataset
        for _position, _source_index in enumerate(_subset.indices):
            _digit = int(_subset.dataset.targets[_source_index])
            digit_indices[_digit].append(_position)
    return (digit_indices,)


@app.cell
def _(digit_indices, digit_picker, mo, notebook_md, target_class_picker):
    matching_indices = digit_indices[digit_picker.value]
    next_example = mo.ui.button(
        value=0, on_click=lambda value: value + 1,
        label="Next example", disabled=not matching_indices,
    )
    mo.vstack([
        notebook_md(r"## Validation digit, prediction & exact pixel attribution" + "\n"
                    r"Browse examples of the selected digit below. For any target class $c$, the exact pixel contribution is "
                    r"$A_{h,w} = x_{h,w} \cdot W_{c,h,w}$, where $\sum_{h,w} A_{h,w} + b_c = \ell_c$ (the exact class logit).", supporting=True),
        mo.hstack([digit_picker, target_class_picker, next_example], justify="start"),
    ])
    return matching_indices, next_example


@app.cell
def _(matching_indices, next_example, notebook_md):
    example_number = next_example.value % len(matching_indices) if matching_indices else 0
    sample_index = matching_indices[example_number] if matching_indices else None
    notebook_md(f"Example **{example_number + 1} of {len(matching_indices)}** for this digit." if matching_indices else
                "No validation examples are available for this digit.")
    return (sample_index,)


@app.cell
def _(classifier, data, data_error, mo, notebook_md, np, plt, sample_index, target_class_picker, torch):
    preview = notebook_md("Prepare the local cache to see digit examples.")
    completeness_note = notebook_md("")
    if data_error is None and sample_index is not None:
        _image, _label = data.val_dataset[sample_index]
        _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4))
        _raw_img = (_image[0] * data.config.std + data.config.mean).numpy()
        _axes[0].imshow(_raw_img, cmap="gray", vmin=0, vmax=1)
        _axes[0].set_title(f"Input digit (true: {_label})")
        _axes[0].axis("off")

        if classifier is not None:
            with torch.inference_mode():
                _batch = _image.unsqueeze(0)
                _logits = classifier(_batch)[0]
                _probs = _logits.softmax(dim=0).numpy()
                _pred = int(_probs.argmax())
                _target_c = _pred if target_class_picker.value == -1 else int(target_class_picker.value)
                _attr = classifier.get_pixel_attribution(_batch, target_class=_target_c)[0, 0].numpy()
                _bias = float(classifier.network.layers.logits.bias[_target_c])
                _attr_sum = float(_attr.sum())
                _logit_val = float(_logits[_target_c])

            _vmax = max(float(np.abs(_attr).max()), 1e-4)
            _im = _axes[1].imshow(_attr, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
            _fig.colorbar(_im, ax=_axes[1], fraction=0.046, pad=0.04)
            _axes[1].set_title(f"Attribution x ⊙ W_{_target_c} (class {_target_c})")
            _axes[1].axis("off")

            _bars = _axes[2].bar(range(10), _probs, color=["#1d4ed8" if i == _target_c else "#94a3b8" for i in range(10)])
            _axes[2].set_xticks(range(10))
            _axes[2].set_ylim(0, 1)
            _axes[2].set_title(f"Prediction: {_pred} ({_probs[_pred]:.1%})")
            _axes[2].set_xlabel("Digit")
            _axes[2].set_ylabel("Probability")

            completeness_note = notebook_md(
                f"**Exact Logit Decomposition (Class {_target_c}):** "
                f"$\\sum_{{h,w}} (x_{{h,w}} \\cdot W_{{{_target_c},h,w}}) = {_attr_sum:+.4f}$, "
                f"bias $b_{_target_c} = {_bias:+.4f}$ $\\implies$ "
                f"logit $\\ell_{_target_c} = {_attr_sum + _bias:+.4f}$ (model output: `{_logit_val:+.4f}`)."
            )
        else:
            _axes[1].axis("off")
            _axes[2].axis("off")

        _fig.tight_layout()
        preview = mo.vstack([mo.as_html(_fig), completeness_note])
        plt.close(_fig)
    preview
    return


@app.cell
def _(classifier, mo, notebook_md, np, plt):
    templates_view = notebook_md("Select a trained run to inspect learned class weight templates.")
    if classifier is not None:
        _templates = classifier.get_class_templates().numpy()
        _vmax = max(float(np.abs(_templates).max()), 1e-4)
        _fig, _axes = plt.subplots(2, 5, figsize=(11, 4.8))
        for _d, _ax in enumerate(_axes.flat):
            _im = _ax.imshow(_templates[_d], cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
            _ax.set_title(f"Template W_{_d}")
            _ax.axis("off")
        _fig.subplots_adjust(right=0.9)
        _cbar_ax = _fig.add_axes([0.92, 0.15, 0.015, 0.7])
        _fig.colorbar(_im, cax=_cbar_ax, label="Weight value")
        templates_view = mo.vstack([
            notebook_md(r"## Glass-Box Class Weight Templates ($W_0 \dots W_9$)" + "\n"
                        r"Each $28 \times 28$ subplot shows the learned linear weights for digit class $c$. "
                        r"Red pixels increase the score for digit $c$; blue pixels decrease it.", supporting=True),
            mo.as_html(_fig),
        ])
        plt.close(_fig)
    templates_view
    return


@app.cell
def _(config, mo, notebook_md, run_error, selected_run, summary):
    mo.vstack([
        notebook_md("## Saved training results"),
        notebook_md(f"Unable to read run: {run_error}" if run_error else
                    "Choose a training run above." if summary is None else
                    f"**{selected_run.name}** · {summary['completed_epochs']} completed epochs · "
                    f"{summary['global_step']} optimizer steps · smoke run: {summary['smoke_run']}"),
        mo.ui.table([summary["metrics"]]) if summary else notebook_md("No saved Logistic Regression runs yet."),
        mo.accordion({"Resolved configuration": notebook_md(f"```yaml\n{config}\n```")}) if config else notebook_md(""),
        mo.image((selected_run / "learning_curves.png").read_bytes())
        if selected_run and (selected_run / "learning_curves.png").is_file() else notebook_md("No learning curve image available."),
    ])
    return


@app.cell
def _(classifier, confusion_matrix, data, data_error, inference_threads):
    validation_counts = None
    if classifier is not None and data_error is None:
        validation_counts = confusion_matrix(classifier, data.val_dataset)
    return (validation_counts,)


@app.cell
def _(mo):
    heatmap_mode = mo.ui.dropdown(
        options=["Counts", "Percent of true digit"], value="Counts", label="Heat map values",
    )
    heatmap_mode
    return (heatmap_mode,)


@app.cell
def _(heatmap_mode, mo, notebook_md, plt, validation_counts):
    matrix_display = notebook_md("Select a checkpoint to see the validation confusion matrix.")
    if validation_counts is not None:
        _counts = validation_counts.numpy()
        _total = int(_counts.sum())
        _accuracy = float(_counts.trace() / _total) if _total else 0.0
        _percent = heatmap_mode.value == "Percent of true digit"
        _values = _counts / _counts.sum(axis=1, keepdims=True).clip(min=1) * 100 if _percent else _counts
        _fig, _ax = plt.subplots(figsize=(8, 7))
        _heat = _ax.imshow(_values, cmap="Blues", vmin=0, vmax=100 if _percent else max(1, _values.max()))
        _fig.colorbar(_heat, ax=_ax, label="Percent of true digit" if _percent else "Examples")
        _ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
                title="Logistic Regression validation heat map — all digits")
        for _row in range(10):
            for _col in range(10):
                _value = _values[_row, _col]
                _ax.text(_col, _row, f"{_value:.1f}%" if _percent else str(int(_value)),
                         ha="center", va="center", fontsize=8,
                         color="white" if _value > _heat.norm.vmax / 2 else "black")
        _fig.tight_layout()
        matrix_display = mo.vstack([
            notebook_md(f"## Full Validation Confusion Matrix\nAll **{_total:,} validation examples**, digits 0–9. "
                        f"Full validation accuracy: **{_accuracy:.2%}**."),
            mo.as_html(_fig),
        ])
        plt.close(_fig)
    matrix_display
    return


@app.cell
def _(LogisticRegression, mo, notebook_md):
    model = LogisticRegression()
    architecture = str(model)
    mo.vstack([
        notebook_md("## Logistic Regression architecture"),
        notebook_md(f"```text\n{architecture}\n```"),
        notebook_md(f"Parameters: **{sum(p.numel() for p in model.parameters()):,}** ($784 \\times 10$ weights $+ 10$ biases). Input: `[N, 1, 28, 28]`; output: ten logits."),
    ])
    return


if __name__ == "__main__":
    app.run()
