"""Interactive inspection of local MNIST data and existing training runs."""
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", css_file="theme.css")


@app.cell
def _():
    import json
    import os
    from dataclasses import replace
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    from models.MNIST.config import DataConfig
    from models.MNIST.data import MNISTDataModule
    from models.MNIST.evaluate import confusion_matrix
    from models.MNIST.model import MNISTCNN, MNISTClassifier
    from models.MNIST.train import train_demo

    def notebook_md(text, *, supporting=False):
        """Render prose with the shared theme; use Markdown heading levels."""
        _classes = "mnist-prose mnist-supporting" if supporting else "mnist-prose"
        return mo.Html(f'<div class="{_classes}">{mo.md(text).text}</div>')

    root = Path(__file__).resolve().parents[1]
    return DataConfig, MNISTCNN, MNISTClassifier, MNISTDataModule, confusion_matrix, json, mo, notebook_md, os, plt, replace, root, torch, train_demo


@app.cell(hide_code=True)
def _(notebook_md):
    notebook_md("""
    # MNIST CNN interactive demo
    Explore cached validation examples and saved training results. Select a run
    to inspect its metrics, curves, and checkpoint predictions. Limited demo
    accuracy does not establish the full baseline target.
    """, supporting=True)
    return


@app.cell
def _(mo, notebook_md, os):
    training_form = mo.ui.dictionary({
        "epochs": mo.ui.number(1, 100, step=1, value=2, label="Training epochs"),
        "threads": mo.ui.number(1, os.cpu_count() or 1, step=1,
                                value=min(8, os.cpu_count() or 1), label="Training CPU threads"),
    }).form(submit_button_label="Train new demo run")
    mo.vstack([
        notebook_md("## Train a demo\nChoose the number of epochs, then press **Train new demo run**. "
                    "Each epoch uses 20 training batches and five validation batches on CPU.", supporting=True),
        training_form,
    ])
    return (training_form,)


@app.cell
def _(mo, notebook_md, root, train_demo, training_form):
    trained_run = None
    training_error = None
    if training_form.value is not None:
        try:
            with mo.status.spinner(title="Training demo…"):
                trained_run = train_demo(root, int(training_form.value["epochs"]), int(training_form.value["threads"]))
        except (OSError, RuntimeError, ValueError) as exc:
            training_error = str(exc)
    notebook_md(f"Training failed: {training_error}" if training_error else
                f"Saved run: `{trained_run}`" if trained_run else "")
    return (trained_run,)


@app.cell
def _(mo, root, trained_run):
    runs = sorted(
        (p.parent for p in (root / "runs").glob("*/training_summary.json")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    run_picker = mo.ui.dropdown(
        options={p.name: str(p) for p in runs},
        value=trained_run.name if trained_run else runs[0].name if runs else None,
        label="Saved training run",
    )
    run_picker
    return (run_picker,)


@app.cell
def _(mo):
    digit_picker = mo.ui.radio(
        options={str(_digit): _digit for _digit in range(10)},
        value="0", inline=True, label="Validation digit",
    )
    return (digit_picker,)


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
    notebook_md(f"Using **{inference_threads} CPU threads** for inference. Changing this control recomputes validation results.")
    return (inference_threads,)


@app.cell
def _(DataConfig, MNISTDataModule, classifier, notebook_md, replace, root):
    _settings = classifier.config.data if classifier is not None else DataConfig(directory=str(root / "data"))
    data = MNISTDataModule(replace(_settings, download=False), accelerator="cpu")
    data_error = None
    try:
        data.setup("fit")
    except (RuntimeError, ValueError, OSError) as exc:
        data_error = str(exc)
    notebook_md(f"**Cache unavailable:** {data_error}. Run demo step 1 first." if data_error else
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
def _(digit_indices, digit_picker, mo, notebook_md):
    matching_indices = digit_indices[digit_picker.value]
    next_example = mo.ui.button(
        value=0, on_click=lambda value: value + 1,
        label="Next example", disabled=not matching_indices,
    )
    mo.vstack([
        notebook_md("## Validation digit and model prediction\nBrowse examples of the selected digit below and compare the true digit with the model’s predicted probabilities.", supporting=True),
        digit_picker,
        next_example,
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
def _(MNISTClassifier, notebook_md, selected_run):
    classifier = None
    checkpoint_error = None
    if selected_run:
        try:
            classifier = MNISTClassifier.load_from_checkpoint(
                selected_run / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=True,
            ).eval()
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            checkpoint_error = str(exc)
    notebook_md(f"Checkpoint unavailable: {checkpoint_error}" if checkpoint_error else
                "Predictions use this run's best checkpoint on CPU." if classifier else "Select a run to enable predictions.")
    return (classifier,)


@app.cell
def _(classifier, data, data_error, mo, notebook_md, plt, sample_index, torch):
    preview = notebook_md("Prepare the local cache to see digit examples.")
    if data_error is None and sample_index is not None:
        _image, _label = data.val_dataset[sample_index]
        _fig, _axes = plt.subplots(1, 2, figsize=(9, 3))
        _axes[0].imshow((_image[0] * data.config.std + data.config.mean).numpy(), cmap="gray", vmin=0, vmax=1)
        _axes[0].set_title(f"Validation digit: {_label}")
        _axes[0].axis("off")
        if classifier is not None:
            with torch.inference_mode():
                _probs = classifier(_image.unsqueeze(0)).softmax(dim=1)[0].numpy()
            _axes[1].bar(range(10), _probs)
            _axes[1].set_xticks(range(10))
            _axes[1].set_ylim(0, 1)
            _axes[1].set_title(f"Prediction: {_probs.argmax()}")
            _axes[1].set_xlabel("Digit")
            _axes[1].set_ylabel("Probability")
        else:
            _axes[1].axis("off")
        _fig.tight_layout()
        preview = mo.as_html(_fig)
        plt.close(_fig)
    preview
    return


@app.cell
def _(config, mo, notebook_md, run_error, selected_run, summary):
    mo.vstack([
        notebook_md("## Saved results"),
        notebook_md(f"Unable to read run: {run_error}" if run_error else
                    "Choose a training run above." if summary is None else
                    f"**{selected_run.name}** · {summary['completed_epochs']} completed epochs · "
                    f"{summary['global_step']} optimizer steps · smoke run: {summary['smoke_run']}"),
        mo.ui.table([summary["metrics"]]) if summary else notebook_md("No saved training runs yet. Run demo step 3, then reopen the notebook."),
        mo.accordion({"Resolved configuration": notebook_md(f"```yaml\n{config}\n```")}) if config else notebook_md(""),
        mo.image((selected_run / "learning_curves.png").read_bytes())
        if selected_run and (selected_run / "learning_curves.png").is_file() else notebook_md("No learning curve image available."),
        notebook_md("") if summary is None else notebook_md(
            "Resume provenance: " + (str(summary.get("source_checkpoint")) if summary.get("source_checkpoint") else "initial training run")
        ),
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
    matrix_display = notebook_md("Select a checkpoint and prepare the cache to see the validation heat map.")
    if validation_counts is not None:
        _counts = validation_counts.numpy()
        _total = int(_counts.sum())
        _accuracy = float(_counts.trace() / _total) if _total else 0.0
        _percent = heatmap_mode.value == "Percent of true digit"
        _values = _counts / _counts.sum(axis=1, keepdims=True).clip(min=1) * 100 if _percent else _counts
        _fig, _ax = plt.subplots(figsize=(8, 7))
        _heat = _ax.imshow(_values, cmap="Blues", vmin=0, vmax=100 if _percent else max(1, _values.max()))
        _fig.colorbar(_heat, ax=_ax, label="Percent of true digit" if _percent else "Examples")
        _ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit", title="Validation heat map — all digits")
        for _row in range(10):
            for _col in range(10):
                _value = _values[_row, _col]
                _ax.text(_col, _row, f"{_value:.1f}%" if _percent else str(int(_value)),
                         ha="center", va="center", fontsize=8,
                         color="white" if _value > _heat.norm.vmax / 2 else "black")
        _fig.tight_layout()
        matrix_display = mo.vstack([
            notebook_md(f"## Validation heat map\nAll **{_total:,} validation examples**, including digits 0–9. "
                        f"Accuracy: **{_accuracy:.2%}**. Rows are true digits; columns are predictions. "
                        "This uses the entire validation set, regardless of demo batch limits."),
            mo.as_html(_fig),
        ])
        plt.close(_fig)
    matrix_display
    return


@app.cell
def _(MNISTCNN, mo, notebook_md):
    model = MNISTCNN()
    architecture = str(model)
    mo.vstack([
        notebook_md("## CNN architecture"),
        notebook_md(f"```text\n{architecture}\n```"),
        notebook_md(f"Parameters: {sum(p.numel() for p in model.parameters()):,}. Input: [N, 1, 28, 28]; output: ten logits."),
    ])
    return


if __name__ == "__main__":
    app.run()
