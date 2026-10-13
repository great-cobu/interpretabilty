2# MNIST CNN project specification

Status: Stages 1–6 implemented; preliminary local demo available now; remaining stages planned  
Date: 2026-09-30

## Objective and scope

Build a command-line Python project that uses PyTorch Lightning to load MNIST, train a convolutional neural network, validate model quality during training, test a selected checkpoint, and generate an evaluation report. Support both standalone local execution and execution in an MLflow harness using the user's existing containers. A new user must be able to reproduce a run from documented commands and saved configuration.

In this specification, **validation** guides checkpoint selection, **testing** measures performance on the held-out test set, and **evaluation** adds class-level metrics and error analysis for that same fixed checkpoint. Software tests verify the implementation separately.

Version one supports CPU and a single available accelerator. Distributed training, hyperparameter search, deployment, a web UI, and custom handwritten-image preprocessing are outside scope.

## Preliminary demo milestone (available after stage 6)

Provide a local command-line walkthrough now using the implemented stages 1–6. This milestone precedes stage 7 and does not depend on evaluation reports, MLflow containers, a full baseline run, or Marimo. The stage-10 Marimo demonstration remains the final walkthrough.

The preliminary demo must show cached MNIST split/batch inspection, synthetic CNN shapes and loss/gradients, two limited CPU training epochs, validation metrics and learning curves, best/last checkpoints with hashes, and a fresh-process resume to a third total epoch. Use local CSV logging with tracking disabled. Label all limited training results as smoke-run evidence; the 98% test-accuracy target applies to the later full baseline only.

Run from the project root after `uv sync --locked --no-editable`. If the cache is missing, first run `uv run --locked --no-editable mnist-cnn prepare-data`. With the cache present, the walkthrough must work with downloads disabled:

```bash
uv run --locked --no-editable mnist-cnn inspect-data --no-download --accelerator cpu --no-tracking-enabled
uv run --locked --no-editable mnist-cnn inspect-model --accelerator cpu --no-tracking-enabled
uv run --locked --no-editable mnist-cnn train --no-download --accelerator cpu --no-tracking-enabled --max-epochs 2 --limit-train-batches 20 --limit-val-batches 5 --output-dir runs/preliminary-demo
uv run --locked --no-editable mnist-cnn train --resume runs/preliminary-demo/checkpoints/last.ckpt --no-download --accelerator cpu --no-tracking-enabled --max-epochs 3 --output-dir runs/preliminary-demo-resumed
```

Named output directories must be new or empty; use new names and update the resume path when repeating the demo. Inspect `logs/metrics.csv`, `learning_curves.png`, `training_summary.json`, and `checkpoints/manifest.json` in each training output. Demonstrate that the resumed epoch/global step advances and that checkpoint selection follows validation loss. Record the actual commands, durations, and results; no fixed accuracy or wall-clock duration is required for this milestone.

Preliminary acceptance requires successful cached CPU commands, finite logged training/validation metrics, readable learning curves, saved best/last checkpoints and hash manifests, and restored training progress in the separate resume process. Standalone checkpoint testing is supported by the existing fixture-based verification recorded in the [stage 5–6 review](../instructions/stage-5-6-review.md). Keep the official test-score measurement reserved for stage 9; this walkthrough does not score or tune against the official test set.

The existing stage-5 smoke and stage-6 resume artifacts may support the walkthrough when available, with their recorded results identified as historical evidence. New runs must report their own results. Passing preliminary acceptance permits this demo immediately and does not declare version one complete. All final acceptance criteria below remain required for the completed project.

## Environment and dependencies

- Use `lightning.pytorch` consistently for Lightning classes and the Trainer.
- Runtime dependencies: `torch`, `torchvision`, `lightning`, `torchmetrics`, `matplotlib`, and `PyYAML`. Development dependency: `pytest`.
- Provide an `mlflow` dependency extra containing the MLflow client and any artifact-store dependencies required by the existing harness. Pin and verify client compatibility against the actual containerized server version.
- Use **uv exclusively** for Python environment and package management. Declare dependencies in `pyproject.toml`, commit `uv.lock`, use `uv add` for dependency changes, and execute project commands with `uv run`. CI and job-image builds must use `uv sync --locked` with the required extras/groups.
- Require **the latest stable CPython 3.14 patch release** at implementation time, with `requires-python = ">=3.14,<3.15"`. Verify the latest patch, install it through uv, and commit its exact version in `.python-version`. Use the same pin in development, CI, and training job containers. Update the pin deliberately as new 3.14 patches are released and rerun compatibility checks. uv supports managed Python installations and project version pins. [uv Python version documentation](https://docs.astral.sh/uv/concepts/python-versions/)
- Verify the complete dependency stack on that Python 3.14 release and record tested versions. Resolve dependency incompatibilities within this requirement; do not downgrade Python or switch to 3.15 as a workaround.
- Provide an optional `notebooks` dependency group containing `marimo`, managed by the same project lockfile. Notebooks must not be required by the runtime package.
- CPU execution is required. Allow explicit CPU, CUDA, or MPS selection where supported; automatic selection must record the resolved device. An explicitly requested unavailable device must produce an actionable error.
- Initial dataset acquisition requires network access; cached-data workflows must work offline.

## Data contract

Use `torchvision.datasets.MNIST` with a configurable cache directory, defaulting to `data/`. Its API provides separate training and test datasets and download/cache behavior. [Torchvision MNIST documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

| Split | Source | Examples | Purpose |
| --- | --- | ---: | --- |
| Training | Official training set | 55,000 | Gradient updates |
| Validation | Official training set | 5,000 | Checkpoint selection and early stopping |
| Test | Official test set | 10,000 | Final measurement and error analysis |

Create the training/validation partition with a dedicated seeded `torch.Generator` and `random_split`. Save the exact subset indices with each training run. Repeated setup calls must preserve membership. Never use test data to fit preprocessing, tune hyperparameters, or choose checkpoints.

Transform each image to a `float32` tensor in `[0, 1]`, then normalize with fixed mean `0.1307` and standard deviation `0.3081`. These are baseline configuration constants, not statistics estimated from the test set. Apply identical preprocessing to every split; no augmentation in version one. Batches have image shape `[N, 1, 28, 28]` and `int64` labels of shape `[N]`, with values from 0 to 9.

Implement `MNISTDataModule` with these responsibilities:

- `prepare_data()`: download/cache both official datasets when downloads are enabled; do not create split state here.
- `setup(stage)`: construct the datasets needed by fit, validation, test, or prediction, loading saved split indices when supplied.
- `train_dataloader()`: shuffle the training subset using seeded randomness.
- `val_dataloader()` and `test_dataloader()`: preserve order and include every example.
- `predict_dataloader()`: expose the ordered test set for evaluation exports.

Use `drop_last=False`, default `num_workers=0`, and enable persistent workers only when workers are positive. Enable pinned memory for CUDA only. Missing cached data with downloads disabled must produce a clear error explaining how to prepare it. Data preparation and stage setup follow the Lightning DataModule lifecycle. [Lightning DataModule reference](https://lightning.ai/docs/pytorch/2.2.0/data/datamodule.html)

## CNN and Lightning module

Keep the network in a plain `torch.nn.Module`; wrap optimization and metric handling in `MNISTClassifier(LightningModule)`.

| Layer | Configuration | Output shape, excluding batch |
| --- | --- | --- |
| Input | Grayscale image | `1 × 28 × 28` |
| Convolution + ReLU | 1 → 32 channels, kernel 3, padding 1 | `32 × 28 × 28` |
| Max pooling | Kernel 2, stride 2 | `32 × 14 × 14` |
| Convolution + ReLU | 32 → 64 channels, kernel 3, padding 1 | `64 × 14 × 14` |
| Max pooling | Kernel 2, stride 2 | `64 × 7 × 7` |
| Flatten | — | `3136` |
| Linear + ReLU | 3136 → 128 | `128` |
| Dropout | Probability 0.2 | `128` |
| Linear | 128 → 10 | `10` |

`forward()` returns unnormalized logits. Use cross-entropy directly on logits and integer labels. Apply softmax only when exporting probabilities; predicted labels are the logits' argmax.

Implement `training_step`, `validation_step`, `test_step`, `predict_step`, and `configure_optimizers`. Save model hyperparameters in the checkpoint so standalone loading reconstructs the network. Maintain separate TorchMetrics instances for training, validation, and testing; reset them between epochs and invocations. Accuracy means total correct predictions divided by total examples, not macro-averaged class accuracy. Aggregate loss weighted by batch size so a partial final batch is handled correctly.

## Training workflow

Use `Trainer.fit(model, datamodule=...)` for training and validation. Seed Python, NumPy, PyTorch, and data workers through `seed_everything(seed, workers=True)` before model/data creation. Enable deterministic execution by default; unsupported deterministic operations must fail with a useful message rather than silently disabling determinism. Exact reproducibility across different hardware or library versions is not promised. [Lightning Trainer documentation](https://lightning.ai/docs/pytorch/stable/common/trainer)

| Setting | Default |
| --- | --- |
| Seed | 42 |
| Batch size | 64 |
| Maximum epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Learning-rate scheduler | None |
| Accelerator / devices | `auto` / 1 |
| Precision | `32-true` |
| Validation frequency | Every epoch |
| Early stopping | Minimum validation loss, patience 3, min_delta 0 |
| Checkpoints | Best validation loss and last training state |
| Logging | Local CSV; additionally MLflow when enabled by the harness profile |

Log `train/loss`, `train/accuracy`, `val/loss`, and `val/accuracy` each epoch. Save checkpoints under stable names `best.ckpt` and `last.ckpt`. Use a filesystem-safe checkpoint filename independent of metric names. Fail on nonfinite loss. Record elapsed training time, completed epochs, best epoch, and checkpoint paths.

For smoke runs, optional positive integer `trainer.limit_train_batches` and `trainer.limit_val_batches` cap batches per epoch. Defaults are null (full loaders). Mark limited runs explicitly; never apply these limits to standalone testing.

Support resuming with `Trainer.fit(..., ckpt_path=...)`, restoring optimizer and training state. Resume must retain the original model, preprocessing, and split configuration; permit explicit changes to operational settings such as output directory and maximum epochs, and record those overrides. Training does not automatically run the test set.

## MLflow harness integration

### Existing infrastructure and execution boundary

Reuse the user's existing MLflow containers, tracking server, backend store, and artifact store. The training process is a client of those services and may run on the host or in a separate job container connected to their network. MLflow tracking does not itself schedule the training job. Document how the existing harness launches the project CLI; add a job image or launch adapter only if that harness requires one. Provisioning a replacement MLflow stack is outside scope.

Before wiring up the integration, inspect the supplied container configuration and establish the server version, tracking URL reachable from each execution context, network name, artifact-access mode, authentication mechanism, dataset/output mounts, and available compute. These values are currently unspecified and must remain configurable. A container service hostname and a host-published URL are distinct; `localhost` inside a job container refers to that container.

### Configuration and connectivity

- Add `configs/mlflow.yaml` as a complete training profile with tracking enabled; retain `configs/default.yaml` for local execution without an MLflow service.
- Support `tracking.enabled`, `tracking.uri`, `tracking.experiment_name` (default `mnist-cnn`), `tracking.run_name`, and non-secret `tracking.tags`.
- Resolve the tracking URI in this order: explicit `--tracking-uri`, `MLFLOW_TRACKING_URI`, YAML. Require a URI when tracking is enabled; never silently create a local MLflow store because a remote URI is missing.
- Obtain credentials from environment variables or the harness's secret injection mechanism. Exclude credentials, tokens, and credential-bearing URLs from saved configuration, parameters, logs, and artifacts.
- Keep experiment artifact placement under the existing server configuration. Determine whether the server proxies artifacts or clients require direct object-store access; supply client endpoint settings and credentials only when needed. A server-local filesystem path must not be assumed accessible to a job container. [MLflow artifact-store architecture](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)
- Before training, check experiment access and round-trip a small artifact through the configured store. Report connection or authorization failures before expensive computation.

### Run ownership and lifecycle

Use Lightning's `MLFlowLogger` alongside `CSVLogger` for training metrics and hyperparameters. Supply an explicit tracking URI, experiment, and run ID. Disable MLflow autologging to avoid duplicate metrics, runs, or checkpoint uploads. The logger supports attaching to an existing run via `run_id`. [Lightning MLFlowLogger reference](https://api.lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.MLFlowLogger.html)

The CLI creates one run per `train`, `test`, or `evaluate` invocation unless the harness supplies an existing run through `--mlflow-run-id` or `MLFLOW_RUN_ID` (CLI takes precedence). Validate that a supplied run is active and belongs to the configured experiment. The harness must allocate a distinct run per invocation; concurrent jobs must not share a run ID.

The component that creates a run owns its final status. For CLI-owned runs, finish only after required artifact uploads succeed; mark failures as failed and handled interruptions as killed. For harness-owned runs, the CLI reports its exit status and leaves finalization to the harness. Ensure Lightning logger finalization respects this policy, using a small adapter if needed. The harness handles hard termination where the process cannot perform cleanup.

Standalone test/evaluation creates a new run tagged with `source_training_run_id` when known and `source_checkpoint_sha256`. Resumed training also creates a new run, or uses a fresh harness-supplied run, and records `resumed_from_run_id`, source checkpoint hash, and the resumed epoch/global step. Do not append changed hyperparameters to a completed source run.

### Logged information and artifacts

Log the resolved non-secret hyperparameters, seed, split sizes, preprocessing constants, optimizer settings, and device. Record command type, Git revision and dirty state when available, Python/package versions, and job image digest when supplied by the harness. Write `tracking.json` locally with experiment ID, run ID, and artifact URI so outputs can be traced back to MLflow.

Preserve the specified training/validation metric names and record epoch/global-step coordinates. Log standalone test and evaluation scalars under `test/` and `evaluation/` respectively. Keep the complete per-class results in the JSON report.

Upload these artifacts to stable paths within each run:

| Path | Contents |
| --- | --- |
| `config/` | Sanitized resolved configuration and exact training/validation indices when applicable |
| `environment/` | Dependency lockfile, installed package versions, and execution metadata |
| `logs/` | CSV metrics and training/validation learning curves |
| `checkpoints/` | Final selected `best.ckpt`, resumable `last.ckpt`, and SHA-256 manifest for training runs |
| `evaluation/` | All metrics, predictions, confusion matrices, error images, and report generated by test/evaluate |

Use explicit checkpoint uploads with `MLFlowLogger(log_model=False)` so paths and upload ownership are unambiguous. A Lightning checkpoint is the version-one model artifact; MLflow Model Registry registration and serving-format model export are optional future work.

Accept local checkpoint paths and `runs:/<run-id>/checkpoints/best.ckpt` or `last.ckpt` URIs for testing, evaluation, and resume. Download remote artifacts into a local cache through the MLflow artifact API and verify against the source manifest before loading. This requires tracking connectivity even if logging for the current command is disabled. [MLflow artifact download API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.artifacts.html)

Keep local checkpoints and reports on a writable persistent output mount. With tracking enabled, bounded connection retries may be used, but an unrecoverable logging/upload error must preserve local results and return nonzero; it must not report a successfully tracked run. Local-only execution remains explicitly available with tracking disabled.

## Testing and evaluation

Both commands require an explicit checkpoint and use its saved architecture and preprocessing configuration. They must run in a fresh process without requiring an earlier in-memory training session. Missing or incompatible checkpoints must fail clearly.

The `test` command uses `Trainer.test` over all 10,000 official test examples and writes `metrics.json` containing test loss, accuracy, sample count, checkpoint SHA-256, configuration, package versions, device, and elapsed test time.

The `evaluate` command runs the same test measurement and then an ordered inference pass for richer reporting. Inference must disable gradients and dropout. Export:

- `metrics.json`: the test fields above plus macro precision, macro recall, macro F1, and per-class precision, recall, F1, and support for all ten digits. Undefined precision/recall values are zero and this policy is stated in the report.
- `confusion_matrix.csv` and `confusion_matrix.png`: a 10 × 10 matrix of integer counts, with rows representing true labels and columns predicted labels.
- `predictions.csv`: exactly one row per test example, including dataset index, true label, predicted label, and probabilities `p_0` through `p_9`.
- `misclassified.png`: up to 25 errors in ascending dataset-index order, with true label, prediction, and confidence. If no errors exist, report that explicitly.
- `report.md`: checkpoint identity, overall metrics, per-class table, plots, and a brief factual account of the most frequent off-diagonal confusions. Do not infer causes unsupported by the images.

Use probabilities in `[0, 1]` and document that confidence is the maximum softmax probability, not a calibrated certainty estimate. Metrics in the report must agree with those recomputed from the predictions. Evaluation must not update model parameters or select a new checkpoint.

## Command-line and configuration contract

Expose an installed `mnist-cnn` console script via `[project.scripts]`, mapped to `mnist_cnn.cli:main`, and an equivalent `python -m mnist_cnn` entry point. All data preparation, training, resume, testing, evaluation, report generation, and MLflow integration must be usable noninteractively from the CLI. CLI handlers delegate to reusable package functions. No workflow may require opening a notebook or manually executing cells.

Stages 1–6 currently provide `check-config`, `prepare-data`, `inspect-data`, synthetic `inspect-model`, `train` (including resume), and `test`; see the [usage guide](../instructions/usage.md). The following end-to-end examples also include evaluation and MLflow commands planned for later stages:

```bash
uv sync --locked
uv run mnist-cnn prepare-data --config configs/default.yaml
uv run mnist-cnn train --config configs/default.yaml --output-dir runs/baseline
uv run mnist-cnn train --config configs/default.yaml --resume runs/baseline/checkpoints/last.ckpt --output-dir runs/resumed
uv run mnist-cnn test --checkpoint runs/baseline/checkpoints/best.ckpt --output-dir runs/test
uv run mnist-cnn evaluate --checkpoint runs/baseline/checkpoints/best.ckpt --output-dir runs/evaluation
uv run pytest
```

Proposed harness commands (the harness supplies the real `MLFLOW_TRACKING_URI` and any credentials; replace the example run ID):

```bash
uv sync --locked --extra mlflow
uv run mnist-cnn train --config configs/mlflow.yaml --output-dir runs/tracked-baseline
uv run mnist-cnn evaluate --tracking-enabled --checkpoint runs:/TRAINING_RUN_ID/checkpoints/best.ckpt --output-dir runs/tracked-evaluation
```

Configuration groups cover data paths/download permission, split seed, batch size, workers, model dropout, optimizer parameters, trainer settings, tracking settings, and output directory. Accept documented CLI overrides for these settings during training, with precedence CLI > YAML > defaults, except for the explicit MLflow environment-variable precedence above. Evaluation permits only operational overrides such as device, data directory, batch size, workers, tracking settings, and output directory. Validate unknown keys, positive counts/rates, and dropout bounds before expensive work. Save the fully resolved, sanitized configuration as `config.yaml` in every run.

Output directories must be unique or empty; reject accidental overwrite. Resume uses a new output directory and records the source checkpoint. Commands return zero on success and nonzero on failure. Every subcommand provides `--help`.

## Demonstration notebooks

Use **Marimo only** for notebooks, stored as Python `.py` files under `notebooks/`; do not add Jupyter notebooks or `.ipynb` files. Demonstrations import the installed `mnist_cnn` package and illustrate data samples, model usage, learning curves, and evaluation artifacts. Keep reusable loading, training, metrics, plotting, and tracking code in `src/`. Notebook cells may provide explanatory text, interactive controls, and calls into those APIs, but must not duplicate pipeline implementations or become dependencies of package code.

Opening a demonstration must not automatically launch full training, download data, or create an MLflow run. Prefer existing local artifacts; any optional small training demonstration requires an explicit user action and calls the same package API as the CLI. Show useful instructions when prerequisite artifacts are missing.

Support editing, presenting, and exporting demonstrations from the command line using the project environment. [Marimo project integration](https://docs.marimo.io/guides/package_management/notebooks_in_projects/), [Marimo script and export workflows](https://docs.marimo.io/guides/scripts/)

```bash
uv sync --locked --group notebooks
uv run --group notebooks marimo edit notebooks/mnist_demo.py
uv run --group notebooks marimo run notebooks/mnist_demo.py
uv run --group notebooks marimo export html notebooks/mnist_demo.py -o runs/mnist_demo.html
```

## Documentation organization

Store maintained project Markdown documents in `doc/` and specifications in `doc/spec/`. Use `doc/README.md` as the documentation index and `doc/instructions/` for setup, command-line usage, MLflow harness operation, and Marimo demonstration instructions. Keep the project specification at `doc/spec/mnist-cnn.md`. Use relative links between documents.

Generated evaluation reports remain run artifacts under `runs/` and in MLflow; they are not maintained project documentation. Keep notebook source files under `notebooks/` and application code under `src/`.

Use **Graphviz** for diagrams where they clarify architecture, data flow, training/evaluation stages, or the MLflow harness. Keep editable DOT source files in `doc/diagrams/` and rendered SVG files alongside them, linked from the relevant Markdown documents. Commit both source and rendered diagrams and regenerate the SVG when its source changes. Statistical plots such as learning curves and confusion matrices remain plotting outputs.

Document the Graphviz system installation prerequisite and reproducible rendering commands in `doc/instructions/diagrams.md`. Diagram generation must work from the command line without a notebook. Graphviz is documentation tooling and must not be required to train or evaluate the model; any Python helper dependencies must be managed through uv.

## Proposed repository layout

```text
pyproject.toml                # Dependencies and development configuration
uv.lock                       # Resolved environment
.python-version               # Exact latest stable 3.14 patch selected and verified
configs/default.yaml          # Baseline settings
configs/mlflow.yaml           # Profile for the existing MLflow harness
src/mnist_cnn/
    __init__.py
    __main__.py               # python -m mnist_cnn entry point
    workflows.py              # Reusable training, resume, test, evaluation APIs
    cli.py                    # Commands and configuration validation
    data.py                   # MNISTDataModule
    model.py                  # CNN network
    module.py                 # Lightning module
    learning.py               # Trainer workflows, testing, and learning curves
    checkpoints.py            # Reconstruction, integrity, and best/last checkpoints
    model_inspection.py       # Synthetic forward/backward demonstration
    evaluation.py             # Metrics, plots, prediction exports, report
    tracking.py               # MLflow configuration, run ownership, artifact IO
notebooks/
    mnist_demo.py             # Marimo demonstration importing package APIs
tests/
    test_data.py
    test_model.py
    test_learning.py          # Lightning training, resume, and standalone tests
    test_workflows.py
    test_evaluation.py
    test_tracking.py
doc/
    README.md                 # Project overview and documentation index
    diagrams/                 # Graphviz DOT sources and rendered SVG diagrams
    spec/
        mnist-cnn.md          # This specification
        implementation-plan.md # Coding/testing stages and review points
    instructions/
        README.md             # Instructions index
        setup.md              # uv, Python version, dependency installation
        usage.md              # CLI commands, outputs, troubleshooting
        mlflow.md             # Existing container integration and launch instructions
        notebooks.md          # Running and exporting Marimo demonstrations
        diagrams.md           # Graphviz installation and CLI rendering instructions
data/                         # Ignored dataset cache
runs/                         # Ignored checkpoints, logs, and reports
```

All application code must live in installable packages under `src/`, initially `src/mnist_cnn/`, with subpackages added when useful. Configure the build backend to discover the src layout so `uv sync` installs `mnist_cnn` and its console script; `src` itself is not the import namespace. Imports must work without `sys.path` changes or `PYTHONPATH` overrides. Replace the current sample `main.py` during implementation; the supported entry points are the installed command and package module. Guard executable entry points for worker spawning. Exclude generated data, runs, notebook exports, virtual environments, and caches from Git.

## Verification and acceptance criteria

These are final version-one criteria. The preliminary demo uses its separate acceptance criteria above and can proceed before stages 7–10 are implemented.

1. A clean environment can install the locked dependencies, import the stack, and run the documented CPU workflow. Record the verified environment in the usage documentation.
2. Data checks confirm 55,000/5,000/10,000 examples, disjoint training/validation indices covering the official training set, reproducible splits, expected tensors and labels, and unchanged membership across setup calls.
3. Model checks confirm `[N, 10]` finite logits for several batch sizes and finite loss/gradients on a small batch.
4. A small CPU integration run exercises Lightning fit, validation, checkpoint saving, standalone checkpoint reload, test, and resume. Use local synthetic fixtures for routine tests so CI does not need a network download; keep real-MNIST integration checks explicitly separate.
5. Handcrafted predictions verify accuracy, per-class and macro metrics, confusion-matrix orientation, absent-class handling, and batch-size-independent aggregation including a partial final batch.
6. A full baseline run targets at least **98% test accuracy** using the validation-selected checkpoint. This is an acceptance target, not an observed result or a unit-test assertion. Record the actual score; do not repeatedly tune against the test set if it is missed.
7. Evaluation produces all specified artifacts, 10,000 unique ordered prediction rows, probabilities summing to one within numerical tolerance, and a confusion matrix summing to 10,000. Exported metrics agree with prediction-derived metrics within floating-point tolerance.
8. Checkpoint round-trip inference yields equivalent logits on a fixed input within tolerance on the same CPU environment; resume demonstrably continues the saved training state.
9. Cached execution succeeds with downloads disabled. Invalid configuration, missing data/checkpoints, and occupied output directories fail with actionable messages.
10. Documentation explains installation, all commands, defaults, split policy, checkpoint selection, evaluation artifacts, and reproducibility limits. Include training/validation learning curves from CSV logs for the completed baseline run.
11. An integration smoke run against the user's existing MLflow containers records one run with matching parameters, metrics, and downloadable artifacts. Test both CLI-owned and harness-owned run lifecycles without duplicate runs or premature finalization.
12. A separate process retrieves a checkpoint via its `runs:/` URI, verifies its hash, and produces a linked evaluation run. Resume creates a distinct run with source lineage and restores training state.
13. Tests cover unavailable tracking services, artifact-upload failures, and sanitized configuration. Tracking failures return nonzero and retain local outputs; local-only execution requires no MLflow service. Remote checks run separately from routine offline tests.
14. Harness documentation records verified server/client versions, reachable URLs for supported launch contexts, network/mount requirements, credential variable names without values, and a successful artifact upload/download check. Containerized jobs persist outputs after container removal and use the requested compute device.
15. The latest stable Python 3.14 patch selected at implementation is pinned in `.python-version`, with a matching project Python range and verified uv lockfile. CI and training jobs use that pin and locked installs.
16. Installed console-script and module entry points provide equivalent help and workflows; the full pipeline runs from a shell without Marimo installed. Application imports resolve from the installed src package without path manipulation.
17. All notebooks are Marimo `.py` demonstrations importing package APIs. With documented local fixtures/artifacts, the demo exports to HTML through the CLI without training, network access, or MLflow side effects.

18. Maintained Markdown documentation is organized under `doc/`, with operational guides in `doc/instructions/` and working relative links from the documentation index.
19. Any explanatory diagrams use Graphviz, have committed DOT sources and matching SVG renderings, and can be regenerated from documented CLI commands. Inspect renderings for readable labels, unclipped content, and clear connections before including them in documentation.

## Implementation sequence

Follow the [supervised implementation and testing plan](implementation-plan.md). Work in these stages, completing focused checks and a user review before advancing unless the user requests continued work:

1. Python 3.14 environment, uv packaging, and CLI skeleton.
2. Validated configuration and run directories.
3. MNIST download, splits, and DataModule.
4. CNN forward pass and loss.
5. Lightning training and validation.
6. Checkpoints, resume, and standalone testing.
   Preliminary local demo and review using stages 1–6, before continuing to stage 7.
7. Evaluation metrics and reports.
8. MLflow tracking and existing container harness integration.
9. Full baseline and reproducibility verification.
10. Marimo demonstration and documentation walkthrough.
