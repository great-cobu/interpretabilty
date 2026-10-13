# Running the preliminary demos

The preliminary demos use the implemented stages 1–6: inspect MNIST and the CNN, train briefly on CPU, view learning curves and checkpoints, then resume in a fresh process. Run all commands from the project root. These are limited smoke demonstrations; the full baseline accuracy measurement, rich evaluation reports and MLflow integration remain for later stages. The [interactive Marimo demo](notebooks.md) can explore the outputs from this walkthrough.

## 1. Prepare the environment and cache

For this checkout, enter the project directory first (use your checkout path on another machine):

```bash
cd /Users/frankelliott/Documents/Projects/Python/interpetability
uv sync --locked --no-editable
uv run --locked --no-editable mnist-cnn prepare-data --data-dir /Users/frankelliott/Documents/Projects/Python/interpetability/data --no-tracking-enabled
```

These commands use built-in defaults, which match `configs/default.yaml`; no configuration file is needed. For a custom profile, pass `--config` with an absolute file path or a path relative to your current directory.

Each step includes the project-directory command and uses a literal absolute cache path. No shell variables or state from an earlier terminal are required. On another machine, replace the checkout prefix in these commands with your project path.

Data preparation downloads MNIST if needed and reuses cached files otherwise. Initial acquisition requires network access. The following demos use `--no-download` and run from the local cache without an MLflow service. See [setup](setup.md) for interpreter installation and [usage](usage.md) for cache paths.

Use `--no-editable` for both setup and every demo command, as shown below. The editable installation failed to import `mnist_cnn` in the current environment; a regular installation was verified with both inspection demos. After changing application source, refresh it with `uv sync --locked --no-editable --reinstall-package mnist-cnn`.

Each command prints `Saved run: PATH`. Commands without `--output-dir` create a unique directory under `runs/`. The named training directory below must be new or empty. When repeating step 3, choose a new training directory and update step 4’s checkpoint path to match. Resume creates its own unique output directory.

## 2. Demo: data and model inspection

```bash
cd /Users/frankelliott/Documents/Projects/Python/interpetability
uv run --locked --no-editable mnist-cnn inspect-data --data-dir /Users/frankelliott/Documents/Projects/Python/interpetability/data --no-download --accelerator cpu --no-tracking-enabled
uv run --locked --no-editable mnist-cnn inspect-model --accelerator cpu --no-tracking-enabled
```

The data command prints split counts and batch summaries and saves `data_summary.json` and `split_indices.json`. Show the 55,000 training, 5,000 validation, and 10,000 test examples, image shape `[N, 1, 28, 28]`, and integer labels. This inspects test data structure without scoring a model.

The model command uses synthetic inputs and saves `model_summary.json`. Show the layer shapes, ten output logits, finite cross-entropy, and gradient checks. It does not train or need the dataset cache. The [CNN walkthrough](stage-4-review.md) explains the shapes and loss.

## 3. Demo: short CPU training

```bash
cd /Users/frankelliott/Documents/Projects/Python/interpetability
uv run --locked --no-editable mnist-cnn train --data-dir /Users/frankelliott/Documents/Projects/Python/interpetability/data --no-download --accelerator cpu --no-tracking-enabled --max-epochs 2 --limit-train-batches 20 --limit-val-batches 5 --output-dir /Users/frankelliott/Documents/Projects/Python/interpetability/runs/preliminary-demo
```

If training reports a missing cache, repeat step 1’s `prepare-data` command, then retry training with a new output-directory name. A failed run may already contain `config.yaml`, so update both the training output and step 4’s resume path rather than reusing that directory.

If you see `data.directory must not be empty`, an older command supplied an empty shell variable. Run the updated step-3 command above, which supplies the cache path directly.

This runs up to two epochs with 20 training batches and five validation batches per epoch. With the default batch size of 64, each completed epoch performs 20 optimizer steps and validates on 320 examples. Training uses shuffled batches; validation uses ordered batches. Training does not score the official test set.

After the command completes, view these local files in your editor or image viewer:

| File under `runs/preliminary-demo/` | What to show |
| --- | --- |
| `config.yaml` | Exact resolved settings and batch limits |
| `logs/metrics.csv` | Actual training/validation loss and accuracy by epoch |
| `learning_curves.png` | Loss and accuracy curves |
| `training_summary.json` | `smoke_run`, elapsed time, completed epochs, global step, and best epoch |
| `checkpoints/best.ckpt` | Model selected by lowest validation loss |
| `checkpoints/last.ckpt` | State for continuing training |
| `checkpoints/manifest.json` | SHA-256 hashes of both checkpoints |

Record the actual metrics and duration from this run. There is no minimum demo accuracy or fixed runtime requirement. Limited results do not establish the later 98% full-baseline target.

## 4. Demo: resume in a fresh process

Use `checkpoints/last.ckpt` inside the successful step-3 run. The command below matches step 3’s output directory. If you chose another training directory, update the literal checkpoint path to match. Each training command is on one line to avoid broken continuation lines.

```bash
cd /Users/frankelliott/Documents/Projects/Python/interpetability
uv run --locked --no-editable mnist-cnn train --resume /Users/frankelliott/Documents/Projects/Python/interpetability/runs/preliminary-demo/checkpoints/last.ckpt --no-download --accelerator cpu --no-tracking-enabled --max-epochs 3
```

Resume creates a unique output directory and prints its path. Keep `best.ckpt` beside the source `last.ckpt`. Resume inherits the data-cache path, model, optimizer, split, and batch-limit settings from the checkpoint, so omit `--config`. If the cache has moved, add `--data-dir` with its absolute path. `--max-epochs 3` means three total epochs: after the two-epoch demo, this runs one more epoch.

If you see `ambiguous option: --no-` followed by `zsh: command not found: tracking-enabled`, a pasted line break split `--no-tracking-enabled`. Run the single-line command above, which keeps every option intact.

Compare both `training_summary.json` files. The resumed summary should record the source checkpoint identity and advance completed epochs from two to three and global step from 40 to 60. Show the resumed learning curves, which contain only the newly executed epoch, and the retained or improved best validation checkpoint. See [training instructions](training.md) for resume constraints and reproducibility limits.

## 5. Review the demo results

The preliminary walkthrough succeeds when cached CPU commands complete, training/validation metrics are finite, curves are readable, checkpoint files and hash manifests exist, and resume restores and advances training progress. Record actual commands, results, and any limitations. Full version-one acceptance remains separate; see the [demo milestone](../spec/mnist-cnn.md#preliminary-demo-milestone-available-after-stage-6).

For an artifact-only walkthrough, existing `runs/stage5-smoke/` and `runs/stage6-resume/` outputs may be used if present. Their measured results are documented in the [stage 5–6 review](stage-5-6-review.md). Identify these as historical evidence; the ignored run directories may be absent in a fresh checkout.

Standalone checkpoint testing is covered by synthetic-fixture verification. To repeat its focused checks without scoring the official MNIST test set:

```bash
cd /Users/frankelliott/Documents/Projects/Python/interpetability
uv run --directory /Users/frankelliott/Documents/Projects/Python/interpetability --locked --no-editable pytest -q /Users/frankelliott/Documents/Projects/Python/interpetability/tests/test_learning.py
```

The test command uses absolute project and test-file paths and sets the working directory with `--directory`, so it also works when executed independently from another directory. `file or directory not found: tests/test_learning.py` means an older relative-path command was run outside the project root.

The `test --checkpoint` CLI measures all official test examples; reserve that model-quality measurement for stage 9. The `evaluate` command is still a placeholder, and the [Marimo demo](notebooks.md) provides an interactive view of the existing data and training artifacts.
