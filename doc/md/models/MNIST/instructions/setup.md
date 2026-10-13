# Setup

Run these commands from the project root. uv manages the interpreter, local `.venv`, dependencies, and package installation.

```bash
uv python install 3.14.7
uv sync --locked
uv run --locked python --version
uv run --locked mnist-cnn --help
uv run --locked python -m mnist_cnn --help
```

The project pins CPython **3.14.7** in `.python-version`, the latest stable 3.14 patch verified on 2026-09-29 against the [Python release page](https://www.python.org/downloads/release/python-3147/). `pyproject.toml` restricts the minor version to 3.14. `uv.lock` records exact package versions; `--locked` rejects an outdated lockfile instead of silently changing it.

The installed `mnist-cnn` command and `python -m mnist_cnn` invoke the same function in `src/mnist_cnn/cli.py`. The package is installed from the src layout; no `PYTHONPATH` or `sys.path` edits are needed. The sample root `main.py` has been replaced by these entry points.

For dependency changes, use `uv add PACKAGE` (or `uv add --dev PACKAGE`), then review changes to both `pyproject.toml` and `uv.lock`. Use `uv sync --locked --no-dev` for a runtime-only environment. A future Python patch upgrade should explicitly update `.python-version` and rerun checks.

The `mlflow` extra and `notebooks` group are defined and locked, but are not installed by the default command. They are for later implementation stages:

```bash
uv sync --locked --extra mlflow
uv sync --locked --group notebooks
```

If uv warns that `VIRTUAL_ENV` points to another project, deactivate that environment first. uv uses this project's `.venv` by default; do not use `--active` to install into another project's environment.

## Tests

Routine tests use tiny in-memory image fixtures and do not download MNIST or connect to MLflow:

```bash
uv run --locked pytest -q
```

After data preparation, include the real cached-data check:

```bash
uv run --locked pytest -q --run-integration
```

The integration check reads `data/` with downloads disabled and iterates all three splits. It checks dataset contracts separately from model accuracy; the suite also includes synthetic Lightning training/resume/test checks.

See [usage](usage.md) and the [stage 1–3 review](stage-1-3-review.md).
