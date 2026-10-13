# Review: stages 1–3

Implemented and verified on 2026-09-29 on macOS, Apple Silicon. These stages stop at data inspection; no CNN has been implemented or trained, and no test-set model accuracy has been measured.

## Stage 1: Environment and entry points

Python is pinned to 3.14.7. uv 0.12.19 produced the committed lockfile. A clean runtime-only installation succeeded with `uv sync --locked --no-dev` in an isolated ignored environment. Imports succeeded after changing the working directory to `/tmp`, with Marimo, MLflow, and pytest absent from that environment.

| Runtime package | Verified version |
| --- | --- |
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| lightning | 2.6.6 |
| torchmetrics | 1.9.0 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| numpy | 2.5.3 |

The console and module entry points produce identical help. Planned model commands fail clearly instead of pretending to train. Learn this stage by reading `pyproject.toml`, then `src/mnist_cnn/__main__.py` and `src/mnist_cnn/cli.py`.

## Stage 2: Configuration

This command demonstrated a CLI override of the default batch size:

```bash
uv run --locked mnist-cnn check-config --config configs/default.yaml --batch-size 32 --output-dir runs/stage2-config
```

Result: saved `runs/stage2-config/config.yaml` with batch size 32. Tests verify precedence, invalid values and types, unknown/duplicate keys, secret redaction, and rejection of occupied output directories. Learn this stage by comparing `configs/default.yaml` with `src/mnist_cnn/config.py` and `src/mnist_cnn/runs.py`.

## Stage 3: Data

Executed:

```bash
uv run --locked mnist-cnn prepare-data --config configs/default.yaml --output-dir runs/stage3-download
uv run --locked mnist-cnn inspect-data --config configs/default.yaml --no-download --accelerator cpu --output-dir runs/stage3-inspect
uv run --locked mnist-cnn inspect-data --no-download --accelerator cpu --num-workers 2 --split-indices runs/stage3-inspect/split_indices.json --output-dir runs/stage3-workers
```

The download populated `data/MNIST/raw/`. The later commands reused the cache with downloads disabled. Both single-process and two-worker loading succeeded, and saved split membership was reused.

| Split | Examples | First batch images | Labels |
| --- | ---: | --- | --- |
| Training | 55,000 | `[64, 1, 28, 28]`, float32 | `[64]`, int64 |
| Validation | 5,000 | `[64, 1, 28, 28]`, float32 | `[64]`, int64 |
| Test | 10,000 | `[64, 1, 28, 28]`, float32 | `[64]`, int64 |

The inspected normalized pixel range was approximately -0.4242 to 2.8215. Training/validation indices are saved in `runs/stage3-inspect/split_indices.json`; the complete batch summary is in the same directory as `data_summary.json`. Git ignore checks confirmed dataset files, run artifacts, and the environment are excluded, with no dataset files tracked.

Learn this stage from the [Graphviz diagram](../diagrams/data-flow.svg), then read `src/mnist_cnn/data.py` and `src/mnist_cnn/workflows.py`.

## Test results

```bash
uv run --locked pytest -q
# 34 passed, 1 integration check skipped

uv run --locked pytest -q --run-integration
# 35 passed
```

Offline tests use small PIL-image fixtures and temporary directories. The real-data integration check visits every example across all three splits, checks image/label contracts, verifies split disjointness and coverage, and reloads saved membership. The two-worker CLI check additionally verifies macOS multiprocessing entry behavior. The Graphviz rendering was visually inspected.

The explicitly named output directories above now contain evidence; rerunning the same commands with those exact names is intentionally rejected. Omit `--output-dir` for a new unique run or choose another empty directory.

## Remaining work and review questions

Stages 4–10 remain planned. The nearby `MLOPS` repository describes MLflow infrastructure, but its inspected documentation/container files did not establish this project's server URL or deployment configuration. No containers or external services were modified. The MLflow extra is locked; live server/client compatibility and credentials will be verified during stage 8 using the actual harness configuration.

Before stage 4, review whether you can explain how the CLI reaches a package function, how an override changes resolved configuration, and why validation membership stays fixed while training batch order shuffles. The next stage introduces the CNN forward pass and loss.
