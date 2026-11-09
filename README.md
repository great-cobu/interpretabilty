# Mechanistic Interpretability, Riemannian Geometry & Declarative MLOps

A unified Python 3.14, PyTorch Lightning, Riemannian differential geometry, and SWI-Prolog MLOps project exploring **glass-box neural network interpretability** and the **Riemannian geometry of activation manifolds**.

---

## 1. Project Architecture

| Subsystem | Location | Description |
| :--- | :--- | :--- |
| **Empirical Models & Glass-Box Interpretability** | [`src/models/MNIST/`](src/models/MNIST/), [`src/models/logistic/`](src/models/logistic/), [`src/models/base.py`](src/models/base.py) | Deterministic 55k/5k/10k MNIST data pipeline (`MNISTDataModule`), `MNISTCNN` with configurable activations (`relu`, `softplus`, `smu`/`SMU`) and feature-map/Grad-CAM hooks, and convex full-batch L-BFGS `LogisticRegression` with exact spatial class templates $W_c$ and pixel attributions $x \odot W_c$. |
| **Riemannian & Information Geometry Theory** | [`theory/`](theory/) | Connects the Linear Representation Hypothesis, superposition, and feature interaction to the pullback Riemannian metric $g_{ij}(u) = (J_f^\top J_f)_{ij}$ and Levi-Civita Christoffel symbols $\Gamma^k_{ij}(u)$ via analytical $S^2$ charts and exact `torch.func.jacrev` second-order automatic differentiation. |
| **Declarative MLOps by Policy** | [`prolog/`](prolog/), [`docker/`](docker/), [`configs/generated/`](configs/generated/) | SWI-Prolog knowledge base (`prolog/config.pl`) and code generators that validate port policies, model-specific hyperparameters, and emit `docker-compose.yml`, `.env` files, and experiment YAML configurations for MariaDB, MLflow, RustFS, and the ML workspace container. |

---

## 2. Quick Start

### Install Dependencies & Run Tests

```bash
# Install root workspace dependencies and run unit tests
uv sync
uv run pytest

# Run the Riemannian geometry theory test suite
cd theory && uv run pytest
```

### Command-Line Interface

The project provides three CLI entry points (`models`, `mnist-cnn`, and `mnist-logistic`):

```bash
# Inspect synthetic CNN shapes & gradients (with C^infinity Smooth Maximum Unit activation)
uv run mnist-cnn inspect-model --accelerator cpu --activation SMU

# Fit convex Multinomial Logistic Regression to global optimality via full-batch L-BFGS
uv run mnist-logistic train --config configs/logistic.yaml --accelerator cpu

# Dispatch to either model via the top-level `models` CLI (--model can appear anywhere)
uv run models check-config --model logistic --config configs/logistic.yaml
uv run models inspect-model --model cnn --activation softplus --accelerator cpu
```

### Interactive Marimo Notebooks

```bash
# Empirical model notebooks
uv run marimo edit notebooks/mnist_demo.py
uv run marimo edit notebooks/logistic_demo.py

# Riemannian activation manifold steering demo (Option A)
uv run marimo edit notebooks/manifold_steering_demo.py

# Riemannian manifold & tangent-plane atlas notebooks
cd theory && uv run marimo edit notebooks/manifold_atlas.py
```

### Regenerating Prolog-Managed Configs & Docker Stack

All container orchestration files and generated experiment configs are derived from [`prolog/config.pl`](prolog/config.pl):

```bash
./scripts/gen_all.sh
```

---

## 3. Documentation Index

* **Container & MLOps Stack**: [`DOCKER_SETUP.md`](DOCKER_SETUP.md) and [`prolog/README.md`](prolog/README.md)
* **MNIST CNN Specification & Walkthroughs**: [`doc/md/models/MNIST/README.md`](doc/md/models/MNIST/README.md)
* **Manifold Steering Roadmap (Option A & Planned Option B LLM Steering)**: [`doc/md/models/MNIST/spec/manifold_steering_roadmap.md`](doc/md/models/MNIST/spec/manifold_steering_roadmap.md)
* **Riemannian Manifolds & Geometry Theory**: [`theory/README.md`](theory/README.md)
