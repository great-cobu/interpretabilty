# Manifold Steering Roadmap: Option A (Implemented) & Option B (Next Phase)

## 1. Option A — MNIST Activation Manifold Steering (Implemented)

* **Module**: [`src/models/MNIST/steering.py`](../../../../src/models/MNIST/steering.py)
* **Interactive Marimo Demo**: [`notebooks/manifold_steering_demo.py`](../../../../notebooks/manifold_steering_demo.py)
* **Core Capabilities**:
  * Orthonormal 2D local affine chart $\varphi^{-1}(u^1, u^2) = x_A + u^1 e_1 + u^2 e_2$ anchored at three MNIST validation digits $(x_A, x_B, x_C)$.
  * Exact pullback activation metric $g_{ij}(u) = (J_h(u)^\top J_h(u))_{ij} + \lambda \delta_{ij}$ and pullback Fisher Information Metric $g^F_{ij}(u) = \mathbb{E}_{p(y \mid u)}[\partial_i \log p \, \partial_j \log p] + \lambda \delta_{ij}$ via `torch.func.jacrev`.
  * Second-order automatic differentiation for Levi-Civita Christoffel symbols $\Gamma^k_{ij}(u)$ on $C^\infty$-smooth (`SMU` and `Softplus`) CNNs.
  * Both **Initial-Value Geodesic Shooting (RK4)** and **Boundary-Value Geodesic Path Energy Minimization**, contrasted directly against Euclidean linear steering.

---

## 2. Option B — LLM Residual-Stream Manifold Steering (Planned Next)

### Objective
Extend the Riemannian manifold steering framework from 2D image charts to **Transformer residual-stream activation manifolds** on Apple Silicon (M4 Max, 48 GB unified memory, MPS acceleration) using a local open-weight language model (e.g., `GPT-2`, `Llama-3.2-1B`, or `Qwen2.5-1.5B`).

### Planned Architecture
1. **Local Tangent Chart Construction ($U_\alpha \subset \mathbb{R}^{d_{\text{model}}}$)**:
   * Collect hidden states $h_\ell \in \mathbb{R}^{d_{\text{model}}}$ at an intermediate transformer layer $\ell$ across contrastive prompt pairs (e.g., *formal vs. informal*, *sycophantic vs. truthful*, or *French vs. English*).
   * Fit a low-dimensional local chart $\varphi_\alpha^{-1} : \mathbb{R}^m \to \mathbb{R}^{d_{\text{model}}}$ ($m \in \{2, 4, 8\}$) via local PCA, a smooth autoencoder chart, or contrastive steering directions $(e_1, \dots, e_m)$.
2. **Pullback Fisher Information Metric on Next-Token Logits**:
   * Map chart coordinates $u \in \mathbb{R}^m$ through the remaining transformer layers $\ell+1 \dots L$ and unembedding $W_U$ to next-token log-probabilities $\log p(y \mid \varphi_\alpha^{-1}(u))$.
   * Compute the pullback Fisher Information Metric $g^F_{ij}(u)$ and Amari's $e$- / $m$-connection or Levi-Civita Christoffel symbols $\Gamma^k_{ij}(u)$ via `torch.func.jacrev` (using top-$K$ active vocabulary tokens for sub-100ms Jacobian evaluation).
3. **Linear vs. Geodesic Activation Patching During Generation**:
   * Compare standard **Contrastive Activation Addition (CAA / RepE)** ($h_\ell \leftarrow h_\ell + \alpha v$) against **Riemannian Geodesic Steering** ($h_\ell \leftarrow \varphi_\alpha^{-1}(\gamma_{\text{geo}}(\alpha))$), measuring perplexity preservation, KL divergence, and steering efficacy in an interactive Marimo notebook.
