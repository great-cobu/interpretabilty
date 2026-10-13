# Supervised implementation and testing plan

Status: Stages 1–6 implemented and verified; preliminary local demo available now; stages 7–10 planned.

Completed work is documented in the [stage 1–3 review](../instructions/stage-1-3-review.md), [stage 4 review](../instructions/stage-4-review.md), and [stage 5–6 review](../instructions/stage-5-6-review.md). The next review point is the preliminary local demo before stage 7.

This plan breaks the [project specification](mnist-cnn.md) into small, reviewable stages. Each stage pairs coding with focused verification so the behavior is demonstrated before more complexity is added.

## How we will work

Work on one stage at a time. Before coding, explain the concept, files to change, and expected behavior. Complete the stage's code, relevant checks, and instructions before presenting it for review. At the review point, show what changed, the exact commands used, their actual results, and any remaining limitations. Pause for the user's feedback before beginning the next stage unless the user has explicitly asked to combine or continue stages.

Fix failures within the current stage before calling it complete. Do not mark unexecuted checks as passing. Keep a short progress record in this document with the stage status and links to useful artifacts; do not commit generated datasets or run directories. Larger changes requested during review should be reflected in the specification and plan.

All commands use uv. Application code belongs in `src/mnist_cnn/`; operational guides belong in `doc/instructions/`. Each stage adds only the tests needed to establish its behavior. Routine tests use local synthetic fixtures, and real-data and live-MLflow checks are explicitly separate. Where a diagram helps learning, use Graphviz DOT and SVG files in `doc/diagrams/`.

## Stage overview

| Stage | Deliverable | Status |
| --- | --- | --- |
| 1 | Python environment, package, and CLI skeleton | Implemented and verified |
| 2 | Validated configuration and run directories | Implemented and verified |
| 3 | MNIST download, splits, and DataModule | Implemented and verified |
| 4 | CNN forward pass and loss | Implemented and verified |
| 5 | Lightning training and validation | Implemented and verified |
| 6 | Checkpoints, resume, and standalone testing | Implemented and verified |
| Demo | Preliminary local CLI walkthrough using stages 1–6 | Available now; presentation pending |
| 7 | Evaluation metrics and reports | Planned |
| 8 | MLflow tracking and container harness integration | Planned |
| 9 | Full baseline and reproducibility verification | Planned |
| 10 | Marimo demonstration and documentation walkthrough | Interactive demo implemented; final walkthrough pending stages 7–9 |

## Stage 1: Environment, package, and CLI skeleton

**Learn:** How uv, the Python interpreter, an installable src package, and a console command fit together.

**Code:** Verify and pin the latest stable Python 3.14 patch in `.python-version`; set the Python range, build configuration, dependencies, optional MLflow extra, and notebook group in `pyproject.toml`; generate `uv.lock`. Create the package, guarded module entry point, and `mnist-cnn` console script. Replace the sample root script. Add setup instructions. Record available MLflow container configuration early, without requiring the service for local stages.

**Verify:** Perform a locked installation; check the interpreter version and imports of the core stack. Run both `uv run mnist-cnn --help` and `uv run python -m mnist_cnn --help`. Confirm imports need no path manipulation and the core environment does not require Marimo. Check that `data/` and generated outputs are ignored by Git. Commands not yet implemented must not pretend to succeed.

**Review:** Show the small package tree, explain the entry point, and demonstrate the help output. Stop here before adding configuration behavior.

## Stage 2: Configuration and run directories

**Learn:** How defaults, YAML, and CLI arguments become one validated configuration.

**Code:** Implement configuration parsing, documented overrides, and output-directory creation in reusable package functions. Save sanitized resolved configuration. Reject unknown keys, invalid values, and accidental overwrite. Document the configuration fields and precedence.

**Verify:** Check representative valid and invalid configurations, CLI precedence, output protection, and removal of secrets from saved metadata. These checks use temporary directories and do not download data or train.

**Review:** Walk through one overridden configuration and one clear failure message. Explain which settings change model behavior and which only change execution.

## Stage 3: MNIST data pipeline

**Learn:** The difference between downloading a dataset, creating splits, transforming samples, and batching them.

**Code:** Implement `MNISTDataModule` and the `prepare-data` command. Cache MNIST in configurable `data/`, implement deterministic 55,000/5,000 training/validation indices, preserve the separate 10,000-example test set, and save split indices. Implement normalization and stage-appropriate loaders. Document data storage and offline reuse.

**Verify:** With local fixtures, check deterministic split membership, disjointness, repeated setup, batch shape, dtype, label range, and inclusion of partial batches. Separately download real MNIST once and verify official split counts and cached loading with downloads disabled. Verify downloaded files remain ignored by Git.

**Review:** Inspect a batch and saved indices. Use a Graphviz data-flow diagram if useful to show why validation and test samples never contribute training gradients.

## Stage 4: CNN and loss

**Learn:** How convolutions, pooling, flattening, and linear layers transform an image into ten logits.

**Code:** Implement the plain PyTorch CNN from the specification, keeping Lightning orchestration for the next stage. Make the forward contract explicit and document logits, argmax, softmax, and cross-entropy.

**Verify:** Run synthetic inputs through the network at several batch sizes; check `[N, 10]` finite output, a finite cross-entropy loss, and finite gradients reaching trainable layers. No full training run is needed.

**Review:** Trace layer shapes and explain why softmax is excluded from the training loss input. Add a Graphviz shape diagram if it helps the walkthrough.

## Stage 5: Lightning training and validation

**Learn:** How a LightningModule defines learning behavior while the Trainer manages execution; how validation differs from training.

**Code:** Add the Lightning module, optimizer, separate phase metrics, seeded Trainer, CSV logging, and `train` workflow. Wire up the specified callbacks and best/last checkpoint outputs. Produce learning curves through package functions used by the CLI. Keep MLflow integration for stage 8.

**Verify:** Run a short CPU integration test with synthetic fixtures to exercise gradient updates, validation, and metric reset/aggregation. Check weighted loss on unequal batch sizes and correct handling of nonfinite loss. Run a short real-data smoke training session, clearly labeled as a smoke run rather than a model-quality result.

**Review:** Inspect training/validation metrics and curves. Explain optimizer steps, epochs, and early stopping. Record the exact smoke-run configuration and duration.

## Stage 6: Checkpoints, resume, and standalone testing

**Learn:** The difference between restoring model weights for inference and restoring full training state to continue optimization.

**Code:** Complete checkpoint metadata and hashing, resume from `last.ckpt`, and the standalone `test` command. Recover architecture and preprocessing in a fresh process. Preserve original split and model settings on resume, record allowed overrides, and use a new output directory.

**Verify:** Test checkpoint round-trip logits on fixed CPU inputs; verify resume advances restored epoch/global-step and optimizer state. Check best-checkpoint selection against validation metrics. Exercise fresh-process testing with fixtures and failures for missing or incompatible checkpoints. Reserve the official test-set performance measurement for stage 9.

**Review:** Load a checkpoint without a preceding in-memory training session, then demonstrate a short resume. Explain why final evaluation uses the validation-selected checkpoint.

## Preliminary local demo: review stages 1–6 now

Use the commands and acceptance criteria in the [preliminary demo milestone](mnist-cnn.md#preliminary-demo-milestone-available-after-stage-6). Present data and CNN inspection, limited CPU training, local learning curves, checkpoint hashes, and fresh-process resume. Reuse recorded artifacts where appropriate or create new output directories for a live run. Record actual results and distinguish historical evidence from the live run.

No stage-7 evaluation, stage-8 MLflow service, stage-9 accuracy result, or stage-10 Marimo notebook is required for this review. Keep the official test-set measurement for stage 9; existing synthetic-fixture tests establish standalone testing behavior. Label limited runs as smoke demonstrations and explain the remaining stages. This milestone is available based on completed stages 1–6; mark it presented only after the walkthrough has actually occurred. Review the preliminary demo before starting stage 7 under the existing supervised workflow.

## Stage 7: Evaluation and reports

**Learn:** What overall accuracy can hide, how to read a confusion matrix, and how per-class metrics describe errors.

**Code:** Add the `evaluate` command, prediction exports, per-class and macro metrics, confusion matrices, misclassified-image output, and Markdown report. Reuse the test measurement and package inference/plotting functions.

**Verify:** Use hand-calculable predictions to test metric values, row/column orientation, undefined metrics, and partial batches. Check export ordering, unique sample indices, probability sums, and agreement between reports and predictions. Check both error-containing and zero-error reports, and visually inspect plots. Use fixtures or validation examples for development rather than repeated official test-set measurements.

**Review:** Explain one small confusion matrix by hand, then inspect a generated report alongside its prediction rows.

## Stage 8: MLflow and the existing container harness

**Learn:** How training jobs, the tracking server, and artifact storage interact; who owns a run and its completion status.

**Code:** Inspect the actual harness configuration and document reachable URLs, server/client compatibility, mounts, credentials, and launch mechanism. Add the tracking profile, logger adapter, explicit artifact uploads, run ownership, lineage, remote checkpoint retrieval, and failure handling. Reuse the existing containers. Add a Graphviz diagram of the verified topology and documented launch commands.

**Verify:** First test configuration, lifecycle, sanitization, and failure behavior with mocked service calls. Then run a small live smoke job against the existing containers. Check one run per invocation, CLI-owned versus harness-owned finalization, persisted local outputs, and artifact upload/download with hashes. In a separate process, evaluate and resume from remote checkpoint URIs using small fixtures. Confirm local-only commands still work without the service.

**Review:** Follow a job from the CLI to its MLflow metrics and downloadable checkpoint, then trace a linked evaluation run. Demonstrate that an upload failure preserves local files and reports failure.

## Stage 9: Full baseline and reproducibility

**Learn:** How to assess a completed experiment and distinguish measured evidence from expected performance.

**Code/run:** Freeze the baseline configuration and run full MNIST training through the harness. Evaluate the validation-selected checkpoint on the official test set and document the measured result. Complete remaining integration fixes exposed by this run.

**Verify:** Check the 98% accuracy target, all 10,000 unique prediction rows, confusion-matrix total, artifact hashes, split provenance, and report consistency. Confirm a fresh process reproduces checkpoint inference on the same CPU environment within tolerance. Verify locked setup and documented commands. If the accuracy target is missed, record the failure and investigate with training/validation evidence; do not repeatedly tune against test results.

**Review:** Present the actual environment, configuration, training curves, test metrics, run ID, and artifacts. State which acceptance criteria passed and which remain unresolved.

## Stage 10: Marimo demonstrations and final walkthrough

The [interactive demo](../instructions/notebooks.md) is implemented for existing stages 1–6. It provides cached validation previews, checkpoint inference, saved run metrics and curves, and model architecture. Marimo checks and local HTML export pass without training or downloads. Rich evaluation and MLflow views and the final acceptance review remain pending their corresponding stages.

**Learn:** How a notebook illustrates a reusable application without becoming its implementation.

**Code:** Add the Marimo `.py` demonstration importing package APIs and reading existing artifacts. Complete documentation indexes and setup, CLI, MLflow, diagram, and notebook instructions. Ensure examples match the implemented commands.

**Verify:** Export the demonstration to HTML from the CLI with local artifacts and no training/network side effects. Inspect the rendered demonstration and Graphviz diagrams. Check documentation links and walk through the documented command sequence. Confirm the application operates without the optional notebook environment.

**Review:** Show the demonstration and the equivalent CLI workflow. Review the specification's acceptance criteria and record any remaining limitations before declaring the project complete.
