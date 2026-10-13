# Review: stages 5–6 — Lightning training and checkpoints

Implemented and verified on 2026-09-29 with the existing Python 3.14.7, PyTorch 2.14.0, and Lightning 2.6.6 environment. No package dependencies were added. The user authorized these two stages together; the next review point is before stage 7.

## Stage 5: What changed

`src/mnist_cnn/module.py` adds `MNISTClassifier`, a LightningModule wrapping the stage-4 CNN. Its training, validation, and test steps share the same cross-entropy calculation but maintain separate accuracy counters. Adam updates parameters only during training. Loss aggregation weights each batch by its example count; accuracy counts correct predictions across the whole phase, rather than averaging class accuracies or batch percentages.

`src/mnist_cnn/learning.py` orchestrates seeded Trainer execution and local CSV logging, then plots learning curves from the recorded epoch metrics. `train` is now a working CLI command. Validation controls early stopping and checkpoint selection. Positive optional batch limits make short demonstrations practical; the default still uses full training and validation loaders.

The CPU smoke command executed was:

```bash
uv run --locked mnist-cnn train --config configs/default.yaml --no-download --accelerator cpu --max-epochs 2 --limit-train-batches 20 --limit-val-batches 5 --output-dir runs/stage5-smoke
```

It performed 40 optimizer steps: 20 batches of 64 training examples in each epoch. Validation measured the first five ordered batches (320 examples) each epoch. Training loader shuffling can select different examples in each limited epoch.

| Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.5416 | 50.47% | 0.7267 | 80.62% |
| 2 | 0.6742 | 78.59% | 0.3396 | 90.31% |

Trainer fit time was approximately 0.76 seconds, excluding data setup, process startup, and plotting. These are limited smoke-run measurements, not the full baseline result. The curves were visually inspected. Artifacts are in `runs/stage5-smoke/`, including `logs/metrics.csv`, `learning_curves.png`, and `training_summary.json`.

## Stage 6: What changed

`src/mnist_cnn/checkpoints.py` adds checked checkpoint loading, saved-configuration reconstruction, hash manifests, and stable best/last filenames. Checkpoints contain model and optimizer states, epoch/global step, callback history, model/preprocessing configuration, exact split membership, and loader generator state. `MNISTDataModule` now implements Lightning's state save/restore hooks.

The `RunCheckpoint` callback retains the best validation score when resuming into a different directory, and writes the latest checkpoint every completed epoch even if its validation score did not improve. Resume requires the source checkpoint and its historical `best.ckpt` together. The source run is not modified.

Executed in a separate process:

```bash
uv run --locked mnist-cnn train --resume runs/stage5-smoke/checkpoints/last.ckpt --max-epochs 3 --output-dir runs/stage6-resume
```

The run restored epoch 2/global step 40 and completed epoch 3/global step 60. The new epoch measured training loss 0.3722, training accuracy 89.38%, validation loss 0.2130, and validation accuracy 92.19% on the same limited validation subset. Trainer fit time was approximately 0.42 seconds. The summary records the source checkpoint hash and explicit overrides. Its learning curves contain only the newly executed epoch.

`test --checkpoint` now reconstructs a fixed model in a fresh process and writes loss, accuracy, sample count, checkpoint hash, configuration, device, and environment to `metrics.json`. Standalone testing was verified on synthetic fixtures, including an uneven final batch. The official MNIST test set was not scored; that remains reserved for stage 9.

See [training instructions](training.md) for exact resume constraints and test commands, and the [Graphviz workflow](../diagrams/training-checkpoints.svg) for the relationship between best, last, resume, and test.

## Verification

The final regression command was:

```bash
uv run --locked pytest -q --run-integration
# 58 passed
```

The checks establish:

- Weighted cross-entropy and micro accuracy agree with hand-calculated expectations across uneven batches; repeating testing resets metric state.
- Nonfinite loss fails clearly.
- A saved checkpoint reproduces the original model's logits exactly on fixed CPU inputs.
- Resume restores optimizer progress: fixture training advances from global step 5 to 10 and all Adam step counters reach 10.
- Fresh interpreter processes can independently test and resume fixture checkpoints.
- Exact split membership survives checkpoint round-trip and resume.
- A deliberately worsening validation sequence keeps epoch 1 as best, still writes later `last.ckpt` files, and stops after the configured patience across resume.
- Model/preprocessing/optimizer overrides are rejected; allowed execution changes are recorded.
- Missing files, incompatible weights/metadata, and hash mismatches fail without reporting successful metrics.
- The earlier model, CLI, configuration, and real cached-data checks still pass.

Lightning emitted informational CPU/worker/checkpoint-directory warnings and a dependency deprecation warning; no checks failed. The nonempty checkpoint-directory warning on resume is expected because the historical best is copied into the new run before fitting.

## What to review

Start with `module.py`: trace an image batch through `training_step`, the shared loss/metric calculation, and `configure_optimizers`. Then compare `best.ckpt` and `last.ckpt` in the workflow diagram. Review `tests/test_learning.py` to see the proof that checkpoint selection and optimizer restoration work.

CPU behavior is verified. Resume restores training state but does not promise bitwise equivalence across hardware, library versions, or worker configurations; GPU RNG state is not currently persisted. Per-class evaluation reports, MLflow connectivity, the full accuracy target, and Marimo demonstrations remain for stages 7–10.

Named run directories in this review already contain evidence. Use a new directory or omit `--output-dir` when repeating a command.
