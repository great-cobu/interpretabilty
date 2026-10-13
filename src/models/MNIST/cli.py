"""Terminal entry points; heavy ML imports occur only for ML workflows."""

import argparse
import json
import sys

import yaml

from models.MNIST import __version__
from models.MNIST.config import ConfigError, config_dict, load_config
from models.MNIST.runs import create_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnist-cnn", description="MNIST CNN: supervised implementation, stages 1–6")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "check-config": "Validate and save resolved configuration (no ML computation)",
        "prepare-data": "Download or verify the MNIST cache",
        "inspect-data": "Inspect splits and batches; save exact split indices",
        "inspect-model": "Trace CNN shapes and check loss/gradients using synthetic inputs",
        "train": "Train with validation, or resume a local checkpoint",
        "test": "Measure a fixed checkpoint on the entire test split",
        "evaluate": "Planned: available after stage 7",
    }
    for name, description in descriptions.items():
        sub = commands.add_parser(name, help=description, description=description)
        if name == "train":
            sub.add_argument("--resume", help="Local last.ckpt (requires best.ckpt alongside it)")
        if name == "test":
            sub.add_argument("--checkpoint", required=True, help="Explicit local checkpoint to measure")
        sub.add_argument("--limit-train-batches", dest="trainer.limit_train_batches", type=int)
        sub.add_argument("--limit-val-batches", dest="trainer.limit_val_batches", type=int)
        sub.add_argument("--config", help="YAML configuration path; otherwise built-in defaults")
        sub.add_argument("--output-dir", help="New or empty directory; default: unique path in runs/")
        sub.add_argument("--data-dir", dest="data.directory")
        sub.add_argument("--download", dest="data.download", action=argparse.BooleanOptionalAction, default=None)
        sub.add_argument("--batch-size", dest="data.batch_size", type=int)
        sub.add_argument("--num-workers", dest="data.num_workers", type=int)
        sub.add_argument("--split-seed", dest="data.split_seed", type=int)
        sub.add_argument("--split-indices", dest="data.split_indices")
        sub.add_argument("--seed", dest="trainer.seed", type=int)
        sub.add_argument("--max-epochs", dest="trainer.max_epochs", type=int)
        sub.add_argument("--accelerator", dest="trainer.accelerator", choices=("auto", "cpu", "cuda", "mps"))
        sub.add_argument("--learning-rate", dest="optimizer.learning_rate", type=float)
        sub.add_argument("--dropout", dest="model.dropout", type=float)
        sub.add_argument("--tracking-enabled", dest="tracking.enabled", action=argparse.BooleanOptionalAction, default=None)
        sub.add_argument("--tracking-uri", dest="tracking.uri")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "evaluate":
        parser.exit(2, f"mnist-cnn: {command} is not implemented yet; see doc/spec/implementation-plan.md\n")
    path = args.pop("config")
    resume = args.pop("resume", None)
    checkpoint_path = args.pop("checkpoint", None)
    saved = None
    changed = {}
    overrides = {k: v for k, v in args.items() if v is not None}
    try:
        if resume or checkpoint_path:
            from models.MNIST.checkpoints import read_checkpoint, checkpoint_config
            saved = read_checkpoint(resume or checkpoint_path)
            config, changed = checkpoint_config(saved, path, overrides, resume=bool(resume))
        else:
            config = load_config(path, overrides)
        if config.tracking.enabled and command != "check-config":
            raise ConfigError("MLflow logging is planned for stage 8; use --no-tracking-enabled for local inspection/preparation")
        if command != "check-config":
            # Check the requested device before creating files or downloading data.
            from models.MNIST.data import resolve_accelerator
            resolve_accelerator(config.trainer.accelerator)
        config, output = create_run(config, command)
        if command == "check-config":
            print(yaml.safe_dump(config_dict(config), sort_keys=False), end="")
        elif command in {"train", "test"}:
            from models.MNIST.learning import train, test_checkpoint
            result = (train(config, output, resume=resume, saved=saved, overrides=changed)
                      if command == "train" else test_checkpoint(config, output, checkpoint_path, saved))
            print(json.dumps(result, indent=2))
        elif command == "inspect-model":
            from models.MNIST.model_inspection import inspect_model
            print(json.dumps(inspect_model(config, output), indent=2))
        else:
            from models.MNIST.workflows import inspect_data, prepare_data
            result = prepare_data(config) if command == "prepare-data" else inspect_data(config, output)
            print(json.dumps(result, indent=2))
        print(f"Saved run: {output}")
        return 0
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        # Configuration messages never echo user-supplied values; OS errors can.
        message = "Filesystem operation failed; check paths and permissions" if isinstance(exc, OSError) else str(exc)
        print(f"mnist-cnn: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
