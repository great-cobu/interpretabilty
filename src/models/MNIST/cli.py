"""Unified command-line interface for models.MNIST."""

import argparse
import json
import sys
import yaml

from models.MNIST import __version__
from models.MNIST.config import ConfigError, config_dict, load_config, prepare_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnist-cnn", description="MNIST CNN CLI")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "check-config": "Validate and save resolved configuration",
        "prepare-data": "Download or verify the MNIST cache",
        "inspect-data": "Inspect splits and batches; save exact split indices",
        "inspect-model": "Trace CNN shapes and check loss/gradients using synthetic inputs",
        "train": "Train with validation, or resume a local checkpoint",
        "test": "Measure a fixed checkpoint on the entire test split",
    }
    for name, description in descriptions.items():
        sub = commands.add_parser(name, help=description, description=description)
        if name == "train":
            sub.add_argument("--resume", help="Local checkpoint (.ckpt) to resume from")
        if name == "test":
            sub.add_argument("--checkpoint", required=True, help="Explicit local checkpoint to measure")
        sub.add_argument("--limit-train-batches", dest="trainer.limit_train_batches", type=int)
        sub.add_argument("--limit-val-batches", dest="trainer.limit_val_batches", type=int)
        sub.add_argument("--config", help="YAML configuration path; otherwise built-in defaults")
        sub.add_argument("--output-dir", help="Output directory; default: unique path in runs/")
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
    path = args.pop("config")
    resume = args.pop("resume", None)
    checkpoint_path = args.pop("checkpoint", None)
    overrides = {k: v for k, v in args.items() if v is not None}

    try:
        base_cfg = None
        if (resume or checkpoint_path) and path is None:
            import torch
            ckpt_file = resume or checkpoint_path
            saved = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            base_cfg = saved.get("hyper_parameters", {}).get("config")
            if isinstance(base_cfg, dict):
                base_cfg = dict(base_cfg, output_dir=None)

        config = load_config(path, overrides, base=base_cfg)
        if command != "check-config":
            from models.MNIST.data import resolve_accelerator
            resolve_accelerator(config.trainer.accelerator)

        config, output = prepare_run_dir(config, command)
        if command == "check-config":
            print(yaml.safe_dump(config_dict(config), sort_keys=False), end="")
        elif command == "train":
            from models.MNIST.train import train
            print(json.dumps(train(config, output, resume=resume), indent=2))
        elif command == "test":
            from models.MNIST.evaluate import test_checkpoint
            print(json.dumps(test_checkpoint(config, output, checkpoint_path), indent=2))
        elif command == "inspect-model":
            from models.MNIST.model import inspect_model
            print(json.dumps(inspect_model(config, output), indent=2))
        elif command == "prepare-data":
            from models.MNIST.data import prepare_data
            print(json.dumps(prepare_data(config), indent=2))
        elif command == "inspect-data":
            from models.MNIST.data import inspect_data
            print(json.dumps(inspect_data(config, output), indent=2))
        print(f"Saved run: {output}")
        return 0
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"mnist-cnn: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
