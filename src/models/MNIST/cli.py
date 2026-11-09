"""Unified command-line interface for models.MNIST."""

import argparse
import json
import sys
import yaml

from models.MNIST import __version__
from models.MNIST.config import VALID_ACTIVATIONS, ConfigError, config_dict, load_config, prepare_run_dir
from models.base import add_common_cli_arguments, load_checkpoint_base_config


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
        add_common_cli_arguments(sub)
        sub.add_argument("--dropout", dest="model.dropout", type=float)
        sub.add_argument("--activation", dest="model.activation", choices=VALID_ACTIVATIONS)
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
            base_cfg = load_checkpoint_base_config(resume or checkpoint_path)

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
