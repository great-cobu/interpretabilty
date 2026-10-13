"""Command-line interface entry point for the models package."""

import argparse
import sys
from models.MNIST.test import main as test_main
from models.MNIST.train import main as train_main


def main() -> None:
    """CLI dispatcher for models package tasks."""
    parser = argparse.ArgumentParser(
        description="Deep learning models toolkit.",
        usage="python -m models.main <command> [<args>]",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("train", help="Train CNN model on MNIST")
    subparsers.add_parser("test", help="Test trained CNN model on MNIST")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    command = sys.argv[1]
    # Re-slice sys.argv so sub-parsers receive remaining arguments
    sys.argv = [f"{sys.argv[0]} {command}"] + sys.argv[2:]

    if command == "train":
        train_main()
    elif command == "test":
        test_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()