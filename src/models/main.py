"""Top-level command-line entry point for the models package."""

import argparse
import sys

from models.MNIST.cli import main as mnist_main
from models.logistic.cli import main as logistic_main

VALID_MODEL_CHOICES = ("mnist", "mnist_cnn", "cnn", "logistic", "logreg")


def main(argv: list[str] | None = None) -> int:
    """Dispatch to either models.MNIST (default / --model mnist) or models.logistic (--model logistic)."""
    args = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    pre_parser.add_argument("--model", default="mnist")
    try:
        parsed, remaining = pre_parser.parse_known_args(args)
    except argparse.ArgumentError as exc:
        print(f"models: {exc}", file=sys.stderr)
        return 2

    model_choice = parsed.model.lower()
    if model_choice not in VALID_MODEL_CHOICES:
        print(
            f"models: unknown --model {parsed.model!r}; choose from {', '.join(VALID_MODEL_CHOICES)}",
            file=sys.stderr,
        )
        return 2

    if model_choice in {"logistic", "logreg"}:
        return logistic_main(remaining)
    return mnist_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
