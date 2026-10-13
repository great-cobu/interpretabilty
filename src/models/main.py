"""Top-level command-line entry point for the models package."""

import sys

from models.MNIST.cli import main as mnist_main
from models.logistic.cli import main as logistic_main


def main(argv: list[str] | None = None) -> int:
    """Dispatch to either models.MNIST (default / --model mnist) or models.logistic (--model logistic)."""
    args = list(sys.argv[1:] if argv is None else argv)
    model_choice = "mnist"
    if len(args) >= 2 and args[0] == "--model":
        model_choice = args[1].lower()
        args = args[2:]

    if model_choice in {"logistic", "logreg"}:
        return logistic_main(args)
    return mnist_main(args)


if __name__ == "__main__":
    raise SystemExit(main())