"""Allow ``python -m mnist_cnn`` with safe multiprocessing entry semantics."""

from models.MNIST.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
