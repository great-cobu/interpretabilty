"""Allow ``python -m models.logistic`` with safe multiprocessing entry semantics."""

from models.logistic.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
