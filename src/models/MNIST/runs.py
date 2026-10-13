"""Create protected run directories and persist resolved configuration."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from models.MNIST.config import Config, config_dict


def create_run(config: Config, command: str) -> tuple[Config, Path]:
    """Claim an empty directory; refuse overwrite, including concurrent claims."""
    path = Path(config.output_dir) if config.output_dir else Path("runs") / (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{command}-{uuid4().hex[:8]}"
    )
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output-dir")
    resolved = replace(config, output_dir=str(path.resolve()))
    # Exclusive creation also prevents another process from claiming the same run.
    with (path / "config.yaml").open("x") as stream:
        yaml.safe_dump(config_dict(resolved), stream, sort_keys=False)
    return resolved, path
