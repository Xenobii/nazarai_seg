from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def run_with_config(entrypoint: Callable[[DictConfig], None]) -> None:
    """Compose the repo config and call an entrypoint."""
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        cfg = compose(
            config_name="config",
            overrides=sys.argv[1:],
        )
    entrypoint(cfg)
