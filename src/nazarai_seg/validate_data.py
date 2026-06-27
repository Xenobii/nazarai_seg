from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from nazarai_seg.config_runner import run_with_config
from nazarai_seg.data import validate_dataset


def main(cfg: DictConfig) -> None:
    """Validate dataset pairing, dimensions, and mask colors."""
    for message in validate_dataset(
        dataset_root=Path(cfg.data.dataset_root),
        image_dir=cfg.data.image_dir,
        mask_dir=cfg.data.mask_dir,
    ):
        print(message)


if __name__ == "__main__":
    run_with_config(main)
