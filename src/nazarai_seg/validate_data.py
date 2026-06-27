from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from nazarai_seg.data import validate_dataset


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate dataset pairing, dimensions, and mask colors."""
    for message in validate_dataset(
        dataset_root=Path(cfg.data.dataset_root),
        image_dir=cfg.data.image_dir,
        mask_dir=cfg.data.mask_dir,
    ):
        print(message)


if __name__ == "__main__":
    main()
