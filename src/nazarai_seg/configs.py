from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Sample:
    """Image and mask paths for one segmentation sample."""

    stem: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class SplitIndices:
    """Fixed study split indices."""

    dev_indices: list[int]
    test_indices: list[int]
