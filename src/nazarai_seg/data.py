from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as alb
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from nazarai_seg.configs import Sample, SplitIndices


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
BACKGROUND_RGB = (0, 0, 0)
IMPRINT_RGB = (255, 255, 255)


def list_samples(dataset_root: Path, image_dir: str, mask_dir: str) -> list[Sample]:
    """Return image/mask samples paired by filename stem."""
    image_root = dataset_root / image_dir
    mask_root = dataset_root / mask_dir
    image_paths = sorted(
        path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    mask_paths = {
        path.stem: path
        for path in mask_root.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }

    samples: list[Sample] = []
    for image_path in image_paths:
        mask_path = mask_paths.get(image_path.stem)
        if mask_path is not None:
            samples.append(
                Sample(
                    stem=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

    return samples


def validate_samples(samples: list[Sample]) -> list[str]:
    """Return dataset validation messages without raising hard assertions."""
    messages: list[str] = []
    if not samples:
        messages.append("No paired image/mask samples found.")
        return messages

    missing_masks = [sample.stem for sample in samples if not sample.mask_path.exists()]
    if missing_masks:
        messages.append(f"Missing masks for {len(missing_masks)} images.")

    mismatched_sizes = 0
    invalid_colors = 0
    valid_colors = {BACKGROUND_RGB, IMPRINT_RGB}
    for sample in samples:
        with Image.open(sample.image_path) as image, Image.open(sample.mask_path) as mask:
            if image.size != mask.size:
                mismatched_sizes += 1

            colors = mask.convert("RGB").getcolors(maxcolors=4096)
            if colors is None:
                invalid_colors += 1
                continue

            mask_colors = {color for _, color in colors}
            if not mask_colors.issubset(valid_colors):
                invalid_colors += 1

    messages.append(f"Paired samples: {len(samples)}")
    messages.append(f"Size mismatches: {mismatched_sizes}")
    messages.append(f"Masks with unexpected colors: {invalid_colors}")
    return messages


def validate_dataset(
    dataset_root: Path,
    image_dir: str,
    mask_dir: str,
) -> list[str]:
    """Return validation messages for raw dataset files."""
    image_root = dataset_root / image_dir
    mask_root = dataset_root / mask_dir
    image_paths = sorted(
        path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    mask_paths = sorted(
        path for path in mask_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    image_stems = {path.stem for path in image_paths}
    mask_stems = {path.stem for path in mask_paths}
    samples = list_samples(
        dataset_root=dataset_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
    )
    messages = [
        f"Images: {len(image_paths)}",
        f"Masks: {len(mask_paths)}",
        f"Missing masks: {len(image_stems - mask_stems)}",
        f"Missing images: {len(mask_stems - image_stems)}",
    ]
    messages.extend(validate_samples(samples))
    return messages


def make_split_indices(
    sample_count: int,
    dev_fraction: float,
    seed: int,
) -> SplitIndices:
    """Create deterministic development and holdout split indices."""
    indices = list(range(sample_count))
    dev_indices, test_indices = train_test_split(
        indices,
        train_size=dev_fraction,
        random_state=seed,
        shuffle=True,
    )
    return SplitIndices(
        dev_indices=sorted(dev_indices),
        test_indices=sorted(test_indices),
    )


def make_cv_folds(
    dev_indices: list[int],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Create deterministic train/validation folds inside the development split."""
    splitter = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    folds: list[tuple[list[int], list[int]]] = []
    dev_array = np.array(dev_indices)
    for train_pos, val_pos in splitter.split(dev_array):
        train_indices = dev_array[train_pos].tolist()
        val_indices = dev_array[val_pos].tolist()
        folds.append((train_indices, val_indices))
    return folds


def build_transform(
    is_train: bool,
    max_size: int,
    pad_height: int,
    pad_width: int,
) -> alb.Compose:
    """Build image/mask transforms for training or evaluation."""
    transforms: list[alb.BasicTransform] = []

    if is_train:
        transforms.extend(
            [
                alb.Affine(
                    translate_percent=0.0,
                    scale=(0.85, 1.15),
                    rotate=(-20, 20),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    p=0.7,
                ),
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.2),
            ]
        )

    transforms.append(
        alb.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    )
    return alb.Compose(transforms)


def resize_to_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and mask into a fixed canvas while preserving aspect ratio."""
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    resized_mask = cv2.resize(
        mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    image_canvas = np.zeros((height, width, 3), dtype=image.dtype)  # (H, W, C)
    mask_canvas = np.zeros((height, width), dtype=mask.dtype)  # (H, W)
    offset_y = (height - resized_height) // 2
    offset_x = (width - resized_width) // 2
    image_canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized_image
    mask_canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized_mask
    return image_canvas, mask_canvas


def decode_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """Decode an RGB mask into class indices with shape `(H, W)`."""
    mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)  # (H, W)
    imprint_pixels = np.all(mask_rgb == np.array(IMPRINT_RGB, dtype=np.uint8), axis=-1)
    mask[imprint_pixels] = 1
    return mask


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset.

    Args:
        samples: Paired image/mask samples.
        transform: Albumentations transform applied to image and mask.
        include_paths: Include path metadata in returned samples.

    Returns:
        Dictionary containing `image` with shape `(C, H, W)` and `mask` with shape
        `(H, W)`.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform: Optional[alb.Compose],
        canvas_height: int,
        canvas_width: int,
        include_paths: bool = False,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.include_paths = include_paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        mask_rgb = np.array(Image.open(sample.mask_path).convert("RGB"))
        mask = decode_mask(mask_rgb)  # (H, W)
        image, mask = resize_to_canvas(
            image=image,
            mask=mask,
            height=self.canvas_height,
            width=self.canvas_width,
        )

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]  # (H, W)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # (C, H, W)
        mask_tensor = torch.from_numpy(mask).long()  # (H, W)
        item: dict[str, object] = {
            "image": image_tensor,
            "mask": mask_tensor,
        }

        if self.include_paths:
            item["stem"] = sample.stem
            item["image_path"] = str(sample.image_path)
            item["mask_path"] = str(sample.mask_path)

        return item


def make_loader(
    samples: list[Sample],
    indices: list[int],
    transform: alb.Compose,
    batch_size: int,
    num_workers: int,
    canvas_height: int,
    canvas_width: int,
    shuffle: bool,
    include_paths: bool = False,
) -> DataLoader:
    """Build a dataloader over selected sample indices."""
    dataset = SegmentationDataset(
        samples=samples,
        transform=transform,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        include_paths=include_paths,
    )
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
