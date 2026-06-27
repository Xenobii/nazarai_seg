from __future__ import annotations

from pathlib import Path
import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from nazarai_seg.data import build_transform, list_samples, make_loader
from nazarai_seg.lightning_module import SegmentationLitModule
from nazarai_seg.models import create_model


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run full-dataset inference for one trained checkpoint."""
    model_name = cfg.model_name or cfg.model.name
    checkpoint_path = Path(
        cfg.checkpoint_path or Path(cfg.paths.checkpoint_dir) / model_name / "best.ckpt"
    )
    limit = cfg.inference.limit
    output_dir = Path(cfg.paths.inference_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(
        dataset_root=Path(cfg.data.dataset_root),
        image_dir=cfg.data.image_dir,
        mask_dir=cfg.data.mask_dir,
    )
    if limit is not None:
        samples = samples[: int(limit)]

    transform = build_transform(
        is_train=False,
        max_size=cfg.data.max_size,
        pad_height=cfg.data.pad_height,
        pad_width=cfg.data.pad_width,
    )
    loader = make_loader(
        samples=samples,
        indices=list(range(len(samples))),
        transform=transform,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        canvas_height=cfg.data.pad_height,
        canvas_width=cfg.data.pad_width,
        shuffle=False,
        include_paths=True,
    )
    model = create_model(
        model_name=model_name,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=None,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    )
    lit_module = SegmentationLitModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=model,
    )
    lit_module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module.to(device)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = lit_module(images)  # (B, C, H, W)
            prediction = torch.argmax(logits, dim=1)[0].cpu()  # (H, W)
            stem = batch["stem"][0]
            image_path = Path(batch["image_path"][0])
            _save_outputs(
                image_path=image_path,
                prediction=prediction,
                output_dir=output_dir,
                stem=stem,
                height=cfg.data.pad_height,
                width=cfg.data.pad_width,
            )


def _save_outputs(
    image_path: Path,
    prediction: Tensor,
    output_dir: Path,
    stem: str,
    height: int,
    width: int,
) -> None:
    mask = prediction.numpy().astype(np.uint8) * 255  # (H, W)
    mask_path = output_dir / f"{stem}_mask.png"
    Image.fromarray(mask).save(mask_path)

    image = np.array(Image.open(image_path).convert("RGB"))
    image = _resize_to_canvas_image(
        image=image,
        height=height,
        width=width,
    )
    overlay = image.copy()
    imprint_pixels = mask > 0
    overlay[imprint_pixels] = (
        0.5 * overlay[imprint_pixels] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)
    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)


def _resize_to_canvas_image(
    image: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=image.dtype)  # (H, W, C)
    offset_y = (height - resized_height) // 2
    offset_x = (width - resized_width) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas


if __name__ == "__main__":
    main()
