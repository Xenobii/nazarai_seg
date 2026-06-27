from __future__ import annotations

from typing import Callable

import segmentation_models_pytorch as smp
import torch.nn as nn


MODEL_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "pspnet": smp.PSPNet,
}


def create_model(
    model_name: str,
    encoder_name: str,
    encoder_weights: str,
    in_channels: int,
    classes: int,
) -> nn.Module:
    """Create a segmentation model from segmentation-models-pytorch."""
    factory = MODEL_FACTORIES[model_name]
    return factory(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
