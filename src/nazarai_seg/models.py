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


def freeze_except_tail_modules(
    model: nn.Module,
    trainable_tail_modules: int,
) -> list[str]:
    """Freeze a model except the final parameter-owning modules."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    parameter_modules: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        direct_parameters = list(module.parameters(recurse=False))
        if direct_parameters:
            parameter_modules.append((name, module))

    selected_modules = parameter_modules[-trainable_tail_modules:]
    trainable_names: list[str] = []
    for name, module in selected_modules:
        trainable_names.append(name)
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = True

    return trainable_names
