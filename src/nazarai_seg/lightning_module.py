from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as func
import pytorch_lightning as pl
from torch import Tensor
from torchmetrics.classification import MulticlassJaccardIndex


class SegmentationLitModule(pl.LightningModule):
    """Lightning module for two-class semantic segmentation."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        loss_name: str,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True,
        )
        self.val_miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
        )
        self.test_miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
        )
        self.save_hyperparameters(ignore=["model"])

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits with shape `(B, C, H, W)`."""
        return self.model(images)

    def training_step(
        self,
        batch: dict[str, Tensor],
        batch_index: int,
    ) -> Tensor:
        """Run one training step."""
        logits = self(batch["image"])  # (B, C, H, W)
        loss = self._loss(logits, batch["mask"])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, Tensor],
        batch_index: int,
    ) -> None:
        """Run one validation step."""
        logits = self(batch["image"])  # (B, C, H, W)
        loss = self._loss(logits, batch["mask"])
        predictions = torch.argmax(logits, dim=1)  # (B, H, W)
        self.val_miou.update(predictions, batch["mask"])
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation mIoU."""
        miou = self.val_miou.compute()
        self.log(
            "val_miou",
            miou,
            prog_bar=True,
            on_epoch=True,
        )
        self.val_miou.reset()

    def test_step(
        self,
        batch: dict[str, Tensor],
        batch_index: int,
    ) -> None:
        """Run one test step."""
        logits = self(batch["image"])  # (B, C, H, W)
        predictions = torch.argmax(logits, dim=1)  # (B, H, W)
        self.test_miou.update(predictions, batch["mask"])

    def on_test_epoch_end(self) -> None:
        """Log test mIoU."""
        miou = self.test_miou.compute()
        self.log(
            "test_miou",
            miou,
            prog_bar=True,
            on_epoch=True,
        )
        self.test_miou.reset()

    def configure_optimizers(self) -> Any:
        """Build the optimizer."""
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        return torch.optim.AdamW(
            trainable_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _loss(self, logits: Tensor, masks: Tensor) -> Tensor:
        """Compute the configured segmentation loss."""
        if self.loss_name == "cross_entropy":
            return func.cross_entropy(logits, masks)

        if self.loss_name == "dice":
            return self.dice_loss(logits, masks)

        ce_loss = func.cross_entropy(logits, masks)
        dice_loss = self.dice_loss(logits, masks)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
