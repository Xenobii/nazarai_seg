from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import optuna
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nazarai_seg.data import build_transform, make_loader
from nazarai_seg.lightning_module import SegmentationLitModule
from nazarai_seg.models import create_model


def train_one_fold(
    cfg: DictConfig,
    model_name: str,
    train_indices: list[int],
    val_indices: list[int],
    trial_params: dict[str, Any],
    fold_index: int,
    trial_number: int,
    samples: list[Any],
) -> tuple[float, Optional[str]]:
    """Train one CV fold and return best validation mIoU."""
    train_transform = build_transform(
        is_train=True,
        max_size=cfg.data.max_size,
        pad_height=cfg.data.pad_height,
        pad_width=cfg.data.pad_width,
    )
    val_transform = build_transform(
        is_train=False,
        max_size=cfg.data.max_size,
        pad_height=cfg.data.pad_height,
        pad_width=cfg.data.pad_width,
    )
    train_loader = make_loader(
        samples=samples,
        indices=train_indices,
        transform=train_transform,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        canvas_height=cfg.data.pad_height,
        canvas_width=cfg.data.pad_width,
        shuffle=True,
    )
    val_loader = make_loader(
        samples=samples,
        indices=val_indices,
        transform=val_transform,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        canvas_height=cfg.data.pad_height,
        canvas_width=cfg.data.pad_width,
        shuffle=False,
    )

    model = create_model(
        model_name=model_name,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    )
    lit_module = SegmentationLitModule(
        model=model,
        learning_rate=trial_params["learning_rate"],
        weight_decay=trial_params["weight_decay"],
        loss_name=trial_params["loss_name"],
        num_classes=cfg.model.classes,
        ce_weight=cfg.training.ce_weight,
        dice_weight=cfg.training.dice_weight,
    )

    checkpoint_dir = (
        Path(cfg.paths.checkpoint_dir)
        / model_name
        / f"trial_{trial_number}"
        / f"fold_{fold_index}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_miou",
        mode="max",
        save_top_k=cfg.training.save_top_k,
    )
    logger = TensorBoardLogger(
        save_dir=cfg.paths.tensorboard_dir,
        name=model_name,
        version=f"trial_{trial_number}_fold_{fold_index}",
    )
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=_resolve_precision(cfg.training.precision),
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_checkpointing=True,
    )
    trainer.fit(
        model=lit_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    best_score = checkpoint_callback.best_model_score
    score = float(best_score.detach().cpu()) if best_score is not None else 0.0
    return score, checkpoint_callback.best_model_path


def tune_model(
    cfg: DictConfig,
    model_name: str,
    samples: list[Any],
    folds: list[tuple[list[int], list[int]]],
    test_indices: list[int],
) -> dict[str, Any]:
    """Tune one model with Optuna over CV folds."""
    fold_limit = cfg.study.max_folds
    active_folds = folds[:fold_limit] if fold_limit is not None else folds

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                cfg.optuna.learning_rate.low,
                cfg.optuna.learning_rate.high,
                log=cfg.optuna.learning_rate.log,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                cfg.optuna.weight_decay.low,
                cfg.optuna.weight_decay.high,
                log=cfg.optuna.weight_decay.log,
            ),
            "loss_name": trial.suggest_categorical(
                "loss_name",
                list(cfg.optuna.loss_names),
            ),
        }
        fold_scores: list[float] = []
        checkpoint_paths: list[str] = []
        for fold_index, (train_indices, val_indices) in enumerate(active_folds):
            score, checkpoint_path = train_one_fold(
                cfg=cfg,
                model_name=model_name,
                train_indices=train_indices,
                val_indices=val_indices,
                trial_params=trial_params,
                fold_index=fold_index,
                trial_number=trial.number,
                samples=samples,
            )
            fold_scores.append(score)
            if checkpoint_path:
                checkpoint_paths.append(checkpoint_path)

        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("checkpoint_paths", checkpoint_paths)
        return sum(fold_scores) / len(fold_scores)

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.study.n_trials)
    best_trial = study.best_trial
    best_checkpoint = _copy_best_checkpoint(
        checkpoint_paths=best_trial.user_attrs.get("checkpoint_paths", []),
        fold_scores=best_trial.user_attrs.get("fold_scores", []),
        model_name=model_name,
        output_dir=Path(cfg.paths.checkpoint_dir),
    )
    test_miou = evaluate_checkpoint(
        cfg=cfg,
        model_name=model_name,
        checkpoint_path=best_checkpoint,
        samples=samples,
        test_indices=test_indices,
    )
    result = {
        "model": model_name,
        "best_miou": best_trial.value,
        "test_miou": test_miou,
        "best_params": best_trial.params,
        "fold_scores": best_trial.user_attrs.get("fold_scores", []),
        "checkpoint_path": str(best_checkpoint) if best_checkpoint is not None else "",
    }
    _write_model_result(cfg=cfg, result=result)
    return result


def write_study_results(cfg: DictConfig, results: list[dict[str, Any]]) -> None:
    """Write study-level CSV and JSON result files."""
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "model_comparison.csv"
    json_path = results_dir / "model_comparison.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "best_miou",
                "test_miou",
                "best_params",
                "fold_scores",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result["model"],
                    "best_miou": result["best_miou"],
                    "test_miou": result["test_miou"],
                    "best_params": json.dumps(result["best_params"]),
                    "fold_scores": json.dumps(result["fold_scores"]),
                    "checkpoint_path": result["checkpoint_path"],
                }
            )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def save_resolved_config(cfg: DictConfig) -> None:
    """Save the resolved Hydra config under outputs."""
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg))


def save_split_files(
    cfg: DictConfig,
    dev_indices: list[int],
    test_indices: list[int],
    folds: list[tuple[list[int], list[int]]],
) -> None:
    """Save development, holdout, and CV split indices."""
    splits_dir = Path(cfg.paths.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "folds": [
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
            }
            for train_indices, val_indices in folds
        ],
    }
    with (splits_dir / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(split_payload, file, indent=2)


def evaluate_checkpoint(
    cfg: DictConfig,
    model_name: str,
    checkpoint_path: Optional[Path],
    samples: list[Any],
    test_indices: list[int],
) -> float:
    """Evaluate a promoted checkpoint on the 30% holdout split."""
    if checkpoint_path is None or not checkpoint_path.exists():
        return 0.0

    transform = build_transform(
        is_train=False,
        max_size=cfg.data.max_size,
        pad_height=cfg.data.pad_height,
        pad_width=cfg.data.pad_width,
    )
    test_loader = make_loader(
        samples=samples,
        indices=test_indices,
        transform=transform,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        canvas_height=cfg.data.pad_height,
        canvas_width=cfg.data.pad_width,
        shuffle=False,
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
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=_resolve_precision(cfg.training.precision),
        logger=False,
        enable_checkpointing=False,
    )
    test_results = trainer.test(
        model=lit_module,
        dataloaders=test_loader,
        verbose=False,
    )
    if not test_results:
        return 0.0

    return float(test_results[0].get("test_miou", 0.0))


def _copy_best_checkpoint(
    checkpoint_paths: list[str],
    fold_scores: list[float],
    model_name: str,
    output_dir: Path,
) -> Optional[Path]:
    if not checkpoint_paths:
        return None

    best_index = 0
    if fold_scores:
        best_index = max(range(len(fold_scores)), key=fold_scores.__getitem__)
        best_index = min(best_index, len(checkpoint_paths) - 1)

    source_path = Path(checkpoint_paths[best_index])
    destination = output_dir / model_name / "best.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source_path.read_bytes()
    destination.write_bytes(data)
    return destination


def _resolve_precision(configured_precision: str) -> str:
    if torch.cuda.is_available():
        return configured_precision

    return "32-true"


def _write_model_result(cfg: DictConfig, result: dict[str, Any]) -> None:
    result_dir = Path(cfg.paths.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{result['model']}_result.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
