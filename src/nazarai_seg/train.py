from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig

from nazarai_seg.config_runner import run_with_config
from nazarai_seg.data import list_samples, make_cv_folds, make_split_indices
from nazarai_seg.training import (
    save_resolved_config,
    save_split_files,
    tune_model,
    write_study_results,
)


def main(cfg: DictConfig) -> None:
    """Run the Optuna cross-validation training study."""
    pl.seed_everything(cfg.seed, workers=True)
    dataset_root = Path(cfg.data.dataset_root)
    samples = list_samples(
        dataset_root=dataset_root,
        image_dir=cfg.data.image_dir,
        mask_dir=cfg.data.mask_dir,
    )
    split_indices = make_split_indices(
        sample_count=len(samples),
        dev_fraction=cfg.study.dev_fraction,
        seed=cfg.seed,
    )
    folds = make_cv_folds(
        dev_indices=split_indices.dev_indices,
        n_folds=cfg.study.n_folds,
        seed=cfg.seed,
    )

    save_resolved_config(cfg)
    save_split_files(
        cfg=cfg,
        dev_indices=split_indices.dev_indices,
        test_indices=split_indices.test_indices,
        folds=folds,
    )
    results = []
    for model_name in cfg.study.models:
        cfg.model.name = model_name
        result = tune_model(
            cfg=cfg,
            model_name=model_name,
            samples=samples,
            folds=folds,
            test_indices=split_indices.test_indices,
        )
        results.append(result)

    write_study_results(cfg=cfg, results=results)


if __name__ == "__main__":
    run_with_config(main)
