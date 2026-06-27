# Quick Semantic Segmentation Study Implementation Plan

## Summary

Build a small PyTorch training study for the dataset under `dataset/`, containing paired RGB images and RGB masks. Masks use two clean colors from `labelmap.txt`: black background and white `Imprint`.

Use a 70/30 fixed holdout split, then run 5-fold cross validation inside the 70% development split. Train UNet, DeepLabV3+, and PSPNet with Hydra configs, PyTorch Lightning, TensorBoard logging, Optuna tuning, mIoU evaluation, saved best checkpoints, and full-dataset inference outputs per model.

## Implementation Changes

- Create a lightweight Python project scaffold with `requirements.txt`, Hydra configs, a `src/nazarai_seg/` package, and module entry points for training, inference, and result summarization.
- Use `segmentation-models-pytorch` for `Unet`, `DeepLabV3Plus`, and `PSPNet` with `resnet34` ImageNet encoders and two output classes.
- Train all model parameters for each segmentation model.
- Decode masks by exact RGB mapping: `(0, 0, 0) -> 0` background and `(255, 255, 255) -> 1` Imprint.
- Use Albumentations for aspect-preserving resize, padding, scale, rotation, and flip augmentation.
- Use AdamW, standard segmentation losses, TorchMetrics mIoU, Optuna trial search, TensorBoard logs, checkpoint saving, and inference overlays.

## Test Plan

- Verify dataset pairing and mask decoding.
- Run a one-epoch smoke test for each model with one Optuna trial and one CV fold.
- Confirm TensorBoard logs and best checkpoints are created.
- Run inference on a small subset and then the full dataset.
- Confirm final comparison tables contain all three models and their best hyperparameters.

## Assumptions

- Use a new CUDA-capable virtual environment for the RTX 4060.
- Use `segmentation-models-pytorch` to keep model imports simple.
- Interpret `70/30 split with five-fold cross validation` as 70% development plus 30% final holdout, with 5-fold CV inside the 70%.
- Treat this as a quick study, so avoid heavyweight packaging, excessive validation framework code, assertions, and warnings.
