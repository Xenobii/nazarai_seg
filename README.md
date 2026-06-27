# NazarAI Segmentation Study

Quick semantic segmentation study for the dataset in `dataset/`.

## Setup

Create a fresh environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The requirements pin CUDA 12.1 PyTorch wheels for an RTX 4060. If your local CUDA driver cannot use these wheels, install the matching PyTorch build from the official PyTorch selector, then install the remaining packages.

## Smoke Checks

```powershell
python -m nazarai_seg.validate_data
python -m nazarai_seg.train "study.models=[unet]" study.n_trials=1 study.max_folds=1 training.max_epochs=1 data.batch_size=1
python -m nazarai_seg.infer model_name=unet checkpoint_path=outputs/checkpoints/unet/best.ckpt inference.limit=5
python -m nazarai_seg.summarize_results
```

## Full Study

```powershell
python -m nazarai_seg.train
python -m nazarai_seg.infer model_name=unet checkpoint_path=outputs/checkpoints/unet/best.ckpt
python -m nazarai_seg.infer model_name=deeplabv3plus checkpoint_path=outputs/checkpoints/deeplabv3plus/best.ckpt
python -m nazarai_seg.infer model_name=pspnet checkpoint_path=outputs/checkpoints/pspnet/best.ckpt
python -m nazarai_seg.summarize_results
```

TensorBoard logs are written under `outputs/tensorboard/`.
