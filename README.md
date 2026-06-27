# NazarAI Segmentation Study

Quick semantic segmentation study for the dataset in `dataset/`.

## Setup

Use the existing `cook` conda environment, then install dependencies:

```powershell
conda activate cook
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin CUDA 12.1 PyTorch wheels for an RTX 4060. If your local CUDA driver cannot use these wheels, install the matching PyTorch build from the official PyTorch selector inside the `cook` conda environment, then install the remaining packages.

## Smoke Checks

```powershell
python validate_data.py
python train.py "study.models=[unet]" study.n_trials=1 study.max_folds=1 training.max_epochs=1 data.batch_size=1
python infer.py model_name=unet checkpoint_path=outputs/checkpoints/unet/best.ckpt inference.limit=5
python summarize_results.py
```

## Full Study

```powershell
python train.py
python infer.py model_name=unet checkpoint_path=outputs/checkpoints/unet/best.ckpt
python infer.py model_name=deeplabv3plus checkpoint_path=outputs/checkpoints/deeplabv3plus/best.ckpt
python infer.py model_name=pspnet checkpoint_path=outputs/checkpoints/pspnet/best.ckpt
python summarize_results.py
```

TensorBoard logs are written under `outputs/tensorboard/`.
