# NazarAI Segmentation Study

Quick semantic segmentation study for the dataset in `dataset/`.

The repo is intentionally simple. Run commands from the repo root:

```powershell
cd G:\Other\nazarai_seg
```

## 1. Local Conda Setup

Use the existing `cook` environment:

```powershell
conda activate cook
python -m pip install -r requirements.txt
```

`requirements.txt` does not install PyTorch. Your `cook` environment already has a working CUDA PyTorch stack. If you ever need to reinstall CUDA PyTorch in this env, use:

```powershell
python -m pip install -r requirements-torch-cu121.txt
```

Validate the dataset:

```powershell
python validate_data.py
```

Run a small smoke training job:

```powershell
python train.py "study.models=[unet]" study.n_trials=1 study.max_folds=1 training.max_epochs=1 data.batch_size=1
```

Run the full study:

```powershell
python train.py
python infer.py model_name=unet checkpoint_path=outputs/checkpoints/unet/best.ckpt
python infer.py model_name=deeplabv3plus checkpoint_path=outputs/checkpoints/deeplabv3plus/best.ckpt
python infer.py model_name=pspnet checkpoint_path=outputs/checkpoints/pspnet/best.ckpt
python summarize_results.py
```

## 2. Docker Setup

The Docker image assumes the dataset folder exists in the repo as:

```text
dataset/
  images/
  masks/
  labelmap.txt
```

Build the image:

```powershell
docker build -t nazarai-seg .
```

Validate the dataset inside the image:

```powershell
docker run --rm nazarai-seg python validate_data.py
```

Run a GPU smoke training job and keep outputs on the host:

```powershell
docker run --rm --gpus all -v "${PWD}\outputs:/workspace/outputs" nazarai-seg python train.py "study.models=[unet]" study.n_trials=1 study.max_folds=1 training.max_epochs=1 data.batch_size=1
```

Run the full study:

```powershell
docker run --rm --gpus all -v "${PWD}\outputs:/workspace/outputs" nazarai-seg python train.py
```

For Linux/macOS shells, replace the volume path with:

```bash
docker run --rm --gpus all -v "$(pwd)/outputs:/workspace/outputs" nazarai-seg python train.py
```

The default config uses `data.num_workers=0` because Docker's default shared memory can kill PyTorch DataLoader workers with large images. If you want to try worker processes, run Docker with a larger shared memory segment and override the config:

```powershell
docker run --rm --gpus all --shm-size=8g -v "${PWD}\outputs:/workspace/outputs" nazarai-seg python train.py data.num_workers=4
```

TensorBoard logs are written under `outputs/tensorboard/`.
