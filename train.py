from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(ROOT / "src"))

from nazarai_seg.config_runner import run_with_config
from nazarai_seg.train import main


if __name__ == "__main__":
    run_with_config(main)
