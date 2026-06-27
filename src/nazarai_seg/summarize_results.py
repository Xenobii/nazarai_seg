from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Write a Markdown comparison table from study results."""
    results_dir = Path(cfg.paths.results_dir)
    csv_path = results_dir / "model_comparison.csv"
    markdown_path = results_dir / "model_comparison.md"
    rows = _read_rows(csv_path)
    lines = [
        "# Model Comparison",
        "",
        "| Model | Best CV mIoU | Holdout mIoU | Best Hyperparameters | Checkpoint |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        params = json.loads(row["best_params"])
        params_text = ", ".join(f"{key}={value}" for key, value in params.items())
        lines.append(
            f"| {row['model']} | {float(row['best_miou']):.5f} | "
            f"{float(row.get('test_miou', 0.0)):.5f} | {params_text} | "
            f"`{row['checkpoint_path']}` |"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {markdown_path}")


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    main()
