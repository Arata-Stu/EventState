"""TensorBoard logging with a compact console mirror."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Mandatory TensorBoard logger used by train and evaluation entry points."""

    def __init__(self, output_directory: str | Path, config_yaml: str | None = None) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_directory / "tensorboard"))
        if config_yaml:
            self.writer.add_text("run/resolved_config", f"```yaml\n{config_yaml}\n```", 0)
            (self.output_directory / "resolved_config.yaml").write_text(
                config_yaml, encoding="utf-8"
            )

    def scalars(self, values: Mapping[str, float | int], step: int) -> None:
        for name, value in values.items():
            numeric = float(value)
            if numeric == numeric:  # Skip NaN, which makes TensorBoard plots misleading.
                self.writer.add_scalar(name, numeric, step)

    def text(self, name: str, value: str, step: int = 0) -> None:
        self.writer.add_text(name, value, step)

    def console(self, stage: str, step: int, values: Mapping[str, Any]) -> None:
        printable = {key: value for key, value in values.items() if isinstance(value, (int, float))}
        summary = " ".join(f"{key}={float(value):.5g}" for key, value in printable.items())
        print(f"[{stage}] step={step} {summary}", flush=True)

    def write_metrics_json(self, name: str, values: Mapping[str, Any]) -> Path:
        path = self.output_directory / name
        path.write_text(json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
