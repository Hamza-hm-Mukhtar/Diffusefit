from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    if x.dim() == 4:
        x = x[0]
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_tensor_image(x: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(x.clamp(0, 1), str(path))


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def linear_decay_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    half = total_epochs // 2
    if epoch < half:
        return base_lr
    progress = (epoch - half) / max(1, total_epochs - half)
    return base_lr * (1.0 - progress)
