from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .utils import read_json


def _load_rgb(path: str | Path, image_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    return TF.to_tensor(image)


def _load_gray(path: str | Path, image_size: tuple[int, int], mode: str = "L") -> torch.Tensor:
    image = Image.open(path).convert(mode)
    interp = Image.NEAREST if mode != "RGB" else Image.BILINEAR
    image = image.resize((image_size[1], image_size[0]), interp)
    return TF.to_tensor(image)


def _one_hot_parse(parse_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    if parse_tensor.dim() == 3 and parse_tensor.size(0) == 1:
        parse_tensor = parse_tensor[0]
    parse_ids = (parse_tensor * 255).round().long().clamp(min=0, max=num_classes - 1)
    return F.one_hot(parse_ids, num_classes=num_classes).permute(2, 0, 1).float()


def _expand_pose_channels(skeleton: torch.Tensor, densepose: torch.Tensor) -> torch.Tensor:
    dense = densepose if (densepose.dim() == 3 and densepose.size(0) == 3) else densepose.repeat(3, 1, 1)
    return torch.cat([skeleton.repeat(3, 1, 1), dense], dim=0)


def _extract_region(image: torch.Tensor, one_hot_parse: torch.Tensor, channel_ids: list[int]) -> torch.Tensor:
    mask = one_hot_parse[channel_ids].sum(dim=0, keepdim=True).clamp(0, 1)
    return image * mask


class ManifestTryOnDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.image_size = tuple(cfg.dataset.image_size)
        self.num_classes = int(cfg.dataset.parse_num_classes)
        self.garment_channel_ids = list(cfg.dataset.garment_channel_ids)
        self.limb_channel_ids = list(cfg.dataset.limb_channel_ids)
        manifest_path = cfg.dataset[f"{split}_manifest"]
        self.items = read_json(manifest_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        person = _load_rgb(item["person_image"], self.image_size)
        garment = _load_rgb(item["garment_image"], self.image_size)
        garment_mask = _load_gray(item["garment_mask"], self.image_size)
        parse_raw = _load_gray(item["parse"], self.image_size)
        skeleton = _load_gray(item["skeleton"], self.image_size)
        densepose_path = item.get("densepose")
        densepose = _load_rgb(densepose_path, self.image_size) if densepose_path else torch.zeros(3, *self.image_size)
        target = _load_rgb(item.get("target_image", item["person_image"]), self.image_size)

        parse_one_hot = _one_hot_parse(parse_raw, self.num_classes)
        limb_map = _extract_region(person, parse_one_hot, self.limb_channel_ids)
        garment_region = _extract_region(person, parse_one_hot, self.garment_channel_ids)
        pose_tensor = _expand_pose_channels(skeleton, densepose)

        return {
            "id": item.get("id", str(idx)),
            "person_image": person,
            "garment_image": garment,
            "garment_mask": garment_mask,
            "parse_map": parse_one_hot,
            "parse_ids": (parse_raw[0] * 255).round().long(),
            "skeleton_map": skeleton,
            "densepose_map": densepose,
            "pose_tensor": pose_tensor,
            "target_image": target,
            "limb_map": limb_map,
            "garment_region_gt": garment_region,
        }


def build_dataset(cfg, split: str):
    return ManifestTryOnDataset(cfg, split=split)
