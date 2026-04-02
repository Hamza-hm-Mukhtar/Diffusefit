from __future__ import annotations

import random

import torch
import torch.nn as nn

from .modules import ConvGNAct, DownsampleBlock


def random_mask_limb(limb_map: torch.Tensor, low: float = 0.2, high: float = 0.75):
    b, c, h, w = limb_map.shape
    masks = []
    masked = limb_map.clone()
    for i in range(b):
        ratio = random.uniform(low, high)
        valid = (limb_map[i].abs().sum(dim=0, keepdim=True) > 0).float()
        rand = torch.rand_like(valid)
        drop = (rand < ratio).float() * valid
        masked[i] = limb_map[i] * (1.0 - drop)
        masks.append(drop)
    return masked, torch.stack(masks, dim=0)


class LimbAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base = int(cfg.model.base_channels)
        self.encoder = nn.Sequential(ConvGNAct(3, base), DownsampleBlock(base, base * 2), DownsampleBlock(base * 2, base * 4))
        self.recon = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(base * 4, base * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(base * 2, base),
            nn.Conv2d(base, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.weight_head = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, limb_map: torch.Tensor, training_mask: bool = True):
        masked, mask = random_mask_limb(limb_map) if training_mask else (limb_map, torch.zeros_like(limb_map[:, :1]))
        latent = self.encoder(masked)
        recon = self.recon(latent)
        weight = self.weight_head(latent)
        return {"limb_masked": masked, "limb_mask": mask, "limb_latent": latent, "limb_recon": recon, "limb_weight": weight}
