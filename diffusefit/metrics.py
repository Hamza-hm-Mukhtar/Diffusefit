from __future__ import annotations

import warnings

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import VGG19_Weights, vgg19


class GarmentFeatureSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        except Exception as exc:  # pragma: no cover - offline fallback
            warnings.warn(f'Falling back to randomly initialized VGG19 features for GFS: {exc}')
            model = vgg19(weights=None).features.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model[:18]

    def forward(self, shop: torch.Tensor, synth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.repeat(1, 3, 1, 1) if mask.size(1) == 1 else mask
            shop = shop * mask
            synth = synth * mask
        f1 = self.model(shop).mean(dim=(2, 3))
        f2 = self.model(synth).mean(dim=(2, 3))
        return F.cosine_similarity(f1, f2, dim=1).mean()


class RunningMetrics:
    def __init__(self, device: str = 'cpu', compute_fid: bool = True):
        self.device = device
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric = lpips.LPIPS(net='vgg').to(device)
        self.gfs_metric = GarmentFeatureSimilarity().to(device)
        self.compute_fid = compute_fid
        self.fid_metric = FrechetInceptionDistance(normalize=True).to(device) if compute_fid else None
        self.values = {'ssim': [], 'psnr': [], 'lpips': [], 'gfs': []}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, garment: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        pred = pred.clamp(0, 1).to(self.device)
        target = target.clamp(0, 1).to(self.device)
        self.values['ssim'].append(float(self.ssim_metric(pred, target).item()))
        self.values['psnr'].append(float(self.psnr_metric(pred, target).item()))
        self.values['lpips'].append(float(self.lpips_metric(pred * 2 - 1, target * 2 - 1).mean().item()))
        if garment is not None:
            self.values['gfs'].append(float(self.gfs_metric(garment.to(self.device), pred, mask.to(self.device) if mask is not None else None).item()))
        if self.fid_metric is not None:
            self.fid_metric.update((target * 255).byte(), real=True)
            self.fid_metric.update((pred * 255).byte(), real=False)

    def compute(self) -> dict[str, float]:
        out = {k: sum(v) / len(v) for k, v in self.values.items() if v}
        if self.fid_metric is not None:
            out['fid'] = float(self.fid_metric.compute().item())
        return out
