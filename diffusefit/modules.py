from __future__ import annotations

from typing import Iterable
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, act: bool = True):
        super().__init__()
        groups = min(32, max(1, out_ch // 4))
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels)
        self.conv2 = ConvGNAct(channels, channels, act=False)

    def forward(self, x):
        return F.silu(x + self.conv2(self.conv1(x)))


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(ConvGNAct(in_ch, out_ch, stride=2), ResidualBlock(out_ch))

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(ConvGNAct(in_ch + skip_ch, out_ch), ResidualBlock(out_ch))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SimpleEncoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 64):
        super().__init__()
        self.stem = nn.Sequential(ConvGNAct(in_ch, base), ResidualBlock(base))
        self.down1 = DownsampleBlock(base, base * 2)
        self.down2 = DownsampleBlock(base * 2, base * 4)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        return [s1, s2, s3]


class SimpleUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 64):
        super().__init__()
        self.enc = SimpleEncoder(in_ch, base)
        self.mid = nn.Sequential(ResidualBlock(base * 4), ResidualBlock(base * 4))
        self.up2 = UpsampleBlock(base * 4, base * 2, base * 2)
        self.up1 = UpsampleBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        s1, s2, s3 = self.enc(x)
        x = self.mid(s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out(x)


class AdaIN2d(nn.Module):
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(style_dim, channels)
        self.to_beta = nn.Linear(style_dim, channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        gamma = self.to_gamma(style).view(b, c, 1, 1)
        beta = self.to_beta(style).view(b, c, 1, 1)
        return ((x - mean) / std) * gamma + beta


class CrossAttentionFusion(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, strategy: str = "cross_attention"):
        super().__init__()
        self.strategy = strategy
        if strategy == "cross_attention":
            self.q = nn.Conv2d(channels, channels, 1)
            self.k = nn.Conv2d(channels, channels, 1)
            self.v = nn.Conv2d(channels, channels, 1)
            self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
            self.proj = nn.Conv2d(channels, channels, 1)
        elif strategy == "concat":
            self.proj = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1), nn.SiLU())
        elif strategy == "add":
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def forward(self, pose_feat: torch.Tensor, garment_feat: torch.Tensor) -> torch.Tensor:
        if self.strategy == "concat":
            return self.proj(torch.cat([pose_feat, garment_feat], dim=1))
        if self.strategy == "add":
            return self.proj(pose_feat + garment_feat)
        b, c, h, w = pose_feat.shape
        q = self.q(pose_feat).flatten(2).transpose(1, 2)
        k = self.k(garment_feat).flatten(2).transpose(1, 2)
        v = self.v(garment_feat).flatten(2).transpose(1, 2)
        out, _ = self.attn(q, k, v)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return self.proj(out)


class MLPStyleEncoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 64, style_dim: int = 256):
        super().__init__()
        self.encoder = SimpleEncoder(in_ch, base)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        style = self.head(feats[-1])
        return feats, style


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_ids: Iterable[int] = (3, 8, 17, 26)):
        super().__init__()
        try:
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        except Exception as exc:  # pragma: no cover - offline fallback
            warnings.warn(f'Falling back to randomly initialized VGG19 features: {exc}')
            model = vgg19(weights=None).features.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.layer_ids = set(layer_ids)

    def forward(self, x: torch.Tensor):
        feats = []
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.layer_ids:
                feats.append(x)
        return feats


class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch: int, latent_ch: int = 4, base: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            ConvGNAct(in_ch, base),
            DownsampleBlock(base, base * 2),
            DownsampleBlock(base * 2, base * 4),
            nn.Conv2d(base * 4, latent_ch, kernel_size=3, padding=1),
        )
        self.dec = nn.Sequential(
            ConvGNAct(latent_ch, base * 4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(base * 4, base * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(base * 2, base),
            nn.Conv2d(base, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
