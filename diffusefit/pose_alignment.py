from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import AdaIN2d, CrossAttentionFusion, MLPStyleEncoder, SimpleEncoder


def masked_channel_mean(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    return (image * mask).sum(dim=(2, 3), keepdim=True) / denom


def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=image.device),
        torch.linspace(-1.0, 1.0, w, device=image.device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    flow_grid = base_grid + flow.permute(0, 2, 3, 1)
    return F.grid_sample(image, flow_grid, mode="bilinear", padding_mode="border", align_corners=False)


class GarmentNormalizer(nn.Module):
    def forward(self, garment: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = masked_channel_mean(garment, mask)
        return torch.clamp(garment - mean + mask, 0.0, 1.0)


class PoseAlignedGarmentAlignment(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base = int(cfg.model.base_channels)
        style_dim = int(cfg.model.style_dim)
        heads = int(cfg.model.attention_heads)
        self.use_pose_guided_flow = bool(cfg.model.use_pose_guided_flow)

        self.normalizer = GarmentNormalizer()
        self.pose_encoder = SimpleEncoder(int(cfg.model.pose_channels), base)
        self.garment_encoder = MLPStyleEncoder(3, base, style_dim)
        self.adain = AdaIN2d(base * 4, style_dim)
        self.shape_decoder = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1),
        )
        self.garment_proj = nn.Conv2d(base * 4, base * 4, kernel_size=1)
        self.fusion = CrossAttentionFusion(base * 4, num_heads=heads, strategy=str(cfg.model.fusion_strategy))
        self.flow_head = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base * 4, 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.shape_head = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, garment, garment_mask, skeleton, densepose, pose_tensor=None):
        pose_tensor = pose_tensor if pose_tensor is not None else torch.cat([skeleton.repeat(1, 3, 1, 1), densepose], dim=1)
        normalized = self.normalizer(garment, garment_mask)
        pose_feats = self.pose_encoder(pose_tensor)
        garment_feats, style = self.garment_encoder(normalized)
        pose_feat = self.adain(pose_feats[-1], style)
        shape_feat = self.shape_decoder(pose_feat)
        garment_feat = self.garment_proj(garment_feats[-1])
        aligned_feat = self.fusion(shape_feat, garment_feat)
        if self.use_pose_guided_flow:
            flow_in = torch.cat([aligned_feat, pose_feat], dim=1)
            flow = self.flow_head(flow_in)
        else:
            flow = torch.zeros(garment.size(0), 2, garment_feat.size(2), garment_feat.size(3), device=garment.device, dtype=garment.dtype)
        flow = F.interpolate(flow, size=garment.shape[-2:], mode="bilinear", align_corners=False)
        warped_garment = warp_with_flow(garment, flow)
        warped_mask = warp_with_flow(garment_mask, flow).clamp(0, 1)
        shape_map = F.interpolate(self.shape_head(shape_feat), size=garment.shape[-2:], mode="bilinear", align_corners=False)
        return {
            "normalized_garment": normalized,
            "style_code": style,
            "pose_feat": pose_feat,
            "shape_feat": shape_feat,
            "aligned_feat": aligned_feat,
            "shape_map": shape_map,
            "flow": flow,
            "warped_garment": warped_garment,
            "warped_mask": warped_mask,
        }


class GPWarpReplacement(nn.Module):
    """
    GP-VTON-style local-flow replacement for ablation use.
    This is an interface-compatible approximation, not the official GP-VTON module.
    """
    def __init__(self, cfg):
        super().__init__()
        base = int(cfg.model.base_channels)
        self.pose_encoder = SimpleEncoder(int(cfg.model.pose_channels), base)
        self.garment_encoder = SimpleEncoder(4, base)
        self.num_parts = 4
        self.flow_head = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base * 4, self.num_parts * 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base * 4, self.num_parts, kernel_size=3, padding=1),
        )

    def forward(self, garment, garment_mask, skeleton, densepose, pose_tensor=None):
        pose_tensor = pose_tensor if pose_tensor is not None else torch.cat([skeleton.repeat(1, 3, 1, 1), densepose], dim=1)
        g_in = torch.cat([garment, garment_mask], dim=1)
        pose_feat = self.pose_encoder(pose_tensor)[-1]
        garment_feat = self.garment_encoder(g_in)[-1]
        fused = torch.cat([pose_feat, garment_feat], dim=1)
        flows = self.flow_head(fused)
        gates = self.gate_head(fused).softmax(dim=1)
        b, _, h, w = gates.shape
        flows = flows.view(b, self.num_parts, 2, h, w)
        flow = (flows * gates.unsqueeze(2)).sum(dim=1)
        flow = F.interpolate(flow, size=garment.shape[-2:], mode="bilinear", align_corners=False)
        warped_garment = warp_with_flow(garment, flow)
        warped_mask = warp_with_flow(garment_mask, flow).clamp(0, 1)
        return {
            "normalized_garment": garment,
            "style_code": None,
            "pose_feat": pose_feat,
            "shape_feat": garment_feat,
            "aligned_feat": fused,
            "shape_map": warped_mask,
            "flow": flow,
            "warped_garment": warped_garment,
            "warped_mask": warped_mask,
        }
