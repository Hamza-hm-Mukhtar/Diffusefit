from __future__ import annotations

import torch
import torch.nn as nn

from .modules import SimpleUNet


class GarmentAwareParsingSynthesis(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = int(cfg.dataset.parse_num_classes)
        self.use_garment_replacement = bool(cfg.model.use_garment_replacement)
        self.net = SimpleUNet(in_ch=self.num_classes + 1 + int(cfg.model.pose_channels), out_ch=self.num_classes, base=int(cfg.model.base_channels))
        self.garment_channel_ids = list(cfg.dataset.garment_channel_ids)

    def replace_layout(self, src_parse: torch.Tensor, warped_mask: torch.Tensor, pose_tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_garment_replacement:
            return src_parse
        rep = src_parse.clone()
        rep[:, self.garment_channel_ids] = 0.0
        rep[:, self.garment_channel_ids[0]] = warped_mask[:, 0]
        pose_prior = pose_tensor[:, :1]
        rep[:, self.garment_channel_ids[0]] = torch.maximum(rep[:, self.garment_channel_ids[0]], pose_prior[:, 0])
        return rep

    def forward(self, src_parse: torch.Tensor, warped_mask: torch.Tensor, pose_tensor: torch.Tensor):
        replaced = self.replace_layout(src_parse, warped_mask, pose_tensor)
        x = torch.cat([replaced, warped_mask, pose_tensor], dim=1)
        logits = self.net(x)
        probs = torch.softmax(logits, dim=1)
        return {"replacement_map": replaced, "layout_logits": logits, "target_layout": probs}
