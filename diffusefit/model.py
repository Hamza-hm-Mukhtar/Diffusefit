from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion import DiffusionTryOnDecoder
from .limb_generator import LimbAutoEncoder
from .parsing import GarmentAwareParsingSynthesis
from .pose_alignment import GPWarpReplacement, PoseAlignedGarmentAlignment


class DiffuseFitModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_target_semantic_layout = bool(cfg.model.use_target_semantic_layout)
        self.use_limb_aware_generator = bool(cfg.model.use_limb_aware_generator)
        self.use_limb_weight_maps = bool(cfg.model.use_limb_weight_maps)
        self.use_co_training = bool(cfg.model.use_co_training)
        self.alignment = GPWarpReplacement(cfg) if bool(cfg.model.use_gpwarp_variant) else PoseAlignedGarmentAlignment(cfg)
        self.parsing = GarmentAwareParsingSynthesis(cfg)
        self.limb = LimbAutoEncoder(cfg)
        cond_channels = 3 + 3 + 3 + 1 + (int(cfg.dataset.parse_num_classes) if self.use_target_semantic_layout else 0)
        self.diffusion = DiffusionTryOnDecoder(cfg, cond_channels=cond_channels)

    def build_condition_image(self, batch: dict, stage1: dict, stage2: dict, limb: dict) -> torch.Tensor:
        person = batch['person_image']
        garment = stage1['warped_garment']
        limb_recon = limb['limb_recon']
        limb_weight = limb['limb_weight'] if self.use_limb_weight_maps else torch.zeros_like(limb['limb_weight'])
        layout = stage2['target_layout'] if self.use_target_semantic_layout else torch.empty(person.size(0), 0, person.size(2), person.size(3), device=person.device)
        return torch.cat([person, garment, limb_recon, limb_weight, layout], dim=1)

    def forward(self, batch: dict) -> dict:
        stage1 = self.alignment(batch['garment_image'], batch['garment_mask'], batch['skeleton_map'], batch['densepose_map'], batch['pose_tensor'])
        parsing_input = stage1['warped_mask'].detach() if not self.use_co_training else stage1['warped_mask']
        stage2 = self.parsing(batch['parse_map'], parsing_input, batch['pose_tensor'])
        if self.use_limb_aware_generator:
            limb = self.limb(batch['limb_map'], training_mask=self.training)
        else:
            zero_weight = torch.zeros(batch['person_image'].size(0), 1, batch['person_image'].size(2), batch['person_image'].size(3), device=batch['person_image'].device)
            limb = {'limb_masked': batch['limb_map'], 'limb_mask': zero_weight, 'limb_latent': None, 'limb_recon': torch.zeros_like(batch['limb_map']), 'limb_weight': zero_weight}
        cond = self.build_condition_image(batch, stage1, stage2, limb)
        stage3 = self.diffusion(cond, batch['target_image'], batch['garment_image'])
        return {'stage1': stage1, 'stage2': stage2, 'limb': limb, 'stage3': stage3, 'condition_image': cond}

    @torch.no_grad()
    def generate(self, batch: dict, steps: int | None = None) -> dict:
        stage1 = self.alignment(batch['garment_image'], batch['garment_mask'], batch['skeleton_map'], batch['densepose_map'], batch['pose_tensor'])
        stage2 = self.parsing(batch['parse_map'], stage1['warped_mask'], batch['pose_tensor'])
        limb = self.limb(batch['limb_map'], training_mask=False) if self.use_limb_aware_generator else {'limb_recon': torch.zeros_like(batch['limb_map']), 'limb_weight': torch.zeros(batch['person_image'].size(0), 1, batch['person_image'].size(2), batch['person_image'].size(3), device=batch['person_image'].device)}
        cond = self.build_condition_image(batch, stage1, stage2, limb)
        image = self.diffusion.generate(cond, batch['garment_image'], steps=steps if steps is not None else int(self.cfg.diffusion.inference_steps))
        return {'warped_garment': stage1['warped_garment'], 'warped_mask': stage1['warped_mask'], 'target_layout': stage2['target_layout'], 'limb_recon': limb['limb_recon'], 'image': image}


def build_model(cfg):
    return DiffuseFitModel(cfg)
