from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import VGGFeatureExtractor


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self.vgg(pred)
        tgt_feats = self.vgg(target)
        return sum(F.l1_loss(a, b) for a, b in zip(pred_feats, tgt_feats))


class DiffuseFitLosses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambda_per = float(cfg.loss.lambda_per)
        self.lambda_vgg = float(cfg.loss.lambda_vgg)
        self.lambda_mask = float(cfg.loss.lambda_mask)
        self.lambda_sem = float(cfg.loss.lambda_sem)
        self.lambda_limb_recon = float(cfg.loss.lambda_limb_recon)
        self.lambda_limb_weight = float(cfg.loss.lambda_limb_weight)
        self.perceptual = PerceptualLoss()
        weights = torch.tensor(list(cfg.loss.sem_class_weights), dtype=torch.float32)
        self.register_buffer('sem_class_weights', weights)

    def stage1(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        gt_garment = batch['garment_region_gt']
        rec = F.l1_loss(outputs['warped_garment'], gt_garment) + F.l1_loss(outputs['warped_mask'], batch['garment_mask'])
        per = self.perceptual(outputs['warped_garment'], gt_garment)
        mask = F.l1_loss(outputs['warped_mask'], batch['garment_mask'])
        total = rec + self.lambda_per * per + self.lambda_mask * mask
        return {'loss': total, 'rec': rec, 'per': per, 'mask': mask}

    def stage2(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        sem = F.cross_entropy(outputs['layout_logits'], batch['parse_ids'].long(), weight=self.sem_class_weights.to(outputs['layout_logits'].device))
        return {'loss': self.lambda_sem * sem, 'sem': sem}

    def stage3(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        diffusion = F.mse_loss(outputs['noise_pred'], outputs['noise'])
        vgg = self.perceptual(outputs['pred_image'], batch['target_image'])
        return {'loss': diffusion + self.lambda_vgg * vgg, 'diffusion': diffusion, 'vgg': vgg}

    def limb(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        recon = F.l1_loss(outputs['limb_recon'], batch['limb_map'])
        weight_target = (batch['limb_map'].abs().sum(dim=1, keepdim=True) > 0).float()
        weight = F.binary_cross_entropy(outputs['limb_weight'], weight_target)
        total = self.lambda_limb_recon * recon + self.lambda_limb_weight * weight
        return {'loss': total, 'limb_recon': recon, 'limb_weight': weight}
