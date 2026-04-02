from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

from .modules import ConvAutoencoder


class CLIPGarmentEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = False):
        super().__init__()
        self.model = None
        self.fallback = None
        try:
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
            if not trainable:
                for p in self.model.parameters():
                    p.requires_grad_(False)
            self.hidden_size = self.model.config.hidden_size
        except Exception as exc:  # pragma: no cover - offline fallback
            warnings.warn(f'Falling back to lightweight garment encoder because CLIP could not be loaded: {exc}')
            self.hidden_size = 768
            self.fallback = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, self.hidden_size, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(1),
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - mean) / std

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            pixel_values = self._normalize(image)
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            return outputs.last_hidden_state
        pooled = self.fallback(image).flatten(1).unsqueeze(1)
        return pooled


class ZeroGarmentEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.zeros(image.size(0), 1, self.hidden_size, device=image.device, dtype=image.dtype)


class DiffusionTryOnDecoder(nn.Module):
    def __init__(self, cfg, cond_channels: int):
        super().__init__()
        latent_channels = int(cfg.model.latent_channels)
        self.image_ae = ConvAutoencoder(3, latent_ch=latent_channels, base=int(cfg.model.base_channels))
        self.cond_ae = ConvAutoencoder(cond_channels, latent_ch=latent_channels, base=int(cfg.model.base_channels))
        self.garment_encoder = CLIPGarmentEncoder(cfg.model.clip_model_name, trainable=bool(cfg.model.clip_trainable)) if bool(cfg.model.use_clip_semantics) else ZeroGarmentEncoder()
        cross_dim = int(getattr(self.garment_encoder, 'hidden_size', 768))
        blocks = tuple(["CrossAttnDownBlock2D"] * (len(cfg.diffusion.block_out_channels) - 1) + ["DownBlock2D"])
        up_blocks = tuple(["UpBlock2D"] + ["CrossAttnUpBlock2D"] * (len(cfg.diffusion.block_out_channels) - 1))
        self.unet = UNet2DConditionModel(
            sample_size=None,
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            down_block_types=blocks,
            up_block_types=up_blocks,
            block_out_channels=tuple(cfg.diffusion.block_out_channels),
            layers_per_block=int(cfg.diffusion.layers_per_block),
            cross_attention_dim=cross_dim,
            attention_head_dim=8,
            norm_num_groups=8,
        )
        self.train_scheduler = DDPMScheduler(num_train_timesteps=int(cfg.diffusion.train_timesteps), beta_schedule=str(cfg.diffusion.beta_schedule))

    def _build_infer_scheduler(self, steps: int):
        sched = DDIMScheduler.from_config(self.train_scheduler.config) if False else DPMSolverMultistepScheduler.from_config(self.train_scheduler.config)
        sched.set_timesteps(steps)
        return sched

    def _predict_x0(self, noisy: torch.Tensor, noise_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas = self.train_scheduler.alphas_cumprod.to(noisy.device)[timesteps].view(-1, 1, 1, 1)
        return (noisy - (1.0 - alphas).sqrt() * noise_pred) / alphas.sqrt().clamp_min(1e-6)

    def forward(self, condition_image: torch.Tensor, target_image: torch.Tensor, garment_image: torch.Tensor):
        z = self.image_ae.encode(target_image)
        cond = self.cond_ae.encode(condition_image)
        tokens = self.garment_encoder(garment_image)
        noise = torch.randn_like(z)
        timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps, (z.size(0),), device=z.device, dtype=torch.long)
        noisy = self.train_scheduler.add_noise(z, noise, timesteps)
        model_input = torch.cat([noisy, cond], dim=1)
        noise_pred = self.unet(model_input, timesteps, encoder_hidden_states=tokens).sample
        x0 = self._predict_x0(noisy, noise_pred, timesteps)
        pred_image = self.image_ae.decode(x0)
        return {"target_latent": z, "cond_latent": cond, "noise": noise, "timesteps": timesteps, "noise_pred": noise_pred, "pred_image": pred_image, "tokens": tokens}

    @torch.no_grad()
    def generate(self, condition_image: torch.Tensor, garment_image: torch.Tensor, steps: int = 50) -> torch.Tensor:
        cond = self.cond_ae.encode(condition_image)
        tokens = self.garment_encoder(garment_image)
        scheduler = self._build_infer_scheduler(steps)
        sample = torch.randn_like(cond)
        for t in scheduler.timesteps:
            model_input = torch.cat([sample, cond], dim=1)
            t_batch = torch.full((sample.size(0),), int(t), device=sample.device, dtype=torch.long)
            noise_pred = self.unet(model_input, t_batch, encoder_hidden_states=tokens).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample
        return self.image_ae.decode(sample)
