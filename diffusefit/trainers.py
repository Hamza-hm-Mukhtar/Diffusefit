from __future__ import annotations

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from tqdm import tqdm

from .losses import DiffuseFitLosses
from .metrics import RunningMetrics
from .utils import AverageMeter, ensure_dir, linear_decay_lr, save_checkpoint


def _move_batch(batch: dict, device: str):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class DiffuseFitTrainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.device = cfg.experiment.device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.losses = DiffuseFitLosses(cfg).to(self.device)
        self.amp = bool(cfg.experiment.amp and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.amp)
        self.out_dir = ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)

    def _make_optimizer(self, stage: str):
        b1, b2 = float(self.cfg.training.beta1), float(self.cfg.training.beta2)
        wd = float(self.cfg.training.weight_decay)
        if stage == 'stage1':
            return Adam(self.model.alignment.parameters(), lr=float(self.cfg.training.lr_stage1), betas=(b1, b2), weight_decay=wd)
        if stage == 'stage2':
            return Adam(self.model.parsing.parameters(), lr=float(self.cfg.training.lr_stage2), betas=(b1, b2), weight_decay=wd)
        if stage == 'stage3':
            params = list(self.model.limb.parameters()) + list(self.model.diffusion.parameters())
            return AdamW(params, lr=float(self.cfg.training.lr_stage3), betas=(b1, b2), weight_decay=wd)
        if stage == 'full':
            return AdamW(self.model.parameters(), lr=float(self.cfg.training.lr_stage3), betas=(b1, b2), weight_decay=wd)
        raise ValueError(stage)

    def _run_epoch(self, loader, optimizer, epoch: int, total_epochs: int, stage: str):
        self.model.train()
        if bool(self.cfg.training.lr_decay_after_half):
            base_lr = optimizer.param_groups[0]['lr']
            lr = linear_decay_lr(base_lr, epoch, total_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        meters = {name: AverageMeter(name) for name in ['loss', 'aux1', 'aux2', 'aux3']}
        pbar = tqdm(loader, desc=f'{stage} epoch {epoch+1}/{total_epochs}')
        for batch in pbar:
            batch = _move_batch(batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                if stage == 'stage1':
                    outputs = self.model.alignment(batch['garment_image'], batch['garment_mask'], batch['skeleton_map'], batch['densepose_map'], batch['pose_tensor'])
                    loss_dict = self.losses.stage1(outputs, batch)
                elif stage == 'stage2':
                    with torch.no_grad():
                        stage1 = self.model.alignment(batch['garment_image'], batch['garment_mask'], batch['skeleton_map'], batch['densepose_map'], batch['pose_tensor'])
                    outputs = self.model.parsing(batch['parse_map'], stage1['warped_mask'], batch['pose_tensor'])
                    loss_dict = self.losses.stage2(outputs, batch)
                elif stage == 'stage3':
                    with torch.no_grad():
                        stage1 = self.model.alignment(batch['garment_image'], batch['garment_mask'], batch['skeleton_map'], batch['densepose_map'], batch['pose_tensor'])
                        stage2 = self.model.parsing(batch['parse_map'], stage1['warped_mask'], batch['pose_tensor'])
                    limb = self.model.limb(batch['limb_map'], training_mask=True)
                    cond = self.model.build_condition_image(batch, stage1, stage2, limb)
                    outputs = self.model.diffusion(cond, batch['target_image'], batch['garment_image'])
                    loss_stage3 = self.losses.stage3(outputs, batch)
                    loss_limb = self.losses.limb(limb, batch)
                    loss_dict = {'loss': loss_stage3['loss'] + loss_limb['loss'], 'aux1': loss_stage3['diffusion'], 'aux2': loss_stage3['vgg'], 'aux3': loss_limb['loss']}
                else:
                    outputs = self.model(batch)
                    loss_stage1 = self.losses.stage1(outputs['stage1'], batch)
                    loss_stage2 = self.losses.stage2(outputs['stage2'], batch)
                    loss_stage3 = self.losses.stage3(outputs['stage3'], batch)
                    loss_limb = self.losses.limb(outputs['limb'], batch)
                    loss_dict = {'loss': loss_stage1['loss'] + loss_stage2['loss'] + loss_stage3['loss'] + loss_limb['loss'], 'aux1': loss_stage1['loss'], 'aux2': loss_stage2['loss'], 'aux3': loss_stage3['loss'] + loss_limb['loss']}
            self.scaler.scale(loss_dict['loss']).backward()
            if float(self.cfg.training.grad_clip_norm) > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.grad_clip_norm))
            self.scaler.step(optimizer)
            self.scaler.update()
            meters['loss'].update(float(loss_dict['loss'].item()))
            for key in ('aux1', 'aux2', 'aux3'):
                if key in loss_dict:
                    meters[key].update(float(loss_dict[key].item()))
            pbar.set_postfix({k: f'{v.avg:.4f}' for k, v in meters.items() if v.count > 0})
        return {k: v.avg for k, v in meters.items() if v.count > 0}

    @torch.no_grad()
    def evaluate(self, loader, steps: int | None = None):
        self.model.eval()
        metrics = RunningMetrics(device=self.device, compute_fid=bool(self.cfg.evaluation.compute_fid))
        for batch in tqdm(loader, desc='evaluate'):
            batch = _move_batch(batch, self.device)
            preds = self.model.generate(batch, steps=steps)
            metrics.update(preds['image'], batch['target_image'], garment=batch['garment_image'], mask=preds['warped_mask'])
        return metrics.compute()

    def fit(self, train_loader, val_loader, stage: str = 'full'):
        optimizer = self._make_optimizer(stage)
        total_epochs = int(self.cfg.training[f'epochs_{stage}']) if stage in ('stage1', 'stage2', 'stage3') else int(self.cfg.training.epochs_stage3)
        best_key = float('inf')
        best_path = self.out_dir / stage / 'best.pt'
        ensure_dir(best_path.parent)
        for epoch in range(total_epochs):
            train_metrics = self._run_epoch(train_loader, optimizer, epoch, total_epochs, stage)
            val_metrics = self.evaluate(val_loader, steps=int(self.cfg.diffusion.inference_steps))
            monitor = val_metrics.get('fid', 0.0)
            state = {'epoch': epoch, 'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'train': train_metrics, 'val': val_metrics, 'cfg': self.cfg}
            save_checkpoint(state, self.out_dir / stage / f'epoch_{epoch+1:03d}.pt')
            if monitor < best_key:
                best_key = monitor
                save_checkpoint(state, best_path)
        return best_path
