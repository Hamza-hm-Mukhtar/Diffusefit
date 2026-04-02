from __future__ import annotations

import argparse
from torch.utils.data import DataLoader

from diffusefit.config import load_config
from diffusefit.datasets import build_dataset
from diffusefit.model import build_model
from diffusefit.trainers import DiffuseFitTrainer
from diffusefit.utils import load_checkpoint, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--extra-config', action='append', default=[])
    parser.add_argument('--override', action='append', default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, extra_paths=args.extra_config, overrides=args.override)
    seed_all(int(cfg.experiment.seed))
    train_loader = DataLoader(build_dataset(cfg, 'train'), batch_size=int(cfg.training.batch_size), shuffle=True, num_workers=int(cfg.experiment.num_workers))
    val_loader = DataLoader(build_dataset(cfg, 'val'), batch_size=max(1, int(cfg.training.batch_size) // 2), shuffle=False, num_workers=int(cfg.experiment.num_workers))
    model = build_model(cfg)
    trainer = DiffuseFitTrainer(cfg, model)
    stage1 = trainer.fit(train_loader, val_loader, stage='stage1')
    model.load_state_dict(load_checkpoint(stage1, map_location='cpu')['model'], strict=False)
    stage2 = trainer.fit(train_loader, val_loader, stage='stage2')
    model.load_state_dict(load_checkpoint(stage2, map_location='cpu')['model'], strict=False)
    trainer.fit(train_loader, val_loader, stage='stage3')


if __name__ == '__main__':
    main()
