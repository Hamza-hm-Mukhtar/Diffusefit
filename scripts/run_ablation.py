from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from diffusefit.config import load_config
from diffusefit.datasets import build_dataset
from diffusefit.model import build_model
from diffusefit.trainers import DiffuseFitTrainer
from diffusefit.utils import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ablation', required=True)
    parser.add_argument('--stage', default='stage3', choices=['stage1', 'stage2', 'stage3', 'full'])
    parser.add_argument('--override', action='append', default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, extra_paths=[args.ablation], overrides=args.override)
    seed_all(int(cfg.experiment.seed))
    trainer = DiffuseFitTrainer(cfg, build_model(cfg))
    train_loader = DataLoader(build_dataset(cfg, 'train'), batch_size=int(cfg.training.batch_size), shuffle=True, num_workers=int(cfg.experiment.num_workers))
    val_loader = DataLoader(build_dataset(cfg, 'val'), batch_size=max(1, int(cfg.training.batch_size) // 2), shuffle=False, num_workers=int(cfg.experiment.num_workers))
    trainer.fit(train_loader, val_loader, stage=args.stage)


if __name__ == '__main__':
    main()
