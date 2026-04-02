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
    parser.add_argument('--stage1-ckpt', default=None)
    parser.add_argument('--extra-config', action='append', default=[])
    parser.add_argument('--override', action='append', default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, extra_paths=args.extra_config, overrides=args.override)
    seed_all(int(cfg.experiment.seed))
    model = build_model(cfg)
    if args.stage1_ckpt:
        model.load_state_dict(load_checkpoint(args.stage1_ckpt, map_location='cpu')['model'], strict=False)
    train_loader = DataLoader(build_dataset(cfg, 'train'), batch_size=int(cfg.training.batch_size), shuffle=True, num_workers=int(cfg.experiment.num_workers))
    val_loader = DataLoader(build_dataset(cfg, 'val'), batch_size=max(1, int(cfg.training.batch_size) // 2), shuffle=False, num_workers=int(cfg.experiment.num_workers))
    trainer = DiffuseFitTrainer(cfg, model)
    trainer.fit(train_loader, val_loader, stage='stage2')


if __name__ == '__main__':
    main()
