from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from diffusefit.config import load_config
from diffusefit.datasets import build_dataset
from diffusefit.model import build_model
from diffusefit.trainers import DiffuseFitTrainer
from diffusefit.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--extra-config', action='append', default=[])
    parser.add_argument('--override', action='append', default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, extra_paths=args.extra_config, overrides=args.override)
    model = build_model(cfg)
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location='cpu')['model'], strict=False)
    loader = DataLoader(build_dataset(cfg, args.split), batch_size=max(1, int(cfg.training.batch_size) // 2), shuffle=False, num_workers=int(cfg.experiment.num_workers))
    trainer = DiffuseFitTrainer(cfg, model)
    metrics = trainer.evaluate(loader, steps=args.steps or int(cfg.diffusion.inference_steps))
    print(json.dumps(metrics, indent=2))
    out = Path(cfg.experiment.output_dir) / cfg.experiment.name / 'eval'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{args.split}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
