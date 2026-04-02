from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusefit.config import load_config
from diffusefit.datasets import build_dataset
from diffusefit.model import build_model
from diffusefit.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[25, 36, 50, 100])
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg)
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location='cpu')['model'], strict=False)
    device = cfg.experiment.device if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    batch = next(iter(DataLoader(build_dataset(cfg, 'test'), batch_size=1, shuffle=False, num_workers=0)))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    results = {}
    with torch.no_grad():
        for step_count in args.steps:
            for _ in range(3):
                _ = model.generate(batch, steps=step_count)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(10):
                _ = model.generate(batch, steps=step_count)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                peak_gb = 0.0
            elapsed = (time.perf_counter() - start) / 10.0 * 1000.0
            results[str(step_count)] = {'latency_ms': elapsed, 'vram_gb': peak_gb}
    print(json.dumps(results, indent=2))
    out = Path(cfg.experiment.output_dir) / cfg.experiment.name / 'benchmark.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
