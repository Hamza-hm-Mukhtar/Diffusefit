from __future__ import annotations

import argparse

import torch

from diffusefit.config import load_config
from diffusefit.datasets import _expand_pose_channels, _load_gray, _load_rgb, _one_hot_parse
from diffusefit.model import build_model
from diffusefit.utils import load_checkpoint, save_tensor_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--person', required=True)
    parser.add_argument('--garment', required=True)
    parser.add_argument('--garment-mask', required=True)
    parser.add_argument('--parse', required=True)
    parser.add_argument('--skeleton', required=True)
    parser.add_argument('--densepose', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    h, w = cfg.dataset.image_size
    person = _load_rgb(args.person, (h, w)).unsqueeze(0)
    garment = _load_rgb(args.garment, (h, w)).unsqueeze(0)
    garment_mask = _load_gray(args.garment_mask, (h, w)).unsqueeze(0)
    parse_raw = _load_gray(args.parse, (h, w)).unsqueeze(0)
    parse_map = _one_hot_parse(parse_raw[0], int(cfg.dataset.parse_num_classes)).unsqueeze(0)
    skeleton = _load_gray(args.skeleton, (h, w)).unsqueeze(0)
    densepose = _load_rgb(args.densepose, (h, w)).unsqueeze(0)
    limb_mask = parse_map[:, list(cfg.dataset.limb_channel_ids)].sum(dim=1, keepdim=True).clamp(0, 1)
    limb_map = person * limb_mask
    garment_mask_gt = parse_map[:, list(cfg.dataset.garment_channel_ids)].sum(dim=1, keepdim=True).clamp(0, 1)
    batch = {
        'person_image': person,
        'garment_image': garment,
        'garment_mask': garment_mask,
        'parse_map': parse_map,
        'parse_ids': (parse_raw[:, 0] * 255).round().long(),
        'skeleton_map': skeleton,
        'densepose_map': densepose,
        'pose_tensor': _expand_pose_channels(skeleton[0], densepose[0]).unsqueeze(0),
        'target_image': person,
        'limb_map': limb_map,
        'garment_region_gt': person * garment_mask_gt,
    }
    model = build_model(cfg)
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location='cpu')['model'], strict=False)
    device = cfg.experiment.device if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model.generate(batch, steps=args.steps or int(cfg.diffusion.inference_steps))
    save_tensor_image(out['image'], args.output)


if __name__ == '__main__':
    main()
