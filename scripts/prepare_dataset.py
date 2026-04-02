from __future__ import annotations

import argparse
from pathlib import Path

from diffusefit.utils import ensure_dir, write_json


def discover_files(root: Path, split: str):
    image_dir = root / split / 'image'
    cloth_dir = root / split / 'cloth'
    mask_dir = root / split / 'cloth-mask'
    parse_dir = root / split / 'image-parse'
    pose_dir = root / split / 'openpose-img'
    densepose_dir = root / split / 'image-densepose'
    items = []
    if not image_dir.exists():
        return items
    for img_path in sorted(image_dir.glob('*')):
        stem = img_path.name
        cloth_path = cloth_dir / stem
        mask_path = mask_dir / stem
        parse_path = parse_dir / stem.replace('.jpg', '.png').replace('.jpeg', '.png')
        pose_path = pose_dir / stem.replace('.jpg', '_rendered.png').replace('.jpeg', '_rendered.png')
        dense_path = densepose_dir / stem.replace('.jpg', '.png').replace('.jpeg', '.png')
        items.append({
            'id': stem,
            'person_image': str(img_path),
            'garment_image': str(cloth_path if cloth_path.exists() else img_path),
            'garment_mask': str(mask_path if mask_path.exists() else mask_dir / stem.replace('.jpg', '.png')),
            'parse': str(parse_path),
            'skeleton': str(pose_path if pose_path.exists() else pose_dir / stem),
            'densepose': str(dense_path) if dense_path.exists() else None,
            'target_image': str(img_path),
        })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['viton', 'viton_hd', 'dresscode'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = ensure_dir(args.out)
    for split in ['train', 'val', 'test']:
        items = discover_files(root, split)
        if items:
            write_json(items, out / f'{split}.json')


if __name__ == '__main__':
    main()
