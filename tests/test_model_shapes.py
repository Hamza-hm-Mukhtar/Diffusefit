import torch

from diffusefit.config import load_config
from diffusefit.model import build_model


def test_forward_shapes():
    cfg = load_config('configs/viton.yaml', overrides=['model.use_clip_semantics=false'])
    model = build_model(cfg)
    model.eval()
    b, h, w = 1, 256, 192
    batch = {
        'person_image': torch.rand(b, 3, h, w),
        'garment_image': torch.rand(b, 3, h, w),
        'garment_mask': torch.rand(b, 1, h, w),
        'parse_map': torch.rand(b, cfg.dataset.parse_num_classes, h, w),
        'parse_ids': torch.randint(0, cfg.dataset.parse_num_classes, (b, h, w)),
        'skeleton_map': torch.rand(b, 1, h, w),
        'densepose_map': torch.rand(b, 3, h, w),
        'pose_tensor': torch.rand(b, cfg.model.pose_channels, h, w),
        'target_image': torch.rand(b, 3, h, w),
        'limb_map': torch.rand(b, 3, h, w),
        'garment_region_gt': torch.rand(b, 3, h, w),
    }
    with torch.no_grad():
        out = model(batch)
    assert out['stage1']['warped_garment'].shape == (b, 3, h, w)
    assert out['stage2']['target_layout'].shape == (b, cfg.dataset.parse_num_classes, h, w)
    assert out['stage3']['pred_image'].shape == (b, 3, h, w)
