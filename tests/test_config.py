from diffusefit.config import load_config


def test_load_config():
    cfg = load_config('configs/viton.yaml')
    assert cfg.dataset.name == 'viton'
    assert cfg.model.fusion_strategy == 'cross_attention'
