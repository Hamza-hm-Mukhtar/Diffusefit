from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


def _merge_default_chain(cfg):
    if "defaults" not in cfg or not cfg.defaults:
        return cfg
    merged = OmegaConf.create()
    cfg_dir = Path(cfg._metadata.resolver_cache.get("config_path", ".")) if hasattr(cfg, "_metadata") else Path(".")
    for item in cfg.defaults:
        if isinstance(item, str):
            sub_cfg = OmegaConf.load(cfg_dir / f"{item}.yaml")
            sub_cfg._metadata.resolver_cache["config_path"] = str((cfg_dir / item).parent)
            merged = OmegaConf.merge(merged, _merge_default_chain(sub_cfg))
    cfg_no_defaults = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg_no_defaults.pop("defaults", None)
    return OmegaConf.merge(merged, cfg_no_defaults)


def load_config(config_path: str, extra_paths: Iterable[str] | None = None, overrides: list[str] | None = None):
    cfg = OmegaConf.load(config_path)
    cfg._metadata.resolver_cache["config_path"] = str(Path(config_path).parent)
    cfg = _merge_default_chain(cfg)
    if extra_paths:
        for path in extra_paths:
            extra = OmegaConf.load(path)
            extra._metadata.resolver_cache["config_path"] = str(Path(path).parent)
            cfg = OmegaConf.merge(cfg, _merge_default_chain(extra))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def to_dict(cfg) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


def save_config(cfg, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
