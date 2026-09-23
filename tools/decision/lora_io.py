"""Saving and loading LoRA adapters: the config travels with the weights."""
from __future__ import annotations

import torch


def save_lora(path, state: dict, cfg: dict, extra: dict | None = None):
    cfg = {**cfg, "layers": sorted(cfg["layers"])}
    torch.save({"state": state, "cfg": cfg, "extra": extra or {}}, path)


def load_lora(path):
    """A training checkpoint (.pt), or an exported adapter directory (adapter.safetensors + adapter.json)."""
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        import json

        from safetensors.torch import load_file

        card = json.loads((p / "adapter.json").read_text())
        cfg = dict(card["lora"])
        cfg["layers"] = set(cfg["layers"])
        cfg["targets"] = tuple(cfg.get("targets", ()))
        state = {k: v.float() for k, v in load_file(str(p / "adapter.safetensors")).items()}
        return state, cfg
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(blob["cfg"])
    cfg["layers"] = set(cfg["layers"])
    return blob["state"], cfg
