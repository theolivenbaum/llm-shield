"""Saving and loading LoRA adapters: the config travels with the weights."""
from __future__ import annotations

import torch


def save_lora(path, state: dict, cfg: dict, extra: dict | None = None):
    cfg = {**cfg, "layers": sorted(cfg["layers"])}
    torch.save({"state": state, "cfg": cfg, "extra": extra or {}}, path)


def load_lora(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(blob["cfg"])
    cfg["layers"] = set(cfg["layers"])
    return blob["state"], cfg
