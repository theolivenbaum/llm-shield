#!/usr/bin/env python3
"""Export a training checkpoint (.pt) as a portable adapter: fp16 safetensors plus a JSON card.

The .pt that train_lora.py writes is a pickle of fp32 tensors and argparse state, which is fine
for resuming a run and wrong for shipping. The export is what tools/build_models.sh folds into
the GGUF. merge_lora_to_gguf.py and eval_jevbench.py --lora read either form.

  python3 export_adapter.py ~/models/lora_v1.pt adapters/jevstral-v1 --note "..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to(torch.float16).contiguous() for k, v in blob["state"].items()},
              str(out / "adapter.safetensors"))
    cfg = dict(blob["cfg"])
    cfg["layers"] = sorted(cfg["layers"])
    cfg["targets"] = list(cfg.get("targets", ()))
    args = (blob.get("extra") or {}).get("args") or {}
    card = {
        "format": "jevstral-lora-v1",
        "base": "mistralai/Shieldstral-1.0-3B (consolidated.safetensors, Mistral format)",
        "lora": cfg,
        "trained_steps": (blob.get("extra") or {}).get("step"),
        "training": {k: args.get(k) for k in ("data", "layout", "lr", "accum", "max_steps", "max_state_tokens",
                                                "layers_from", "rank", "alpha", "seed")},
        "note": a.note,
    }
    (out / "adapter.json").write_text(json.dumps(card, indent=1))
    n = sum(v.numel() for v in blob["state"].values())
    print(f"wrote {out}/adapter.safetensors ({n / 1e6:.2f}M parameters, fp16) and adapter.json")


if __name__ == "__main__":
    main()
