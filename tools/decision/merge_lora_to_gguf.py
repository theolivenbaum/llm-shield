#!/usr/bin/env python3
"""Write a GGUF with a LoRA folded into the weights, for the C# runtime.

The adapter is merged on the fly as tools/convert_shieldstral_to_gguf.py reads each tensor:
W' = W + (alpha / r) * B @ A, in float32, before quantization. Nothing is written to disk
but the GGUF itself. A merged bf16 checkpoint would be another 7 GiB.

The result is still a Shieldstral: same architecture, same vocabulary, same verdict tokens.
ShieldstralDecider opens it like any other GGUF.

  python3 merge_lora_to_gguf.py MODEL_DIR ADAPTER.pt --outtype q8_0 --outfile OUT.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import convert_shieldstral_to_gguf as conv  # noqa: E402
from lora_io import load_lora  # noqa: E402

MODULE_TO_TENSOR = {
    "wq": "attention.wq", "wk": "attention.wk", "wv": "attention.wv", "wo": "attention.wo",
    "w1": "feed_forward.w1", "w2": "feed_forward.w2", "w3": "feed_forward.w3",
}


def deltas(adapter_path):
    state, cfg = load_lora(adapter_path)
    scale = cfg.get("alpha", 16.0) / cfg["rank"]
    out = {}
    for name, a in state.items():
        if not name.endswith(".lora_a"):
            continue
        prefix = name[: -len(".lora_a")]            # layers.14.wq
        _, layer, module = prefix.split(".")
        b = state[prefix + ".lora_b"]
        out[f"layers.{layer}.{MODULE_TO_TENSOR[module]}.weight"] = (b.float() @ a.float()).numpy() * scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("adapter", type=Path)
    ap.add_argument("--outtype", default="q8_0", choices=sorted(conv.OUT_TYPES))
    ap.add_argument("--outfile", type=Path, required=True)
    a = ap.parse_args()

    merge = deltas(a.adapter)
    print(f"merging {len(merge)} LoRA deltas from {a.adapter}")
    original = conv.SafetensorsFile.get_tensor

    def get_tensor(self, name):
        w = original(self, name)
        d = merge.get(name)
        if d is None:
            return w
        rel = float(np.linalg.norm(d) / max(np.linalg.norm(w), 1e-12))
        print(f"    + delta {name} (relative norm {rel:.2e})")
        return (w.astype(np.float32) + d.astype(np.float32)).astype(np.float32)

    conv.SafetensorsFile.get_tensor = get_tensor
    import json

    d = a.model_dir
    params = json.loads((d / "params.json").read_text())
    tekken = json.loads((d / "tekken.json").read_text())
    template_path = d / "chat_template.jinja"
    template = template_path.read_text() if template_path.exists() else None
    conv.convert_text_model(d / "consolidated.safetensors", a.outfile, params, tekken,
                            conv.OUT_TYPES[a.outtype], template)


if __name__ == "__main__":
    main()
