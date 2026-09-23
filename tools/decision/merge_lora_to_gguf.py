#!/usr/bin/env python3
"""Write a GGUF with a LoRA folded into the weights, for the C# runtime.

The adapter is merged on the fly as tools/convert_shieldstral_to_gguf.py reads each tensor:
W' = W + (alpha / r) * B @ A, in float32, before quantization. Nothing is written to disk
but the GGUF itself. A merged bf16 checkpoint would be another 7 GiB.

The result is still a Shieldstral: same architecture, same vocabulary, same verdict tokens.
JevstralDecider opens it like any other GGUF.

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

MODULE_TO_TENSOR = {
    "wq": "attention.wq", "wk": "attention.wk", "wv": "attention.wv", "wo": "attention.wo",
    "w1": "feed_forward.w1", "w2": "feed_forward.w2", "w3": "feed_forward.w3",
}


def deltas(adapter_path):
    """
    Per-tensor W deltas, (alpha / r) * B @ A, in float32. An exported adapter directory
    (export_adapter.py: adapter.safetensors + adapter.json) needs only numpy and safetensors, so
    tools/build_models.sh runs without PyTorch. A training checkpoint (.pt) needs torch.
    """
    path = Path(adapter_path)
    if path.is_dir():
        import json

        from safetensors.numpy import load_file

        cfg = json.loads((path / "adapter.json").read_text())["lora"]
        state = {k: v.astype(np.float32) for k, v in load_file(str(path / "adapter.safetensors")).items()}
    else:
        from lora_io import load_lora

        tensors, cfg = load_lora(adapter_path)
        state = {k: v.float().numpy() for k, v in tensors.items()}
    scale = cfg.get("alpha", 16.0) / cfg["rank"]
    # Only the rank-r factors are kept; each dense delta is formed when its tensor is converted.
    # All 84 of them at once would be ~6 GB of float32.
    out = {}
    for name, a in state.items():
        if not name.endswith(".lora_a"):
            continue
        prefix = name[: -len(".lora_a")]            # layers.14.wq
        _, layer, module = prefix.split(".")
        out[f"layers.{layer}.{MODULE_TO_TENSOR[module]}.weight"] = (state[prefix + ".lora_b"], a, scale)
    return out


def dense(factors):
    b, a, scale = factors
    return (b @ a) * scale


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
        f = merge.get(name)
        if f is None:
            return w
        d = dense(f)
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
