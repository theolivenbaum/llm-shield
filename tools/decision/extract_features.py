#!/usr/bin/env python3
"""Residual-stream features at each option's verdict position, for probing heads and exit layers.

For every decision the option reads run as usual, but the state after selected layers is
kept at each branch's last token (where the verdict is read). It is RMS-normalised without
the learned norm weight, so layers are comparable, and stored as fp16. The verdict margin
the zero-shot decider uses is stored next to it.

  python3 extract_features.py --data td:PATH@N syn:PATH@N --out feats.pt
  python3 extract_features.py --jevbench easy,original,hard --out jev_feats.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

LAYERS = (8, 12, 14, 16, 18, 20, 22, 24, 26)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/user/models/shieldstral")
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--jevbench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-state-tokens", type=int, default=1536)
    a = ap.parse_args()

    from engine import Decider
    from lora_io import load_lora
    from jevstral_torch import Shieldstral
    from train_lora import load_sources

    torch.set_num_threads(a.threads)
    state, cfg = load_lora(a.lora) if a.lora else (None, None)
    m = Shieldstral(a.model_dir, lora=cfg).eval()
    if state:
        m.load_state_dict(state, strict=False)
    d = Decider(m, max_state_tokens=a.max_state_tokens)

    items = load_sources(a.data, seed=7) if a.data else []
    if a.jevbench:
        from eval_jevbench import load_tasks
        for tier, t in load_tasks(a.jevbench.split(",")):
            labels = ["no", "yes"] if t.question["type"] == "noul" else t.labels
            exp = "yes" if (t.question["type"] == "noul" and t.expected == "yes") else str(t.expected)
            tgt = {k: (1.0 if k == exp else 0.0) for k in labels}
            gold = (t.provenance or {}).get("gold_probs")
            if gold:
                tgt = {k: float(gold[k]) for k in labels}
            items.append({"id": t.id, "src": "jev:" + tier + ":" + t.family, "state": t.state,
                          "question": t.question, "labels": labels, "target": tgt})

    out = []
    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items):
            m.tap_layers = None
            from decision_prompts import build_reads
            prefix, suffixes, kind = build_reads(it["question"], it["labels"], d.clip(it["state"]), d.layout,
                                                 d.style, d.noul)
            past = d.run_prefix(d.encode_prefix(prefix))
            sfx = [m.encode(s) for s in suffixes]
            m.tap_layers, m.taps = set(LAYERS), {}
            h = d.run_suffixes(past, sfx)
            last = torch.tensor([len(s) - 1 for s in sfx])
            feats = []
            for L in LAYERS:
                x = m.taps[L][torch.arange(len(sfx)), last].float()
                feats.append((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)).half())
            margin = m.verdict(h).float()
            out.append({"id": it["id"], "src": it["src"], "qtype": it["question"]["type"],
                        "labels": it["labels"], "target": [it["target"][k] for k in it["labels"]],
                        "feats": torch.stack(feats), "margin": margin})
            if i % 50 == 0:
                print(f"{i}/{len(items)} {(time.time() - t0) / (i + 1):.2f}s/decision", flush=True)
    torch.save({"layers": LAYERS, "items": out}, a.out)
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
