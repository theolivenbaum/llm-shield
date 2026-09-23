#!/usr/bin/env python3
"""Accuracy on the LocalLLaMA/typed-decisions test split, the benchmark laya's fine-tune reports (0.766).

Accuracy is argmax against the gold argmax. For a score question the argmax is the
modal level, not the expected one, matching how laya's notebook reports it.

  python3 eval_td.py --data /home/user/models/td/test.parquet@100 --out td.jsonl [--lora A.pt]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["qtype"], []).append(r)
    tot = [r["correct"] for r in rows]
    print(f"all     n={len(rows)} acc={sum(tot) / len(tot):.3f} "
          f"soft_ce={sum(r['ce'] for r in rows) / len(rows):.3f} tvd={sum(r['tvd'] for r in rows) / len(rows):.3f}")
    for k, v in sorted(by.items()):
        print(f"{k:7s} n={len(v)} acc={sum(r['correct'] for r in v) / len(v):.3f} "
              f"soft_ce={sum(r['ce'] for r in v) / len(v):.3f} tvd={sum(r['tvd'] for r in v) / len(v):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/user/models/shieldstral")
    ap.add_argument("--data", default="/home/user/models/td/test.parquet@100")
    ap.add_argument("--out")
    ap.add_argument("--lora")
    ap.add_argument("--report")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.report:
        summarize(a.report)
        return

    import torch
    from engine import Decider
    from lora_io import load_lora
    from jevstral_torch import Shieldstral
    from train_lora import load_sources

    torch.set_num_threads(a.threads)
    state, cfg = load_lora(a.lora) if a.lora else (None, None)
    m = Shieldstral(a.model_dir, lora=cfg).eval()
    if state:
        m.load_state_dict(state, strict=False)
    d = Decider(m)
    items = load_sources([f"td:{a.data}"], seed=123)
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    with open(a.out, "a") as f, torch.no_grad():
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            t0 = time.perf_counter()
            res = d.decide(it["question"], it["labels"], it["state"])
            p = res["probs"]
            labs = it["labels"]
            gold = max(it["target"], key=it["target"].get)
            pred = max(p, key=p.get)
            ce = -sum(it["target"][k] * math.log(max(p[k], 1e-9)) for k in labs)
            tvd = 0.5 * sum(abs(it["target"][k] - p[k]) for k in labs)
            f.write(json.dumps({"id": it["id"], "qtype": it["question"]["type"], "correct": pred == gold,
                                "ce": ce, "tvd": tvd, "probs": p, "margins": res["margins"],
                                "labels": labs, "target": it["target"],
                                "latency_s": time.perf_counter() - t0}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(items)}", flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
