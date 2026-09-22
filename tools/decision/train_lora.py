#!/usr/bin/env python3
"""LoRA on Shieldstral's own yes/no verdict, trained listwise over the options of a typed decision.

No new head. The option score is the log-odds the model already produces
(yes minus no) for "is option X the correct answer?". The softmax over those
log-odds is trained against the gold distribution. So the adapted model is still
a Shieldstral, can be merged into the base weights, and runs in the C# runtime
unchanged.

Losses, per decision:
  noul    binary cross-entropy of sigmoid(z) against the gold P(yes), soft
  choice  soft cross-entropy of softmax(z) against the gold distribution
  score   the same, plus the ranked probability score (ordinal distance matters), as laya does

Only the top layers carry LoRA, and autograd starts there (`grad_from`), so the
backward never walks the frozen bottom of the stack.

Data sources (never JevBench items):
  td:<parquet>     LocalLLaMA/typed-decisions, soft gold from its annotators
  syn:<jsonl>      gen_synthetic.py output
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine import Decider  # noqa: E402
from lora_io import save_lora  # noqa: E402
from shieldstral_torch import Shieldstral  # noqa: E402


# ------------------------------------------------------------------ data

def load_td(path, limit_cases=0, seed=0):
    import pandas as pd

    d = pd.read_parquet(path)
    rows = list(d.itertuples())
    if limit_cases:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit_cases]
    items = []
    for r in rows:
        qs = json.loads(r.questions) if isinstance(r.questions, str) else r.questions
        gold = json.loads(r.gold) if isinstance(r.gold, str) else r.gold
        state = r.state
        try:
            state = json.loads(state)
        except (TypeError, ValueError):
            pass
        for key, q in qs.items():
            g = gold[key]["probabilities"]
            if q["type"] == "noul":
                labels = ["no", "yes"]
                target = {"no": float(g["false"]), "yes": float(g["true"])}
            elif q["type"] == "score":
                labels = [str(i) for i in range(len(q["criteria"]))]
                target = {k: float(g[k]) for k in labels}
            else:
                labels = sorted(q["criteria"].keys())
                target = {k: float(g[k]) for k in labels}
            items.append({"id": f"{r.id}:{key}", "src": "td", "state": state,
                          "question": q, "labels": labels, "target": target})
    return items


def load_syn(path, limit=0, seed=0):
    items = []
    for line in open(path):
        x = json.loads(line)
        q = x["question"]
        labels = ["no", "yes"] if q["type"] == "noul" else x["labels"]
        items.append({"id": x["id"], "src": "syn:" + x["family"], "state": x["state"], "question": q,
                      "labels": labels, "target": {k: float(x["target"][k]) for k in labels}})
    if limit:
        random.Random(seed).shuffle(items)
        items = items[:limit]
    return items


def load_sources(specs, seed):
    items = []
    for spec in specs:
        kind, _, rest = spec.partition(":")
        path, _, lim = rest.partition("@")
        lim = int(lim) if lim else 0
        items += load_td(path, lim, seed) if kind == "td" else load_syn(path, lim, seed)
    return items


# ------------------------------------------------------------------ loss

def decision_loss(z, item, rps_weight=1.0):
    qt = item["question"]["type"]
    tgt = torch.tensor([item["target"][k] for k in item["labels"]], dtype=torch.float32)
    tgt = tgt / tgt.sum()
    if qt == "noul" and z.numel() == 1:
        return F.binary_cross_entropy_with_logits(z[0].float(), tgt[1])
    logp = torch.log_softmax(z.float(), 0)
    loss = -(tgt * logp).sum()
    if qt == "score" and rps_weight:
        cp, ct = logp.exp().cumsum(0)[:-1], tgt.cumsum(0)[:-1]
        loss = loss + rps_weight * ((cp - ct) ** 2).mean()
    return loss


def predicted_correct(z, item):
    if item["question"]["type"] == "noul" and z.numel() == 1:
        pred = "yes" if float(z[0]) > 0 else "no"
    else:
        pred = item["labels"][int(z.argmax())]
    gold = max(item["target"], key=item["target"].get)
    return pred == gold


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/user/models/shieldstral")
    ap.add_argument("--data", nargs="+", required=True, help="td:PATH[@cases] or syn:PATH[@items]")
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--layers-from", type=int, default=14, help="LoRA on layers [from, 26)")
    ap.add_argument("--targets", default="wq,wk,wv,wo,w1,w2,w3")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", default="card", choices=["card", "docfirst"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(a.threads)

    lora = {"rank": a.rank, "alpha": a.alpha, "layers": set(range(a.layers_from, 26)),
            "targets": tuple(a.targets.split(",")), "dropout": 0.0}
    m = Shieldstral(a.model_dir, lora=lora)
    m.train()
    d = Decider(m, layout=a.layout, style="listed", max_state_tokens=a.max_state_tokens)

    train = load_sources(a.data, a.seed)
    val = load_sources(a.val, a.seed + 1) if a.val else []
    print(f"train {len(train)} decisions, val {len(val)}", flush=True)

    params = m.lora_parameters()
    print(f"LoRA parameters: {sum(p.numel() for p in params) / 1e6:.2f}M on layers {a.layers_from}-25", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    total = int(len(train) * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps * a.accum)
    n_updates = max(1, total // a.accum)
    warm = max(1, n_updates // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_updates))))

    order = []
    while len(order) < total:
        idx = list(range(len(train)))
        random.shuffle(idx)
        order += idx
    order = order[:total]

    t0 = time.time()
    run_loss, run_acc, n_run = 0.0, 0, 0
    step = 0
    for i, j in enumerate(order):
        item = train[j]
        z, kind, _ = d.margins(item["question"], item["labels"], item["state"], grad_from=a.layers_from)
        loss = decision_loss(z, item)
        (loss / a.accum).backward()
        run_loss += float(loss.detach())
        run_acc += predicted_correct(z.detach(), item)
        n_run += 1
        if (i + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                el = time.time() - t0
                print(f"step {step}/{n_updates} loss {run_loss / n_run:.4f} acc {run_acc / n_run:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {el / (i + 1):.2f}s/decision "
                      f"eta {(total - i - 1) * el / (i + 1) / 60:.0f}min", flush=True)
                run_loss, run_acc, n_run = 0.0, 0, 0
            if step % a.save_every == 0:
                save_lora(a.out, m.lora_state(), lora, {"step": step, "args": vars(a)})
    save_lora(a.out, m.lora_state(), lora, {"step": step, "args": vars(a)})
    print(f"saved {a.out} after {step} updates in {(time.time() - t0) / 60:.1f} min", flush=True)

    if val:
        m.eval()
        with torch.no_grad():
            L, C = 0.0, 0
            for item in val:
                z, _, _ = d.margins(item["question"], item["labels"], item["state"])
                L += float(decision_loss(z, item))
                C += predicted_correct(z, item)
        print(f"val loss {L / len(val):.4f} acc {C / len(val):.3f} (n={len(val)})", flush=True)


if __name__ == "__main__":
    main()
