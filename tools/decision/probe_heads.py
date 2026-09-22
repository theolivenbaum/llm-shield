#!/usr/bin/env python3
"""Which output head, on which layer? Listwise linear probes over extracted features.

For every tapped layer L, score_i = w · f_L(option i) + b[type], softmax over a decision's
options, soft cross-entropy against the gold distribution. Two variants:

  probe      a fresh head: what a new output layer on layer L can decode
  residual   verdict margin + w · f_26: whether a learned correction improves on the model's
             own yes/no read (a head that starts at the verdict)

Held-out rows are scored with the raw verdict margin (T fitted on train) as the baseline. A
probe on layer L < 26 that matches the verdict means the top 26 − L layers can be skipped:
early exit.

  python3 probe_heads.py --train train_feats.pt --val val_feats.pt [--jev jev_feats.pt]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
QTYPES = {"noul": 0, "choice": 1, "score": 2}


def pack(items, layer_index):
    """Flatten a list of decisions into (features, group ids, targets, type ids, margins)."""
    f, g, t, q, z = [], [], [], [], []
    for n, it in enumerate(items):
        k = len(it["labels"])
        f.append(it["feats"][layer_index].float() if layer_index is not None else torch.zeros(k, 1))
        g += [n] * k
        tgt = torch.tensor(it["target"], dtype=torch.float32)
        t.append(tgt / tgt.sum())
        q += [QTYPES[it["qtype"]]] * k
        z.append(it["margin"].float())
    return torch.cat(f), torch.tensor(g), torch.cat(t), torch.tensor(q), torch.cat(z)


def group_logsoftmax(s, g, n):
    m = torch.full((n,), -1e30).scatter_reduce(0, g, s, "amax", include_self=True)
    e = (s - m[g]).exp()
    denom = torch.zeros(n).index_add(0, g, e)
    return s - m[g] - denom.log()[g]


def metrics(s, g, t, items, name):
    n = len(items)
    lp = group_logsoftmax(s, g, n)
    nll = float(-(t * lp).sum() / n)
    p = lp.exp()
    correct, confs = [], []
    start = 0
    for it in items:
        k = len(it["labels"])
        pi, ti = p[start:start + k], t[start:start + k]
        correct.append(int(pi.argmax()) == int(ti.argmax()))
        confs.append(float(pi.max()))
        start += k
    acc = sum(correct) / n
    bins = [[0, 0.0, 0] for _ in range(10)]
    for c, ok in zip(confs, correct):
        b = bins[min(int(c * 10), 9)]
        b[0] += 1; b[1] += c; b[2] += ok
    ece = sum(abs(b[2] / b[0] - b[1] / b[0]) * b[0] / n for b in bins if b[0])
    return {"name": name, "acc": acc, "nll": nll, "ece": ece, "correct": correct}


def fit(feats, g, t, q, z, n, residual, epochs=300, wd=1e-3, lr=0.05):
    dim = feats.shape[1]
    w = torch.zeros(dim, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    ls = torch.zeros(1, requires_grad=True)          # log-temperature on the margin (residual)
    opt = torch.optim.Adam([w, b, ls], lr=lr)
    for _ in range(epochs):
        s = feats @ w + b[q] + (z * ls.exp() if residual else 0)
        loss = -(t * group_logsoftmax(s, g, n)).sum() / n + wd * w.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach(), ls.detach()


def score(feats, q, z, w, b, ls, residual):
    return feats @ w + b[q] + (z * ls.exp() if residual else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--jev")
    ap.add_argument("--wd", type=float, default=1e-3)
    a = ap.parse_args()

    tr = torch.load(a.train, weights_only=False)
    va = torch.load(a.val, weights_only=False)
    jv = torch.load(a.jev, weights_only=False) if a.jev else None
    layers = tr["layers"]
    tri, vai = tr["items"], va["items"]

    # Baseline: the zero-shot verdict margin, with one temperature fitted on train.
    _, g, t, q, z = pack(tri, None)
    best = min(((float(-(t * group_logsoftmax(z / T, g, len(tri))).sum()), T)
                for T in [math.exp(x / 20) for x in range(-40, 41)]))
    T = best[1]
    _, gv, tv, qv, zv = pack(vai, None)
    rows = [metrics(zv / T, gv, tv, vai, f"verdict margin (T={T:.2f})")]
    jrows = {}
    if jv:
        _, gj, tj, qj, zj = pack(jv["items"], None)
        jrows["verdict"] = metrics(zj / T, gj, tj, jv["items"], "verdict")

    for li, L in enumerate(layers):
        for residual in ([False, True] if L == layers[-1] else [False]):
            f, g, t, q, z = pack(tri, li)
            w, b, ls = fit(f, g, t, q, z, len(tri), residual, wd=a.wd)
            fv, gv, tv, qv, zv = pack(vai, li)
            name = f"{'residual' if residual else 'probe'} L{L}"
            rows.append(metrics(score(fv, qv, zv, w, b, ls, residual), gv, tv, vai, name))
            if jv:
                fj, gj, tj, qj, zj = pack(jv["items"], li)
                jrows[name] = metrics(score(fj, qj, zj, w, b, ls, residual), gj, tj, jv["items"], name)

    print(f"held-out non-JevBench: n={len(vai)}")
    for r in rows:
        print(f"  {r['name']:28s} acc {r['acc']:.3f}  nll {r['nll']:.3f}  ece {r['ece']:.3f}")
    if jv:
        tiers = sorted({it["src"].split(":")[1] for it in jv["items"]})
        print("public JevBench (not trained on; reported per tier):")
        for name, r in jrows.items():
            per = {}
            for it, ok in zip(jv["items"], r["correct"]):
                per.setdefault(it["src"].split(":")[1], []).append(ok)
            print(f"  {name:28s} " + "  ".join(f"{tr_}={sum(v) / len(v):.3f}" for tr_, v in sorted(per.items())))


if __name__ == "__main__":
    main()
