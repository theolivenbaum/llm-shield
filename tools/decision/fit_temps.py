#!/usr/bin/env python3
"""Fit per-question-type temperatures on stored margins, as laya's notebook does after training.

Input: JSONL written by eval_td.py (margins, labels, target) from a held-out split that is not
JevBench. Output: {"noul": T, "choice": T, "score": T, "choice:6-10": T, ...}. That is the map
eval_jevbench.py --temps, the jevbench adapter and JevstralDecider all take.

Each temperature minimises the soft cross-entropy of softmax(z / T) against the gold
distribution. A 1-D golden-section search over log T in [0.1, 10] suffices.

  python3 fit_temps.py td_val.jsonl [more.jsonl ...] --out temps.json
"""
from __future__ import annotations

import argparse
import json
import math


def bucket(k):
    return "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"


def nll(rows, t):
    tot = 0.0
    for z, tgt in rows:
        m = max(z)
        e = [math.exp((x - m) / t) for x in z]
        s = sum(e)
        tot -= sum(g * math.log(max(v / s, 1e-12)) for g, v in zip(tgt, e))
    return tot / len(rows)


def fit(rows):
    lo, hi = math.log(0.1), math.log(10.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(rows, math.exp(a)), nll(rows, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(rows, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(rows, math.exp(b))
    return math.exp((lo + hi) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-bucket", type=int, default=40, help="fit a <type>:<bucket> key only with this many items")
    a = ap.parse_args()

    by_type, by_bucket = {}, {}
    for path in a.inputs:
        for line in open(path):
            r = json.loads(line)
            labels = r["labels"]
            z = r["margins"]
            if len(z) != len(labels):
                continue
            tgt = [float(r["target"][k]) for k in labels]
            s = sum(tgt)
            tgt = [x / s for x in tgt]
            by_type.setdefault(r["qtype"], []).append((z, tgt))
            by_bucket.setdefault(f"{r['qtype']}:{bucket(len(labels))}", []).append((z, tgt))

    temps = {}
    for qt, rows in sorted(by_type.items()):
        t = fit(rows)
        temps[qt] = round(t, 4)
        print(f"{qt:7s} n={len(rows):4d} T={t:.3f} nll {nll(rows, 1.0):.4f} -> {nll(rows, t):.4f}")
    for key, rows in sorted(by_bucket.items()):
        if len(rows) >= a.min_bucket and key.split(":")[1] != "2":
            t = fit(rows)
            temps[key] = round(t, 4)
            print(f"{key:12s} n={len(rows):4d} T={t:.3f}")
    with open(a.out, "w") as f:
        json.dump(temps, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
