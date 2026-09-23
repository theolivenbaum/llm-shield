#!/usr/bin/env python3
"""Fit how to combine reason-then-decide with the verdict decider, on non-JevBench items only.

For each item, the combined distribution is softmax(s / T + w * z), where s are the reasoning
path's mean label log-probabilities and z the verdict decider's per-option log-odds. T and w are
fitted by grid search to minimise the soft cross-entropy against the synthetic items' gold
distributions. w = 0 means the reasoning path alone, with temperature T.

  python3 fit_ensemble.py --reason syn_reason.jsonl --verdict syn_verdict.jsonl --out ensemble.json
  python3 fit_ensemble.py --apply ensemble.json --reason jev_reason.jsonl --verdict jev_verdict.jsonl
"""
from __future__ import annotations

import argparse
import json
import math


def load(path, key="margins"):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def combine(s, z, T, w):
    x = [a / T + w * b for a, b in zip(s, z)]
    m = max(x)
    e = [math.exp(v - m) for v in x]
    t = sum(e)
    return [v / t for v in e]


def align(r, v):
    """Both paths' scores in the reasoning record's label order; None when they do not line up."""
    rl, vl = r["labels"], v["labels"]
    if sorted(rl) != sorted(vl) or len(v["margins"]) != len(vl):
        return None
    zmap = dict(zip(vl, v["margins"]))
    return r["margins"], [zmap[l] for l in rl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", required=True)
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--out")
    ap.add_argument("--apply")
    a = ap.parse_args()
    R, V = load(a.reason), load(a.verdict)
    rows = []
    for i, r in R.items():
        if i not in V:
            continue
        al = align(r, V[i])
        if al is None:
            continue
        tgt = r.get("target") or V[i].get("target")
        rows.append((r, al[0], al[1], [tgt[l] for l in r["labels"]] if tgt else None))

    if a.apply:
        p = json.loads(open(a.apply).read())
        for name, T, w in (("reasoning alone", 1.0, 0.0), ("reasoning, fitted T", p["T_reason_only"], 0.0),
                           ("ensemble", p["T"], p["w"])):
            ok = sum(max(range(len(s)), key=lambda k: combine(s, z, T, w)[k]) == r["labels"].index(
                r["expected"] if r["qtype"] != "noul" else r["expected"]) for r, s, z, _ in rows)
            print(f"{name:22s} acc {ok / len(rows):.3f} (n={len(rows)})")
        return

    def nll(T, w):
        tot = 0.0
        for _, s, z, tgt in rows:
            q = combine(s, z, T, w)
            t = sum(tgt)
            tot -= sum(g / t * math.log(max(x, 1e-12)) for g, x in zip(tgt, q))
        return tot / len(rows)

    Ts = [math.exp(k / 10) for k in range(-25, 26)]
    ws = [k / 20 for k in range(0, 41)]
    best = min((nll(T, w), T, w) for T in Ts for w in ws)
    solo = min((nll(T, 0.0), T) for T in Ts)
    out = {"T": best[1], "w": best[2], "nll": best[0], "T_reason_only": solo[1], "nll_reason_only": solo[0],
           "nll_raw": nll(1.0, 0.0), "n": len(rows)}
    print(json.dumps(out, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
