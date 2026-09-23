#!/usr/bin/env python3
"""Thinking budget against precision, from one high-budget run.

Decoding is greedy, so a run capped at B thinking tokens produces exactly the first B tokens
of an uncapped run, then a forced [/THINK]. One run at a high budget, with its token ids saved
(reason.py writes gen_ids), therefore contains every lower budget. This script truncates each
item's reasoning at each budget, closes the thinking, and re-scores the labels: a prefill per
(item, budget), no generation.

  python3 budget_study.py --runs reason_es.jsonl reason_hard.jsonl --budgets 0,256,512,1024,1536,2048,4096 \\
      --out budget.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ece(pairs, bins=10):
    b = [[0, 0.0, 0] for _ in range(bins)]
    for c, ok in pairs:
        x = b[min(int(c * bins), bins - 1)]
        x[0] += 1; x[1] += c; x[2] += ok
    n = len(pairs)
    return sum(abs(x[2] / x[0] - x[1] / x[0]) * x[0] / n for x in b if x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/user/models/ministral3b-reasoning")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--budgets", default="0,256,512,1024,1536,2048,4096")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    from eval_jevbench import load_tasks, to_task_probs
    from jevbench.scoring import score_task
    from jevstral_torch import Shieldstral
    from reason import render, score_labels

    torch.set_num_threads(a.threads)
    m = Shieldstral(a.model_dir).eval()
    system = (Path(a.model_dir) / "SYSTEM_PROMPT.txt").read_text().strip()
    end_think = m.tok.special_by_str["[/THINK]"]
    tasks = {t.id: (tier, t) for tier, t in load_tasks(["easy", "original", "hard"])}
    budgets = [int(x) for x in a.budgets.split(",")]

    # A continued run (--continue-from) appends a row with the prior and the new tokens: the last
    # row for an id is the longest reasoning.
    latest = {}
    for path in a.runs:
        for line in open(path):
            r = json.loads(line)
            if "gen_ids" in r and r["id"] in tasks:
                latest[r["id"]] = r
    rows = list(latest.values())
    print(f"{len(rows)} items with saved reasoning", flush=True)

    results = {B: [] for B in budgets}
    for i, r in enumerate(rows):
        tier, t = tasks[r["id"]]
        labels = ["no", "yes"] if t.question["type"] == "noul" else t.labels
        prompt = m.encode(render(t.question, labels, t.state, system) + "[THINK]", bos=True)
        gen = [x for x in r["gen_ids"] if x != end_think]
        closed_at = len(gen) if r.get("closed") else None
        for B in budgets:
            used = gen[:B]
            ctx = prompt + used + [end_think] + m.encode("\nFinal answer:")
            with torch.no_grad():
                sc = score_labels(m, ctx, labels)
            mx = max(sc)
            e = [math.exp(x - mx) for x in sc]
            probs = {lab: v / sum(e) for lab, v in zip(labels, e)}
            tp = to_task_probs(t, probs)
            res = score_task(tp, t)
            results[B].append({"id": r["id"], "tier": tier, "correct": bool(res["correct"]),
                               "conf": max(res["probs"].values()), "think": len(used),
                               "natural_end": closed_at is not None and closed_at <= B})
        if i % 10 == 0:
            print(f"{i}/{len(rows)}", flush=True)

    summary = {}
    for B in budgets:
        per = {}
        for x in results[B]:
            per.setdefault(x["tier"], []).append(x)
        summary[B] = {}
        for tier, xs in sorted(per.items()) + [("all", results[B])]:
            summary[B][tier] = {
                "n": len(xs), "acc": sum(x["correct"] for x in xs) / len(xs),
                "ece": ece([(x["conf"], x["correct"]) for x in xs]),
                "mean_think_tokens": sum(x["think"] for x in xs) / len(xs),
                "finished_within_budget": sum(x["natural_end"] for x in xs) / len(xs)}
    Path(a.out).write_text(json.dumps({"summary": summary, "items": results}, indent=1))
    print(f"{'budget':>7s} " + " ".join(f"{t:>28s}" for t in summary[budgets[0]]))
    for B in budgets:
        print(f"{B:7d} " + " ".join(
            f"acc {v['acc']:.3f} ece {v['ece']:.3f} tok {v['mean_think_tokens']:5.0f}" for v in summary[B].values()))


if __name__ == "__main__":
    main()
