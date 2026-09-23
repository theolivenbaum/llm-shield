#!/usr/bin/env python3
"""Score Shieldstral decision strategies on the public JevBench tiers.

Uses jevbench's own loader, `score_task`, Brier, ECE and TVD, so the numbers here
are computed the way the leaderboard computes them. The judge tier and the
held-out halves are private, so only public easy / standard / hard can run here.
Anything reported from this script is a public-item estimate, and says so.

  python3 eval_jevbench.py --model-dir M --out run.jsonl [--mode verify|list] [--layout docfirst|card]
          [--tiers easy,original,hard] [--limit N] [--lora adapter.pt] [--temps temps.json]
  python3 eval_jevbench.py --report run.jsonl [--temps temps.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
JEVBENCH = Path(os.environ.get("JEVBENCH", "/home/user/jevbench"))
sys.path.insert(0, str(JEVBENCH))

from jevbench.metrics import brier_score, ece_top_label  # noqa: E402
from jevbench.scoring import score_task  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402
from jevbench import composite_v13 as c13  # noqa: E402

TIER_FILES = {"easy": "easy.jsonl", "original": "original.jsonl", "hard": "hard.jsonl"}
TIER_NAME = {"easy": "easy", "original": "standard", "hard": "hard"}


def load_tasks(tiers):
    out = []
    for t in tiers:
        for task in load_jsonl(str(JEVBENCH / "datasets/public" / TIER_FILES[t])):
            out.append((TIER_NAME[t], task))
    return out


def to_task_probs(task, probs):
    """Map the decider's labels onto the task's (noul is always no/yes in JevBench)."""
    if task.question["type"] == "noul":
        p = probs.get("yes", probs.get("true"))
        return {"no": 1.0 - p, "yes": p}
    return {lab: float(probs[lab]) for lab in task.labels}


def apply_temps(rec, temps):
    """Re-derive probabilities from stored margins under a temperature map, without re-running the model."""
    if not temps or rec.get("mode") != "verify":
        return rec["probs"]
    qt, k = rec["qtype"], len(rec["labels"])
    b = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    t = float(temps.get(f"{qt}:{b}", temps.get(qt, 1.0)))
    z = rec["margins"]
    if qt == "noul" and len(z) == 1:
        p = 1 / (1 + math.exp(-z[0] / t))
        return {"no": 1 - p, "yes": p}
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    labs = ["no", "yes"] if qt == "noul" else rec["labels"]
    return {lab: v / s for lab, v in zip(labs, e)}


def report(path, temps=None, quiet=False):
    recs = [json.loads(l) for l in open(path)]
    tasks = {t.id: (tier, t) for tier, t in load_tasks(["easy", "original", "hard"])}
    by_tier = {}
    for r in recs:
        tier, task = tasks[r["id"]]
        probs = apply_temps(r, temps)
        sc = score_task(probs, task)
        conf = max(sc["probs"].values()) if sc["valid"] else 0.0
        row = {"correct": bool(sc["correct"]), "conf": conf, "family": task.family,
               "brier": brier_score(sc["probs"], str(task.expected), task.labels) if sc["valid"] else 2.0,
               "latency": r.get("latency_s", 0.0), "k": len(task.labels)}
        gold = (task.provenance or {}).get("gold_probs")
        if gold:
            row["tvd"] = c13.tvd(sc["probs"], gold, task.labels)
        by_tier.setdefault(tier, []).append(row)

    out = {}
    for tier, rows in by_tier.items():
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n
        chance = sum(1 / r["k"] for r in rows) / n
        ece = ece_top_label([(r["conf"], r["correct"]) for r in rows])["ece"]
        tv = [r["tvd"] for r in rows if "tvd" in r]
        fam = {}
        for r in rows:
            fam.setdefault(r["family"], []).append(r["correct"])
        out[tier] = {"n": n, "acc": acc, "chance": chance, "ece": ece,
                     "brier": sum(r["brier"] for r in rows) / n,
                     "mean_tvd": (sum(tv) / len(tv)) if tv else None,
                     "p50_s": sorted(r["latency"] for r in rows)[n // 2],
                     "families": {f: round(sum(v) / len(v), 3) for f, v in sorted(fam.items())}}
    # Intelligence over the public tiers present, reweighted without the judge tier.
    w = {"easy": .14, "standard": .28, "hard": .30}
    have = [t for t in w if t in out]
    if have:
        tot = sum(w[t] for t in have)
        intel = sum(w[t] / tot * c13.chance_corrected_accuracy(out[t]["acc"], c13.TIER_CHANCES[t]) for t in have)
        out["intelligence_public_no_judge"] = intel
    if "hard" in out:
        out["calibration_hard_public"] = c13.calibration(out["hard"]["ece"], out["hard"]["mean_tvd"])
    if not quiet:
        for tier in ("easy", "standard", "hard"):
            if tier in out:
                o = out[tier]
                print(f"{tier:9s} n={o['n']:3d} acc={o['acc']:.3f} chance={o['chance']:.3f} ece={o['ece']:.3f} "
                      f"brier={o['brier']:.3f} tvd={o['mean_tvd'] if o['mean_tvd'] is None else round(o['mean_tvd'], 3)} "
                      f"p50={o['p50_s']:.2f}s")
                print("          ", o["families"])
        for k in ("intelligence_public_no_judge", "calibration_hard_public"):
            if k in out:
                print(f"{k}: {out[k]:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/user/models/shieldstral")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--mode", default="verify", choices=["verify", "list"])
    ap.add_argument("--layout", default="docfirst", choices=["docfirst", "card", "qcache"])
    ap.add_argument("--style", default="listed", choices=["listed", "bare"])
    ap.add_argument("--noul", default="contrast", choices=["contrast", "direct"])
    ap.add_argument("--query", default="short", choices=["full", "short"])
    ap.add_argument("--tiers", default="easy,original,hard")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-state-tokens", type=int, default=0)
    ap.add_argument("--lora")
    ap.add_argument("--temps")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    temps = json.loads(Path(a.temps).read_text()) if a.temps else None
    if a.report:
        report(a.report, temps)
        return

    import torch
    from engine import Decider
    from lora_io import load_lora
    from jevstral_torch import Shieldstral

    torch.set_num_threads(a.threads)
    lora_cfg = None
    lora_state = None
    if a.lora:
        lora_state, lora_cfg = load_lora(a.lora)
    m = Shieldstral(a.model_dir, lora=lora_cfg).eval()
    if lora_state:
        m.load_state_dict(lora_state, strict=False)
    d = Decider(m, layout=a.layout, style=a.style, max_state_tokens=a.max_state_tokens or None,
                noul=a.noul, query=a.query)

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    tasks = load_tasks(a.tiers.split(","))
    if a.limit:
        per = {}
        tasks = [x for x in tasks if per.setdefault(x[0], []).append(1) or len(per[x[0]]) <= a.limit]
    with open(a.out, "a") as f:
        for i, (tier, task) in enumerate(tasks):
            if task.id in done:
                continue
            q = task.question
            labels = ["no", "yes"] if q["type"] == "noul" else task.labels
            t0 = time.perf_counter()
            res = d.decide(q, labels, task.state) if a.mode == "verify" else d.decide_list(q, labels, task.state)
            dt = time.perf_counter() - t0
            probs = to_task_probs(task, res["probs"])
            sc = score_task(probs, task)
            rec = {"id": task.id, "tier": tier, "family": task.family, "qtype": q["type"], "labels": labels,
                   "mode": a.mode, "layout": a.layout, "noul": a.noul, "probs": probs, "margins": res["margins"],
                   "expected": task.expected, "correct": sc["correct"], "latency_s": dt,
                   **({"top_token": res["top_token"], "letter_mass": res["letter_mass"]} if a.mode == "list" else {})}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(tasks)}] {tier} {task.family:16s} {q['type']:6s} "
                  f"{'OK ' if sc['correct'] else 'bad'} pred={sc['predicted']} exp={task.expected} {dt:.1f}s",
                  flush=True)
    report(a.out, temps)


if __name__ == "__main__":
    main()
