#!/usr/bin/env python3
"""Where the time goes in reason-then-decide, in tokens per second.

A decision is three phases: prefill the prompt, decode the thinking greedily, score the labels
after "Final answer:". Decode dominates and is bound by streaming the weights, so its cost per
step barely depends on how many rows share it: this measures a step at several row counts and
reports tokens/s per sequence and in aggregate. Run it on an idle machine.

  python3 speed_reason.py --rows 1,4,8,12 --steps 64 --out speed.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reason import PAD, Stream, render, score_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/user/models/ministral3b-reasoning")
    ap.add_argument("--rows", default="1,4,8,12")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--context", type=int, default=1024, help="thinking already cached before the timed steps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()

    from eval_jevbench import load_tasks
    from jevstral_torch import Shieldstral

    torch.set_num_threads(a.threads)
    m = Shieldstral(a.model_dir).eval()
    system = (Path(a.model_dir) / "SYSTEM_PROMPT.txt").read_text().strip()
    tasks = [t for _, t in load_tasks(["original"])]
    prompts = []
    for t in tasks:
        labels = ["no", "yes"] if t.question["type"] == "noul" else t.labels
        prompts.append((t, labels, m.encode(render(t.question, labels, t.state, system) + "[THINK]", bos=True)))
    prompts.sort(key=lambda x: len(x[2]))
    median = prompts[len(prompts) // 2]
    out = {"threads": a.threads, "prompt_tokens": len(median[2])}

    # Prefill: one prompt, 128-token chunks as reason.py runs it.
    st = Stream(m, 1, len(median[2]) + 8)
    st.prefill(0, median[2][:64])  # warm-up
    t0 = time.perf_counter()
    st.prefill(0, median[2])
    dt = time.perf_counter() - t0
    out["prefill"] = {"tokens": len(median[2]), "seconds": dt, "tokens_per_s": len(median[2]) / dt}
    print(f"prefill {len(median[2])} tokens: {dt:.2f}s = {len(median[2]) / dt:.0f} tok/s", flush=True)

    # Decode: every row holds the median prompt plus --context tokens of prior thinking, so attention
    # spans what a mid-reasoning step sees.
    out["decode"] = []
    ctx = median[2] + [PAD] * a.context
    for R in [int(x) for x in a.rows.split(",")]:
        st = Stream(m, R, len(ctx) + a.steps + 8)
        st.prefill(0, ctx)
        for r in range(1, R):  # the cache contents do not change the cost; copy row 0
            for l in range(len(m.layers)):
                st.k[l][r] = st.k[l][0]
                st.v[l][r] = st.v[l][0]
            st.len[r] = st.len[0]
        live = torch.ones(R, dtype=torch.bool)
        tok = torch.full((R,), 1000, dtype=torch.long)
        for _ in range(4):  # warm-up
            m.full_logits(st.step(tok, live), fast=True).argmax(-1)
        t0 = time.perf_counter()
        for _ in range(a.steps):
            tok = m.full_logits(st.step(tok, live), fast=True).argmax(-1)
        dt = (time.perf_counter() - t0) / a.steps
        row = {"rows": R, "step_s": dt, "per_sequence_tokens_per_s": 1 / dt, "aggregate_tokens_per_s": R / dt}
        out["decode"].append(row)
        print(f"decode rows={R:2d}: {dt * 1000:.0f} ms/step, {1 / dt:.1f} tok/s per sequence, "
              f"{R / dt:.1f} tok/s aggregate", flush=True)
        del st

    # Scoring: a prefill of prompt + reasoning + "Final answer:" and one short branch per label.
    t, labels, p = median
    ctx = p + [PAD] * 512 + [m.tok.special_by_str["[/THINK]"]] + m.encode("\nFinal answer:")
    score_labels(m, ctx[:200], labels)  # warm-up
    t0 = time.perf_counter()
    score_labels(m, ctx, labels)
    dt = time.perf_counter() - t0
    out["score"] = {"context_tokens": len(ctx), "labels": len(labels), "seconds": dt}
    print(f"score {len(labels)} labels over {len(ctx)} tokens: {dt:.2f}s", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
