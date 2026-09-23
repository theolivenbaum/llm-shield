#!/usr/bin/env python3
"""Reason, then decide: a Ministral-3 reasoning model thinks, then the labels are scored.

Shieldstral's single verdict token cannot compute a date window or chain three tables; the
hard tier is mostly that. Ministral-3-3B-Reasoning shares Shieldstral's architecture and
tokenizer (the same Jevstral runtime runs it), and it thinks in [THINK]…[/THINK] before it
answers. This script:

  1. prompts it with the state, the question and the options, asking for one label;
  2. decodes the reasoning greedily, several items at once (left-padded batch, per-row
     positions, a preallocated KV cache), up to a thinking budget, then closes [/THINK];
  3. scores every label as a continuation of "Final answer: " with the reasoning in context:
     a trunk plus one branch per label, the same shape as Jevstral's option reads. The
     distribution is the softmax of the mean per-token log-probability (length-normalised),
     so a label's probability is not its first token's;
  4. writes JSONL in eval_jevbench's format, so --report and the probe tools read it.

  python3 reason.py --model-dir ~/models/ministral3b-reasoning --tiers hard --out hard_reason.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decision_prompts import option_lines, state_text  # noqa: E402
from jevstral_torch import Shieldstral, rmsnorm  # noqa: E402

PAD = 11


def render(q, labels, state, system):
    qtype = q["type"]
    if qtype == "noul":
        c = q.get("criteria") or {}
        opts = [("no", c.get("false", "the statement does not hold")), ("yes", c.get("true", "the statement holds"))]
    else:
        opts = option_lines(qtype, labels, q.get("criteria"))
    body = "\n".join(f"- {lab}: {desc}" if desc else f"- {lab}" for lab, desc in opts)
    user = (f"{state_text(state)}\n\n---\nQuestion: {q.get('instructions', '').strip()}\n\n"
            f"Possible answers (choose exactly one label):\n{body}\n\n"
            f"Work it out from the facts above, applying every stated rule, condition and exception. "
            f"End with a line of the form 'Final answer: <label>'.")
    return f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]"


class Batch:
    """A left-padded batch with a preallocated KV cache, for greedy decoding."""

    def __init__(self, m: Shieldstral, prompts: list[list[int]], max_new: int):
        self.m = m
        B = len(prompts)
        L = max(len(p) for p in prompts)
        self.cap = L + max_new + 64
        self.pad = torch.tensor([L - len(p) for p in prompts])
        ids = torch.full((B, L), PAD, dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p)
        self.k = [torch.zeros(B, m.n_kv, self.cap, m.hd, dtype=torch.bfloat16) for _ in m.layers]
        self.v = [torch.zeros(B, m.n_kv, self.cap, m.hd, dtype=torch.bfloat16) for _ in m.layers]
        self.len = 0
        self.key_ok = torch.zeros(B, self.cap, dtype=torch.bool)
        self.key_ok[:, :L] = torch.arange(L)[None] >= self.pad[:, None]
        self.ids = ids

    @torch.no_grad()
    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """Appends ids [B, T]; returns the last position's hidden state [B, dim]."""
        m, B, T = self.m, ids.shape[0], ids.shape[1]
        s = self.len
        pos = (torch.arange(s, s + T)[None] - self.pad[:, None]).clamp_min(0)
        causal = torch.ones(T, s + T, dtype=torch.bool).tril(s)
        mask = (causal[None] & self.key_ok[:, None, :s + T])[:, None]
        h = m.embed[ids]
        for l, layer in enumerate(m.layers):
            a = rmsnorm(h, layer.attn_norm, m.eps)
            q = layer.wq(a).view(B, T, m.n_heads, m.hd).transpose(1, 2)
            k = layer.wk(a).view(B, T, m.n_kv, m.hd).transpose(1, 2)
            v = layer.wv(a).view(B, T, m.n_kv, m.hd).transpose(1, 2)
            q, k = m.rope(q, pos), m.rope(k, pos)
            self.k[l][:, :, s:s + T] = k
            self.v[l][:, :, s:s + T] = v
            o = F.scaled_dot_product_attention(q, self.k[l][:, :, :s + T], self.v[l][:, :, :s + T],
                                               attn_mask=mask, enable_gqa=True)
            h = h + layer.wo(o.transpose(1, 2).reshape(B, T, -1))
            f = rmsnorm(h, layer.ffn_norm, m.eps)
            h = h + layer.w2(F.silu(layer.w1(f)) * layer.w3(f))
        self.len += T
        return h[:, -1]

    def prefill(self, chunk=512):
        h = None
        for c in range(0, self.ids.shape[1], chunk):
            h = self.step(self.ids[:, c:c + chunk])
        return h

    def mark(self, ok: torch.Tensor):
        """Key validity for the position just written (False for rows that already finished)."""
        self.key_ok[:, self.len - 1] = ok


@torch.no_grad()
def reason_batch(m, prompts, max_think, end_think, eos, answer_ids):
    """Greedy reasoning for a batch; returns each row's generated token ids (think + answer line)."""
    B = len(prompts)
    batch = Batch(m, prompts, max_think + 24)
    h = batch.prefill()
    out = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool)
    for step in range(max_think):
        logits = m.full_logits(h)
        nxt = logits.argmax(-1)
        for b in range(B):
            if done[b]:
                continue
            t = int(nxt[b])
            out[b].append(t)
            if t == eos or t == end_think:
                done[b] = True
        if bool(done.all()):
            break
        feed = torch.where(done, torch.full_like(nxt, PAD), nxt)
        h = batch.step(feed[:, None])
        batch.mark(~done)
    return out


def text_of(m, ids):
    return b"".join(m.tok.id_to_bytes.get(i, b"") for i in ids).decode("utf-8", "replace")


@torch.no_grad()
def score_labels(m, context_ids, labels):
    """
    Mean log-probability of " <label>" after the context, per label. The context minus its
    last token is a shared trunk. Each branch is that last token followed by the label, so
    branch position j predicts label token j.
    """
    from engine import Decider

    past = Decider(m).run_prefix(context_ids[:-1])
    branches = [[context_ids[-1]] + m.encode(" " + lab) for lab in labels]
    n, T = len(branches), max(len(b) for b in branches)
    ids = torch.zeros(n, T, dtype=torch.long)
    valid = torch.zeros(n, T, dtype=torch.bool)
    for i, b in enumerate(branches):
        ids[i, :len(b)] = torch.tensor(b)
        valid[i, :len(b)] = True
    h, _ = m(ids, valid=valid, past=past)
    scores = []
    for i, b in enumerate(branches):
        k = len(b) - 1
        lg = torch.log_softmax(m.full_logits(h[i, :k]), -1)
        scores.append(float(lg[torch.arange(k), torch.tensor(b[1:])].sum()) / k)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/user/models/ministral3b-reasoning")
    ap.add_argument("--tiers", default="hard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16, help="most rows per batch")
    ap.add_argument("--kv-tokens", type=int, default=44000,
                    help="rows x (prompt + think budget) per batch; bounds the KV cache (~106 KB per token)")
    ap.add_argument("--max-think", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ids", help="comma-separated task ids to run")
    ap.add_argument("--syn", help="gen_synthetic.py JSONL instead of JevBench (for calibration fits)")
    a = ap.parse_args()

    from eval_jevbench import load_tasks, to_task_probs
    from jevbench.scoring import score_task

    torch.set_num_threads(a.threads)
    m = Shieldstral(a.model_dir).eval()
    system = (Path(a.model_dir) / "SYSTEM_PROMPT.txt").read_text().strip()
    end_think = m.tok.special_by_str.get("[/THINK]")
    eos = 2

    if a.syn:
        from types import SimpleNamespace
        tasks = []
        for line in open(a.syn):
            x = json.loads(line)
            labels = ["no", "yes"] if x["question"]["type"] == "noul" else x["labels"]
            tasks.append(("syn", SimpleNamespace(id=x["id"], family=x["family"], question=x["question"],
                                                 labels=labels, state=x["state"],
                                                 expected=max(x["target"], key=x["target"].get),
                                                 target=x["target"])))
    else:
        tasks = load_tasks(a.tiers.split(","))
    if a.ids:
        want = set(a.ids.split(","))
        tasks = [x for x in tasks if x[1].id in want]
    if a.limit:
        tasks = tasks[:a.limit]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    tasks = [x for x in tasks if x[1].id not in done]
    # Similar lengths batch together, so padding stays small.
    tasks.sort(key=lambda x: len(state_text(x[1].state)))

    with open(a.out, "a") as f:
        batches, cur = [], []
        for x in tasks:
            n = len(m.encode(render(x[1].question, x[1].labels, x[1].state, system), bos=True)) + a.max_think
            if cur and (len(cur) >= a.batch or (len(cur) + 1) * max(n, cur_max) > a.kv_tokens):
                batches.append(cur)
                cur = []
            cur_max = max(n, cur_max) if cur else n
            cur.append(x)
        if cur:
            batches.append(cur)
        for chunk in batches:
            t0 = time.time()
            prompts, labs = [], []
            for tier, t in chunk:
                labels = ["no", "yes"] if t.question["type"] == "noul" else t.labels
                labs.append(labels)
                prompts.append(m.encode(render(t.question, labels, t.state, system) + "[THINK]", bos=True))
            gens = reason_batch(m, prompts, a.max_think, end_think, eos, None)
            gen_s = time.time() - t0
            for (tier, t), labels, p, g in zip(chunk, labs, prompts, gens):
                g = [x for x in g if x != eos]
                if end_think not in g:
                    g = g + [end_think]
                think_end = g.index(end_think) + 1
                ctx = p + g[:think_end] + m.encode("\nFinal answer:")
                sc = score_labels(m, ctx, labels)
                mx = max(sc)
                e = [math.exp(x - mx) for x in sc]
                probs = {lab: v / sum(e) for lab, v in zip(labels, e)}
                if a.syn:
                    tp = probs
                    pred = max(tp, key=tp.get)
                    res = {"correct": pred == t.expected, "predicted": pred}
                else:
                    tp = to_task_probs(t, probs)
                    res = score_task(tp, t)
                rec = {"id": t.id, "tier": tier, "family": t.family, "qtype": t.question["type"], "labels": labels,
                       "mode": "reason", "probs": tp, "margins": sc, "expected": t.expected,
                       "correct": res["correct"], **({"target": t.target} if a.syn else {}), "think_tokens": think_end, "closed": end_think in gens[chunk.index((tier, t))],
                       "answer_text": text_of(m, g[think_end:])[-200:], "reasoning": text_of(m, g[:think_end])[-3000:],
                       "latency_s": gen_s / len(chunk)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{t.id:40s} {'OK ' if res['correct'] else 'bad'} pred={res['predicted']} exp={t.expected} "
                      f"think={think_end} {gen_s / len(chunk):.0f}s/item", flush=True)


if __name__ == "__main__":
    main()
