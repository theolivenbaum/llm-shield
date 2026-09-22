"""Runs decision reads through the PyTorch Shieldstral: shared prefix once, options batched."""
from __future__ import annotations

import math

import torch

from decision_prompts import build_list_read, build_reads

PREFIX_CHUNK = 1024


class Decider:
    def __init__(self, model, layout="docfirst", style="listed", temperature=None, max_state_tokens=None,
                 noul="contrast"):
        self.noul = noul
        self.m = model
        self.layout = layout
        self.style = style
        # Per question type, or "<type>:<bucket>" as laya does; a divisor on the log-odds.
        self.temperature = temperature or {}
        self.max_state_tokens = max_state_tokens
        self._letter_ids = None

    # ---------------------------------------------------------------- core

    def encode_prefix(self, text: str) -> list[int]:
        return self.m.encode(text, bos=True)

    def run_prefix(self, ids: list[int], grad_from: int = 0):
        """Chunked prefill, so a 4k-token policy never builds a 4k x 4k mask per head at once."""
        past = None
        for s in range(0, len(ids), PREFIX_CHUNK):
            chunk = torch.tensor([ids[s:s + PREFIX_CHUNK]])
            _, kv = self.m(chunk, past=past, grad_from=grad_from)
            if past is None:
                past = (kv, chunk.shape[1])
            else:
                merged = [(torch.cat([pk, k], 2), torch.cat([pv, v], 2)) for (pk, pv), (k, v) in zip(past[0], kv)]
                past = (merged, past[1] + chunk.shape[1])
        return past

    def run_suffixes(self, past, suffix_ids: list[list[int]], grad_from: int = 0):
        """Final pre-norm hidden state at the last token of each suffix: [N, dim]."""
        n, T = len(suffix_ids), max(len(s) for s in suffix_ids)
        ids = torch.zeros(n, T, dtype=torch.long)
        valid = torch.zeros(n, T, dtype=torch.bool)
        for i, s in enumerate(suffix_ids):
            ids[i, :len(s)] = torch.tensor(s)
            valid[i, :len(s)] = True
        h, _ = self.m(ids, valid=valid, past=past, grad_from=grad_from)
        last = torch.tensor([len(s) - 1 for s in suffix_ids])
        return h[torch.arange(n), last]

    def margins(self, q, labels, state, grad_from: int = 0):
        """Yes-minus-no log-odds for each read of this decision, and the read kind."""
        prefix, suffixes, kind = build_reads(q, labels, self.clip(state), self.layout, self.style, self.noul)
        past = self.run_prefix(self.encode_prefix(prefix), grad_from)
        sfx = [self.m.encode(s) for s in suffixes]
        h = self.run_suffixes(past, sfx, grad_from)
        return self.m.verdict(h), kind, h

    def clip(self, state):
        if not self.max_state_tokens:
            return state
        from decision_prompts import state_text
        text = state_text(state)
        ids = self.m.encode(text)
        if len(ids) <= self.max_state_tokens:
            return state
        raw = b"".join(self.m.tok.id_to_bytes.get(i, b"") for i in ids[:self.max_state_tokens])
        return raw.decode("utf-8", errors="ignore") + " [...]"

    # ----------------------------------------------------------- decisions

    def temp_for(self, qtype, k):
        b = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        return float(self.temperature.get(f"{qtype}:{b}", self.temperature.get(qtype, 1.0)))

    @torch.no_grad()
    def decide(self, q, labels, state) -> dict:
        z, kind, _ = self.margins(q, labels, state)
        return self.to_probs(z, kind, q["type"], labels)

    def to_probs(self, z, kind, qtype, labels):
        t = self.temp_for(qtype, len(labels))
        if kind == "binary":
            p = 1.0 / (1.0 + math.exp(-float(z[0]) / t))
            probs = {"yes": p, "no": 1.0 - p}
            if "true" in labels:
                probs = {"true": p, "false": 1.0 - p}
            return {"probs": probs, "margins": [float(z[0])]}
        pr = torch.softmax(z.float() / t, 0).tolist()
        labs = ["no", "yes"] if qtype == "noul" else labels
        return {"probs": dict(zip(labs, pr)), "margins": [float(x) for x in z]}

    @torch.no_grad()
    def decide_list(self, q, labels, state) -> dict:
        prefix, suffix, letters = build_list_read(q, labels, self.clip(state))
        past = self.run_prefix(self.encode_prefix(prefix))
        h = self.run_suffixes(past, [self.m.encode(suffix)])
        ids = [self.m.encode(L)[0] for L in letters]
        z = self.m.head(h, ids)[0]
        full = self.m.full_logits(h)[0]
        top = self.m.tok.decode_one(int(full.argmax()))
        pr = torch.softmax(z, 0).tolist()
        labs = labels if q["type"] != "noul" else ["no", "yes"]
        return {"probs": dict(zip(labs, pr)), "margins": [float(x) for x in z], "top_token": top,
                "letter_mass": float(torch.logsumexp(z, 0) - torch.logsumexp(full, 0))}
