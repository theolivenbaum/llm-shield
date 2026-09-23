#!/usr/bin/env python3
"""PyTorch Shieldstral 1.0 3B, for decision research: prefix reuse, batched reads, LoRA.

This is not the runtime. `src/Jevstral` is. This file exists for work
the C# path cannot do yet: training (it has no backward pass), and running
thousands of evaluation reads fast enough to compare prompt designs. On a CPU with
AMX, bf16 GEMM through PyTorch runs about 60x faster than the managed kernels.

It follows `tools/reference_shieldstral.py` exactly. RoPE rotates adjacent pairs,
YaRN keeps a magnitude scale of 1.0, and the embeddings are tied. `check` mode
reproduces `tests/fixtures/scores.json`, so a divergence shows up before any
decision number is trusted.

Weights stay bf16. Norms, the RoPE rotation, the attention softmax and the verdict
logits are computed in fp32.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from reference_shieldstral import TekkenTokenizer, yarn_corr_dims  # noqa: E402

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruction provided. Note that the answer can only be "yes" or "no".'
)
SYSTEM_BLOCK = f"[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT]"

LORA_TARGETS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")




class LoRALinear(nn.Module):
    """y = x W^T + (alpha/r) * (x A^T) B^T. W is frozen, and B starts at zero, so step 0 is the base model."""

    def __init__(self, weight: torch.Tensor, rank: int = 0, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.rank = rank
        if rank > 0:
            out_f, in_f = weight.shape
            self.lora_a = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32))
            self.lora_b = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.scale = alpha / rank
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        if self.rank > 0:
            d = self.drop(x).to(torch.bfloat16)
            y = y + (F.linear(F.linear(d, self.lora_a.to(torch.bfloat16)), self.lora_b.to(torch.bfloat16))
                     * self.scale)
        return y

    def merged(self) -> torch.Tensor:
        if self.rank == 0:
            return self.weight.data
        delta = (self.lora_b @ self.lora_a) * self.scale
        return (self.weight.data.float() + delta).to(self.weight.dtype)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


class Block(nn.Module):
    def __init__(self, sd: dict, i: int, cfg: dict, lora: dict | None):
        super().__init__()
        p = f"layers.{i}."
        r = lora["rank"] if lora and i in lora["layers"] else 0
        a = lora.get("alpha", 16.0) if lora else 16.0
        dr = lora.get("dropout", 0.0) if lora else 0.0
        tg = lora.get("targets", LORA_TARGETS) if lora else ()

        def lin(name, key):
            return LoRALinear(sd.pop(p + key), r if name in tg else 0, a, dr)

        self.attn_norm = nn.Parameter(sd.pop(p + "attention_norm.weight").float(), requires_grad=False)
        self.ffn_norm = nn.Parameter(sd.pop(p + "ffn_norm.weight").float(), requires_grad=False)
        self.wq = lin("wq", "attention.wq.weight")
        self.wk = lin("wk", "attention.wk.weight")
        self.wv = lin("wv", "attention.wv.weight")
        self.wo = lin("wo", "attention.wo.weight")
        self.w1 = lin("w1", "feed_forward.w1.weight")
        self.w2 = lin("w2", "feed_forward.w2.weight")
        self.w3 = lin("w3", "feed_forward.w3.weight")
        self.cfg = cfg


class Shieldstral(nn.Module):
    def __init__(self, model_dir: str | Path, lora: dict | None = None):
        super().__init__()
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        p = json.loads((model_dir / "params.json").read_text())
        self.dim, self.n_layers = p["dim"], p["n_layers"]
        self.n_heads, self.n_kv, self.hd = p["n_heads"], p["n_kv_heads"], p["head_dim"]
        self.eps = p["norm_eps"]

        sd = load_file(str(model_dir / "consolidated.safetensors"))
        sd = {k: v for k, v in sd.items() if not k.startswith(("vision_", "pre_mm", "patch_merger"))}
        self.embed = nn.Parameter(sd.pop("tok_embeddings.weight"), requires_grad=False)
        self.norm = nn.Parameter(sd.pop("norm.weight").float(), requires_grad=False)
        self.layers = nn.ModuleList(Block(sd, i, p, lora) for i in range(self.n_layers))
        del sd

        yarn = p["yarn"]
        factor, n_orig = float(yarn["factor"]), int(yarn["original_max_position_embeddings"])
        low, high = yarn_corr_dims(self.hd, n_orig, p["rope_theta"], float(yarn["beta"]), float(yarn["alpha"]))
        i = torch.arange(self.hd // 2, dtype=torch.float32)
        extrap = 1.0 / (p["rope_theta"] ** (2 * i / self.hd))
        interp = extrap / factor
        ramp = ((i - low) / max(0.001, high - low)).clamp(0, 1)
        mix = 1.0 - ramp
        self.register_buffer("freqs", interp * (1 - mix) + extrap * mix, persistent=False)
        self.beta4 = float((p.get("llama_4_scaling") or {}).get("beta", 0.0))
        self.n_orig4 = int((p.get("llama_4_scaling") or {}).get("original_max_position_embeddings", 16384))

        # Residual-stream taps: set tap_layers to {k, ...} and each forward leaves the state
        # after layer k (1-based; 26 = the last layer, pre final norm) in self.taps[k].
        self.tap_layers: set[int] | None = None
        self.taps: dict[int, torch.Tensor] = {}

        self.tok = TekkenTokenizer(model_dir / "tekken.json")
        self.yes_ids, self.no_ids = verdict_ids(self.tok)

    # --------------------------------------------------------------- pieces

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, hd]; pos: [B, T]. Rotates adjacent pairs (x[2i], x[2i+1])."""
        th = pos.float()[:, None, :, None] * self.freqs  # [B,1,T,hd/2]
        cos, sin = th.cos(), th.sin()
        xf = x.float()
        ev, od = xf[..., 0::2], xf[..., 1::2]
        out = torch.stack((ev * cos - od * sin, ev * sin + od * cos), dim=-1).flatten(-2)
        if self.beta4:
            temp = 1.0 + self.beta4 * torch.log1p(torch.floor(pos.float() / self.n_orig4))
            out = out * temp[:, None, :, None]
        return out.to(x.dtype)

    def forward(self, ids: torch.Tensor, valid: torch.Tensor | None = None, past=None,
                grad_from: int = 0, mask: torch.Tensor | None = None, positions: torch.Tensor | None = None):
        """
        ids   [B, T] tokens, right-padded; valid [B, T] bool (True = real token).
        past  None, or (kv list, prefix_len) from `prefix()`, with batch 1 (shared) or B.
        Returns the final hidden states [B, T, dim] (pre-norm) and this chunk's kv list.
        Layers below `grad_from` run without autograd, which saves the backward
        through a frozen stack. LoRA should only live at or above it.
        """
        B, T = ids.shape
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool)
        P = 0 if past is None else past[1]
        pos = (P + torch.arange(T))[None].expand(B, T) if positions is None else positions

        # Attention mask [B, 1, T, P+T]: the whole prefix, plus a causal mask over valid suffix tokens.
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        m = causal[None] & valid[:, None, :]
        if P:
            m = torch.cat([torch.ones(B, T, P, dtype=torch.bool), m], dim=-1)
        m = m[:, None] if mask is None else mask

        h = self.embed[ids]
        new_kv = []
        for l, layer in enumerate(self.layers):
            ctx = torch.enable_grad() if l >= grad_from and torch.is_grad_enabled() else torch.no_grad()
            with ctx:
                a = rmsnorm(h, layer.attn_norm, self.eps)
                q = layer.wq(a).view(B, T, self.n_heads, self.hd).transpose(1, 2)
                k = layer.wk(a).view(B, T, self.n_kv, self.hd).transpose(1, 2)
                v = layer.wv(a).view(B, T, self.n_kv, self.hd).transpose(1, 2)
                q, k = self.rope(q, pos), self.rope(k, pos)
                new_kv.append((k, v))
                if P:
                    pk, pv = past[0][l]
                    if pk.shape[0] != B:
                        pk, pv = pk.expand(B, -1, -1, -1), pv.expand(B, -1, -1, -1)
                    k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)
                h = h + layer.wo(o.transpose(1, 2).reshape(B, T, -1))
                f = rmsnorm(h, layer.ffn_norm, self.eps)
                h = h + layer.w2(F.silu(layer.w1(f)) * layer.w3(f))
            if self.tap_layers is not None and l + 1 in self.tap_layers:
                self.taps[l + 1] = h
        return h, new_kv

    def prefix(self, ids: list[int], grad_from: int = 0):
        """Run a shared prefix once. Returns (kv, length) to pass as `past`."""
        _, kv = self.forward(torch.tensor([ids]), grad_from=grad_from)
        return (kv, len(ids))

    def head(self, h_last: torch.Tensor, token_ids) -> torch.Tensor:
        """Logits of a handful of vocabulary rows, in fp32, from [N, dim] pre-norm states."""
        n = rmsnorm(h_last, self.norm, self.eps).float()
        return n @ self.embed[token_ids].float().T

    def verdict(self, h_last: torch.Tensor) -> torch.Tensor:
        """yes-minus-no logit margin, the best surface form of each (as the moderator reads it)."""
        z = self.head(h_last, self.yes_ids + self.no_ids)
        ny = len(self.yes_ids)
        return z[:, :ny].max(-1).values - z[:, ny:].max(-1).values

    def full_logits(self, h_last: torch.Tensor, fast: bool = False) -> torch.Tensor:
        """
        Full-vocabulary logits. The exact path widens the bf16 embedding table to fp32, a 1.6 GB
        conversion that cost ~1.7 s per call: fine for scoring a handful of label tokens, ruinous once
        per decode step. fast=True multiplies in bf16 and widens only the output (0.06 s). Its
        rounding (~0.1 on logits in the 20s) is harmless for choosing the next greedy token, and it
        is never used to score labels.
        """
        n = rmsnorm(h_last, self.norm, self.eps)
        if fast:
            return F.linear(n.to(torch.bfloat16), self.embed).float()
        return n.float() @ self.embed.float().T

    def encode(self, text: str, bos: bool = False) -> list[int]:
        return self.tok.encode(text, add_bos=bos)

    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if "lora_" in n]

    def lora_state(self) -> dict:
        return {n: p.detach().clone() for n, p in self.named_parameters() if "lora_" in n}


def verdict_ids(tok: TekkenTokenizer) -> tuple[list[int], list[int]]:
    yes, no = [], []
    for i in range(tok.vocab_size):
        s = tok.decode_one(i).strip().strip("\"'.").lower()
        if s == "yes":
            yes.append(i)
        elif s == "no":
            no.append(i)
    return yes, no


def main():
    """Reproduce tests/fixtures/scores.json with bf16 weights."""
    import time

    model_dir = sys.argv[1]
    torch.set_num_threads(4)
    t = time.time()
    m = Shieldstral(model_dir).eval()
    print(f"loaded in {time.time() - t:.1f}s")
    cases = json.loads((TOOLS.parent / "tests/fixtures/scores.json").read_text())
    with torch.no_grad():
        for c in cases:
            user = f"<Instruct>: {c['instruct']}\n\n<Query>: {c['query']}\n\n<Document>: {c['document']}"
            ids = m.encode(f"{SYSTEM_BLOCK}[INST]{user}[/INST]", bos=True)
            t = time.time()
            h, _ = m(torch.tensor([ids]))
            margin = m.verdict(h[:, -1]).item()
            p = 1 / (1 + math.exp(-margin))
            print(f"{c['name']:22s} tokens={len(ids)} (ref {c['tokens']}) score={p:.5f} ref={c['score']:.5f} "
                  f"margin={margin:.3f} ref={c['yes_logit'] - c['no_logit']:.3f} {time.time() - t:.2f}s")


if __name__ == "__main__":
    main()
