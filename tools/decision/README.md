# Typed decisions with Shieldstral

This directory is the research path behind `ShieldstralDecider`. It measures how well a 3B
policy-safety classifier does the kind of typed, calibrated decisions that
[laya](https://github.com/theolivenbaum/laya) and [djev](https://github.com/theolivenbaum/djev-dev)
make, scored on the public tiers of JevBench. It also trains LoRA adapters that the C#
runtime can load once merged.

It runs on a CPU. PyTorch bf16 matmuls on AMX make it about 60× faster than the managed
kernels were before `PanelGemm`, and still several times faster now, which is what makes
training possible here.

| file | what it does |
|---|---|
| `shieldstral_torch.py` | PyTorch Shieldstral: prefix KV reuse, batched option reads, LoRA hooks. `python3 shieldstral_torch.py MODEL_DIR` reproduces `tests/fixtures/scores.json`. |
| `decision_prompts.py` | noul / choice / score → yes/no reads. The prompt the C# `ShieldstralDecider` renders byte for byte. |
| `engine.py` | `Decider`: one shared prefix, one batched pass over the option queries, softmax over log-odds |
| `eval_jevbench.py` | public JevBench tiers, scored with jevbench's own `score_task`, Brier, ECE and TVD |
| `eval_td.py` | `LocalLLaMA/typed-decisions` test split (laya's fine-tune benchmark) |
| `gen_synthetic.py` | 12 procedurally generated decision families, gold labels computed rather than annotated |
| `train_lora.py` | LoRA on the verdict itself, listwise over options, soft targets |
| `fit_temps.py` | per-type temperature, fitted on non-JevBench held-out data |
| `merge_lora_to_gguf.py` | folds an adapter into a GGUF for the C# runtime |
| `jevbench_adapter.py` | a JevBench adapter (`shieldstral_local`) for either backend, plus a runner |

## How a typed decision becomes Shieldstral reads

Shieldstral answers exactly one kind of question: does the Document meet the requirement in
the Query, yes or no. The first token after `[/INST]` holds the whole answer. laya adds a
trained head over an encoder, and djev reads label-token logprobs from a diffusion LM's
answer slots. Here, the model's own verdict is read once per option:

```
[SYSTEM_PROMPT]Judge whether the Document meets…[/SYSTEM_PROMPT][INST]<Instruct>: <preamble>
Question: <instructions>
Exactly one of these answers is correct:
- option <label>: <criterion>
…

<Document>: <state>                                    ← shared prefix, prefilled once
\n\n<Query>: Is option <label> (<criterion>) the correct answer to the question?[/INST]   ← one per option
```

Each option gives a log-odds z_i = logit(yes) − logit(no), and the probabilities are
softmax(z / T). Three findings shaped this:

- **Only verdict reads work.** A single read listing lettered options and asking for the
  letter scores below chance (0.25 / 0.28 on easy / standard). Shieldstral puts nothing on
  letter tokens.
- **A noul must be a contrast.** Asking the question directly and reading P(yes) carries a
  topical yes-bias: a state that is merely about the question leans yes. On easy "fact" items,
  the true cases scored +1.9 to +4.7 and the false ones −0.6 to +1.3, so 5 of the 6 false
  cases came out "yes". Reading the false and the true criterion as two competing options cancels
  the bias: easy-tier fact accuracy 0.58 → 0.92.
- **Document order is a trade.** The model card's order (Instruct, Query, Document) is better
  zero-shot on standard (0.847 vs 0.778), mostly on ordinal and routing. But every option has
  to re-read the document. Instruct, Document, Query lets all options share the document
  prefix, the only affordable way to run a 9-way choice over a 4k-token policy on a CPU. The
  adapter is trained on this order.

## Reproducing

```bash
pip install torch safetensors numpy regex pandas pyarrow gguf
python3 tools/download_shieldstral.py ~/models/shieldstral        # original bf16 weights, 7.2 GiB
python3 tools/decision/shieldstral_torch.py ~/models/shieldstral   # parity with the fixtures

# zero-shot on public JevBench
python3 tools/decision/eval_jevbench.py --out ~/runs/zs.jsonl --layout docfirst
python3 tools/decision/eval_jevbench.py --report ~/runs/zs.jsonl

# training data: typed-decisions (parquet from the HF dataset) + synthetic
python3 tools/decision/gen_synthetic.py --n 3000 --seed 1 --out ~/models/syn_train.jsonl
python3 tools/decision/train_lora.py --data td:~/models/td/train.parquet@300 syn:~/models/syn_train.jsonl@2000 \
    --out ~/models/lora_v1.pt --layout docfirst --max-steps 200 --accum 8 --layers-from 14 --rank 16

# C# runtime: merge and run
python3 tools/decision/merge_lora_to_gguf.py ~/models/shieldstral ~/models/lora_v1.pt --outfile ~/models/decision-q8_0.gguf
dotnet run --project src/LlmShield.Shieldstral.Cli -c Release -- decide ~/models/decision-q8_0.gguf --tasks tasks.jsonl
```

## Contamination

Nothing here trains on JevBench. The training data is the `LocalLLaMA/typed-decisions` train
split plus `gen_synthetic.py`. The generator was implemented from a written spec of decision
*families*, by an agent that never opened the JevBench repository: policy conditions,
final-value extraction, ordinal rubrics, answer adequacy, routing, date arithmetic, rule
precedence, untrusted-content injection and calibrated probability. The spec itself was written
after reading a sample of *public* JevBench items (one per family), and its families mirror the
ones JevBench names. No item text, entity or number was copied. Still, an adapter trained on
this mix is "public-benchmark-directed" in JevBench's sense and has to be disclosed as such.
The honest reading is the held-out half, which only the JevBench maintainers can run. Temperatures are
fitted on held-out typed-decisions and synthetic items, never on JevBench.
