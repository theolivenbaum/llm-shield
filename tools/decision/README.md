# Typed decisions with Shieldstral

This directory is the research path behind `JevstralDecider`. It measures how well a 3B
policy-safety classifier does the kind of typed, calibrated decisions that
[laya](https://github.com/theolivenbaum/laya) and [djev](https://github.com/theolivenbaum/djev-dev)
make, scored on the public tiers of JevBench. It also trains LoRA adapters that the C#
runtime can load once merged.

It runs on a CPU. PyTorch bf16 matmuls on AMX make it about 60× faster than the managed
kernels were before `PanelGemm`, and still several times faster now, which is what makes
training possible here.

| file | what it does |
|---|---|
| `jevstral_torch.py` | PyTorch Shieldstral: prefix KV reuse, batched option reads, LoRA hooks. `python3 jevstral_torch.py MODEL_DIR` reproduces `tests/fixtures/scores.json`. |
| `decision_prompts.py` | noul / choice / score → yes/no reads. The prompt the C# `JevstralDecider` renders byte for byte. |
| `engine.py` | `Decider`: one shared prefix, one batched pass over the option queries, softmax over log-odds |
| `eval_jevbench.py` | public JevBench tiers, scored with jevbench's own `score_task`, Brier, ECE and TVD |
| `eval_td.py` | `LocalLLaMA/typed-decisions` test split (laya's fine-tune benchmark) |
| `gen_synthetic.py` | 12 procedurally generated decision families, gold labels computed rather than annotated |
| `train_lora.py` | LoRA on the verdict itself, listwise over options, soft targets |
| `fit_temps.py` | per-type temperature, fitted on non-JevBench held-out data |
| `merge_lora_to_gguf.py` | folds an adapter into a GGUF for the C# runtime |
| `jevbench_adapter.py` | a JevBench adapter (`jevstral_local`) for either backend, plus a runner |
| `reason.py` | reason, then decide: Ministral-3-3B-Reasoning thinks, then every label is scored after "Final answer:" (continuous batching, resumable) |
| `fit_ensemble.py` | softmax(s/T + w·z) over reasoning label scores and verdict log-odds, fitted on synthetic items only |
| `budget_study.py` | accuracy at every thinking budget below the one a run used, from its saved token ids |
| `speed_reason.py` | prefill, decode (per sequence and aggregate, by row count) and label scoring, in tokens/s |
| `run_reasoning_eval.sh` | all of the reasoning evaluation in order, resumable |

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
\n\n<Query>: Is option <label> correct?[/INST]   ← one per option
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
- **The per-option query only names the option.** Its criterion is already listed in the
  instruction. With the v1 adapter, "Is option X correct?" scores 1.000 / 0.847 on easy /
  standard, against 0.979 / 0.819 when the query restates the criterion, and its branch is a third
  as long. Moving the queries before the document (`qcache`, one token per option) keeps accuracy
  (0.958 / 0.847) but is badly overconfident (ECE 0.41 / 0.34).

## Reproducing

```bash
pip install torch safetensors numpy regex pandas pyarrow gguf
python3 tools/download_shieldstral.py ~/models/shieldstral        # original bf16 weights, 7.2 GiB
python3 tools/decision/jevstral_torch.py ~/models/shieldstral   # parity with the fixtures

# zero-shot on public JevBench
python3 tools/decision/eval_jevbench.py --out ~/runs/zs.jsonl --layout docfirst
python3 tools/decision/eval_jevbench.py --report ~/runs/zs.jsonl

# training data: typed-decisions (parquet from the HF dataset) + synthetic
python3 tools/decision/gen_synthetic.py --n 3000 --seed 1 --out ~/models/syn_train.jsonl
python3 tools/decision/train_lora.py --data td:~/models/td/train.parquet@300 syn:~/models/syn_train.jsonl@2000 \
    --out ~/models/lora_v1.pt --layout docfirst --max-steps 200 --accum 8 --layers-from 14 --rank 16

# C# runtime: merge and run
python3 tools/decision/merge_lora_to_gguf.py ~/models/shieldstral ~/models/lora_v1.pt --outfile ~/models/decision-q8_0.gguf
dotnet run --project src/Jevstral.Cli -c Release -- decide ~/models/decision-q8_0.gguf --tasks tasks.jsonl
```

## Reason, then decide

The verdict decider reads one token per option. That token cannot compute a date window or chain
three rules, which is most of the hard tier. `mistralai/Ministral-3-3B-Reasoning-2512` has the same
architecture and tokenizer, so the same runtime runs it (`ReasoningDecider`, `jev reason`). It
thinks greedily up to a budget, then each label is scored as the continuation of "Final answer:".

Public JevBench, measured on a 4-core VM before the runs were moved to a bigger machine:

| | easy | standard | hard |
|---|---|---|---|
| verdict decider, v1 adapter (docfirst) | 0.979 | 0.819 | 0.423 |
| reasoning, 4096-token budget (easy/std), 1536 (hard) | 1.000 | 0.931 | 0.640 |
| reasoning, hard: the 50 items cut off at 1536 continued to 2560 | | | **0.667** |

- **The hard tier is budget-bound.** At 1536 tokens only 30% of hard items closed their
  thinking. `temporal_numeric` (0.27) and `long_policy` (0.42), the weakest families, almost never
  did. Continuing the cut-off items to 2560 tokens took them from 23 to 26 correct.
- **The two paths fail on different items.** Reasoning is weak on routing (0.667 on standard,
  where the verdict decider has 1.0). On hard, 0.748 of items are right under one path or the
  other: roughly the ceiling for `fit_ensemble.py`.
- **Synthetic calibration items (150, seed 99):** reasoning at 4096 gets 0.90 (routing 0.75).
- Latency at these budgets is minutes per item on 4 cores: a batch of 3–6 rows took 60–1900 s.
  `speed_reason.py` breaks that down.

Still to run (`run_reasoning_eval.sh` does all of it, in order):
1. hard at the full 4096-token budget for every item;
2. the verdict decider on the synthetic items, then the ensemble fit, applied to JevBench;
3. the budget study (low / medium / high = 512 / 1536 / 4096 and every point between);
4. tokens/s for torch and for `jev reason`.

```bash
MODELS=~/models JEVBENCH=~/jevbench THREADS=32 KV_TOKENS=120000 BATCH=32 \
    tools/decision/run_reasoning_eval.sh ~/runs/reasoning
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
