# Handover: Jevstral on JevBench

State of the work as of 2026-09-24, and what to do next. The detail behind each point is in
`tools/decision/README.md` (the research path), `README.md` (runtime and models) and `CLAUDE.md`
(invariants).

## The goal and where it stands

Goal: typed decisions (noul / choice / score) from a 3B model on a CPU, scoring **above 0.7 on
JevBench's hard tier** without giving up easy and standard. **Not reached: hard is at 0.667.**

| public JevBench | easy (48) | standard (72) | hard (111) |
|---|---|---|---|
| verdict decider, zero-shot | 0.979 | 0.778 | 0.396 |
| verdict decider, `adapters/jevstral-v1` | 0.979 | 0.819 | 0.423 |
| verdict decider, v1, short query (C# default) | 1.000 | 0.847 | — |
| reasoning (Ministral-3-3B-Reasoning), 4096 easy/std, 1536 hard | 1.000 | 0.931 | 0.640 |
| reasoning, hard, the 50 cut-off items continued to 2560 tokens | | | **0.667** |

Over all 231 public items that is about 0.82. The judge tier is private and has not been run.

## Two decision paths

1. **Verdict decider** (`JevstralDecider`, `jev decide`): Shieldstral 1.0 3B plus the v1 LoRA,
   merged into `Jevstral-1.0-3B-<Q>.gguf`. One yes/no read per option and a softmax over the
   log-odds. Takes seconds, uses a layered cache (question, then data, then short option
   branches), and is bit-identical to a cold call. Strong on routing and ordinal questions; one
   token cannot do arithmetic or multi-step rules.
2. **Reason, then decide** (`ReasoningDecider`, `jev reason`): Mistral's
   Ministral-3-3B-Reasoning, unchanged; the same architecture and tokenizer run on the same
   runtime. It thinks greedily up to a budget, then scores each label after "Final answer:".
   Minutes per item on 4 cores. Carries the hard tier, but is weak on routing (0.667 on
   standard, where the verdict decider gets 1.0).

Both are published at `https://models.curiosity.ai/jevstral/`: Q8_0, Q5_1 and Q4_0 of each, the
reasoning system prompt, and `SHA256SUMS`. `jev download [decider|reasoning|all] [q8_0|q5_1|q4_0]`
fetches them and checks each file's SHA-256. `tools/build_models.sh` rebuilds them bit for bit:
the reasoning Q8_0 built on two machines has the same checksum.

## Hard tier, per family

With the 2560-token continuation merged. "either" means right under at least one of the two
paths, roughly the ceiling for an ensemble.

| family | n | reasoning | verdict v1 | either |
|---|---|---|---|---|
| adversarial | 6 | 1.00 | 0.50 | 1.00 |
| ambiguous | 7 | 0.71 | 0.29 | 0.86 |
| judge_hard | 17 | 0.94 | 0.76 | 0.94 |
| long_policy | 19 | **0.42** | 0.21 | 0.63 |
| multi_hop | 18 | 0.61 | 0.33 | 0.78 |
| probability | 10 | 0.90 | 0.40 | 1.00 |
| routing_hard | 5 | 1.00 | 1.00 | 1.00 |
| temporal_numeric | 15 | **0.27** | 0.13 | 0.33 |
| tradeoff | 6 | 0.50 | 0.33 | 0.67 |
| trap | 8 | 0.88 | 0.75 | 1.00 |
| **all** | 111 | **0.667** | 0.423 | 0.775 |

What the runs showed:
- **Hard is budget-bound, but not uniformly.** At 1536 tokens only 30% of hard items finished
  thinking. Continuing the 50 cut-off items to 2560 took them from 23 to 26 correct, and 56 of 111
  items now close. But a fresh 4096-token rerun of 10 `judge_hard` items (it was stopped there)
  got 8 right, against 10 at the shorter budget. Longer is not free: the budget study should
  decide per tier, or per type.
- **The two paths fail on different items.** The ensemble (`fit_ensemble.py`) has room from
  0.667 up to 0.775. Fit it on synthetic items only.
- **temporal_numeric and long_policy are the real problem.** Even the ceiling is 0.33 on
  temporal_numeric. These need more than a budget: see "If hard is still below 0.7".
- Synthetic calibration items (150, seed 99): reasoning at 4096 gets 0.90 (routing 0.75).

## What has not been done

- **Hard at the full 4096-token budget for all items.** 54 hard items have no saved token ids,
  because they came from an early run that did not record them.
- **The ensemble fit.** `fit_ensemble.py` is written, and its apply path has been run end to end
  on saved runs with untuned parameters. It has not been fitted: the verdict decider has not been
  run on the synthetic calibration items.
- **The budget study** (`budget_study.py`: low / medium / high and every point between) has not
  been run.
- **Tokens/s** has not been measured cleanly. Every timing so far was taken under load. What is
  known: batches of 3–6 rows took 60–1900 s per item on 4 cores, and the verdict decider's layered
  cache takes a 1-thread decision from 15.3 s cold to 7.5–9.3 s warm. `speed_reason.py` and the
  `jev reason` step of the pipeline measure prefill, decode by row count, and label scoring.
- Temperatures for the C# runtime, and porting the ensemble into C#.

## How to run it on a bigger machine

```bash
# weights (bf16, ~7.2 GiB each) for the research path; the GGUF only for the C# speed step
python3 tools/download_shieldstral.py ~/models/shieldstral
python3 tools/download_shieldstral.py ~/models/ministral3b-reasoning --repo mistralai/Ministral-3-3B-Reasoning-2512
dotnet run --project src/Jevstral.Cli -c Release -- download reasoning --to ~/models
git clone https://github.com/theolivenbaum/jevbench ~/jevbench

pip install torch safetensors numpy regex pandas pyarrow gguf
MODELS=~/models JEVBENCH=~/jevbench THREADS=32 KV_TOKENS=120000 BATCH=32 \
    tools/decision/run_reasoning_eval.sh ~/runs/reasoning
```

`run_reasoning_eval.sh` runs, in order:
1. reasoning on the public tiers;
2. reasoning on the synthetic calibration items;
3. the verdict decider on both;
4. the ensemble, fitted on the synthetic items and applied to JevBench;
5. the budget study;
6. speed measurements.

Every step appends to its output and skips the ids already there, so run it again after an
interruption. The KV cache is ~106 KB per token in bf16; `KV_TOKENS` × that is the cache's memory.

### Reusing what this VM already computed

`tools/decision/runs/2026-09-vm/` holds the outputs (gzipped JSONL, with the token ids of every
reasoning run that saved them):

| file | what it is |
|---|---|
| `reason3b_es` | reasoning, easy + standard, 4096 budget (120 items) |
| `reason3b_hard` | reasoning, hard, 1536 budget (111 items; 54 without `gen_ids`) |
| `reason3b_hard_ext` | the 50 cut-off hard items continued to 2560 (`--continue-from`) |
| `reason3b_hard_rerun` | 10 of the 54 rerun from scratch at 4096 (stopped there) |
| `reason3b_synval` | reasoning on the 150 synthetic calibration items, 4096 |
| `lora1_all` | verdict decider, v1, docfirst, full query, all tiers |
| `lora1_short_es` | verdict decider, v1, short query, easy + standard |

To skip regenerating them, unpack into the pipeline's output directory under its names:
- `reason3b_synval` → `reason_syn.jsonl`;
- `lora1_all` → `verdict_jev.jsonl`;
- `reason3b_es` → the first lines of `reason_jev.jsonl`.

The pipeline then runs only what is missing. For the budget study, give `budget_study.py`
several `--runs`: the last row per id wins.

```bash
mkdir -p ~/runs/reasoning && cd tools/decision/runs/2026-09-vm
zcat reason3b_synval.jsonl.gz > ~/runs/reasoning/reason_syn.jsonl
zcat lora1_all.jsonl.gz       > ~/runs/reasoning/verdict_jev.jsonl
zcat reason3b_es.jsonl.gz     > ~/runs/reasoning/reason_jev.jsonl   # hard then runs fresh at 4096
```

## If hard is still below 0.7

In order of expected value:
1. **The ensemble**, fitted on synthetic items. Its ceiling is 0.775.
2. **Budget per type**, from the budget study: longer where it helps (cut-off items), shorter
   where it hurts (`judge_hard` above).
3. **temporal_numeric / long_policy.** Two ideas, both untried:
   - Sample several reasoning chains and vote (self-consistency). This costs N× the decode, which
     batches well because decode is bound by streaming the weights.
   - Fine-tune the reasoning model with a LoRA on synthetic date-arithmetic and long-policy items
     with computed answers (`gen_synthetic.py` has both families; they need longer, harder
     variants). `train_lora.py` trains the verdict head today; the reasoning model needs an SFT
     objective on its own "Final answer:" scoring.
4. A second verdict adapter (v2), trained on the short query that is now the default. v1 was
   trained on the full query. This helps standard and routing, not hard.

**Contamination rule:** nothing is fitted or trained on JevBench items. Temperatures, the
ensemble and adapters use synthetic items (`gen_synthetic.py`, seed 99 for validation, seed 1 for
training) and typed-decisions. The synthetic families were written from a spec of JevBench's
family names, which has to be disclosed ("public-benchmark-directed"; see
`tools/decision/README.md`).

## Things that bit this work

- **Memory.** The weights are memory-mapped. When anonymous memory grows (the KV cache, the
  allocator), Linux evicts them and every step reads from disk: a 5 s step became 15 s. What
  helps: `MALLOC_ARENA_MAX=2`, `malloc_trim` between batches (reason.py and train_lora.py do
  this), `KV_TOKENS` sized to leave the weights resident, and 128-token prefill chunks.
- **`pkill -f pattern` matches the shell running it.** Kill by PID, or use a `[p]attern` so the
  pattern does not match itself.
- **Disk.** A build needs ~15 GiB. On WSL, write to the Linux side: `/mnt/<drive>` returned short
  writes and EIO under multi-GB writes (`build_models.sh` stages in `JEVSTRAL_WORK` for this).
- **Continuous batching:** a refilled row's first token has to be recorded before it is fed. The
  bug that did not do this shifted every later item by one token. It is fixed, with a comment in
  `reason_stream`.
- **Decode logits:** do not widen the 131k×3072 embedding to fp32 every step (1.7 s per step).
  `full_logits(fast=True)` stays in bf16.
- **Republishing models:** the checksums are in `JevstralModels.PublishedSha256`. Rebuild, then
  update the table from the new `SHA256SUMS`, or every download is rejected.
