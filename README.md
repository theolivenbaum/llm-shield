# Jevstral

Typed, calibrated decisions from a 3B language model, on a CPU, in 100% managed .NET 10.

You give Jevstral a piece of state (a ticket, a policy and a claim, a request and a
response) and a typed question. It returns a probability distribution over the question's
exact labels. The question types are the System One ones used by Jev, laya and djev, and
scored by [JevBench](https://github.com/theolivenbaum/jevbench):

| type | question | answer |
|---|---|---|
| `noul` | yes or no, with a criterion for each | P(yes) |
| `choice` | one of N labelled options, each with a description | a distribution over the labels |
| `score` | an ordinal level with a description per level | a distribution over the levels |

```csharp
using var jev = await JevstralDecider.OpenAsync("Jevstral-q8_0.gguf");

DecisionResult r = await jev.DecideAsync(
    DecisionQuestion.Choice("Which intent does the user's message express?",
        new("track_order", "Wants to know where an order is"),
        new("cancel_order", "Wants to cancel an order"),
        new("billing_question", "Asks about a charge or invoice")),
    state: "Please cancel order 5521, I no longer need it.");

Console.WriteLine($"{r.Label} {r.Confidence:P0}");      // cancel_order, and how sure
```

## How it decides

Jevstral is built on Mistral's
[Shieldstral 1.0 3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B), a Ministral-3
decoder trained to answer one kind of question: does this document meet this requirement,
yes or no. Every typed question is reduced to that primitive:

- **one read per option**: "is option X (its criterion) the correct answer?" The log-odds
  z_i = logit(yes) − logit(no) of each read compete in a softmax(z / T).
- **a noul is a two-option contrast** between its false and its true criterion. Asking the
  question directly leaves a topical bias, because a document that is merely *about* the
  question leans yes. Two reads share the bias, and the softmax cancels it.
- **one forward pass per decision**: the instruction and the state are a shared causal
  trunk, and each option's query is a branch that attends to the trunk and to itself
  (`ForwardTreeAsync`). A 9-way choice over a 4k-token policy costs one pass, not nine.
- **only the verdict rows** of the 131072-row LM head are evaluated.

`tools/decision` holds the PyTorch research path that trains adapters for this format:
LoRA on the verdict itself, listwise over the options, against soft targets. Its
[README](tools/decision/README.md) has the findings and the contamination notes.

## Results on public JevBench

Zero-shot, public items only. The judge tier and the held-out halves are private. The
reference rows are from JevBench v1.3's leaderboard, measured on all items.

| system | easy | standard | hard | notes |
|---|---|---|---|---|
| Jevstral zero-shot, card order (Instruct, Query, Document) | 0.958 | 0.847 | — | each option re-reads the state |
| Jevstral zero-shot, shared-prefix order (Instruct, Document, Query) | 0.979 | 0.778 | 0.396 | hard ECE 0.22, TVD 0.33 |
| laya (ModernBERT-large 421M, fine-tuned) | 0.944 | 0.729 | 0.341 | |
| djev (DiffusionGemma 26B-A4B, GPU) | 1.00 | 0.979 | 0.695 | |
| Jev 1.13 | 1.00 | 0.990 | 0.741 | |

The shallow families are already strong: intent, extraction and routing, and on hard,
routing_hard at 1.0 and adversarial at 0.67. Multi-hop tables (0.22) and date and number
reasoning (0.20) are near chance zero-shot. That is what the adapters target.

## Performance

A 4-core Sapphire Rapids VM, Q8_0 / Q5_1, managed code only:

| | before | now |
|---|---|---|
| prefill, 256 tokens, 4 threads | 8 tokens/s | 61–65 tokens/s |
| decode, 4 threads | 0.9 tokens/s | 3.1–4.1 tokens/s |
| 3072×3072 matmul, 256 tokens, 1 thread | 15–18 GFLOP/s | 120–129 GFLOP/s |
| a 5-option decision over a short message | | ~4 s |
| a decision over a 4k-token policy | | ~90 s |

Prefill is a register-tiled GEMM (`PanelGemm`):
- 64 weight rows are decoded and transposed in registers into an L2-resident panel;
- the panel is streamed against six tokens at a time, with 24 `Vector512` accumulators live.

Dequantization is vectorised, and attention runs on transposed keys with tiled queries.
`CLAUDE.md` explains each step and the invariants that keep the prefix cache and the
branched pass bit-identical to running things one at a time.

## Command line

```bash
dotnet run --project src/Jevstral.Cli -c Release -- decide  model.gguf --tasks tasks.jsonl [--temps temps.json]
dotnet run --project src/Jevstral.Cli -c Release -- decide  model.gguf --serve   # JSONL on stdin/stdout
dotnet run --project src/Jevstral.Cli -c Release -- verdict model.gguf --instruct … --query … --document …
dotnet run --project src/Jevstral.Cli -c Release -- bench   model.gguf [--threads N]
dotnet run --project src/Jevstral.Cli -c Release -- download q5_1                 # the base checkpoint
```

`decide` reads JevBench-format records: `{"id", "state", "question": {"type", "instructions",
"criteria"}, "labels"}`. `tools/decision/jevbench_adapter.py` runs it under JevBench's own
runner and scoring.

## Models

A GGUF is the whole model. `download` fetches the base Shieldstral builds from
models.curiosity.ai. `tools/convert_shieldstral_to_gguf.py` converts the original weights.
`tools/decision/merge_lora_to_gguf.py` folds a trained adapter in. Every GGUF type is
readable:

```
F32 F16 BF16 F64  I8 I16 I32 I64
Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1
Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K
IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS
TQ1_0 TQ2_0 MXFP4
```

Each type is checked against the reference `gguf` package. Weights stay quantized in the
memory-mapped file: a 3.4 GiB checkpoint loads instantly and costs its on-disk size in
shared page cache.

## Validation

- The runtime's per-layer activations and yes/no verdicts match an independent NumPy
  implementation of the base checkpoint (`tools/reference_shieldstral.py`, bf16) and
  llama.cpp (worst disagreement 3.8e-3 on the same quantization).
- The PyTorch research path reproduces the same fixtures. On typed decisions, C# Q8_0
  margins agree with PyTorch bf16 to about 0.05, with identical predictions.
- The branched pass is bit-identical to running each option alone after the prefix.

```bash
dotnet test                                      # everything that needs no weights
JEVSTRAL_MODEL=/path/to/model.gguf dotnet test   # plus the model-backed parity tests
```

## Requirements

.NET 10 SDK. No native dependencies, no GPU. The fast paths need AVX-512 and fall back to
AVX2 or portable code. Python with `torch` only for `tools/decision`, and with `gguf`,
`numpy` and `regex` for conversion and fixtures.

## Licence

BSD-3-Clause. Portions derived from TensorSharp, © Zhongkai Fu, also BSD-3-Clause; see
`third-party/TensorSharp-LICENSE`. The base weights are Mistral's and carry their own
licence.
