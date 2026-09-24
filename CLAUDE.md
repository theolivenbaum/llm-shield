# CLAUDE.md

Guidance for working in this repository.

## What this is

Jevstral: typed, calibrated decisions (noul / choice / score, the schema of Jev, laya and
djev, scored by JevBench) from a 3B decoder, on a CPU, in managed .NET 10. It is a fork of
a runtime for Mistral's Shieldstral 1.0 3B safety classifier. Shieldstral is still the base
checkpoint: its weights, its tokenizer and its yes/no verdict are what Jevstral reads and
adapts. The moderation product is gone. `VerdictScorer` keeps the base checkpoint's native
instruct/query/document question as the primitive the fixtures check against.

The numerics are a reduction of [TensorSharp](https://github.com/zhongkaifu/TensorSharp)
(BSD-3-Clause) to what this one architecture needs. There is no native dependency and no
backend abstraction: one CPU path, written around `System.Numerics.Tensors`, `Vector<T>` and
`Vector512`.

This repository is not bound to Shieldstral's behaviour: the architecture, the output layers
and the prompt may all change when that gets closer to good decisions. What must hold is
parity between `tools/decision` (where adapters are trained) and the C# runtime (where they
run).

Files carrying logic derived from TensorSharp say so in their header. Keep that
attribution when you move code between files; `third-party/TensorSharp-LICENSE`
is the licence it is carried under.

## Layout

```
src/Jevstral/
  Gguf/            GgmlType.cs (block geometry), GgufFile.cs (mmap reader)
  Quantization/    Dequantizer.cs (every GGML type), QuantGrids.g.cs (generated)
  Numerics/        Kernels.cs, QuantMatMul.cs, PanelGemm.cs, AttentionKernels.cs, WeightMatrix.cs
  Tokenization/    TekkenTokenizer.cs
  Model/           ModelConfig.cs, Rope.cs, KvCache.cs, MinistralModel.cs
  ChatTemplate.cs, SystemPromptCache.cs, VerdictScorer.cs, JevstralDecider.cs,
  ModelDownloader.cs
src/Jevstral.Cli/    the `jev` command
tests/Jevstral.Tests/
tests/fixtures/                   generated oracles (JSON), committed
tools/                            Python: conversion, reference impl, fixtures
tools/decision/                   PyTorch research path: typed decisions, JevBench eval, LoRA
```

Dependency direction is one way: `Gguf` → `Quantization` → `Numerics` →
`Model` → the top-level API. Nothing lower reaches up.

## Invariants worth knowing before changing anything

These are the things that are easy to "fix" into being wrong. Each has a test;
the test names are given so you can see what would catch you.

**RoPE pairs adjacent elements.** `(x[2i], x[2i+1])`, ggml's default mode — not
Hugging Face's `(x[i], x[i+d/2])`. The Mistral-format checkpoint is stored for
the former. Introducing a permutation in the converter, or `rotate_half` in the
model, produces a model that loads and runs and is completely wrong.
→ `RopeAndCacheTests.RotationPairsAdjacentElements`

**YaRN's magnitude scale is exactly 1.0 here.** `params.json` says
`"apply_scale": false`, which llama.cpp encodes as `mscale == mscale_all_dim`, so
the two terms cancel. llama.cpp splits this into an "attention factor" and a
fixed `1 + 0.1·ln(factor)` inside its RoPE kernel; `Rope` computes the product
directly instead. Getting it wrong scales every q and k by ~1.277.
→ `RopeAndCacheTests.MagnitudeScaleIsOneWhenApplyScaleIsOff`

**Prefill and decode share one frequency table.** If they ever diverge, a newly
generated token is rotated differently from the prompt already in the KV cache
and attention degrades silently. `Rope` exists to make that impossible.

**The attention context is `heads × head_dim`, not `hidden_size`.** For
Shieldstral those are 4096 and 3072; `attn_output` projects one to the other.
They coincide in most Llama-family models, which is why assuming they are equal
is a natural mistake and a confusing one to debug.

**The LM head is tied to the embedding table.** `output.weight` is legitimately
absent. It is evaluated only at the last position — 131072 rows is a third of a
forward pass and every other position's logits are discarded.

**Control tokens are matched literally, before any BPE.** `[SYSTEM_PROMPT]` and
friends must land on their own ids. The `_specialTokens` list is sorted
longest-first so `[/SYSTEM_PROMPT]` is not shadowed by a prefix of itself.

**The prompt is bytes the model was trained on.** `ChatTemplate` reproduces the
published `chat_template.jinja` for the roles Shieldstral accepts. An extra
space is not cosmetic.
→ `ChatTemplateTests`

**The prefix cache must be a pure optimisation.** Cached and uncached paths
produce bit-identical logits, and the tests assert that over all 131072 values
rather than over the score.
→ `SystemPromptCacheTests.CachedAndUncachedProduceIdenticalLogits`

**A file at the download destination is always complete.** Everything else treats
a path that exists as a file worth memory-mapping, so `ModelDownloader` writes to
a `.download` sidecar and renames only at the end. Its resume path is the part
that needs care: a partial file is reused only when the recorded size *and* entity
tag still match the server, because resuming into a republished model yields a
GGUF of exactly the right length that is wrong from the resume point on — which no
loader can detect. models.curiosity.ai answers HEAD with 405, so the size, the
tag and range support all come from a one-byte ranged GET; drop that fallback and
nothing fails, downloads just silently stop resuming. The resume checks catch a
republished file, not a bad byte inside one, so the Jevstral files are also hashed
against `JevstralModels.PublishedSha256` before the rename, and a mismatch deletes
the partial file instead of leaving it to be resumed. Rebuilding the models changes
those checksums; update the table with the new `SHA256SUMS` when you publish.
→ `ModelDownloaderTests`, against a loopback socket rather than the real host

**A token's result must not depend on the batch around it.** The prefix cache and
the branched forward are only pure optimisations because every kernel computes a
token's output with the same operation sequence whether the call has 5 tokens or
500. `PanelGemm` runs one FMA chain per output over k in ascending order, in the
6-token tile and in the 1-token tail alike. The attention kernels run one fused
chain per (query, key), including the scalar tail (`MathF.FusedMultiplyAdd`, not
`a += x * y`). Softmax runs over exactly the keys a query may see, never over a
padded row. Break any of these and the prefix cache stops being bit-identical:
nothing fails loudly, the tests below just stop passing.
→ `PanelGemmTests.ATokensResultDoesNotDependOnItsBatch`,
`AttentionKernelTests.AKeysScoreDoesNotDependOnWhereItsRangeStarts`,
`DeciderTests.BranchesMatchRunningEachContinuationAlone`

**Branches restart RoPE at the end of the shared part.** `ForwardTreeAsync` runs a
causal trunk and then N branches in one pass. Branch tokens are written to
consecutive cache slots, but each branch's positions start again where the trunk
ends. Each branch attends to the trunk and to itself, never to a sibling. The slot
is not the position; confusing them gives every option after the first a shifted
rotation.

**Softmax subtracts the maximum; `TensorPrimitives.SoftMax` does not.** That one
evaluates `exp(x) / Σexp(x)` directly. Attention scores here reach the 90s in the
deepest layers, `exp` overflows float32, and the division returns NaN — for the
positions that mattered most, in layer 24 of 26, on prompts long enough to have
sharp attention. It survives short prompts and shallow layers, which is precisely
what makes it dangerous. `Kernels.Softmax` does the max-shift; do not "simplify"
it back.
→ `KernelTests.SoftmaxSurvivesLargeLogits`

## Testing

```bash
dotnet test                                          # no weights needed
JEVSTRAL_MODEL=/path/to/model.gguf dotnet test    # + model-backed parity
```

Tests that change `QuantMatMul.Strategy` must join
`[Collection(MatMulStrategyCollection.Name)]` *and* restore it. The strategy is
process-wide and xunit runs test classes in parallel, so restoring alone is not
enough — one class pinning it to Float while another pins it to Integer makes both
measure whatever the scheduler left behind, and it fails intermittently, which is
worse than failing. The collection disables parallelism between them.

`JEVSTRAL_MODEL` may be a `.gguf` or a directory to search. Model-backed tests
write a line explaining the skip and pass when it is unset — keep that pattern
for new ones, so a checkout without a 3.4 GiB download stays green.

Comparisons use `Numeric.Close` (relative error), never xunit's decimal-places
overload: that one compares rounded strings, so two float32 values a single ULP
apart can straddle a boundary and "fail" at four decimal places while agreeing to
seven significant figures.

## Benchmarking

```bash
dotnet run --project src/Jevstral.Cli -c Release -- \
  bench /path/to/models --json benchmark.json
```

One process does everything: environment, per-type dequantize throughput, the
float-versus-integer matmul comparison, and an end-to-end sweep over every GGUF
it finds, once per strategy. That is not a convenience — comparing a number
measured now against one measured in a separate invocation compares two machine
states as much as two kernels, and on a shared VM the machine state moves.

Two habits keep the micro-numbers honest: each timed sample repeats the work
until it spans at least 250 ms (a one-token matmul takes a few milliseconds, and
timing that directly measures the scheduler), and the reported figure is the
*fastest* sample, since everything that makes a run slower is noise.

## Regenerating fixtures

`tests/fixtures/*.json` are committed oracles. They are generated, not
hand-written; regenerate rather than edit.

```bash
pip install gguf numpy regex

# the source weights, if you do not already have them (~7.2 GiB, not the 15 GB
# the full repository would be — see the script's header for what it skips)
python3 tools/download_shieldstral.py models/shieldstral

# i-quant codebooks -> Quantization/QuantGrids.g.cs
python3 tools/gen_quant_grids.py

# per-type dequantization oracle
python3 tools/gen_quant_fixtures.py

# tokenizer.json, scores.json, activations.json (needs the original weights)
python3 tools/reference_shieldstral.py fixtures /path/to/models/shieldstral
```

`tools/reference_shieldstral.py` is the NumPy reference: an independent
implementation of the architecture reading the original bfloat16 weights. It
deliberately shares no code with the runtime — including its tokenizer, which
walks tekken's ranked vocabulary the way `mistral-common` does rather than
applying the derived merge table. Keep it that way; the value of the fixtures is
entirely in that independence.

`activations.json` records the last row and the sum of squares of each tensor for
the first few layers. Full rows for 26 layers would be tens of megabytes, and the
last row is the one every downstream value depends on.

## Parallelism

Every loop that spreads across cores — the row chunks in `QuantMatMul` and the
attention heads in `MinistralModel` — takes its fan-out from a `ParallelOptions`
threaded down from the caller, so a host bounds the whole runtime with one
setting. That is why the forward path is asynchronous: `Parallel.ForAsync` is
what carries the options, and `await` cannot appear in an `unsafe` context, which
is in turn why `QuantMatMul` is no longer an `unsafe` class — only the individual
pointer-using helpers are, and they are called from inside the loop body rather
than wrapping it.

Two consequences worth remembering before "simplifying" any of it back:

- **A `fixed` region cannot span an `await`.** The parallel regions therefore pass
  `Memory<T>` (which a lambda *can* capture) and take their spans inside the body,
  or pin with a `GCHandle` where a raw pointer is genuinely needed — see
  `PinnedMatMul` in the tests and `Benchmark.MatMulWork`.
- **Scratch buffers come from `ArrayPool`, not from a per-head field.** The
  attention scores buffer used to be a `float[][]` on the model, one slot per head.
  That only works while the fan-out is fixed at construction; renting per iteration
  costs nothing measurable and stops the model pinning a buffer per head forever.

The result must not depend on the fan-out — row chunks write to disjoint slices,
so no accumulation can be reordered.
→ `KernelTests.MaxDegreeOfParallelismDoesNotChangeTheResult`

## Performance notes

A prompt of more than one token goes through `PanelGemm`, a register-tiled GEMM. The
structure is laya's `PackedMatrix` (see that repository's CLAUDE.md for the tile sweep):
- A worker takes 64 output rows and decodes them for one 2048-column K block.
- The block is transposed into a `[k][64]` panel (512 KiB, resident in L2) with in-register
  16×16 transposes.
- The panel is streamed against the tokens six at a time, with 24 `Vector512` accumulators
  live.

Single-threaded on the 4-core Sapphire Rapids VM, 3072×3072:

| tokens | row-wise dot (before) | panel GEMM | int8 row-wise |
|---|---|---|---|
| 8 | 21 | 24 GFLOP/s | 10 |
| 64 | 28 | 83–91 | 16–22 |
| 256 | 15–18 | 120–129 | 15–25 |

The old loop did one `TensorPrimitives.Dot` per (row, token): two loads per FMA, a horizontal
reduction per output, and an activation slab too big for L2, re-streamed from L3 for every
weight row. With the panel GEMM, end-to-end prefill on 4 threads went from 8 to about 64
tokens/s.

Measured and kept:
- **Scalar scatter into the panel was as slow as the multiply at 64 tokens.** Every store hit a
  different cache line. The 16×16 unpack/shuffle transpose fixed it (Q5_1 at 64 tokens: 30 → 51,
  then 83 with vector dequant).
- **The dequantizers for Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 are vectorised under AVX-512.** Every
  prefill decodes every weight once, at about a nanosecond per weight in scalar code. They keep
  the scalar multiply-then-add, so `DequantizerParityTests` still holds them byte-for-byte.
- **With vector dequant, float beats int8 even at one token** for everything but Q8_0 (Q5_1:
  7.6 vs 2.4 GFLOP/s). `Auto` follows that. The int8 path remains for Q8_0, for hardware without
  AVX-512, and for `Strategy = Integer`.
- **Only the verdict rows of the LM head are evaluated** (`ForwardSelectedAsync`). The full
  131072-row head cost more than a short suffix's whole pass.
- **Attention keys are transposed once per head.** A score vector then ends in a store, not a
  shuffle chain. Queries are taken four at a time for scores and two at a time for the weighted
  sum. Beyond about 2k tokens the transposed keys and the values outgrow L2, and one query at a
  time re-streamed them from L3 for every query: a 4k-token decision went from 126 to 93 s.

The decode path (one token) is still row-wise: it is bound by streaming the weights, and the
panel transpose would cost more than it saves.

`QuantMatMul.Strategy = Integer` keeps the int8 kernels reachable for tests and benchmarks.
Three details in them are load-bearing:

1. **Unpack once per row, dot once per (row, token).** Unpacking nibbles is per *weight* work.
   Fused into the dot, a 64-token prefill unpacks the same row 64 times.
2. **One horizontal reduction per row, not per block.**
3. **`vpmaddubsw` + `vpmaddwd`** (`Avx2.MultiplyAddAdjacent`) do 32 8-bit MACs in two
   instructions. The first operand must be unsigned, hence the `|w| · sign(w)·a` trick. That is
   also why activations clamp to ±127, so every pairwise sum stays inside int16.

Quantizing activations adds about 3e-3 of relative L2 error per matmul; `IntegerDotTests`
bounds both the noise and any systematic bias.

## Typed decisions

`JevstralDecider` answers the Jev / laya / djev typed questions (noul, choice, score) with
Shieldstral's own yes/no verdict. It makes one read per option, "is option X the correct
answer?", and runs a softmax over the per-option log-odds. A noul is read as a two-option
choice between its false and true criteria. Asking it directly leaves a topical yes-bias
(easy-tier fact accuracy: 0.58 direct, 0.92 as a contrast).

The cache is layered for the usual case, a fixed question over many inputs:
1. **The question** (system prompt, instruction, option list) stays resident between calls. A
   small LRU of snapshots (`QuestionCacheSize`) covers several questions interleaved.
2. **The data** runs once per call, in the trunk.
3. **One short branch per option** ("Is option X correct?", ~8 tokens) attends to both. It is the
   only per-option work. Restating the criterion there was ~27 tokens and scored lower (v1 adapter,
   public easy / standard: 0.979 / 0.819 against 1.000 / 0.847).

Every layer is bit-identical to a cold call
(`DeciderTests.ReusingTheQuestionPrefixIsBitIdentical`,
`DeciderTests.InterleavedQuestionsAreRestoredFromTheirSnapshots`). Putting the option queries
before the data (`--layout qcache` in tools/decision) would cut layer 3 to one token, but it is
badly overconfident (ECE 0.3–0.4), because the data never sees the option. The
prompt must match `tools/decision/decision_prompts.py` byte for byte: that is where adapters
are trained (`tools/decision/README.md`), and a LoRA is only valid for its prompt.
→ `DeciderTests.PrefixAndSuffixRenderTheTrainedPrompt`

## Releasing

`.devops/azure-pipelines.yml` builds `main`, runs the tests and pushes
`Jevstral` to nuget.org through the `nuget-curiosity-org` service
connection. Versions are CalVer — `yy.M.<buildId mod 65536>`, stamped by
`/p:Version` at build time, the modulo because the build counter has to fit an
int16. `Directory.Build.props` keeps `IsPackable` false, so a new project is not
published by accident: opt in on the project *and* add a build step, or it will
never leave the agent.

## Conventions

- `unsafe` and pointers are fine in `Gguf`, `Quantization` and `Numerics`; the
  model layer should stay in spans where it can. Put `unsafe` on the member, not
  the class, so the file can still contain an `await`.
- Buffers come from `ArrayPool<float>.Shared`. Return them in `finally`.
- A `Span<T>` cannot be captured by a lambda, and cannot live across an `await` —
  the parallel regions hold `Memory<T>` and take `.Span` inside the body.
- Public API is documented with `///`; explain *why*, not what the code says.
- Errors name the file and what to do: "re-download this file", not "bad header".
