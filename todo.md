# todo — Jevstral

Checked items are done and covered by tests or by a committed measurement.

## 1. Decisions

- [x] `JevstralDecider`: noul / choice / score from the verdict. One read per option,
      softmax over log-odds, noul read as a contrast. → `DeciderTests`
- [x] One pass per decision: trunk plus branches (`ForwardTreeAsync`), bit-identical to
      running each branch alone. → `DeciderTests.BranchesMatchRunningEachContinuationAlone`
- [x] `jev decide` over JevBench-format JSONL, and `--serve` for a persistent harness
- [x] PyTorch research path in `tools/decision` that matches the C# margins (about 0.05
      on Q8_0 vs bf16)
- [x] Zero-shot on public JevBench: easy 0.979, standard 0.778, hard 0.396 (shared-prefix
      order)
- [ ] LoRA v1 (typed-decisions + synthetic): evaluate on public JevBench and on the
      typed-decisions test split, against laya's 0.766
- [ ] Temperatures fitted on non-JevBench held-out data, shipped with the model
- [ ] Merge the adapter into a GGUF and check C# parity on it
- [ ] A dedicated decision head in place of the LM head's yes/no rows: one 3072-wide
      vector, initialised to e_yes − e_no so step 0 is the verdict, trained with the
      adapter, stored in the GGUF as `jev.head.*`
- [ ] Early exit: measure the verdict read from intermediate layers. If a head on layer
      ~18 holds up, a decision costs 70% of a pass.
- [ ] An act / abstain signal like laya's `act_head`, from the calibration features
      (top-1, margin, entropy, k)
- [x] Publish Jevstral GGUFs on models.curiosity.ai/jevstral/ (all six plus SYSTEM_PROMPT and SHA256SUMS; Q8_0 checksums verified against a local build)
- [ ] A JevBench submission: held-out tiers can only be run by the maintainers, and
      training on the synthetic families must be disclosed (see tools/decision/README.md)

## 2. Runtime

- [x] GGUF reader, every quantization type, tokenizer, Ministral-3 forward, and parity
      with NumPy and llama.cpp (inherited from the Shieldstral runtime)
- [x] `PanelGemm` prefill: 120–129 GFLOP/s single-threaded at 256 tokens, up from 15–18
- [x] AVX-512 dequantizers for the legacy quants
- [x] Transposed-key, query-tiled attention
- [x] Verdict-rows-only LM head
- [ ] Long prompts: a 4k-token decision is ~90 s on 4 cores. The FLOPs say about 60.
      Attention is O(n²) and still streams the transposed keys per query tile; a
      flash-style key-blocked loop is the next step.
- [ ] int8 with VNNI (`vpdpbusd`) for the panel GEMM: halves the weight traffic and doubles
      the MACs per instruction against float, at the int8 path's 3e-3 error
- [ ] Integer kernels for the k-quants
- [ ] Vision: the converter emits the Pixtral tower, but its forward pass is not implemented
