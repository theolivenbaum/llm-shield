// Ministral-3 forward pass — the language backbone of Shieldstral 1.0 3B.
//
// Derived from TensorSharp's Mistral3Model (https://github.com/zhongkaifu/TensorSharp),
// BSD-3-Clause. See third-party/TensorSharp-LICENSE.
using System.Buffers;
using Jevstral.Gguf;
using Jevstral.Numerics;
using Jevstral.Tokenization;

namespace Jevstral.Model;

/// <summary>
/// A dense pre-norm transformer: RMSNorm → grouped-query attention with YaRN
/// RoPE → RMSNorm → SwiGLU MLP, 26 times, then a tied LM head.
///
/// Weights stay in their GGUF quantization inside the memory-mapped file and are
/// decoded one row at a time inside the matmul, so loading is near-instant and
/// the resident set is the file's size in shared page cache rather than that
/// plus a private float32 copy.
///
/// Not thread-safe: an instance owns one KV cache and one logits buffer. Use one
/// instance per concurrent request, or serialise calls.
/// </summary>
public sealed class MinistralModel : IDisposable
{
    private sealed class Layer
    {
        public required VectorParameter AttentionNorm { get; init; }
        public required WeightMatrix Q { get; init; }
        public required WeightMatrix K { get; init; }
        public required WeightMatrix V { get; init; }
        public required WeightMatrix O { get; init; }
        public required VectorParameter FfnNorm { get; init; }
        public required WeightMatrix Gate { get; init; }
        public required WeightMatrix Up { get; init; }
        public required WeightMatrix Down { get; init; }
    }

    private readonly GgufFile _gguf;
    private readonly bool _ownsFile;
    private readonly Layer[] _layers;
    private readonly WeightMatrix _tokenEmbeddings;
    private readonly WeightMatrix _lmHead;      // empty when embeddings are tied
    private readonly VectorParameter _outputNorm;
    private readonly Rope _rope;
    private readonly float[] _logits;

    public ModelConfig Config { get; }
    public TekkenTokenizer Tokenizer { get; }
    public KvCache KvCache { get; }
    public string ModelPath => _gguf.Path;

    /// <summary>Positions currently held in the KV cache.</summary>
    public int CachedTokenCount => KvCache.Length;

    public MinistralModel(string ggufPath, int initialCacheCapacity = 512)
        : this(new GgufFile(ggufPath), ownsFile: true, initialCacheCapacity) { }

    public MinistralModel(GgufFile gguf, bool ownsFile = false, int initialCacheCapacity = 512)
    {
        _gguf = gguf;
        _ownsFile = ownsFile;

        // A GGUF that is missing a tensor or disagrees with its own hyperparameters throws from
        // here, and then nobody holds the mapping this constructor opened. Windows still will not
        // let a mapped file be deleted, and deleting it is what the callers that catch these do —
        // VerdictScorer.CreateAsync drops a corrupt cached model and downloads it again.
        try
        {
            Config = ModelConfig.FromGguf(gguf);
            Tokenizer = TekkenTokenizer.FromGguf(gguf);
            _rope = new Rope(Config);

            _tokenEmbeddings = WeightMatrix.From(gguf, "token_embd.weight");
            _outputNorm = VectorParameter.From(gguf, "output_norm.weight");
            // Shieldstral ties the LM head to the embedding table, so `output.weight`
            // is legitimately absent; fall back to the embeddings in that case.
            _lmHead = WeightMatrix.From(gguf, "output.weight", required: false);

            _layers = new Layer[Config.LayerCount];
            for (int l = 0; l < Config.LayerCount; l++)
            {
                string p = $"blk.{l}.";
                _layers[l] = new Layer
                {
                    AttentionNorm = VectorParameter.From(gguf, p + "attn_norm.weight"),
                    Q = WeightMatrix.From(gguf, p + "attn_q.weight"),
                    K = WeightMatrix.From(gguf, p + "attn_k.weight"),
                    V = WeightMatrix.From(gguf, p + "attn_v.weight"),
                    O = WeightMatrix.From(gguf, p + "attn_output.weight"),
                    FfnNorm = VectorParameter.From(gguf, p + "ffn_norm.weight"),
                    Gate = WeightMatrix.From(gguf, p + "ffn_gate.weight"),
                    Up = WeightMatrix.From(gguf, p + "ffn_up.weight"),
                    Down = WeightMatrix.From(gguf, p + "ffn_down.weight"),
                };
            }

            ValidateShapes();

            KvCache = new KvCache(Config.LayerCount, Config.KvHeadCount, Config.HeadDim, initialCacheCapacity);
            _logits = new float[Config.VocabSize];
        }
        catch
        {
            if (ownsFile) gguf.Dispose();
            throw;
        }
    }

    private void ValidateShapes()
    {
        if (_tokenEmbeddings.Cols != Config.HiddenSize)
            throw new InvalidDataException(
                $"token_embd is {_tokenEmbeddings.Cols} wide but the model's hidden size is {Config.HiddenSize}.");
        Layer first = _layers[0];
        if (first.Q.Rows != Config.QDim)
            throw new InvalidDataException($"attn_q has {first.Q.Rows} rows; expected {Config.QDim}.");
        if (first.K.Rows != Config.KvDim)
            throw new InvalidDataException($"attn_k has {first.K.Rows} rows; expected {Config.KvDim}.");
        if (first.Gate.Rows != Config.FeedForwardSize)
            throw new InvalidDataException(
                $"ffn_gate has {first.Gate.Rows} rows; expected {Config.FeedForwardSize}.");
    }

    // ----------------------------------------------------------------- forward

    /// <summary>
    /// Appends <paramref name="tokens"/> to whatever is already cached and returns
    /// the logits for the final position. The returned memory is reused between
    /// calls — copy it if you need to keep it.
    /// <para>
    /// <paramref name="options"/> is handed straight to the row-parallel matmuls
    /// and the per-head attention loop, which between them are the only work this
    /// spreads across cores. Pass one with a bounded
    /// <see cref="ParallelOptions.MaxDegreeOfParallelism"/> to leave the rest of
    /// the machine alone.
    /// </para>
    /// </summary>
    public ValueTask<ReadOnlyMemory<float>> ForwardAsync(ReadOnlyMemory<int> tokens, ParallelOptions options)
        => ForwardAsync(tokens, capture: null, options);

    /// <summary>
    /// Runs <paramref name="tokens"/> through the model for their KV state only,
    /// skipping the 131072-row LM head. Used to warm a prefix whose logits nobody
    /// will look at — for Shieldstral that is the fixed system prompt, where the
    /// head would otherwise be a third of the work for a discarded result.
    /// </summary>
    public async ValueTask PrefillAsync(ReadOnlyMemory<int> tokens, ParallelOptions options)
        => await ForwardAsync(tokens, capture: null, options, computeLogits: false).ConfigureAwait(false);

    /// <summary>
    /// Forward pass with an optional observer for intermediate tensors, used by
    /// the parity tests to diff every layer against the Python reference.
    /// </summary>
    public ValueTask<ReadOnlyMemory<float>> ForwardAsync(
        ReadOnlyMemory<int> tokens, IActivationSink? capture, ParallelOptions options)
        => ForwardAsync(tokens, capture, options, computeLogits: true);

    /// <summary>
    /// Appends <paramref name="tokens"/> and returns the logits of just <paramref name="vocabRows"/>
    /// at the final position. A verdict that reads a handful of tokens has no use for the other
    /// 131 thousand rows of the LM head. Decoding and multiplying them was more work than a
    /// short suffix's whole pass through the layers.
    /// </summary>
    public async ValueTask<float[]> ForwardSelectedAsync(
        ReadOnlyMemory<int> tokens, int[] vocabRows, ParallelOptions options)
    {
        var result = new float[1][];
        await ForwardAsync(tokens, null, options, computeLogits: false, branchOffsets: null,
            selectedRows: vocabRows, selectedOut: result).ConfigureAwait(false);
        return result[0];
    }

    /// <summary>
    /// Runs several continuations of the cached prefix in one pass, as independent
    /// branches: each attends to the whole prefix and to its own earlier tokens, never to
    /// another branch's. RoPE positions restart at the prefix length in every branch, so
    /// each branch sees exactly what it would if it were run alone after the prefix.
    /// This is how a decision's option reads, a few dozen tokens each, become one GEMM
    /// instead of N passes that each re-stream every weight.
    /// <para>
    /// Returns the logits of <paramref name="vocabRows"/> at each branch's last token. The
    /// branch tokens are left in the KV cache; <see cref="KvCache.Truncate"/> back to the
    /// prefix before the cache is used for anything else.
    /// </para>
    /// </summary>
    public ValueTask<float[][]> ForwardBranchesAsync(
        IReadOnlyList<int[]> branches, int[] vocabRows, ParallelOptions options)
        => ForwardTreeAsync(ReadOnlyMemory<int>.Empty, branches, vocabRows, options);

    /// <summary>
    /// <see cref="ForwardBranchesAsync"/> with a shared <paramref name="trunk"/> in the same
    /// pass: the trunk is appended causally, then every branch continues from its end. A
    /// decision is then one pass, prefix and options together. Every weight is decoded once
    /// per decision instead of twice, and the GEMM sees one tall batch of tokens instead of
    /// a long one and a short one.
    /// </summary>
    public async ValueTask<float[][]> ForwardTreeAsync(
        ReadOnlyMemory<int> trunk, IReadOnlyList<int[]> branches, int[] vocabRows, ParallelOptions options)
    {
        ArgumentNullException.ThrowIfNull(branches);
        if (branches.Count == 0) throw new ArgumentException("No branches to forward.", nameof(branches));
        var offsets = new int[branches.Count];
        int total = trunk.Length;
        for (int b = 0; b < branches.Count; b++)
        {
            if (branches[b].Length == 0) throw new ArgumentException($"Branch {b} is empty.", nameof(branches));
            offsets[b] = total;
            total += branches[b].Length;
        }
        var tokens = new int[total];
        trunk.Span.CopyTo(tokens);
        for (int b = 0; b < branches.Count; b++) branches[b].CopyTo(tokens, offsets[b]);

        var result = new float[branches.Count][];
        await ForwardAsync(tokens, null, options, computeLogits: false, offsets, vocabRows, result).ConfigureAwait(false);
        return result;
    }

    private ValueTask<ReadOnlyMemory<float>> ForwardAsync(
        ReadOnlyMemory<int> tokens, IActivationSink? capture, ParallelOptions options, bool computeLogits)
        => ForwardAsync(tokens, capture, options, computeLogits, null, null, null);

    private async ValueTask<ReadOnlyMemory<float>> ForwardAsync(
        ReadOnlyMemory<int> tokens, IActivationSink? capture, ParallelOptions options, bool computeLogits,
        int[]? branchOffsets, int[]? selectedRows, float[][]? selectedOut)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (tokens.Length == 0) throw new ArgumentException("No tokens to forward.", nameof(tokens));

        int seq = tokens.Length;
        int startPos = KvCache.Length;
        Branches? layout = branchOffsets is null ? null : new Branches(branchOffsets, seq, startPos);
        int hidden = Config.HiddenSize;
        int ff = Config.FeedForwardSize;

        KvCache.EnsureCapacity(startPos + seq);

        float[] states = ArrayPool<float>.Shared.Rent(seq * hidden);
        float[] normed = ArrayPool<float>.Shared.Rent(seq * hidden);
        float[] block = ArrayPool<float>.Shared.Rent(seq * hidden);
        float[] queries = ArrayPool<float>.Shared.Rent(seq * Config.QDim);
        float[] gate = ArrayPool<float>.Shared.Rent(seq * ff);
        float[] up = ArrayPool<float>.Shared.Rent(seq * ff);
        try
        {
            Memory<float> h = states.AsMemory(0, seq * hidden);
            Memory<float> norm = normed.AsMemory(0, seq * hidden);
            Memory<float> scratch = block.AsMemory(0, seq * hidden);
            Memory<float> g = gate.AsMemory(0, seq * ff);
            Memory<float> u = up.AsMemory(0, seq * ff);

            QuantMatMul.GatherRows(_tokenEmbeddings, tokens.Span, h.Span);
            capture?.Observe("embeddings", h.Span, seq, hidden);

            for (int l = 0; l < Config.LayerCount; l++)
            {
                Layer layer = _layers[l];

                NormRows(h.Span, layer.AttentionNorm, norm.Span, seq, hidden);
                capture?.Observe($"blk.{l}.attn_norm", norm.Span, seq, hidden);

                await AttentionAsync(layer, l, norm, queries, scratch, seq, startPos, layout, capture, options).ConfigureAwait(false);

                capture?.Observe($"blk.{l}.attn_out", scratch.Span, seq, hidden);

                Kernels.Add(h.Span, scratch.Span);
                capture?.Observe($"blk.{l}.post_attn", h.Span, seq, hidden);

                NormRows(h.Span, layer.FfnNorm, norm.Span, seq, hidden);
                capture?.Observe($"blk.{l}.ffn_norm", norm.Span, seq, hidden);

                await QuantMatMul.ForwardAsync(layer.Gate, norm, seq, g, options).ConfigureAwait(false);
                await QuantMatMul.ForwardAsync(layer.Up, norm, seq, u, options).ConfigureAwait(false);
                Kernels.SwiGlu(g.Span, u.Span, g.Span);
                await QuantMatMul.ForwardAsync(layer.Down, g, seq, scratch, options).ConfigureAwait(false);
                capture?.Observe($"blk.{l}.ffn_out", scratch.Span, seq, hidden);

                Kernels.Add(h.Span, scratch.Span);
                capture?.Observe($"blk.{l}.output", h.Span, seq, hidden);
            }

            KvCache.Advance(seq);

            if (selectedRows is not null && selectedOut is not null)
            {
                WeightMatrix table = _lmHead.IsEmpty ? _tokenEmbeddings : _lmHead;
                float[] row = ArrayPool<float>.Shared.Rent(hidden);
                try
                {
                    Span<float> n = norm.Span[..hidden];
                    for (int b = 0; b < selectedOut.Length; b++)
                    {
                        int lastToken = layout is null ? seq - 1 : layout.Value.LastToken(b);
                        Kernels.RmsNorm(h.Span.Slice(lastToken * hidden, hidden), _outputNorm, Config.RmsNormEps, n);
                        var logits = new float[selectedRows.Length];
                        for (int i = 0; i < selectedRows.Length; i++)
                        {
                            table.DequantizeRow(selectedRows[i], row.AsSpan(0, hidden));
                            logits[i] = Kernels.Dot(n, row.AsSpan(0, hidden));
                        }
                        selectedOut[b] = logits;
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(row);
                }
            }
            if (!computeLogits) return default;

            // Only the last position feeds the LM head: Shieldstral's verdict is a
            // single token, and the head is a 131072-row matmul we would rather not
            // run once per prompt token.
            Memory<float> last = norm[..hidden];
            Kernels.RmsNorm(h.Span.Slice((seq - 1) * hidden, hidden), _outputNorm, Config.RmsNormEps, last.Span);
            capture?.Observe("final_norm", last.Span, 1, hidden);

            WeightMatrix head = _lmHead.IsEmpty ? _tokenEmbeddings : _lmHead;
            await QuantMatMul.ForwardAsync(head, last, _logits, options).ConfigureAwait(false);
            capture?.Observe("logits", _logits, 1, Config.VocabSize);

            return _logits;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(states);
            ArrayPool<float>.Shared.Return(normed);
            ArrayPool<float>.Shared.Return(block);
            ArrayPool<float>.Shared.Return(queries);
            ArrayPool<float>.Shared.Return(gate);
            ArrayPool<float>.Shared.Return(up);
        }
    }

    private void NormRows(ReadOnlySpan<float> source, VectorParameter weight, Span<float> destination,
        int rows, int width)
    {
        for (int t = 0; t < rows; t++)
            Kernels.RmsNorm(source.Slice(t * width, width), weight, Config.RmsNormEps,
                destination.Slice(t * width, width));
    }

    /// <param name="output">Receives the attention block's contribution, seq × hidden.</param>
    private async ValueTask AttentionAsync(
        Layer layer, int layerIndex, ReadOnlyMemory<float> input,
        float[] queryBuffer, Memory<float> output, int seq, int startPos, Branches? layout,
        IActivationSink? capture, ParallelOptions options)
    {
        int heads = Config.HeadCount, kvHeads = Config.KvHeadCount, headDim = Config.HeadDim;
        int group = Config.GroupSize, kvDim = Config.KvDim, qDim = Config.QDim;
        float scale = 1f / MathF.Sqrt(headDim);
        int cacheLength = startPos + seq;

        Memory<float> q = queryBuffer.AsMemory(0, seq * qDim);
        await QuantMatMul.ForwardAsync(layer.Q, input, seq, q, options).ConfigureAwait(false);
        RotateQueries(q.Span, heads, headDim, qDim, seq, startPos, layout);
        capture?.Observe($"blk.{layerIndex}.q_rope", q.Span, seq, qDim);

        // K and V land straight in their cache slots, so the cache always holds
        // post-RoPE keys exactly as the score loop will read them back.
        float[] staging = ArrayPool<float>.Shared.Rent(seq * kvDim);
        try
        {
            Memory<float> kv = staging.AsMemory(0, seq * kvDim);

            await QuantMatMul.ForwardAsync(layer.K, input, seq, kv, options).ConfigureAwait(false);
            StoreKeys(kv.Span, layerIndex, kvHeads, headDim, kvDim, seq, startPos, layout);
            capture?.Observe($"blk.{layerIndex}.k_rope", kv.Span, seq, kvDim);

            await QuantMatMul.ForwardAsync(layer.V, input, seq, kv, options).ConfigureAwait(false);
            StoreValues(kv.Span, layerIndex, kvHeads, headDim, kvDim, seq, startPos);
            capture?.Observe($"blk.{layerIndex}.v", kv.Span, seq, kvDim);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(staging);
        }

        // The attention context is heads x head_dim wide (4096 for Shieldstral),
        // which is *not* the model's hidden size (3072) — attn_output projects the
        // one down to the other. They coincide in most Llama-family models, so it
        // is an easy assumption to make and a confusing one to debug.
        float[] contextBuffer = ArrayPool<float>.Shared.Rent(seq * qDim);
        try
        {
            Memory<float> context = contextBuffer.AsMemory(0, seq * qDim);
            KvCache cache = KvCache;

            // Grouped-query attention. Heads are independent, so each rents its own
            // scores buffer for the duration and no synchronisation is needed. The
            // pool is what makes that free: a per-head cache on the model would have
            // to be sized for the widest fan-out any caller ever asks for, and would
            // pin every one of those buffers for the model's lifetime.
            if (headDim == AttentionKernels.HeadDim && AttentionKernels.Supported)
            {
                await Parallel.ForAsync(0, heads, options, (head, _) =>
                {
                    AttendHead(cache, layerIndex, head / group, head, q, context, seq, startPos, cacheLength, qDim, scale, layout);
                    return ValueTask.CompletedTask;
                }).ConfigureAwait(false);
            }
            else await Parallel.ForAsync(0, heads, options, (head, _) =>
            {
                float[] scores = ArrayPool<float>.Shared.Rent(cacheLength);
                try
                {
                    int kvHead = head / group;
                    ReadOnlySpan<float> keys = cache.KeyHistory(layerIndex, kvHead, cacheLength);
                    ReadOnlySpan<float> values = cache.ValueHistory(layerIndex, kvHead, cacheLength);

                    for (int t = 0; t < seq; t++)
                    {
                        // Visible keys: everything shared [0, shared), then this token's own
                        // segment up to itself. For a plain sequence both are startPos and the
                        // two ranges are simply the causal mask [0, startPos + t].
                        int self = startPos + t;
                        int shared = layout?.Shared(t) ?? startPos;
                        int segment = layout?.SegmentStart(t) ?? startPos;
                        int limit = shared + (self - segment + 1);
                        ReadOnlySpan<float> query = q.Span.Slice(t * qDim + head * headDim, headDim);
                        Span<float> row = scores.AsSpan(0, limit);
                        for (int i = 0; i < limit; i++)
                        {
                            int p = i < shared ? i : segment + (i - shared);
                            row[i] = Kernels.Dot(query, keys.Slice(p * headDim, headDim)) * scale;
                        }
                        Kernels.Softmax(row);

                        Span<float> sink = context.Span.Slice(t * qDim + head * headDim, headDim);
                        sink.Clear();
                        for (int i = 0; i < limit; i++)
                        {
                            float w = row[i];
                            int p = i < shared ? i : segment + (i - shared);
                            if (w != 0f) Kernels.AddScaled(sink, values.Slice(p * headDim, headDim), w);
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scores);
                }

                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);

            capture?.Observe($"blk.{layerIndex}.attn_weighted", context.Span, seq, qDim);

            await QuantMatMul.ForwardAsync(layer.O, context, seq, output, options).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(contextBuffer);
        }
    }

    /// <summary>
    /// One query head over its KV head: transpose the cached keys once, then score and
    /// accumulate every token of this call with <see cref="AttentionKernels"/>. Visibility is
    /// the same two ranges as the portable loop: [0, shared) and the token's own segment.
    /// </summary>
    private static unsafe void AttendHead(KvCache cache, int layer, int kvHead, int head, Memory<float> q,
        Memory<float> context, int seq, int startPos, int cacheLength, int qDim, float scale, Branches? layout)
    {
        const int hd = AttentionKernels.HeadDim;
        int ldk = (cacheLength + 15) & ~15;
        float[] kt = ArrayPool<float>.Shared.Rent(hd * ldk);
        // Four score rows: queries are taken four at a time while they share a segment.
        float[] scores = ArrayPool<float>.Shared.Rent(4 * cacheLength);
        try
        {
            fixed (float* keys = cache.KeyHistory(layer, kvHead, cacheLength))
            fixed (float* values = cache.ValueHistory(layer, kvHead, cacheLength))
            fixed (float* k = kt)
            fixed (float* rows = scores)
            fixed (float* qs = q.Span)
            fixed (float* ctx = context.Span)
            {
                AttentionKernels.TransposeKeys(keys, cacheLength, k, ldk);
                int t = 0;
                while (t < seq)
                {
                    int shared = layout?.Shared(t) ?? startPos;
                    int segment = layout?.SegmentStart(t) ?? startPos;

                    // Up to four consecutive tokens of the same segment. They see the same
                    // shared range, and their own ranges are prefixes of the last one's.
                    int n = 1;
                    while (n < 4 && t + n < seq
                           && (layout?.Shared(t + n) ?? startPos) == shared
                           && (layout?.SegmentStart(t + n) ?? startPos) == segment) n++;

                    int ownLast = startPos + t + n - 1 - segment + 1;
                    float* r0 = rows, r1 = rows + cacheLength, r2 = rows + 2 * cacheLength, r3 = rows + 3 * cacheLength;
                    float* q0 = qs + (long)t * qDim + head * hd;
                    float* q1 = n > 1 ? q0 + qDim : q0;
                    float* q2 = n > 2 ? q0 + 2 * qDim : q0;
                    float* q3 = n > 3 ? q0 + 3 * qDim : q0;
                    if (n == 1)
                    {
                        AttentionKernels.Scores(q0, k, ldk, 0, shared, scale, r0);
                        AttentionKernels.Scores(q0, k, ldk, segment, ownLast, scale, r0 + shared);
                    }
                    else
                    {
                        AttentionKernels.Scores4(q0, q1, q2, q3, k, ldk, 0, shared, scale, r0, r1, r2, r3);
                        AttentionKernels.Scores4(q0, q1, q2, q3, k, ldk, segment, ownLast, scale,
                            r0 + shared, r1 + shared, r2 + shared, r3 + shared);
                    }

                    // Each softmax runs over exactly the keys its query may see.
                    int c0 = shared + ownLast - (n - 1);
                    for (int j = 0; j < n; j++)
                        Kernels.Softmax(new Span<float>(rows + j * cacheLength, c0 + j));

                    float* o = ctx + (long)t * qDim + head * hd;
                    int j2 = 0;
                    for (; j2 + 2 <= n; j2 += 2)
                        AttentionKernels.WeightedSum2(rows + j2 * cacheLength, c0 + j2, rows + (j2 + 1) * cacheLength,
                            c0 + j2 + 1, shared, segment, values, o + j2 * qDim, o + (j2 + 1) * qDim);
                    if (j2 < n)
                        AttentionKernels.WeightedSum(rows + j2 * cacheLength, c0 + j2, shared, segment, values, o + j2 * qDim);
                    t += n;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(kt);
            ArrayPool<float>.Shared.Return(scores);
        }
    }

    private void RotateQueries(Span<float> q, int heads, int headDim, int qDim, int seq, int startPos, Branches? layout)
    {
        for (int t = 0; t < seq; t++)
        {
            Span<float> row = q.Slice(t * qDim, qDim);
            int position = layout?.Position(t) ?? startPos + t;
            _rope.Apply(row, heads, headDim, position);
            ApplyPositionScale(row, position);
        }
    }

    private void StoreKeys(Span<float> kv, int layerIndex, int kvHeads, int headDim, int kvDim, int seq, int startPos,
        Branches? layout)
    {
        for (int t = 0; t < seq; t++)
        {
            Span<float> row = kv.Slice(t * kvDim, kvDim);
            _rope.Apply(row, kvHeads, headDim, layout?.Position(t) ?? startPos + t);
            for (int kh = 0; kh < kvHeads; kh++)
                row.Slice(kh * headDim, headDim).CopyTo(KvCache.Key(layerIndex, kh, startPos + t));
        }
    }

    private void StoreValues(Span<float> kv, int layerIndex, int kvHeads, int headDim, int kvDim, int seq, int startPos)
    {
        for (int t = 0; t < seq; t++)
            for (int kh = 0; kh < kvHeads; kh++)
                kv.Slice(t * kvDim + kh * headDim, headDim)
                    .CopyTo(KvCache.Value(layerIndex, kh, startPos + t));
    }

    /// <summary>
    /// Llama-4 attention temperature: <c>q *= 1 + beta·ln(1 + floor(pos / n_orig))</c>.
    /// Inside the 16384-token training window the floor is 0 and this is exactly 1,
    /// so it costs nothing on the prompts Shieldstral actually sees.
    /// </summary>
    private void ApplyPositionScale(Span<float> query, int position)
    {
        if (Config.RopeOriginalContextLength <= 0 || Config.AttentionTemperatureScale == 0f) return;
        float interval = MathF.Floor((float)position / Config.RopeOriginalContextLength);
        if (interval == 0f) return;
        Kernels.Scale(query, 1f + Config.AttentionTemperatureScale * MathF.Log(1f + interval));
    }

    // ------------------------------------------------------------------- state

    public void ResetKvCache() => KvCache.Reset();

    public void Dispose()
    {
        if (_ownsFile) _gguf.Dispose();
    }
}

/// <summary>
/// Where each token of a branched forward sits. Tokens before the first offset are a
/// causal trunk; each offset starts a branch that sees the cache, the whole trunk and its
/// own earlier tokens. Token t is always written to cache slot startPos + t; what varies is
/// its RoPE position and which slots it may attend to.
/// </summary>
internal readonly struct Branches(int[] offsets, int tokens, int startPos)
{
    private int Trunk => offsets[0];

    private int BranchOf(int t)
    {
        int i = Array.BinarySearch(offsets, t);
        return i >= 0 ? i : ~i - 1;
    }

    public int Position(int t) => t < Trunk ? startPos + t : startPos + Trunk + (t - offsets[BranchOf(t)]);

    /// <summary>Slots [0, Shared) are visible in full: the cache for a trunk token, cache plus trunk for a branch token.</summary>
    public int Shared(int t) => t < Trunk ? startPos : startPos + Trunk;

    public int SegmentStart(int t) => t < Trunk ? startPos : startPos + offsets[BranchOf(t)];

    public int LastToken(int branch) => (branch + 1 < offsets.Length ? offsets[branch + 1] : tokens) - 1;
}

/// <summary>
/// Receives intermediate tensors during a forward pass. Implemented by the tests
/// to diff every layer against the Python reference; production code passes null.
/// </summary>
public interface IActivationSink
{
    void Observe(string name, ReadOnlySpan<float> values, int rows, int columns);
}
