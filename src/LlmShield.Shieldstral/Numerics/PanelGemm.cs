// Register-tiled prefill GEMM against GGUF-quantized weights.
// The tile shape and the "one panel in L2, streamed against every token" structure
// follow laya's PackedMatrix (github.com/theolivenbaum/laya), where they were swept.
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Quantization;

namespace LlmShield.Shieldstral.Numerics;

/// <summary>
/// <c>y = x · Wᵀ</c> for a block of tokens, as a real GEMM rather than a grid of dot products.
///
/// The per-(row, token) dot product that <see cref="QuantMatMul"/> started with costs two
/// loads per FMA and a horizontal reduction per output. Worse, a prefill's activation slab
/// (up to 4 MiB) does not fit in L2, so every one of the thousands of weight rows re-streamed
/// all the tokens from L3. On a 4-core Xeon that measured about 12 GFLOP/s per core, a tenth of
/// what the FMA units sustain.
///
/// Here each worker takes a panel of <see cref="PanelWidth"/> output rows and decodes it once.
/// For every K block of <see cref="KBlock"/> it is transposed into a <c>[k][64]</c> buffer
/// (512 KiB, resident in L2). Tokens are then consumed six at a time: 24 vector accumulators
/// (6 tokens × 4 × 16 lanes) stay in registers for the whole K block. Each step is four weight
/// loads, six broadcasts and twenty-four FMAs.
///
/// Every output element is one FMA chain over k in ascending order, whichever tile its token
/// lands in and however many tokens the call has. So a token's result does not depend on the
/// batch around it. That keeps the prefix cache a pure optimisation: a prompt run whole and
/// one run as cached prefix plus suffix produce bit-identical logits.
/// </summary>
public static class PanelGemm
{
    /// <summary>Output rows per panel: four 512-bit vectors, or eight 256-bit ones.</summary>
    public const int PanelWidth = 64;

    /// <summary>
    /// Input columns per packed block. 2048 × 64 floats is 512 KiB, which leaves room in a
    /// 2 MiB L2 for the activations passing through. It is a multiple of 256, so a block
    /// boundary is also a quant-block boundary for every GGML type, k-quants included.
    /// </summary>
    public const int KBlock = 2048;

    private const int RowBlock = 6;

    /// <summary>
    /// Which kernel runs. 512 when the JIT accelerates <see cref="Vector512"/>, 256 when only
    /// AVX2 is available, 0 when neither is (the caller falls back to the dot-product path).
    /// <c>LLMSHIELD_VECTOR_BITS=256|512</c> overrides it, for benchmarking one against the other.
    /// </summary>
    public static int VectorBits { get; } = Resolve();

    private static int Resolve()
    {
        string? forced = Environment.GetEnvironmentVariable("LLMSHIELD_VECTOR_BITS");
        if (forced == "512" && Vector512.IsHardwareAccelerated) return 512;
        if (forced == "256" && Vector256.IsHardwareAccelerated) return 256;
        if (forced == "0") return 0;
        if (Vector512.IsHardwareAccelerated) return 512;
        if (Vector256.IsHardwareAccelerated) return 256;
        return 0;
    }

    /// <summary>Whether this GEMM can take the product: the shape has to tile into whole panels and quant blocks.</summary>
    public static bool Supports(in WeightMatrix w) => Supports(w.Type, w.Rows, w.Cols);

    public static bool Supports(GgmlType type, int rows, int cols)
        => VectorBits != 0 && rows % PanelWidth == 0 && cols % GgmlTypeInfo.BlockSize(type) == 0
           && (cols <= KBlock || KBlock % GgmlTypeInfo.BlockSize(type) == 0);

    /// <summary>
    /// Computes <paramref name="destination"/>[t, r] = dot(x[t, :], W[r, :]) for all
    /// <paramref name="tokens"/> tokens, spreading panels across <paramref name="options"/>.
    /// </summary>
    public static async ValueTask ForwardAsync(
        WeightMatrix w, ReadOnlyMemory<float> x, int tokens, Memory<float> destination, ParallelOptions options)
    {
        int panels = w.Rows / PanelWidth;
        await Parallel.ForAsync(0, panels, options, (panel, _) =>
        {
            RunPanel(w, x, tokens, destination, panel);
            return ValueTask.CompletedTask;
        }).ConfigureAwait(false);
    }

    private static unsafe void RunPanel(in WeightMatrix w, ReadOnlyMemory<float> x, int tokens, Memory<float> y, int panel)
    {
        int cols = w.Cols, rows = w.Rows;
        int r0 = panel * PanelWidth;
        float[] packed = ArrayPool<float>.Shared.Rent(Math.Min(cols, KBlock) * PanelWidth);
        float[] row = ArrayPool<float>.Shared.Rent(16 * Math.Min(cols, KBlock));
        try
        {
            using var xPin = x.Pin();
            using var yPin = y.Pin();
            fixed (float* pk = packed)
            fixed (float* rw = row)
            {
                float* xp = (float*)xPin.Pointer;
                float* yp = (float*)yPin.Pointer;
                for (int k0 = 0; k0 < cols; k0 += KBlock)
                {
                    int kc = Math.Min(KBlock, cols - k0);
                    long byteOffset = GgmlTypeInfo.RowBytes(w.Type, k0);
                    Pack(w, r0, k0 == 0 ? 0 : byteOffset, kc, rw, pk);

                    bool accumulate = k0 != 0;
                    float* xk = xp + k0;
                    float* yr = yp + r0;
                    int t = 0;
                    if (VectorBits == 512)
                    {
                        for (; t + RowBlock <= tokens; t += RowBlock)
                            Tile512x6(xk + (long)t * cols, cols, pk, kc, yr + (long)t * rows, rows, accumulate);
                        for (; t < tokens; t++)
                            Tile512x1(xk + (long)t * cols, pk, kc, yr + (long)t * rows, accumulate);
                    }
                    else
                    {
                        for (; t + RowBlock <= tokens; t += RowBlock)
                            for (int half = 0; half < PanelWidth; half += 16)
                                Tile256x6(xk + (long)t * cols, cols, pk + half, kc, yr + (long)t * rows + half, rows, accumulate);
                        for (; t < tokens; t++)
                            Tile256x1(xk + (long)t * cols, pk, kc, yr + (long)t * rows, accumulate);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packed);
            ArrayPool<float>.Shared.Return(row);
        }
    }

    /// <summary>
    /// Decodes 64 rows' slice [k0, k0+kc) and transposes it into <c>packed[k * 64 + j]</c>.
    /// Sixteen rows are decoded row-major into <paramref name="rows16"/> and then read back
    /// one column at a time with a gather, so every write to the panel is a whole vector.
    /// Scattering scalars at a 256-byte stride touched a cache line per element. At 64
    /// tokens that made packing as expensive as the multiply.
    /// </summary>
    private static unsafe void Pack(in WeightMatrix w, int r0, long byteOffset, int kc, float* rows16, float* packed)
    {
        if (Avx512F.IsSupported && kc % 16 == 0)
        {
            for (int g = 0; g < PanelWidth; g += 16)
            {
                for (int j = 0; j < 16; j++)
                    Dequantizer.Dequantize(w.Type, w.Row(r0 + g + j) + byteOffset, rows16 + (long)j * kc, kc);
                for (int k = 0; k < kc; k += 16)
                    Transpose16x16(rows16 + k, kc, packed + (long)k * PanelWidth + g, PanelWidth);
            }
            return;
        }

        for (int j = 0; j < PanelWidth; j++)
        {
            Dequantizer.Dequantize(w.Type, w.Row(r0 + j) + byteOffset, rows16, kc);
            float* dst = packed + j;
            for (int k = 0; k < kc; k++, dst += PanelWidth) *dst = rows16[k];
        }
    }

    /// <summary>
    /// dst[c * ldd + r] = src[r * lds + c] for a 16 x 16 block, in registers: unpack pairs of
    /// rows, unpack again as 64-bit lanes, then two rounds of 128-bit lane shuffles.
    /// 64 shuffles move 256 elements.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void Transpose16x16(float* src, int lds, float* dst, int ldd)
    {
        var r0 = Vector512.Load(src); var r1 = Vector512.Load(src + lds);
        var r2 = Vector512.Load(src + 2 * lds); var r3 = Vector512.Load(src + 3 * lds);
        var r4 = Vector512.Load(src + 4 * lds); var r5 = Vector512.Load(src + 5 * lds);
        var r6 = Vector512.Load(src + 6 * lds); var r7 = Vector512.Load(src + 7 * lds);
        var r8 = Vector512.Load(src + 8 * lds); var r9 = Vector512.Load(src + 9 * lds);
        var ra = Vector512.Load(src + 10 * lds); var rb = Vector512.Load(src + 11 * lds);
        var rc = Vector512.Load(src + 12 * lds); var rd = Vector512.Load(src + 13 * lds);
        var re = Vector512.Load(src + 14 * lds); var rf = Vector512.Load(src + 15 * lds);

        // Stage 1: interleave 32-bit elements of row pairs.
        var t0 = Avx512F.UnpackLow(r0, r1); var t1 = Avx512F.UnpackHigh(r0, r1);
        var t2 = Avx512F.UnpackLow(r2, r3); var t3 = Avx512F.UnpackHigh(r2, r3);
        var t4 = Avx512F.UnpackLow(r4, r5); var t5 = Avx512F.UnpackHigh(r4, r5);
        var t6 = Avx512F.UnpackLow(r6, r7); var t7 = Avx512F.UnpackHigh(r6, r7);
        var t8 = Avx512F.UnpackLow(r8, r9); var t9 = Avx512F.UnpackHigh(r8, r9);
        var ta = Avx512F.UnpackLow(ra, rb); var tb = Avx512F.UnpackHigh(ra, rb);
        var tc = Avx512F.UnpackLow(rc, rd); var td = Avx512F.UnpackHigh(rc, rd);
        var te = Avx512F.UnpackLow(re, rf); var tf = Avx512F.UnpackHigh(re, rf);

        // Stage 2: interleave 64-bit pairs, giving 4 x 4 blocks per 128-bit lane.
        r0 = Avx512F.UnpackLow(t0.AsDouble(), t2.AsDouble()).AsSingle();
        r1 = Avx512F.UnpackHigh(t0.AsDouble(), t2.AsDouble()).AsSingle();
        r2 = Avx512F.UnpackLow(t1.AsDouble(), t3.AsDouble()).AsSingle();
        r3 = Avx512F.UnpackHigh(t1.AsDouble(), t3.AsDouble()).AsSingle();
        r4 = Avx512F.UnpackLow(t4.AsDouble(), t6.AsDouble()).AsSingle();
        r5 = Avx512F.UnpackHigh(t4.AsDouble(), t6.AsDouble()).AsSingle();
        r6 = Avx512F.UnpackLow(t5.AsDouble(), t7.AsDouble()).AsSingle();
        r7 = Avx512F.UnpackHigh(t5.AsDouble(), t7.AsDouble()).AsSingle();
        r8 = Avx512F.UnpackLow(t8.AsDouble(), ta.AsDouble()).AsSingle();
        r9 = Avx512F.UnpackHigh(t8.AsDouble(), ta.AsDouble()).AsSingle();
        ra = Avx512F.UnpackLow(t9.AsDouble(), tb.AsDouble()).AsSingle();
        rb = Avx512F.UnpackHigh(t9.AsDouble(), tb.AsDouble()).AsSingle();
        rc = Avx512F.UnpackLow(tc.AsDouble(), te.AsDouble()).AsSingle();
        rd = Avx512F.UnpackHigh(tc.AsDouble(), te.AsDouble()).AsSingle();
        re = Avx512F.UnpackLow(td.AsDouble(), tf.AsDouble()).AsSingle();
        rf = Avx512F.UnpackHigh(td.AsDouble(), tf.AsDouble()).AsSingle();

        // Stage 3: gather 128-bit lanes {0,2} and {1,3} across rows 4 apart.
        t0 = Avx512F.Shuffle4x128(r0, r4, 0x88); t1 = Avx512F.Shuffle4x128(r1, r5, 0x88);
        t2 = Avx512F.Shuffle4x128(r2, r6, 0x88); t3 = Avx512F.Shuffle4x128(r3, r7, 0x88);
        t4 = Avx512F.Shuffle4x128(r0, r4, 0xDD); t5 = Avx512F.Shuffle4x128(r1, r5, 0xDD);
        t6 = Avx512F.Shuffle4x128(r2, r6, 0xDD); t7 = Avx512F.Shuffle4x128(r3, r7, 0xDD);
        t8 = Avx512F.Shuffle4x128(r8, rc, 0x88); t9 = Avx512F.Shuffle4x128(r9, rd, 0x88);
        ta = Avx512F.Shuffle4x128(ra, re, 0x88); tb = Avx512F.Shuffle4x128(rb, rf, 0x88);
        tc = Avx512F.Shuffle4x128(r8, rc, 0xDD); td = Avx512F.Shuffle4x128(r9, rd, 0xDD);
        te = Avx512F.Shuffle4x128(ra, re, 0xDD); tf = Avx512F.Shuffle4x128(rb, rf, 0xDD);

        // Stage 4: the same across the two halves.
        Avx512F.Shuffle4x128(t0, t8, 0x88).Store(dst + 0 * ldd);
        Avx512F.Shuffle4x128(t1, t9, 0x88).Store(dst + 1 * ldd);
        Avx512F.Shuffle4x128(t2, ta, 0x88).Store(dst + 2 * ldd);
        Avx512F.Shuffle4x128(t3, tb, 0x88).Store(dst + 3 * ldd);
        Avx512F.Shuffle4x128(t4, tc, 0x88).Store(dst + 4 * ldd);
        Avx512F.Shuffle4x128(t5, td, 0x88).Store(dst + 5 * ldd);
        Avx512F.Shuffle4x128(t6, te, 0x88).Store(dst + 6 * ldd);
        Avx512F.Shuffle4x128(t7, tf, 0x88).Store(dst + 7 * ldd);
        Avx512F.Shuffle4x128(t0, t8, 0xDD).Store(dst + 8 * ldd);
        Avx512F.Shuffle4x128(t1, t9, 0xDD).Store(dst + 9 * ldd);
        Avx512F.Shuffle4x128(t2, ta, 0xDD).Store(dst + 10 * ldd);
        Avx512F.Shuffle4x128(t3, tb, 0xDD).Store(dst + 11 * ldd);
        Avx512F.Shuffle4x128(t4, tc, 0xDD).Store(dst + 12 * ldd);
        Avx512F.Shuffle4x128(t5, td, 0xDD).Store(dst + 13 * ldd);
        Avx512F.Shuffle4x128(t6, te, 0xDD).Store(dst + 14 * ldd);
        Avx512F.Shuffle4x128(t7, tf, 0xDD).Store(dst + 15 * ldd);
    }

    // The kernels hold their accumulators as locals and store inline. Handing them to a
    // helper by value makes them address-exposed, and the JIT then spills every one of
    // them after every FMA (laya's CLAUDE.md, "Things that have bitten us").

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile512x6(float* x, int ldx, float* p, int kc, float* y, int ldy, bool acc)
    {
        Vector512<float> c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23,
                         c30, c31, c32, c33, c40, c41, c42, c43, c50, c51, c52, c53;
        if (acc)
        {
            float* d = y;
            c00 = Vector512.Load(d); c01 = Vector512.Load(d + 16); c02 = Vector512.Load(d + 32); c03 = Vector512.Load(d + 48); d += ldy;
            c10 = Vector512.Load(d); c11 = Vector512.Load(d + 16); c12 = Vector512.Load(d + 32); c13 = Vector512.Load(d + 48); d += ldy;
            c20 = Vector512.Load(d); c21 = Vector512.Load(d + 16); c22 = Vector512.Load(d + 32); c23 = Vector512.Load(d + 48); d += ldy;
            c30 = Vector512.Load(d); c31 = Vector512.Load(d + 16); c32 = Vector512.Load(d + 32); c33 = Vector512.Load(d + 48); d += ldy;
            c40 = Vector512.Load(d); c41 = Vector512.Load(d + 16); c42 = Vector512.Load(d + 32); c43 = Vector512.Load(d + 48); d += ldy;
            c50 = Vector512.Load(d); c51 = Vector512.Load(d + 16); c52 = Vector512.Load(d + 32); c53 = Vector512.Load(d + 48);
        }
        else
        {
            c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = c20 = c21 = c22 = c23 = Vector512<float>.Zero;
            c30 = c31 = c32 = c33 = c40 = c41 = c42 = c43 = c50 = c51 = c52 = c53 = Vector512<float>.Zero;
        }

        float* a0 = x, a1 = x + ldx, a2 = x + 2 * ldx, a3 = x + 3 * ldx, a4 = x + 4 * ldx, a5 = x + 5 * ldx;
        float* w = p;
        for (int k = 0; k < kc; k++, w += PanelWidth)
        {
            var b0 = Vector512.Load(w);
            var b1 = Vector512.Load(w + 16);
            var b2 = Vector512.Load(w + 32);
            var b3 = Vector512.Load(w + 48);
            var s = Vector512.Create(a0[k]);
            c00 = Vector512.FusedMultiplyAdd(s, b0, c00); c01 = Vector512.FusedMultiplyAdd(s, b1, c01);
            c02 = Vector512.FusedMultiplyAdd(s, b2, c02); c03 = Vector512.FusedMultiplyAdd(s, b3, c03);
            s = Vector512.Create(a1[k]);
            c10 = Vector512.FusedMultiplyAdd(s, b0, c10); c11 = Vector512.FusedMultiplyAdd(s, b1, c11);
            c12 = Vector512.FusedMultiplyAdd(s, b2, c12); c13 = Vector512.FusedMultiplyAdd(s, b3, c13);
            s = Vector512.Create(a2[k]);
            c20 = Vector512.FusedMultiplyAdd(s, b0, c20); c21 = Vector512.FusedMultiplyAdd(s, b1, c21);
            c22 = Vector512.FusedMultiplyAdd(s, b2, c22); c23 = Vector512.FusedMultiplyAdd(s, b3, c23);
            s = Vector512.Create(a3[k]);
            c30 = Vector512.FusedMultiplyAdd(s, b0, c30); c31 = Vector512.FusedMultiplyAdd(s, b1, c31);
            c32 = Vector512.FusedMultiplyAdd(s, b2, c32); c33 = Vector512.FusedMultiplyAdd(s, b3, c33);
            s = Vector512.Create(a4[k]);
            c40 = Vector512.FusedMultiplyAdd(s, b0, c40); c41 = Vector512.FusedMultiplyAdd(s, b1, c41);
            c42 = Vector512.FusedMultiplyAdd(s, b2, c42); c43 = Vector512.FusedMultiplyAdd(s, b3, c43);
            s = Vector512.Create(a5[k]);
            c50 = Vector512.FusedMultiplyAdd(s, b0, c50); c51 = Vector512.FusedMultiplyAdd(s, b1, c51);
            c52 = Vector512.FusedMultiplyAdd(s, b2, c52); c53 = Vector512.FusedMultiplyAdd(s, b3, c53);
        }

        float* o = y;
        c00.Store(o); c01.Store(o + 16); c02.Store(o + 32); c03.Store(o + 48); o += ldy;
        c10.Store(o); c11.Store(o + 16); c12.Store(o + 32); c13.Store(o + 48); o += ldy;
        c20.Store(o); c21.Store(o + 16); c22.Store(o + 32); c23.Store(o + 48); o += ldy;
        c30.Store(o); c31.Store(o + 16); c32.Store(o + 32); c33.Store(o + 48); o += ldy;
        c40.Store(o); c41.Store(o + 16); c42.Store(o + 32); c43.Store(o + 48); o += ldy;
        c50.Store(o); c51.Store(o + 16); c52.Store(o + 32); c53.Store(o + 48);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile512x1(float* x, float* p, int kc, float* y, bool acc)
    {
        Vector512<float> c0, c1, c2, c3;
        if (acc) { c0 = Vector512.Load(y); c1 = Vector512.Load(y + 16); c2 = Vector512.Load(y + 32); c3 = Vector512.Load(y + 48); }
        else c0 = c1 = c2 = c3 = Vector512<float>.Zero;
        float* w = p;
        for (int k = 0; k < kc; k++, w += PanelWidth)
        {
            var s = Vector512.Create(x[k]);
            c0 = Vector512.FusedMultiplyAdd(s, Vector512.Load(w), c0);
            c1 = Vector512.FusedMultiplyAdd(s, Vector512.Load(w + 16), c1);
            c2 = Vector512.FusedMultiplyAdd(s, Vector512.Load(w + 32), c2);
            c3 = Vector512.FusedMultiplyAdd(s, Vector512.Load(w + 48), c3);
        }
        c0.Store(y); c1.Store(y + 16); c2.Store(y + 32); c3.Store(y + 48);
    }

    /// <summary>
    /// AVX2: 16 registers, so 6 tokens × 2 vectors (12 accumulators) over a 16-wide slice of the
    /// panel, called four times per panel. Same FMA chain per element as the 512-bit kernel.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile256x6(float* x, int ldx, float* p, int kc, float* y, int ldy, bool acc)
    {
        Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        if (acc)
        {
            float* d = y;
            c00 = Vector256.Load(d); c01 = Vector256.Load(d + 8); d += ldy;
            c10 = Vector256.Load(d); c11 = Vector256.Load(d + 8); d += ldy;
            c20 = Vector256.Load(d); c21 = Vector256.Load(d + 8); d += ldy;
            c30 = Vector256.Load(d); c31 = Vector256.Load(d + 8); d += ldy;
            c40 = Vector256.Load(d); c41 = Vector256.Load(d + 8); d += ldy;
            c50 = Vector256.Load(d); c51 = Vector256.Load(d + 8);
        }
        else
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector256<float>.Zero;
        }
        float* a0 = x, a1 = x + ldx, a2 = x + 2 * ldx, a3 = x + 3 * ldx, a4 = x + 4 * ldx, a5 = x + 5 * ldx;
        float* w = p;
        for (int k = 0; k < kc; k++, w += PanelWidth)
        {
            var b0 = Vector256.Load(w);
            var b1 = Vector256.Load(w + 8);
            var s = Vector256.Create(a0[k]);
            c00 = Vector256.FusedMultiplyAdd(s, b0, c00); c01 = Vector256.FusedMultiplyAdd(s, b1, c01);
            s = Vector256.Create(a1[k]);
            c10 = Vector256.FusedMultiplyAdd(s, b0, c10); c11 = Vector256.FusedMultiplyAdd(s, b1, c11);
            s = Vector256.Create(a2[k]);
            c20 = Vector256.FusedMultiplyAdd(s, b0, c20); c21 = Vector256.FusedMultiplyAdd(s, b1, c21);
            s = Vector256.Create(a3[k]);
            c30 = Vector256.FusedMultiplyAdd(s, b0, c30); c31 = Vector256.FusedMultiplyAdd(s, b1, c31);
            s = Vector256.Create(a4[k]);
            c40 = Vector256.FusedMultiplyAdd(s, b0, c40); c41 = Vector256.FusedMultiplyAdd(s, b1, c41);
            s = Vector256.Create(a5[k]);
            c50 = Vector256.FusedMultiplyAdd(s, b0, c50); c51 = Vector256.FusedMultiplyAdd(s, b1, c51);
        }
        float* o = y;
        c00.Store(o); c01.Store(o + 8); o += ldy;
        c10.Store(o); c11.Store(o + 8); o += ldy;
        c20.Store(o); c21.Store(o + 8); o += ldy;
        c30.Store(o); c31.Store(o + 8); o += ldy;
        c40.Store(o); c41.Store(o + 8); o += ldy;
        c50.Store(o); c51.Store(o + 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile256x1(float* x, float* p, int kc, float* y, bool acc)
    {
        for (int half = 0; half < PanelWidth; half += 32)
        {
            float* yy = y + half;
            Vector256<float> c0, c1, c2, c3;
            if (acc) { c0 = Vector256.Load(yy); c1 = Vector256.Load(yy + 8); c2 = Vector256.Load(yy + 16); c3 = Vector256.Load(yy + 24); }
            else c0 = c1 = c2 = c3 = Vector256<float>.Zero;
            float* w = p + half;
            for (int k = 0; k < kc; k++, w += PanelWidth)
            {
                var s = Vector256.Create(x[k]);
                c0 = Vector256.FusedMultiplyAdd(s, Vector256.Load(w), c0);
                c1 = Vector256.FusedMultiplyAdd(s, Vector256.Load(w + 8), c1);
                c2 = Vector256.FusedMultiplyAdd(s, Vector256.Load(w + 16), c2);
                c3 = Vector256.FusedMultiplyAdd(s, Vector256.Load(w + 24), c3);
            }
            c0.Store(yy); c1.Store(yy + 8); c2.Store(yy + 16); c3.Store(yy + 24);
        }
    }
}
