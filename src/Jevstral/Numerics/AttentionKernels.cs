// Attention inner loops for 128-wide heads, the transposed-key layout from laya's
// attention units (github.com/theolivenbaum/laya): adjacent keys in adjacent lanes,
// so a score vector finishes with a store instead of a shuffle chain.
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Jevstral.Numerics;

/// <summary>
/// Scores and the weighted value sum for one query against a head's cached keys and values.
///
/// The plain loop took one 128-long dot product per key, and each dot product ended in a
/// horizontal reduction. At a few thousand tokens that made attention about half of a
/// prefill, although it is an eighth of the FLOPs. Here the keys are transposed once per
/// head into <c>kt[d][p]</c>. A block of 64 keys is then 128 broadcast-FMA steps into four
/// independent accumulators, plus a store. The weighted sum holds the whole 128-wide output
/// in eight registers across every key.
///
/// Every key's score is one fused chain over d in ascending order, whether the key falls in a
/// vector block or in the scalar tail (which uses <see cref="MathF.FusedMultiplyAdd"/> for
/// that reason). So a score does not depend on where its range starts. That keeps a branch
/// bit-identical to the same continuation run alone.
/// </summary>
public static class AttentionKernels
{
    public const int HeadDim = 128;

    public static bool Supported => Vector512.IsHardwareAccelerated;

    /// <summary>kt[d * ldk + p] = keys[p * 128 + d] for p in [0, length).</summary>
    public static unsafe void TransposeKeys(float* keys, int length, float* kt, int ldk)
    {
        int p = 0;
        for (; p + 16 <= length; p += 16)
            for (int d = 0; d < HeadDim; d += 16)
                PanelGemm.Transpose16x16(keys + (long)p * HeadDim + d, HeadDim, kt + (long)d * ldk + p, ldk);
        for (; p < length; p++)
            for (int d = 0; d < HeadDim; d++)
                kt[(long)d * ldk + p] = keys[(long)p * HeadDim + d];
    }

    /// <summary>dst[i] = scale · dot(q, key[from + i]) for i in [0, count).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Scores(float* q, float* kt, int ldk, int from, int count, float scale, float* dst)
    {
        var vs = Vector512.Create(scale);
        int i = 0;
        for (; i + 64 <= count; i += 64)
        {
            float* k = kt + from + i;
            Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default;
            for (int d = 0; d < HeadDim; d++, k += ldk)
            {
                var s = Vector512.Create(q[d]);
                a0 = Vector512.FusedMultiplyAdd(s, Vector512.Load(k), a0);
                a1 = Vector512.FusedMultiplyAdd(s, Vector512.Load(k + 16), a1);
                a2 = Vector512.FusedMultiplyAdd(s, Vector512.Load(k + 32), a2);
                a3 = Vector512.FusedMultiplyAdd(s, Vector512.Load(k + 48), a3);
            }
            (a0 * vs).Store(dst + i); (a1 * vs).Store(dst + i + 16);
            (a2 * vs).Store(dst + i + 32); (a3 * vs).Store(dst + i + 48);
        }
        for (; i + 16 <= count; i += 16)
        {
            float* k = kt + from + i;
            Vector512<float> a = default;
            for (int d = 0; d < HeadDim; d++, k += ldk)
                a = Vector512.FusedMultiplyAdd(Vector512.Create(q[d]), Vector512.Load(k), a);
            (a * vs).Store(dst + i);
        }
        for (; i < count; i++)
        {
            float a = 0f;
            float* k = kt + from + i;
            for (int d = 0; d < HeadDim; d++, k += ldk) a = MathF.FusedMultiplyAdd(q[d], *k, a);
            dst[i] = a * scale;
        }
    }

    /// <summary>
    /// dst = Σ w[i] · values[position(i)], where positions [0, shared) are 0..shared-1 and the
    /// rest continue from <paramref name="segment"/>. The eight accumulators never leave registers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void WeightedSum(float* w, int count, int shared, int segment, float* values, float* dst)
    {
        Vector512<float> c0 = default, c1 = default, c2 = default, c3 = default,
                         c4 = default, c5 = default, c6 = default, c7 = default;
        for (int i = 0; i < count; i++)
        {
            int p = i < shared ? i : segment + (i - shared);
            float* v = values + (long)p * HeadDim;
            var s = Vector512.Create(w[i]);
            c0 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v), c0);
            c1 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 16), c1);
            c2 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 32), c2);
            c3 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 48), c3);
            c4 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 64), c4);
            c5 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 80), c5);
            c6 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 96), c6);
            c7 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 112), c7);
        }
        c0.Store(dst); c1.Store(dst + 16); c2.Store(dst + 32); c3.Store(dst + 48);
        c4.Store(dst + 64); c5.Store(dst + 80); c6.Store(dst + 96); c7.Store(dst + 112);
    }

    /// <summary>
    /// <see cref="Scores"/> for four queries at once: each key vector loaded from the transposed
    /// cache feeds four FMA chains. At a few thousand tokens the transposed keys outgrow L2, and
    /// one query at a time re-streamed all of them from L3 for every query. Each (query, key)
    /// chain is unchanged, so the scores are bit-identical to four separate calls.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Scores4(float* q0, float* q1, float* q2, float* q3, float* kt, int ldk, int from, int count,
        float scale, float* d0, float* d1, float* d2, float* d3)
    {
        var vs = Vector512.Create(scale);
        int i = 0;
        for (; i + 32 <= count; i += 32)
        {
            float* k = kt + from + i;
            Vector512<float> a0 = default, a1 = default, b0 = default, b1 = default,
                             c0 = default, c1 = default, e0 = default, e1 = default;
            for (int d = 0; d < HeadDim; d++, k += ldk)
            {
                var k0 = Vector512.Load(k);
                var k1 = Vector512.Load(k + 16);
                var s = Vector512.Create(q0[d]);
                a0 = Vector512.FusedMultiplyAdd(s, k0, a0); a1 = Vector512.FusedMultiplyAdd(s, k1, a1);
                s = Vector512.Create(q1[d]);
                b0 = Vector512.FusedMultiplyAdd(s, k0, b0); b1 = Vector512.FusedMultiplyAdd(s, k1, b1);
                s = Vector512.Create(q2[d]);
                c0 = Vector512.FusedMultiplyAdd(s, k0, c0); c1 = Vector512.FusedMultiplyAdd(s, k1, c1);
                s = Vector512.Create(q3[d]);
                e0 = Vector512.FusedMultiplyAdd(s, k0, e0); e1 = Vector512.FusedMultiplyAdd(s, k1, e1);
            }
            (a0 * vs).Store(d0 + i); (a1 * vs).Store(d0 + i + 16);
            (b0 * vs).Store(d1 + i); (b1 * vs).Store(d1 + i + 16);
            (c0 * vs).Store(d2 + i); (c1 * vs).Store(d2 + i + 16);
            (e0 * vs).Store(d3 + i); (e1 * vs).Store(d3 + i + 16);
        }
        if (i < count)
        {
            Scores(q0, kt, ldk, from + i, count - i, scale, d0 + i);
            Scores(q1, kt, ldk, from + i, count - i, scale, d1 + i);
            Scores(q2, kt, ldk, from + i, count - i, scale, d2 + i);
            Scores(q3, kt, ldk, from + i, count - i, scale, d3 + i);
        }
    }

    /// <summary>
    /// <see cref="WeightedSum"/> for two queries, the second seeing at least as many keys as
    /// the first. Each value row is loaded once for both. Each query's chain runs over its own
    /// keys in the same order as alone.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void WeightedSum2(float* w0, int count0, float* w1, int count1, int shared, int segment,
        float* values, float* dst0, float* dst1)
    {
        Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default,
                         a4 = default, a5 = default, a6 = default, a7 = default;
        Vector512<float> b0 = default, b1 = default, b2 = default, b3 = default,
                         b4 = default, b5 = default, b6 = default, b7 = default;
        int i = 0;
        for (; i < count0; i++)
        {
            int p = i < shared ? i : segment + (i - shared);
            float* v = values + (long)p * HeadDim;
            var v0 = Vector512.Load(v); var v1 = Vector512.Load(v + 16);
            var v2 = Vector512.Load(v + 32); var v3 = Vector512.Load(v + 48);
            var s = Vector512.Create(w0[i]);
            a0 = Vector512.FusedMultiplyAdd(s, v0, a0); a1 = Vector512.FusedMultiplyAdd(s, v1, a1);
            a2 = Vector512.FusedMultiplyAdd(s, v2, a2); a3 = Vector512.FusedMultiplyAdd(s, v3, a3);
            s = Vector512.Create(w1[i]);
            b0 = Vector512.FusedMultiplyAdd(s, v0, b0); b1 = Vector512.FusedMultiplyAdd(s, v1, b1);
            b2 = Vector512.FusedMultiplyAdd(s, v2, b2); b3 = Vector512.FusedMultiplyAdd(s, v3, b3);
            v0 = Vector512.Load(v + 64); v1 = Vector512.Load(v + 80);
            v2 = Vector512.Load(v + 96); v3 = Vector512.Load(v + 112);
            s = Vector512.Create(w0[i]);
            a4 = Vector512.FusedMultiplyAdd(s, v0, a4); a5 = Vector512.FusedMultiplyAdd(s, v1, a5);
            a6 = Vector512.FusedMultiplyAdd(s, v2, a6); a7 = Vector512.FusedMultiplyAdd(s, v3, a7);
            s = Vector512.Create(w1[i]);
            b4 = Vector512.FusedMultiplyAdd(s, v0, b4); b5 = Vector512.FusedMultiplyAdd(s, v1, b5);
            b6 = Vector512.FusedMultiplyAdd(s, v2, b6); b7 = Vector512.FusedMultiplyAdd(s, v3, b7);
        }
        for (; i < count1; i++)
        {
            int p = i < shared ? i : segment + (i - shared);
            float* v = values + (long)p * HeadDim;
            var s = Vector512.Create(w1[i]);
            b0 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v), b0);
            b1 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 16), b1);
            b2 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 32), b2);
            b3 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 48), b3);
            b4 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 64), b4);
            b5 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 80), b5);
            b6 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 96), b6);
            b7 = Vector512.FusedMultiplyAdd(s, Vector512.Load(v + 112), b7);
        }
        a0.Store(dst0); a1.Store(dst0 + 16); a2.Store(dst0 + 32); a3.Store(dst0 + 48);
        a4.Store(dst0 + 64); a5.Store(dst0 + 80); a6.Store(dst0 + 96); a7.Store(dst0 + 112);
        b0.Store(dst1); b1.Store(dst1 + 16); b2.Store(dst1 + 32); b3.Store(dst1 + 48);
        b4.Store(dst1 + 64); b5.Store(dst1 + 80); b6.Store(dst1 + 96); b7.Store(dst1 + 112);
    }
}
