using Jevstral.Numerics;
using Xunit;

namespace Jevstral.Tests;

/// <summary>
/// The transposed-key attention kernels against a scalar definition, over lengths that
/// exercise the 64-key blocks, the 16-key blocks and the scalar tail. Includes the
/// property the branched forward depends on: a key's score does not depend on where
/// its range starts.
/// </summary>
public class AttentionKernelTests
{
    private const int D = AttentionKernels.HeadDim;

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(15)]
    [InlineData(17)]
    [InlineData(64)]
    [InlineData(211)]
    public unsafe void ScoresAndWeightedSumMatchTheDefinition(int length)
    {
        if (!AttentionKernels.Supported) return;   // the portable loop serves every call then
        float[] keys = Random(length * D, 1), values = Random(length * D, 2), q = Random(D, 3);
        int ldk = (length + 15) & ~15;
        var kt = new float[D * ldk];
        var scores = new float[length];
        var sum = new float[D];
        var w = Random(length, 4);
        fixed (float* k = keys) fixed (float* t = kt) fixed (float* s = scores) fixed (float* qq = q)
        fixed (float* v = values) fixed (float* ww = w) fixed (float* o = sum)
        {
            AttentionKernels.TransposeKeys(k, length, t, ldk);
            AttentionKernels.Scores(qq, t, ldk, 0, length, 0.5f, s);
            AttentionKernels.WeightedSum(ww, length, length, 0, v, o);
        }

        var expected = new float[length];
        var expectedSum = new double[D];
        for (int p = 0; p < length; p++)
        {
            double a = 0;
            for (int d = 0; d < D; d++) a += (double)q[d] * keys[p * D + d];
            expected[p] = (float)(a * 0.5);
            for (int d = 0; d < D; d++) expectedSum[d] += (double)w[p] * values[p * D + d];
        }
        Numeric.Close(expected, scores, 1e-4, $"scores({length})");
        Numeric.Close(expectedSum.Select(x => (float)x).ToArray(), sum, 1e-4, $"weighted sum({length})");
    }

    [Fact]
    public unsafe void AKeysScoreDoesNotDependOnWhereItsRangeStarts()
    {
        if (!AttentionKernels.Supported) return;
        const int length = 150;
        float[] keys = Random(length * D, 5), q = Random(D, 6);
        int ldk = (length + 15) & ~15;
        var kt = new float[D * ldk];
        var whole = new float[length];
        var part = new float[length - 37];
        fixed (float* k = keys) fixed (float* t = kt) fixed (float* qq = q) fixed (float* a = whole) fixed (float* b = part)
        {
            AttentionKernels.TransposeKeys(k, length, t, ldk);
            AttentionKernels.Scores(qq, t, ldk, 0, length, 1f, a);
            AttentionKernels.Scores(qq, t, ldk, 37, length - 37, 1f, b);
        }
        Assert.Equal(whole[37..], part);
    }
}
