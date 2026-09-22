using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Numerics;
using Xunit;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// The register-tiled prefill GEMM. Three properties matter. It computes the product.
/// A token's output does not depend on how many other tokens share the call: that is
/// what keeps the prefix cache a pure optimisation. And the fan-out does not change a bit.
/// </summary>
[Collection(MatMulStrategyCollection.Name)]
public class PanelGemmTests
{
    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private static float[] Reference(float[] w, float[] x, int rows, int cols, int tokens)
    {
        var y = new float[tokens * rows];
        for (int t = 0; t < tokens; t++)
            for (int r = 0; r < rows; r++)
            {
                double s = 0;
                for (int k = 0; k < cols; k++) s += (double)x[t * cols + k] * w[r * cols + k];
                y[t * rows + r] = (float)s;
            }
        return y;
    }

    // 3072 and 4096 cross the 2048-column K block; 9216 is the FFN down projection's width.
    // Token counts cover the 6-row tile, its ragged tail, and a single token.
    [Theory]
    [InlineData(128, 256, 1)]
    [InlineData(64, 3072, 5)]
    [InlineData(128, 3072, 13)]
    [InlineData(64, 4096, 6)]
    [InlineData(64, 9216, 7)]
    public async Task MatchesTheDefinition(int rows, int cols, int tokens)
    {
        if (PanelGemm.VectorBits == 0) return;   // no 256- or 512-bit vectors: the row-wise paths serve every call
        float[] w = Random(rows * cols, 1), x = Random(tokens * cols, 2);
        var y = new float[tokens * rows];

        int saved = QuantMatMul.PanelMinTokens;
        try
        {
            QuantMatMul.PanelMinTokens = 1;
            await PinnedMatMul.ForwardAsync(GgmlType.F32, w, rows, cols, x, tokens, y);
        }
        finally { QuantMatMul.PanelMinTokens = saved; }

        Numeric.Close(Reference(w, x, rows, cols, tokens), y, 1e-4, $"panel {rows}x{cols}x{tokens}");
    }

    [Fact]
    public async Task ATokensResultDoesNotDependOnItsBatch()
    {
        if (PanelGemm.VectorBits == 0) return;   // no 256- or 512-bit vectors: the row-wise paths serve every call
        const int rows = 128, cols = 3072, tokens = 17;
        float[] w = Random(rows * cols, 3), x = Random(tokens * cols, 4);
        var all = new float[tokens * rows];
        await PinnedMatMul.ForwardAsync(GgmlType.F32, w, rows, cols, x, tokens, all);

        // The last 5 tokens alone: a cached prefix followed by a short suffix.
        var tail = new float[5 * rows];
        await PinnedMatMul.ForwardAsync(GgmlType.F32, w, rows, cols, x[((tokens - 5) * cols)..], 5, tail);
        Assert.Equal(all[((tokens - 5) * rows)..], tail);
    }

    [Fact]
    public async Task FanOutDoesNotChangeTheResult()
    {
        if (PanelGemm.VectorBits == 0) return;   // no 256- or 512-bit vectors: the row-wise paths serve every call
        const int rows = 256, cols = 3072, tokens = 11;
        float[] w = Random(rows * cols, 5), x = Random(tokens * cols, 6);
        var one = new float[tokens * rows];
        var many = new float[tokens * rows];
        await PinnedMatMul.ForwardAsync(GgmlType.F32, w, rows, cols, x, tokens, one,
            new ParallelOptions { MaxDegreeOfParallelism = 1 });
        await PinnedMatMul.ForwardAsync(GgmlType.F32, w, rows, cols, x, tokens, many,
            new ParallelOptions { MaxDegreeOfParallelism = 8 });
        Assert.Equal(one, many);
    }
}
