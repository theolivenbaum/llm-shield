using Jevstral.Model;
using Xunit;
using Xunit.Abstractions;

namespace Jevstral.Tests;

/// <summary>
/// Typed decisions over Shieldstral's verdict. The branched forward is an optimisation
/// with a strict contract: a branch must see exactly what it would see if it ran alone
/// after the prefix, in the same bits. The prompt renderer must produce the bytes the
/// research path trains on.
/// </summary>
public class DeciderTests
{
    private readonly ITestOutputHelper _output;
    public DeciderTests(ITestOutputHelper output) => _output = output;

    private static readonly DecisionQuestion Refund = DecisionQuestion.Noul(
        "Under the stated policy, is the requested action permitted?",
        whenTrue: "Every required condition is established.",
        whenFalse: "A condition is missing or a prohibition applies.");

    [Fact]
    public void PrefixAndSuffixRenderTheTrainedPrompt()
    {
        string prefix = JevstralDecider.RenderPrefix(Refund, "Policy: refunds need a receipt. No receipt.");
        Assert.Equal(
            "[SYSTEM_PROMPT]Judge whether the Document meets the requirements based on the Query and the Instruction " +
            "provided. Note that the answer can only be \"yes\" or \"no\".[/SYSTEM_PROMPT][INST]<Instruct>: " +
            JevstralDecider.Preamble +
            "\nQuestion: Under the stated policy, is the requested action permitted?" +
            "\nExactly one of these answers is correct:" +
            "\n- option no: A condition is missing or a prohibition applies." +
            "\n- option yes: Every required condition is established." +
            "\n\n<Document>: Policy: refunds need a receipt. No receipt.", prefix);
        Assert.Equal(
            "\n\n<Query>: Is option yes (Every required condition is established.) the correct answer to the question?[/INST]",
            JevstralDecider.RenderSuffix(Refund, Refund.Options[1]));
    }

    [Fact]
    public void PerOptionLayoutPutsTheDocumentInEachBranch()
    {
        const string state = "Policy: refunds need a receipt. No receipt.";
        string prefix = JevstralDecider.RenderPrefix(Refund, state, DecisionLayout.PerOption);
        Assert.DoesNotContain("<Document>", prefix);
        Assert.EndsWith("\n- option yes: Every required condition is established.", prefix);
        Assert.Equal(
            "\n\n<Query>: Is option no (A condition is missing or a prohibition applies.) the correct answer to the question?" +
            "\n\n<Document>: " + state + "[/INST]",
            JevstralDecider.RenderSuffix(Refund, Refund.Options[0], DecisionLayout.PerOption, state));
    }

    [Fact]
    public async Task BranchesMatchRunningEachContinuationAlone()
    {
        if (Fixtures.ModelPath is not { } path) { _output.WriteLine("JEVSTRAL_MODEL is not set; skipping"); return; }
        var options = new ParallelOptions();
        using var model = new MinistralModel(path);

        int[] prefix = [.. model.Tokenizer.Encode(JevstralDecider.RenderPrefix(Refund,
            "Policy: refunds require a receipt and purchase within 30 days. Bought 12 days ago, no receipt."), addSpecial: true)];
        int[][] branches = [.. Refund.Options.Select(o =>
            model.Tokenizer.Encode(JevstralDecider.RenderSuffix(Refund, o), addSpecial: false).ToArray())];
        int[] rows = [13059, 1956, 2638];   // any few vocabulary rows will do

        // Reference: prefill the prefix, then each branch on its own from the same cache.
        model.ResetKvCache();
        await model.PrefillAsync(prefix, options);
        var alone = new List<float[]>();
        foreach (int[] branch in branches)
        {
            model.KvCache.Truncate(prefix.Length);
            alone.Add(await model.ForwardSelectedAsync(branch, rows, options));
        }

        // One pass: prefix as trunk, every branch after it.
        model.ResetKvCache();
        float[][] tree = await model.ForwardTreeAsync(prefix, branches, rows, options);

        // And branches only, off a cached prefix.
        model.ResetKvCache();
        await model.PrefillAsync(prefix, options);
        float[][] cached = await model.ForwardBranchesAsync(branches, rows, options);

        for (int b = 0; b < branches.Length; b++)
        {
            Assert.Equal(alone[b], tree[b]);
            Assert.Equal(alone[b], cached[b]);
        }
    }

    /// <summary>
    /// With a fixed question and changing data, the instruction and option list stay in the cache
    /// between calls. Reusing them must change nothing: a warm decider and a cold one give
    /// bit-identical margins.
    /// </summary>
    [Fact]
    public async Task ReusingTheQuestionPrefixIsBitIdentical()
    {
        if (Fixtures.ModelPath is not { } path) { _output.WriteLine("JEVSTRAL_MODEL is not set; skipping"); return; }
        const string first = "Policy: refunds require a receipt. The customer has a receipt.";
        const string second = "Policy: refunds require a receipt. The customer has no receipt.";

        using var warm = await JevstralDecider.OpenAsync(path);
        await warm.DecideAsync(Refund, first);
        DecisionResult reused = await warm.DecideAsync(Refund, second);

        using var cold = await JevstralDecider.OpenAsync(path);
        DecisionResult fresh = await cold.DecideAsync(Refund, second);

        _output.WriteLine($"reused {reused.CachedTokens} of {reused.PrefixTokens} prefix tokens (cold: {fresh.CachedTokens})");
        Assert.True(reused.CachedTokens > fresh.CachedTokens, "the question prefix was not reused");
        Assert.Equal(fresh.Margins, reused.Margins);
    }

    /// <summary>Two questions interleaved: each comes back from its snapshot, bit-identical to a cold call.</summary>
    [Fact]
    public async Task InterleavedQuestionsAreRestoredFromTheirSnapshots()
    {
        if (Fixtures.ModelPath is not { } path) { _output.WriteLine("JEVSTRAL_MODEL is not set; skipping"); return; }
        DecisionQuestion intent = DecisionQuestion.Choice("Which intent does the message express?",
            new("refund", "Wants money back"), new("cancel", "Wants to end a subscription"), new("other", "Anything else"));
        const string data = "Please stop my plan at the end of the month.";

        using var warm = await JevstralDecider.OpenAsync(path);
        await warm.DecideAsync(intent, "I was charged twice.");
        await warm.DecideAsync(Refund, "Policy: refunds require a receipt. The customer has a receipt.");
        DecisionResult restored = await warm.DecideAsync(intent, data);

        using var cold = await JevstralDecider.OpenAsync(path);
        DecisionResult fresh = await cold.DecideAsync(intent, data);

        _output.WriteLine($"restored {restored.CachedTokens} of {restored.PrefixTokens}; snapshot hits {warm.QuestionCacheHits}");
        Assert.Equal(1, warm.QuestionCacheHits);
        Assert.True(restored.CachedTokens > fresh.CachedTokens);
        Assert.Equal(fresh.Margins, restored.Margins);
    }

    [Fact]
    public async Task ADecisionIsADistributionOverItsOptions()
    {
        if (Fixtures.ModelPath is not { } path) { _output.WriteLine("JEVSTRAL_MODEL is not set; skipping"); return; }
        using var decider = await JevstralDecider.OpenAsync(path);

        DecisionResult no = await decider.DecideAsync(Refund,
            "Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago but has no receipt. Issue a refund.");
        _output.WriteLine($"no-receipt case: {string.Join(", ", no.Labels.Zip(no.Probabilities, (l, p) => $"{l}={p:F3}"))}");
        // Only the shape is asserted here: whether the base model gets a policy item right is a
        // question for the benchmark, not for a unit test.
        Numeric.Close(1.0, no.Probabilities.Sum(), 1e-5, "probabilities sum");
        Assert.Equal(["no", "yes"], no.Labels);

        DecisionQuestion intent = DecisionQuestion.Choice("Which intent does the user's message express?",
            new("track_order", "Wants to know where an order is"), new("cancel_order", "Wants to cancel an order"),
            new("billing_question", "Asks about a charge or invoice"));
        DecisionResult result = await decider.DecideAsync(intent, "Please cancel order 5521, I no longer need it.");
        Assert.Equal("cancel_order", result.Label);
    }
}
