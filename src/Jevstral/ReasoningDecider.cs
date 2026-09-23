using System.Text;
using Jevstral.Model;
using Jevstral.Numerics;

namespace Jevstral;

/// <summary>A decision answered after the model has reasoned about it.</summary>
public sealed record ReasonedDecision(DecisionResult Decision, string Reasoning, int ThinkTokens, bool FinishedThinking);

/// <summary>
/// Reason, then decide: a Ministral-3 reasoning checkpoint thinks in <c>[THINK]…[/THINK]</c>,
/// and every label is then scored as the continuation of "Final answer:".
///
/// The verdict decider reads one token per option, and that token cannot compute a date
/// window, sum a table or chain three rules. On JevBench's hard tier that is most of the items.
/// Ministral-3-3B-Reasoning shares Shieldstral's architecture and tokenizer, so it runs on the
/// same forward pass. Two things make it affordable on a CPU:
/// - several items decode together, each in its own region of the KV cache
///   (<see cref="MinistralModel.ForwardRowsAsync"/>), so a step decodes the weights once for
///   all of them;
/// - the labels are scored against the row's own cached reasoning, with no re-prefill.
///
/// A label's score is its mean token log-probability, and the distribution is their softmax
/// divided by <see cref="Temperature"/>. The prompt matches <c>tools/decision/reason.py</c>
/// byte for byte.
///
/// Not thread-safe: one instance owns one KV cache.
/// </summary>
public sealed class ReasoningDecider : IDisposable
{
    private const int EndOfSequence = 2;

    private readonly MinistralModel _model;
    private readonly ParallelOptions _options;
    private readonly string _systemPrompt;
    private readonly int _think;
    private readonly int _endThink;

    /// <summary>Most thinking tokens per item; a row that has not closed its thinking by then is closed for it.</summary>
    public int MaxThinkTokens { get; set; } = 1536;

    /// <summary>Most rows decoded together.</summary>
    public int MaxRows { get; set; } = 12;

    /// <summary>Rows × (prompt + thinking budget) per batch. The cache holds fp32 keys and values, ~213 KB per token.</summary>
    public int KvTokenBudget { get; set; } = 24_000;

    /// <summary>Divisor on the per-label mean log-probabilities; fitted on held-out data, never on JevBench.</summary>
    public float Temperature { get; set; } = 1f;

    public MinistralModel Model => _model;

    private ReasoningDecider(MinistralModel model, string systemPrompt, ParallelOptions options)
    {
        _model = model;
        _options = options;
        _systemPrompt = systemPrompt;
        int[] think = [.. model.Tokenizer.Encode("[THINK]", addSpecial: false)];
        int[] endThink = [.. model.Tokenizer.Encode("[/THINK]", addSpecial: false)];
        if (think.Length != 1 || endThink.Length != 1)
        {
            model.Dispose();
            throw new InvalidDataException(
                "This GGUF has no [THINK] / [/THINK] control tokens: it is not a reasoning checkpoint. " +
                "Convert mistralai/Ministral-3-3B-Reasoning-2512 with tools/convert_shieldstral_to_gguf.py.");
        }
        _think = think[0];
        _endThink = endThink[0];
    }

    /// <param name="systemPrompt">The checkpoint's SYSTEM_PROMPT.txt, which is part of its trained format.</param>
    public static ReasoningDecider Open(string ggufPath, string systemPrompt, ParallelOptions? options = null)
        => new(new MinistralModel(ggufPath, initialCacheCapacity: 4096), systemPrompt.Trim(), options ?? new ParallelOptions());

    /// <summary>Downloads the published reasoning checkpoint and its system prompt if needed, then opens it.</summary>
    public static async Task<ReasoningDecider> CreateAsync(
        JevstralQuantization quantization = JevstralQuantization.Q8_0, string? downloadToPath = null,
        ParallelOptions? options = null, Action<DownloadProgress>? reportProgress = null,
        CancellationToken cancellationToken = default)
    {
        string path = await JevstralModels.EnsureAsync(JevstralModel.Reasoning, quantization, downloadToPath,
            reportProgress, cancellationToken).ConfigureAwait(false);
        string system = await JevstralModels.EnsureReasoningSystemPromptAsync(path, cancellationToken).ConfigureAwait(false);
        return Open(path, system, options);
    }

    /// <summary>The prompt, up to and including the opening <c>[THINK]</c>.</summary>
    public string RenderPrompt(DecisionQuestion question, string state)
    {
        var sb = new StringBuilder(state.Length + 1024);
        sb.Append(ChatTemplate.SystemOpen).Append(_systemPrompt).Append(ChatTemplate.SystemClose)
          .Append(ChatTemplate.InstructionOpen)
          .Append(state).Append("\n\n---\nQuestion: ").Append(question.Instructions.Trim())
          .Append("\n\nPossible answers (choose exactly one label):");
        foreach (DecisionOption o in question.Options)
        {
            sb.Append("\n- ").Append(o.Label);
            if (o.Description.Length > 0) sb.Append(": ").Append(o.Description);
        }
        sb.Append("\n\nWork it out from the facts above, applying every stated rule, condition and exception. ")
          .Append("End with a line of the form 'Final answer: <label>'.")
          .Append(ChatTemplate.InstructionClose).Append("[THINK]");
        return sb.ToString();
    }

    public async ValueTask<ReasonedDecision> DecideAsync(DecisionQuestion question, string state)
        => (await DecideAsync([(question, state)]).ConfigureAwait(false))[0];

    /// <summary>Reasons about every item, as many at a time as the row and KV budgets allow.</summary>
    public async ValueTask<IReadOnlyList<ReasonedDecision>> DecideAsync(IReadOnlyList<(DecisionQuestion Question, string State)> items)
    {
        var prompts = items.Select(i => _model.Tokenizer.Encode(RenderPrompt(i.Question, i.State), addSpecial: true).ToArray()).ToArray();
        var results = new ReasonedDecision[items.Count];

        // Similar lengths together keep a batch's regions, and its padding of wasted steps, small.
        int[] order = [.. Enumerable.Range(0, items.Count).OrderBy(i => prompts[i].Length)];
        var batch = new List<int>();
        int widest = 0;
        foreach (int i in order)
        {
            int need = prompts[i].Length + MaxThinkTokens + 8;
            if (batch.Count > 0 && (batch.Count >= MaxRows || (batch.Count + 1) * Math.Max(widest, need) > KvTokenBudget))
            {
                await RunBatchAsync(batch, items, prompts, results).ConfigureAwait(false);
                batch.Clear();
                widest = 0;
            }
            batch.Add(i);
            widest = Math.Max(widest, need);
        }
        if (batch.Count > 0) await RunBatchAsync(batch, items, prompts, results).ConfigureAwait(false);
        return results;
    }

    private async ValueTask RunBatchAsync(List<int> batch, IReadOnlyList<(DecisionQuestion Question, string State)> items,
        int[][] prompts, ReasonedDecision[] results)
    {
        int rows = batch.Count;
        var rowBase = new int[rows];
        var length = new int[rows];
        for (int r = 1; r < rows; r++) rowBase[r] = rowBase[r - 1] + prompts[batch[r - 1]].Length + MaxThinkTokens + 8;

        _model.ResetKvCache();
        var next = new int[rows];
        // Prefill one row at a time: a single long GEMM each, and the activation buffers stay one prompt wide.
        for (int r = 0; r < rows; r++)
        {
            int[] p = prompts[batch[r]];
            float[][] logits = await _model.ForwardRowsAsync(0, [rowBase[r]], [0], [p], null, _options).ConfigureAwait(false);
            length[r] = p.Length;
            next[r] = Kernels.ArgMax(logits[0]);
        }

        var thought = new List<int>[rows];
        var done = new bool[rows];
        for (int r = 0; r < rows; r++) thought[r] = [];
        for (int step = 0; step < MaxThinkTokens; step++)
        {
            var active = new List<int>();
            for (int r = 0; r < rows; r++)
            {
                if (done[r]) continue;
                thought[r].Add(next[r]);
                if (next[r] == _endThink || next[r] == EndOfSequence) done[r] = true;
                else active.Add(r);
            }
            if (active.Count == 0 || step == MaxThinkTokens - 1) break;

            float[][] logits = await _model.ForwardRowsAsync(0,
                [.. active.Select(r => rowBase[r])], [.. active.Select(r => length[r])],
                [.. active.Select(r => new[] { next[r] })], null, _options).ConfigureAwait(false);
            for (int k = 0; k < active.Count; k++)
            {
                int r = active[k];
                length[r]++;
                next[r] = Kernels.ArgMax(logits[k]);
            }
        }

        for (int r = 0; r < rows; r++)
        {
            List<int> t = thought[r];
            if (t.Count > 0 && t[^1] == EndOfSequence) t.RemoveAt(t.Count - 1);
            bool finished = t.Count > 0 && t[^1] == _endThink;
            // The last token decoded was sampled but never fed; close the thinking on top of what is cached.
            int cached = length[r];
            var tail = new List<int>();
            if (t.Count > 0) tail.Add(t[^1]);
            if (!finished) tail.Add(_endThink);
            tail.AddRange(_model.Tokenizer.Encode("\nFinal answer:", addSpecial: false));
            (DecisionQuestion question, _) = items[batch[r]];
            results[batch[r]] = await ScoreAsync(question, rowBase[r], cached, tail, t, finished).ConfigureAwait(false);
        }
        _model.ResetKvCache();
    }

    /// <summary>
    /// Mean log-probability of " label" after the row's reasoning plus <paramref name="tail"/>.
    /// The tail is appended to the row once; its last logits give every label's first token.
    /// A longer label then runs as a continuation written into the slots after it. The row's
    /// length does not advance for labels, so each one overwrites the last.
    /// </summary>
    private async ValueTask<ReasonedDecision> ScoreAsync(DecisionQuestion question, int rowBase, int cached,
        List<int> tail, List<int> thought, bool finished)
    {
        float[][] tailLogits = await _model.ForwardRowTokensAsync(0, rowBase, cached, [.. tail], _options).ConfigureAwait(false);
        float[] first = tailLogits[^1];
        int after = cached + tail.Count;

        var scores = new float[question.Options.Count];
        for (int i = 0; i < scores.Length; i++)
        {
            int[] label = [.. _model.Tokenizer.Encode(" " + question.Options[i].Label, addSpecial: false)];
            double sum = LogSoftmaxAt(first, label[0]);
            if (label.Length > 1)
            {
                float[][] rest = await _model.ForwardRowTokensAsync(0, rowBase, after, label[..^1], _options).ConfigureAwait(false);
                for (int j = 1; j < label.Length; j++) sum += LogSoftmaxAt(rest[j - 1], label[j]);
            }
            scores[i] = (float)(sum / label.Length);
        }

        var probs = new float[scores.Length];
        for (int i = 0; i < probs.Length; i++) probs[i] = scores[i] / Temperature;
        Kernels.Softmax(probs);
        string reasoning = _model.Tokenizer.Decode(thought);
        var decision = new DecisionResult([.. question.Options.Select(o => o.Label)], probs, scores, cached, tail.Count);
        return new ReasonedDecision(decision, reasoning, thought.Count, finished);
    }

    private static double LogSoftmaxAt(float[] logits, int id)
    {
        float max = float.NegativeInfinity;
        foreach (float v in logits) if (v > max) max = v;
        double sum = 0;
        foreach (float v in logits) sum += Math.Exp(v - max);
        return logits[id] - max - Math.Log(sum);
    }

    public void Dispose() => _model.Dispose();
}
