using System.Text;
using Jevstral.Model;
using Jevstral.Numerics;

namespace Jevstral;

/// <summary>The three typed questions of a System One decision (the Jev / laya / djev schema).</summary>
public enum DecisionType
{
    /// <summary>Yes or no. The options are the false and the true criterion, in that order.</summary>
    Noul,
    /// <summary>Exactly one of N labelled options.</summary>
    Choice,
    /// <summary>An ordinal level; option i is level i.</summary>
    Score,
}

/// <summary>One answer a decision can take, with the criterion that makes it correct.</summary>
public sealed record DecisionOption(string Label, string Description = "");

/// <summary>A typed question about a piece of state.</summary>
public sealed record DecisionQuestion(DecisionType Type, string Instructions, IReadOnlyList<DecisionOption> Options)
{
    /// <summary>A yes/no question. The criteria read as "answer yes if …" / "answer no if …".</summary>
    public static DecisionQuestion Noul(string instructions, string? whenTrue = null, string? whenFalse = null)
        => new(DecisionType.Noul, instructions,
        [
            new("no", string.IsNullOrWhiteSpace(whenFalse) ? "no, the statement does not hold" : whenFalse),
            new("yes", string.IsNullOrWhiteSpace(whenTrue) ? "yes, the statement holds" : whenTrue),
        ]);

    public static DecisionQuestion Choice(string instructions, params DecisionOption[] options)
        => new(DecisionType.Choice, instructions, options);

    /// <summary>Levels 0..n-1, one description each.</summary>
    public static DecisionQuestion Score(string instructions, params string[] levels)
        => new(DecisionType.Score, instructions,
            [.. levels.Select((d, i) => new DecisionOption(i.ToString(System.Globalization.CultureInfo.InvariantCulture), d))]);
}

/// <summary>
/// A distribution over a question's options, in option order.
/// <see cref="Margins"/> are the raw yes-minus-no log-odds of each option's read,
/// before temperature, which is what a calibration fit wants.
/// </summary>
public sealed record DecisionResult(
    IReadOnlyList<string> Labels,
    float[] Probabilities,
    float[] Margins,
    int PrefixTokens,
    int SuffixTokens)
{
    public string Label => Labels[Kernels.ArgMax(Probabilities)];
    public float Confidence => Probabilities.Max();
    public float ProbabilityOf(string label) => Probabilities[IndexOf(label)];

    /// <summary>For a score question: the expected level, Σ i·p_i.</summary>
    public float ExpectedLevel => Probabilities.Select((p, i) => p * i).Sum();

    private int IndexOf(string label)
    {
        for (int i = 0; i < Labels.Count; i++) if (Labels[i] == label) return i;
        throw new KeyNotFoundException($"'{label}' is not one of this question's options.");
    }
}

/// <summary>
/// Typed decisions (noul / choice / score) answered with Shieldstral's yes/no verdict.
///
/// Shieldstral answers one question: does the document meet the requirement in the
/// query. A typed decision is reduced to that primitive with one read per option,
/// "is option X the correct answer?". The per-option log-odds (yes minus no) then
/// compete in a softmax. A yes/no question is read the same way, as a two-option
/// choice between its false and true criteria. Asking the question directly leaves
/// a topical bias in the verdict: a document that is merely about the question leans
/// yes. With two reads the bias is common to both and cancels in the softmax.
///
/// The instruction and the document come first and are prefilled once. Each option
/// then adds only its query (a few dozen tokens), restarting from the same KV prefix.
/// This order (Instruct, Document, Query) differs from the model card's (Instruct,
/// Query, Document). Without it every option would re-read the whole document. It
/// also lets the document be read with the question already in view.
///
/// The prompt text must match <c>tools/decision/decision_prompts.py</c> byte for
/// byte. A LoRA trained there is only valid for the prompt it was trained on.
///
/// Not thread-safe: one instance owns one KV cache.
/// </summary>
public sealed class JevstralDecider : IDisposable
{
    public const string Preamble =
        "You are a careful decision engine. Answer strictly from the facts in the Document, " +
        "applying every rule, condition and exception it states.";

    private readonly MinistralModel _model;
    private readonly bool _ownsModel;
    private readonly ParallelOptions _options;
    private readonly int[] _yesTokens;
    private readonly int[] _noTokens;
    private readonly int[] _verdictRows;
    private readonly IReadOnlyDictionary<string, float> _temperature;
    private SystemPromptCache? _system;

    public MinistralModel Model => _model;

    private JevstralDecider(MinistralModel model, bool ownsModel, ParallelOptions options,
        IReadOnlyDictionary<string, float>? temperature)
    {
        _model = model;
        _ownsModel = ownsModel;
        _options = options;
        _temperature = temperature ?? new Dictionary<string, float>();
        _yesTokens = Resolve("yes");
        _noTokens = Resolve("no");
        _verdictRows = [.. _yesTokens, .. _noTokens];
        if (_yesTokens.Length == 0 || _noTokens.Length == 0)
        {
            if (ownsModel) model.Dispose();
            throw new InvalidDataException(
                "The model's vocabulary has no 'yes'/'no' tokens; this does not look like a Shieldstral checkpoint.");
        }
    }

    /// <param name="temperature">
    /// Divisors on the log-odds, keyed by question type (<c>noul</c>, <c>choice</c>,
    /// <c>score</c>) or by type and option-count bucket (<c>choice:3-5</c>), as laya
    /// keys them. The bucket key wins. A missing key means 1.
    /// </param>
    public static async ValueTask<JevstralDecider> OpenAsync(
        string ggufPath, IReadOnlyDictionary<string, float>? temperature = null, ParallelOptions? options = null)
    {
        var decider = new JevstralDecider(new MinistralModel(ggufPath, initialCacheCapacity: 2048),
            ownsModel: true, options ?? new ParallelOptions(), temperature);
        try
        {
            int[] system = [.. decider._model.Tokenizer.Encode(SystemBlock, addSpecial: true)];
            decider._system = await SystemPromptCache.CaptureAsync(decider._model, system, decider._options)
                .ConfigureAwait(false);
            return decider;
        }
        catch
        {
            decider.Dispose();
            throw;
        }
    }

    private static string SystemBlock
        => ChatTemplate.SystemOpen + VerdictScorer.SystemPrompt + ChatTemplate.SystemClose;

    /// <summary>The shared prefix: system prompt, instruction, option list, document.</summary>
    public static string RenderPrefix(DecisionQuestion question, string state)
    {
        var sb = new StringBuilder(state.Length + 512);
        sb.Append(SystemBlock).Append(ChatTemplate.InstructionOpen).Append("<Instruct>: ").Append(Preamble)
          .Append("\nQuestion: ").Append(question.Instructions.Trim());
        string noun = question.Type == DecisionType.Score ? "level" : "option";
        sb.Append("\nExactly one of these answers is correct:");
        foreach (DecisionOption o in question.Options)
        {
            sb.Append("\n- ").Append(noun).Append(' ').Append(o.Label);
            if (o.Description.Length > 0) sb.Append(": ").Append(o.Description);
        }
        sb.Append("\n\n<Document>: ").Append(state);
        return sb.ToString();
    }

    /// <summary>The per-option query, which closes the instruction turn.</summary>
    public static string RenderSuffix(DecisionQuestion question, DecisionOption option)
    {
        string noun = question.Type == DecisionType.Score ? "level" : "option";
        string d = option.Description.Length > 0 ? $" ({option.Description})" : "";
        return $"\n\n<Query>: Is {noun} {option.Label}{d} the correct answer to the question?{ChatTemplate.InstructionClose}";
    }

    public async ValueTask<DecisionResult> DecideAsync(DecisionQuestion question, string state, ParallelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(question);
        if (question.Options.Count < 2)
            throw new ArgumentException("A decision needs at least two options.", nameof(question));
        ParallelOptions po = options ?? _options;

        int[] prefix = [.. _model.Tokenizer.Encode(RenderPrefix(question, state), addSpecial: true)];
        int start = 0;
        if (_system is not null && _system.IsPrefixOf(prefix))
        {
            _system.RestoreInto(_model);
            start = _system.TokenCount;
        }
        else
        {
            _model.ResetKvCache();
        }

        // The prefix and every option's query in one pass: a causal trunk, then one branch per
        // option, reading only the verdict rows of the LM head.
        var branches = new int[question.Options.Count][];
        int suffixTokens = 0;
        for (int i = 0; i < branches.Length; i++)
        {
            branches[i] = [.. _model.Tokenizer.Encode(RenderSuffix(question, question.Options[i]), addSpecial: false)];
            suffixTokens += branches[i].Length;
        }
        float[][] verdicts = await _model.ForwardTreeAsync(prefix.AsMemory(start), branches, _verdictRows, po)
            .ConfigureAwait(false);
        _model.KvCache.Truncate(prefix.Length);

        var margins = new float[branches.Length];
        for (int i = 0; i < margins.Length; i++)
            margins[i] = Best(verdicts[i], 0, _yesTokens.Length) - Best(verdicts[i], _yesTokens.Length, _noTokens.Length);

        float t = TemperatureFor(question);
        var probs = new float[margins.Length];
        for (int i = 0; i < probs.Length; i++) probs[i] = margins[i] / t;
        Kernels.Softmax(probs);
        return new DecisionResult([.. question.Options.Select(o => o.Label)], probs, margins, prefix.Length, suffixTokens);
    }

    private float TemperatureFor(DecisionQuestion q)
    {
        string type = q.Type switch { DecisionType.Noul => "noul", DecisionType.Score => "score", _ => "choice" };
        int k = q.Options.Count;
        string bucket = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
        if (_temperature.TryGetValue($"{type}:{bucket}", out float t) || _temperature.TryGetValue(type, out t))
            return MathF.Max(t, 1e-3f);
        return 1f;
    }

    private static float Best(float[] logits, int start, int count)
    {
        float best = float.NegativeInfinity;
        for (int i = start; i < start + count; i++) if (logits[i] > best) best = logits[i];
        return best;
    }

    private int[] Resolve(string form)
    {
        var ids = new List<int>();
        for (int id = 0; id < _model.Tokenizer.VocabSize; id++)
            if (_model.Tokenizer.Decode(id).Trim().Trim('"', '\'', '.').ToLowerInvariant() == form) ids.Add(id);
        return [.. ids];
    }

    public void Dispose()
    {
        if (_ownsModel) _model.Dispose();
    }
}
