using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Jevstral.Cli;

/// <summary>
/// `decide`: answers typed decisions from a JSONL file. Each line is a JevBench-style record:
/// <c>{"id", "state", "question": {"type", "instructions", "criteria"}, "labels"}</c>.
/// It writes one line per record with the distribution, the raw per-option margins and the
/// timing. The format is the one <c>tools/decision</c> reads, so the C# runtime and the
/// PyTorch research path can be compared item for item.
/// </summary>
internal static class Decide
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length == 0) { Console.Error.WriteLine("error: decide needs a model path"); return 2; }
        string model = args[0];
        string? tasks = null, output = null, temps = null;
        int limit = 0, threads = -1;
        bool serve = false;
        var layout = DecisionLayout.SharedDocument;
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--tasks": tasks = args[++i]; break;
                case "--out": output = args[++i]; break;
                case "--temps": temps = args[++i]; break;
                case "--limit": limit = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--threads": threads = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--serve": serve = true; break;
                case "--layout":
                    layout = args[++i] switch
                    {
                        "shared" or "docfirst" => DecisionLayout.SharedDocument,
                        "per-option" or "card" => DecisionLayout.PerOption,
                        var other => throw new ArgumentException($"unknown layout '{other}': expected shared or per-option"),
                    };
                    break;
                default: Console.Error.WriteLine($"error: unexpected argument '{args[i]}'"); return 2;
            }
        }
        if (tasks is null && !serve) { Console.Error.WriteLine("error: --tasks or --serve is required"); return 2; }

        Dictionary<string, float>? temperature = temps is null ? null
            : JsonSerializer.Deserialize<Dictionary<string, float>>(File.ReadAllText(temps));
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        var sw = Stopwatch.StartNew();
        using var decider = await JevstralDecider.OpenAsync(model, temperature, options, layout).ConfigureAwait(false);
        Console.Error.WriteLine($"loaded in {sw.Elapsed.TotalSeconds:F1}s");

        using TextWriter writer = output is null ? Console.Out : new StreamWriter(output, append: false);
        int n = 0, correct = 0, scored = 0;
        // --serve answers one request line on stdin with one line on stdout, for a harness that
        // keeps the model loaded between decisions (tools/decision/jevbench_adapter.py).
        foreach (string line in serve ? ReadStdin() : File.ReadLines(tasks!))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (limit > 0 && n >= limit) break;
            JsonNode record = JsonNode.Parse(line)!;
            (DecisionQuestion question, string state) = Parse(record);

            sw.Restart();
            DecisionResult result = await decider.DecideAsync(question, state).ConfigureAwait(false);
            double seconds = sw.Elapsed.TotalSeconds;

            string? expected = record["expected"]?.ToString();
            bool? ok = expected is null ? null : result.Label == expected;
            if (ok is not null) { scored++; if (ok.Value) correct++; }
            n++;

            var probs = new JsonObject();
            for (int i = 0; i < result.Labels.Count; i++) probs[result.Labels[i]] = result.Probabilities[i];
            writer.WriteLine(new JsonObject
            {
                ["id"] = record["id"]?.ToString(),
                ["qtype"] = record["question"]!["type"]!.ToString(),
                ["labels"] = new JsonArray([.. result.Labels.Select(l => (JsonNode)l)]),
                ["probs"] = probs,
                ["margins"] = new JsonArray([.. result.Margins.Select(m => (JsonNode)m)]),
                ["predicted"] = result.Label,
                ["expected"] = expected,
                ["correct"] = ok,
                ["prefix_tokens"] = result.PrefixTokens,
                ["cached_tokens"] = result.CachedTokens,
                ["suffix_tokens"] = result.SuffixTokens,
                ["latency_s"] = seconds,
            }.ToJsonString());
            writer.Flush();
            Console.Error.WriteLine($"[{n}] {record["id"]} {result.Label} ({expected}) " +
                                    $"{result.PrefixTokens}+{result.SuffixTokens} tok {seconds:F2}s");
        }
        if (scored > 0) Console.Error.WriteLine($"accuracy {correct}/{scored} = {(double)correct / scored:F3}");
        return 0;
    }

    /// <summary>
    /// `reason`: the reason-then-decide path over the same JSONL records. Items are batched, so
    /// the whole file is read first and the lines are written as each batch completes.
    /// </summary>
    public static async Task<int> Reason(string[] args)
    {
        if (args.Length == 0) { Console.Error.WriteLine("error: reason needs a model path"); return 2; }
        string model = args[0];
        string? tasks = null, output = null, system = null;
        int limit = 0, threads = -1, maxThink = 1536, rows = 12;
        float temperature = 1f;
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--tasks": tasks = args[++i]; break;
                case "--out": output = args[++i]; break;
                case "--system": system = args[++i]; break;
                case "--limit": limit = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--threads": threads = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--max-think": maxThink = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--rows": rows = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--temperature": temperature = float.Parse(args[++i], CultureInfo.InvariantCulture); break;
                default: Console.Error.WriteLine($"error: unexpected argument '{args[i]}'"); return 2;
            }
        }
        // `jev download reasoning` puts the system prompt next to the model.
        system ??= File.Exists(JevstralModels.SystemPromptPathFor(model)) ? JevstralModels.SystemPromptPathFor(model) : null;
        if (tasks is null) { Console.Error.WriteLine("error: --tasks is required"); return 2; }
        if (system is null)
        {
            Console.Error.WriteLine($"error: no --system given and no {JevstralModels.ReasoningSystemPromptFile} next to the model; " +
                                    "`jev download reasoning` fetches both");
            return 2;
        }

        var records = File.ReadLines(tasks).Where(l => !string.IsNullOrWhiteSpace(l)).Select(l => JsonNode.Parse(l)!).ToList();
        if (limit > 0) records = records.Take(limit).ToList();
        var items = records.Select(Parse).ToList();

        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        using var decider = ReasoningDecider.Open(model, File.ReadAllText(system), options);
        decider.MaxThinkTokens = maxThink;
        decider.MaxRows = rows;
        decider.Temperature = temperature;
        var sw = Stopwatch.StartNew();
        IReadOnlyList<ReasonedDecision> results = await decider.DecideAsync(items).ConfigureAwait(false);
        double seconds = sw.Elapsed.TotalSeconds;

        using TextWriter writer = output is null ? Console.Out : new StreamWriter(output, append: false);
        int correct = 0, scored = 0;
        for (int i = 0; i < records.Count; i++)
        {
            DecisionResult d = results[i].Decision;
            string? expected = records[i]["expected"]?.ToString();
            bool? ok = expected is null ? null : d.Label == expected;
            if (ok is not null) { scored++; if (ok.Value) correct++; }
            var probs = new JsonObject();
            for (int k = 0; k < d.Labels.Count; k++) probs[d.Labels[k]] = d.Probabilities[k];
            writer.WriteLine(new JsonObject
            {
                ["id"] = records[i]["id"]?.ToString(),
                ["qtype"] = records[i]["question"]!["type"]!.ToString(),
                ["labels"] = new JsonArray([.. d.Labels.Select(l => (JsonNode)l)]),
                ["probs"] = probs,
                ["margins"] = new JsonArray([.. d.Margins.Select(m => (JsonNode)m)]),
                ["predicted"] = d.Label,
                ["expected"] = expected,
                ["correct"] = ok,
                ["think_tokens"] = results[i].ThinkTokens,
                ["finished_thinking"] = results[i].FinishedThinking,
                ["reasoning"] = results[i].Reasoning,
                ["latency_s"] = seconds / records.Count,
            }.ToJsonString());
        }
        if (scored > 0) Console.Error.WriteLine($"accuracy {correct}/{scored} = {(double)correct / scored:F3}, {seconds / records.Count:F1}s/item");
        return 0;
    }

    private static IEnumerable<string> ReadStdin()
    {
        while (Console.In.ReadLine() is { } line) yield return line;
    }

    /// <summary>JevBench's noul labels are no/yes over false/true criteria; score criteria are a list of levels.</summary>
    internal static (DecisionQuestion, string) Parse(JsonNode record)
    {
        JsonNode q = record["question"]!;
        string type = q["type"]!.ToString();
        string instructions = q["instructions"]?.ToString() ?? "";
        JsonNode? criteria = q["criteria"];
        JsonNode stateNode = record["state"]!;
        string state = stateNode is JsonValue v && v.TryGetValue(out string? text)
            ? text
            : PythonJson(stateNode);

        DecisionQuestion question = type switch
        {
            "noul" => DecisionQuestion.Noul(instructions, criteria?["true"]?.ToString(), criteria?["false"]?.ToString()),
            "score" => DecisionQuestion.Score(instructions,
                [.. criteria!.AsArray().Select(c => c!.ToString())]),
            _ => DecisionQuestion.Choice(instructions,
                [.. record["labels"]!.AsArray().Select(l => new DecisionOption(l!.ToString(), criteria?[l!.ToString()]?.ToString() ?? ""))]),
        };
        return (question, state);
    }

    /// <summary>
    /// A structured state rendered as Python's <c>json.dumps(ensure_ascii=False)</c> does, with ", " and ": "
    /// separators. That is what the research path feeds the model, and a LoRA is only valid for the
    /// bytes it was trained on.
    /// </summary>
    internal static string PythonJson(JsonNode node) => node switch
    {
        JsonObject o => "{" + string.Join(", ", o.Select(kv => JsonSerializer.Serialize(kv.Key, Unescaped) + ": " + PythonJson(kv.Value!))) + "}",
        JsonArray a => "[" + string.Join(", ", a.Select(x => PythonJson(x!))) + "]",
        _ => node.ToJsonString(Unescaped),
    };

    private static readonly JsonSerializerOptions Unescaped = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };
}
