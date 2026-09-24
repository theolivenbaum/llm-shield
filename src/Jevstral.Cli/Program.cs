using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using Jevstral;
using Jevstral.Gguf;
using Jevstral.Model;

namespace Jevstral.Cli;

/// <summary>
/// Command-line front end: score content, inspect a GGUF, or dump activations for
/// the parity harness.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Usage();
            return args.Length == 0 ? 2 : 0;
        }

        try
        {
            return args[0] switch
            {
                "download" => await Download(args[1..]).ConfigureAwait(false),
                "verdict" => await Verdict(args[1..]).ConfigureAwait(false),
                "decide" => await Decide.Run(args[1..]).ConfigureAwait(false),
                "reason" => await Decide.Reason(args[1..]).ConfigureAwait(false),
                "inspect" => Inspect(args[1..]),
                "tokenize" => Tokenize(args[1..]),
                "dump" => await Dump(args[1..]).ConfigureAwait(false),
                "bench" => await Benchmark.Run(args[1..]).ConfigureAwait(false),
                _ => Unknown(args[0]),
            };
        }
        catch (Exception e) when (e is FileNotFoundException or KeyNotFoundException
                                    or InvalidDataException or NotSupportedException
                                    or IOException or HttpRequestException)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 1;
        }
    }

    /// <summary>
    /// Named literals so a run that produced Infinity or NaN still writes a file
    /// you can look at — the whole point of `dump` is diagnosing exactly that.
    /// </summary>
    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    private static int Unknown(string verb)
    {
        Console.Error.WriteLine($"error: unknown command '{verb}'");
        Usage();
        return 2;
    }

    private static void Usage() => Console.Error.WriteLine("""
        jev — Jevstral: typed decisions (noul / choice / score) on a CPU

          decide   <model.gguf> --tasks FILE.jsonl | --serve [--out FILE.jsonl] [--temps FILE.json]
                   [--layout shared|per-option] [--limit N] [--threads N]
                                                         typed decisions, JevBench-format JSONL
          reason   <reasoning.gguf> --tasks FILE.jsonl [--system SYSTEM_PROMPT.txt] [--out FILE.jsonl]
                   [--max-think N] [--rows N] [--temperature T] [--limit N] [--threads N]
                                                         reason, then decide (Ministral-3 reasoning checkpoints)
          download [decider|reasoning|all] [q8_0|q5_1|q4_0] [--to DIR]
                                                         fetch Jevstral models from models.curiosity.ai
                                                         (resumable, SHA-256 checked; reasoning brings its
                                                         system prompt). Default: decider q8_0
          download base [q5_1|q5_0|q4_0] [--to PATH]     the unadapted Shieldstral checkpoint
          verdict  <model.gguf> --instruct TEXT --query TEXT --document TEXT [--json]
                                [--document-file PATH] [--no-prefix-cache] [--prefix-cache PATH]
                                [--threads N]
          inspect  <model.gguf>                          print metadata and tensor summary
          tokenize <model.gguf> <text>                   print token ids and pieces
          dump     <model.gguf> <prompt-file> <out.json> record activations for parity checks
          bench    [<model.gguf|dir> ...] [--json out.json] [--no-micro] [--no-model]
                   [--prefill-tokens N] [--decode-tokens N] [--warmups N] [--repeats N]
                   [--threads N]
        """);

    // ------------------------------------------------------------- download

    private static async Task<int> Download(string[] args)
    {
        // `jev download base [q5_1|q5_0|q4_0] [--to PATH]`: the unadapted Shieldstral checkpoint.
        if (args.Length > 0 && args[0].Equals("base", StringComparison.OrdinalIgnoreCase)) return await DownloadBase(args[1..]).ConfigureAwait(false);

        var models = new List<JevstralModel> { JevstralModel.Decider };
        var quantization = JevstralQuantization.Q8_0;
        string? directory = null;
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a == "--to") directory = Next(args, ref i);
            else if (a.Equals("all", StringComparison.OrdinalIgnoreCase)) models = [JevstralModel.Decider, JevstralModel.Reasoning];
            else if (Enum.TryParse(a, ignoreCase: true, out JevstralModel m)) models = [m];
            else if (!Enum.TryParse(a, ignoreCase: true, out quantization))
            {
                Console.Error.WriteLine(
                    $"error: unexpected '{a}' — expected decider, reasoning, all or base, and one of " +
                    string.Join(", ", Enum.GetNames<JevstralQuantization>()));
                return 2;
            }
        }

        foreach (JevstralModel model in models)
        {
            string name = JevstralModels.FileNameFor(model, quantization);
            string path = directory is null ? JevstralModels.DefaultPathFor(name) : Path.Combine(directory, name);
            if (File.Exists(path))
            {
                Console.Error.WriteLine($"{path} is already there ({new FileInfo(path).Length / (1024.0 * 1024 * 1024):F2} GiB)");
            }
            else
            {
                Console.Error.WriteLine($"{JevstralModels.UrlFor(model, quantization)}\n  -> {path}");
                await JevstralModels.EnsureAsync(model, quantization, path, DownloadProgressPrinter()).ConfigureAwait(false);
                Console.Error.WriteLine(JevstralModels.Sha256For(name) is null ? "  (mirror: no published checksum to verify)" : "  SHA-256 verified");
            }
            Console.WriteLine(path);
            if (model == JevstralModel.Reasoning)
            {
                await JevstralModels.EnsureReasoningSystemPromptAsync(path).ConfigureAwait(false);
                Console.WriteLine(JevstralModels.SystemPromptPathFor(path));
            }
        }
        return 0;
    }

    private static async Task<int> DownloadBase(string[] args)
    {
        var quantization = ShieldstralQuantization.Q5_1;
        string? destination = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--to": destination = Next(args, ref i); break;
                default:
                    if (!Enum.TryParse(args[i], ignoreCase: true, out quantization))
                    {
                        Console.Error.WriteLine(
                            $"error: unknown model '{args[i]}' — expected one of {string.Join(", ", Enum.GetNames<ShieldstralQuantization>())}");
                        return 2;
                    }
                    break;
            }
        }

        string path = destination ?? ModelDownloader.DefaultPathFor(quantization);
        if (File.Exists(path))
        {
            Console.WriteLine($"{path} is already there ({new FileInfo(path).Length / (1024.0 * 1024 * 1024):F2} GiB)");
            return 0;
        }

        Console.Error.WriteLine($"{ModelDownloader.UrlFor(quantization)}\n  -> {path}");
        await ModelDownloader.EnsureModelAsync(quantization, path, DownloadProgressPrinter()).ConfigureAwait(false);
        Console.WriteLine(path);
        return 0;
    }

    private static Action<DownloadProgress> DownloadProgressPrinter()
    {
        // Without a terminal the carriage return would produce one enormous line
        // in a log file, so redirected output gets a line every 5%.
        bool tty = !Console.IsErrorRedirected;
        long lastLine = -1;
        return p =>
        {
            long bucket = tty ? 0 : (long)(p.Fraction * 20);
            if (!tty && bucket == lastLine) return;
            lastLine = bucket;

            string total = p.TotalBytes is long t ? $" / {t / (1024.0 * 1024 * 1024):F2} GiB" : string.Empty;
            string line = $"  {p.DownloadedBytes / (1024.0 * 1024 * 1024):F2} GiB{total}  {p.Fraction:P1}";
            bool last = p.TotalBytes is long all && p.DownloadedBytes >= all;
            Console.Error.Write(tty ? $"\r{line}   " + (last ? "\n" : "") : line + "\n");
        };
    }

    // -------------------------------------------------------------- verdict

    /// <summary>One yes/no question in the base checkpoint's native format.</summary>
    private static async Task<int> Verdict(string[] args)
    {
        if (args.Length == 0) { Usage(); return 2; }
        string model = args[0];
        string instruct = "Answer strictly from the facts in the document.";
        string query = "Does the document answer the question?";
        string? document = null;
        bool json = false, cache = true;
        string? cachePath = null;
        int threads = -1;

        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--instruct": instruct = Next(args, ref i); break;
                case "--query": query = Next(args, ref i); break;
                case "--document": document = Next(args, ref i); break;
                case "--document-file": document = File.ReadAllText(Next(args, ref i)); break;
                case "--prefix-cache": cachePath = Next(args, ref i); break;
                case "--no-prefix-cache": cache = false; break;
                case "--threads": threads = int.Parse(Next(args, ref i), CultureInfo.InvariantCulture); break;
                case "--json": json = true; break;
                default:
                    Console.Error.WriteLine($"error: unexpected argument '{args[i]}'");
                    return 2;
            }
        }

        if (document is null)
        {
            if (Console.IsInputRedirected) document = Console.In.ReadToEnd();
            else { Console.Error.WriteLine("error: --document or --document-file is required"); return 2; }
        }

        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };

        var sw = Stopwatch.StartNew();
        using var scorer = await VerdictScorer.OpenAsync(model, cache, cachePath, options).ConfigureAwait(false);
        TimeSpan load = sw.Elapsed;

        sw.Restart();
        VerdictResult result = await scorer.ScoreAsync(instruct, query, document).ConfigureAwait(false);
        TimeSpan elapsed = sw.Elapsed;

        if (json)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                score = result.Score,
                yes = result.IsYes,
                yes_logit = result.YesLogit,
                no_logit = result.NoLogit,
                prompt_tokens = result.PromptTokens,
                prefilled_tokens = result.PrefilledTokens,
                load_ms = load.TotalMilliseconds,
                inference_ms = elapsed.TotalMilliseconds,
            }, JsonOptions));
        }
        else
        {
            Console.WriteLine(result);
            Console.WriteLine($"  {result.PromptTokens} prompt tokens " +
                              $"({result.PrefilledTokens} from the cached system prompt), " +
                              $"load {load.TotalMilliseconds:F0} ms, inference {elapsed.TotalMilliseconds:F0} ms");
        }
        return 0;
    }

    // -------------------------------------------------------------- inspect

    private static int Inspect(string[] args)
    {
        if (args.Length == 0) { Usage(); return 2; }
        using var gguf = new GgufFile(args[0]);

        Console.WriteLine($"{args[0]}");
        Console.WriteLine($"  GGUF v{gguf.Version}, {gguf.Tensors.Count} tensors, data at {gguf.DataOffset}");
        Console.WriteLine();
        Console.WriteLine("metadata:");
        foreach ((string key, object value) in gguf.Metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            Console.WriteLine($"  {key,-52} {Describe(value)}");

        var byType = gguf.Tensors.Values
            .GroupBy(t => t.Type)
            .OrderByDescending(g => g.Sum(t => t.ByteCount));
        Console.WriteLine();
        Console.WriteLine("tensors by type:");
        foreach (var group in byType)
            Console.WriteLine($"  {group.Key,-10} {group.Count(),5} tensors  " +
                              $"{group.Sum(t => t.ByteCount) / (1024.0 * 1024):N1} MiB");

        if (gguf.GetString("general.architecture") is "mistral3" or "ministral3")
        {
            Console.WriteLine();
            Console.WriteLine("config: " + ModelConfig.FromGguf(gguf));
        }
        return 0;
    }

    private static string Describe(object value) => value switch
    {
        string s => s.Length > 90 ? $"\"{s[..90]}...\" ({s.Length} chars)" : $"\"{s}\"",
        string[] a => $"string[{a.Length}]  e.g. {string.Join(", ", a.Take(4).Select(x => $"\"{Escape(x)}\""))}",
        Array a => $"{a.GetType().GetElementType()!.Name}[{a.Length}]",
        _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? "",
    };

    private static string Escape(string s) => s.Length > 16 ? s[..16] + "..." : s;

    // ------------------------------------------------------------- tokenize

    private static int Tokenize(string[] args)
    {
        if (args.Length < 2) { Usage(); return 2; }
        using var gguf = new GgufFile(args[0]);
        var tokenizer = Tokenization.TekkenTokenizer.FromGguf(gguf);

        string text = string.Join(' ', args[1..]);
        List<int> ids = tokenizer.Encode(text);
        Console.WriteLine($"{ids.Count} tokens");
        Console.WriteLine(string.Join(", ", ids));
        foreach (int id in ids)
            Console.WriteLine($"  {id,7}  {JsonSerializer.Serialize(tokenizer.Decode(id))}");
        return 0;
    }

    // ----------------------------------------------------------------- dump

    /// <summary>
    /// Records the same tensors the Python reference records, so the two JSON
    /// files can be diffed directly.
    /// </summary>
    private sealed class JsonActivationSink : IActivationSink
    {
        public Dictionary<string, object> Tensors { get; } = [];

        public void Observe(string name, ReadOnlySpan<float> values, int rows, int columns)
        {
            double checksum = 0;
            for (int i = 0; i < values.Length; i++) checksum += (double)values[i] * values[i];
            var lastRow = new float[columns];
            values.Slice((rows - 1) * columns, columns).CopyTo(lastRow);
            Tensors[name] = new { rows, columns, last_row = lastRow, checksum };
        }
    }

    private static async Task<int> Dump(string[] args)
    {
        if (args.Length < 3) { Usage(); return 2; }
        using var model = new MinistralModel(args[0]);
        string prompt = File.ReadAllText(args[1]);
        int[] tokens = [.. model.Tokenizer.Encode(prompt, addSpecial: true)];

        var sink = new JsonActivationSink();
        model.ResetKvCache();
        float[] logits = (await model.ForwardAsync(tokens, sink, new ParallelOptions()).ConfigureAwait(false)).ToArray();

        int[] top20 = [.. Enumerable.Range(0, logits.Length)
            .OrderByDescending(i => logits[i]).Take(20)];
        var values = new Dictionary<string, float>();
        foreach (int i in top20) values[i.ToString(CultureInfo.InvariantCulture)] = logits[i];

        File.WriteAllText(args[2], JsonSerializer.Serialize(new
        {
            prompt,
            tokens,
            tensors = sink.Tensors,
            logits = new { top20, values },
        }, JsonOptions));
        Console.WriteLine($"wrote {args[2]} ({sink.Tensors.Count} tensors, {tokens.Length} tokens)");
        return 0;
    }

    private static string Next(string[] args, ref int i)
    {
        if (++i >= args.Length) throw new InvalidDataException($"'{args[i - 1]}' needs a value");
        return args[i];
    }
}
