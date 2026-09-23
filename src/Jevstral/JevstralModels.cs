namespace Jevstral;

/// <summary>The two models a Jevstral deployment uses.</summary>
public enum JevstralModel
{
    /// <summary>Shieldstral 1.0 3B with the Jevstral adapter folded in: <see cref="JevstralDecider"/>.</summary>
    Decider,
    /// <summary>Mistral's Ministral-3-3B-Reasoning, unchanged: <see cref="ReasoningDecider"/>.</summary>
    Reasoning,
}

/// <summary>The quantizations tools/build_models.sh produces by default.</summary>
public enum JevstralQuantization
{
    /// <summary>Closest to the bf16 weights; the build that parity is measured on.</summary>
    Q8_0,
    Q5_1,
    /// <summary>Smallest.</summary>
    Q4_0,
}

/// <summary>
/// Where the published Jevstral files live and what they are called: the layout
/// <c>tools/build_models.sh</c> writes and models.curiosity.ai serves.
/// <c>JEVSTRAL_MODEL_BASE_URL</c> points it at a mirror.
/// </summary>
public static class JevstralModels
{
    public const string DefaultBaseUrl = "https://models.curiosity.ai/jevstral/";

    /// <summary>The reasoning checkpoint's trained system prompt, published next to its GGUF.</summary>
    public const string ReasoningSystemPromptFile = "Ministral-3-3B-Reasoning.SYSTEM_PROMPT.txt";

    public static string BaseUrl
        => Environment.GetEnvironmentVariable("JEVSTRAL_MODEL_BASE_URL") is { Length: > 0 } url
            ? (url.EndsWith('/') ? url : url + "/")
            : DefaultBaseUrl;

    public static string FileNameFor(JevstralModel model, JevstralQuantization quantization) => model switch
    {
        JevstralModel.Decider => $"Jevstral-1.0-3B-{quantization}.gguf",
        JevstralModel.Reasoning => $"Ministral-3-3B-Reasoning-{quantization}.gguf",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    public static string UrlFor(JevstralModel model, JevstralQuantization quantization) => BaseUrl + FileNameFor(model, quantization);

    public static string DefaultPathFor(string fileName) => Path.Combine(Path.GetTempPath(), "Jevstral", fileName);

    /// <summary>Makes sure the model is on disk (resumable, see <see cref="ModelDownloader"/>) and returns its path.</summary>
    public static async Task<string> EnsureAsync(
        JevstralModel model, JevstralQuantization quantization = JevstralQuantization.Q8_0, string? downloadToPath = null,
        Action<DownloadProgress>? reportProgress = null, CancellationToken cancellationToken = default)
    {
        string path = downloadToPath ?? DefaultPathFor(FileNameFor(model, quantization));
        await ModelDownloader.DownloadFileAsync(UrlFor(model, quantization), path, reportProgress, cancellationToken)
            .ConfigureAwait(false);
        return path;
    }

    /// <summary>The reasoning system prompt, fetched once next to the model.</summary>
    public static async Task<string> EnsureReasoningSystemPromptAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath))!, ReasoningSystemPromptFile);
        await ModelDownloader.DownloadFileAsync(BaseUrl + ReasoningSystemPromptFile, path, null, cancellationToken)
            .ConfigureAwait(false);
        return await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
    }
}
