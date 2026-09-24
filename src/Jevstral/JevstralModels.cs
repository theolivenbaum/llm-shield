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

    /// <summary>
    /// SHA256SUMS of the files published at <see cref="DefaultBaseUrl"/>. A download is hashed
    /// against these before it is renamed into place. A mirror (<c>JEVSTRAL_MODEL_BASE_URL</c>)
    /// serving a different build is not checked, since its files have their own checksums; a
    /// rebuild with <c>tools/build_models.sh</c> reproduces these bit for bit.
    /// </summary>
    public static IReadOnlyDictionary<string, string> PublishedSha256 { get; } = new Dictionary<string, string>
    {
        ["Jevstral-1.0-3B-Q8_0.gguf"] = "3393d3f75880ba78b2ac479efa2d469cb602b0ada86af3ffd46dcc896644dfbc",
        ["Jevstral-1.0-3B-Q5_1.gguf"] = "84f42fb862accf8baecb57b8d78b2aee098cdc56218baef8fe31e17d7b41ace8",
        ["Jevstral-1.0-3B-Q4_0.gguf"] = "9a63736344fd301b406d952367c605b974e237aea6afbcff84b8058993531f93",
        ["Ministral-3-3B-Reasoning-Q8_0.gguf"] = "c09140947eb236a35e3511586347c5482a4f32c2dc02fd73fcea7dc01dc2b0e1",
        ["Ministral-3-3B-Reasoning-Q5_1.gguf"] = "c118e35cb4b9c49af8b8e15ae3eef51e6ac489a76d95e701eeb31da6b3ee4b68",
        ["Ministral-3-3B-Reasoning-Q4_0.gguf"] = "d6b4ef09eb0ceec74edc29efcdeb43e079decc6e7688f7d27f70c6b08558e7b4",
        [ReasoningSystemPromptFile] = "aba1efaff0bdc73f4a864139e2c07dce8fc1e3df7b5d5c2b2623df6816697156",
    };

    /// <summary>The checksum a download of <paramref name="fileName"/> is held to: none from a mirror.</summary>
    public static string? Sha256For(string fileName)
        => BaseUrl == DefaultBaseUrl && PublishedSha256.TryGetValue(fileName, out string? sha) ? sha : null;

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
        await ModelDownloader.DownloadFileAsync(UrlFor(model, quantization), path, reportProgress, cancellationToken,
            Sha256For(FileNameFor(model, quantization))).ConfigureAwait(false);
        return path;
    }

    /// <summary>Where the reasoning system prompt sits: next to the model it belongs to.</summary>
    public static string SystemPromptPathFor(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath))!, ReasoningSystemPromptFile);

    /// <summary>The reasoning system prompt, fetched once next to the model.</summary>
    public static async Task<string> EnsureReasoningSystemPromptAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        string path = SystemPromptPathFor(modelPath);
        await ModelDownloader.DownloadFileAsync(BaseUrl + ReasoningSystemPromptFile, path, null, cancellationToken,
            Sha256For(ReasoningSystemPromptFile)).ConfigureAwait(false);
        return await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
    }
}
