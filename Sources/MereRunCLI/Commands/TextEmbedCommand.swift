import ArgumentParser
import Foundation
import MereRunCore

struct TextEmbed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Generate text embeddings using native Qwen3-Embedding-0.6B.",
        discussion: """
        Uses the native MLX Qwen encoder (no Python runtime).

        Example:
          mere.run text embed "hello world"
          mere.run text embed "foo" "bar" --max-tokens 1024
          mere.run text embed "semantic search query" --output embeddings.json --pretty
        """
    )

    @Argument(help: "One or more texts to embed.")
    var texts: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Model path or model id (default: text-embed-qwen3-0.6b).")
    var model: String?

    @Option(name: [.long], help: "Maximum token length per input (clamped to model max).")
    var maxTokens: Int?

    @Option(name: [.customShort("o"), .long], help: "Optional output JSON path.")
    var output: String?

    @Flag(name: [.long], help: "Pretty-print JSON output.")
    var pretty: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let resolvedModelPath = try resolveModelPath()
        let resources = Qwen3EmbeddingResources(rootURL: URL(fileURLWithPath: resolvedModelPath))
        let model = try Qwen3EmbeddingModel(resources: resources)
        let result = try model.embed(texts: texts, maxTokens: maxTokens)

        let payload = EmbeddingResponse(
            model: Qwen3EmbeddingCatalog.modelId,
            data: result.embeddings.enumerated().map { index, vector in
                EmbeddingDatum(index: index, embedding: vector)
            },
            usage: EmbeddingUsage(
                promptTokens: result.tokenCounts.reduce(0, +),
                totalTokens: result.tokenCounts.reduce(0, +)
            )
        )

        let encoder = JSONEncoder()
        if pretty {
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        }
        let data = try encoder.encode(payload)

        if let outputPath = output {
            let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: outputURL, options: [.atomic])
        }

        if let text = String(data: data, encoding: .utf8) {
            print(text)
        } else {
            throw ValidationError("Failed to encode embedding output as UTF-8.")
        }
    }

    private func resolveModelPath() throws -> String {
        let fm = FileManager.default

        if let model, !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
            let asPath = URL(fileURLWithPath: trimmed).standardizedFileURL
            if fm.fileExists(atPath: asPath.path) {
                return asPath.path
            }

            let normalized = trimmed.lowercased()
            if normalized == Qwen3EmbeddingCatalog.modelId {
                if let resolved = Qwen3EmbeddingCatalog.resolveModelPath(fileManager: fm) {
                    return resolved
                }
                throw ValidationError(
                    """
                    Embedding model '\(Qwen3EmbeddingCatalog.modelId)' not found.
                    Upload/download the model archive (\(Qwen3EmbeddingCatalog.archiveKey)) first.
                    """
                )
            }

            throw ValidationError("Model path not found: \(trimmed)")
        }

        if let resolved = Qwen3EmbeddingCatalog.resolveModelPath(fileManager: fm) {
            return resolved
        }

        throw ValidationError(
            """
            Embedding model not found.
            Expected model id: \(Qwen3EmbeddingCatalog.modelId)
            Upload/download archive: \(Qwen3EmbeddingCatalog.archiveKey)
            """
        )
    }
}

private struct EmbeddingResponse: Codable, Sendable {
    let object = "list"
    let model: String
    let data: [EmbeddingDatum]
    let usage: EmbeddingUsage
}

private struct EmbeddingDatum: Codable, Sendable {
    let object = "embedding"
    let index: Int
    let embedding: [Float]
}

private struct EmbeddingUsage: Codable, Sendable {
    let promptTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }
}
