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

        let resolvedModelRoot = try await resolveModelRoot()
        let resources = Qwen3EmbeddingResources(rootURL: resolvedModelRoot)
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

    private func resolveModelRoot() async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: model,
                defaultModelID: Qwen3EmbeddingCatalog.modelId,
                progress: nil
            )
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw ValidationError(error.localizedDescription)
        }
    }
}

private struct EmbeddingResponse: Encodable, Sendable {
    let object = "list"
    let model: String
    let data: [EmbeddingDatum]
    let usage: EmbeddingUsage
}

private struct EmbeddingDatum: Encodable, Sendable {
    let object = "embedding"
    let index: Int
    let embedding: [Float]
}

private struct EmbeddingUsage: Encodable, Sendable {
    let promptTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }
}
