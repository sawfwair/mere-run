import ArgumentParser
import Foundation
import MereRunCore

struct TextEmbed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Generate native text or EmbeddingGemma 2 multimodal embeddings.",
        discussion: """
        Uses native MLX encoders. EmbeddingGemma 2 supports text, code, images, audio, and video,
        task prefixes, and 128/256/512/768-dimensional normalized vectors.

        Example:
          mere.run text embed "hello world"
          mere.run text embed "foo" "bar" --max-tokens 1024
          mere.run text embed "semantic search query" --output embeddings.json --pretty
          mere.run text embed "find a sorting function" --model text-embed-embeddinggemma2 --task code-retrieval --dimensions 256
          mere.run text embed --image ./photo.png --audio ./clip.wav --model text-embed-embeddinggemma2
          mere.run text embed --input-json ./mixed-inputs.json --model text-embed-embeddinggemma2

        Direct texts and media paths each produce an independent embedding.
        JSON uses {"inputs":[{"content":[{"type":"text","text":"caption"},
        {"type":"image","path":"photo.png"}]}]}; content preserves media order.
        """
    )

    @Argument(help: "One or more texts to embed.")
    var texts: [String] = []

    @Option(name: .long, parsing: .upToNextOption, help: "Independent local images (EmbeddingGemma 2).")
    var image: [String] = []

    @Option(name: .long, parsing: .upToNextOption, help: "Independent local audio segments, up to 30 seconds each (EmbeddingGemma 2).")
    var audio: [String] = []

    @Option(name: .long, parsing: .upToNextOption, help: "Independent local videos, sampled at 1 fps up to 32 frames (EmbeddingGemma 2).")
    var video: [String] = []

    @Option(name: .long, help: "Ordered mixed-input JSON document, or - for stdin (EmbeddingGemma 2).")
    var inputJSON: String?

    @Option(name: [.customShort("m"), .long], help: "Model path or model id (default: text-embed-qwen3-0.6b).")
    var model: String?

    @Option(name: [.long], help: "Maximum token length per input (clamped to model max).")
    var maxTokens: Int?

    @Option(name: .long, help: "EmbeddingGemma 2 task: raw, query, document, code-retrieval, question-answering, fact-checking, classification, clustering, similarity.")
    var task: String = "raw"

    @Option(name: .long, help: "EmbeddingGemma 2 document title (requires --task document).")
    var title: String?

    @Option(name: .long, help: "EmbeddingGemma 2 output dimensions: 128, 256, 512, or 768.")
    var dimensions: Int?

    @Option(name: [.customShort("o"), .long], help: "Optional output JSON path.")
    var output: String?

    @Flag(name: [.long], help: "Pretty-print JSON output.")
    var pretty: Bool = false

    func validate() throws {
        guard !texts.isEmpty || !image.isEmpty || !audio.isEmpty || !video.isEmpty || inputJSON != nil else {
            throw ValidationError("Provide text, --image, --audio, --video, or --input-json.")
        }
        guard inputJSON == nil || texts.isEmpty && image.isEmpty && audio.isEmpty && video.isEmpty else {
            throw ValidationError("--input-json cannot be combined with direct inputs.")
        }
        guard (image + audio + video).allSatisfy({ !$0.isEmpty && !$0.contains("://") }) else {
            throw ValidationError("Media paths must refer to local files.")
        }
        guard EmbeddingGemma2Task(rawValue: task) != nil else { throw ValidationError("Unknown embedding task: \(task).") }
        guard title == nil || task == "document" else { throw ValidationError("--title requires --task document.") }
        if let dimensions, !EmbeddingGemma2Config.outputDimensions.contains(dimensions) {
            throw ValidationError("--dimensions must be 128, 256, 512, or 768.")
        }
        if let maxTokens, maxTokens <= 0 { throw ValidationError("--max-tokens must be positive.") }
        if model == EmbeddingGemma2Catalog.modelID {
            if let maxTokens, maxTokens < 2 { throw ValidationError("EmbeddingGemma 2 requires at least two tokens (BOS and EOS).") }
        } else if model == nil || model == Qwen3EmbeddingCatalog.modelId {
            try validateQwenOptions()
        }
    }

    private func validateQwenOptions() throws {
        guard dimensions == nil, task == "raw", title == nil,
              image.isEmpty, audio.isEmpty, video.isEmpty, inputJSON == nil else {
            throw ValidationError("Embedding task, dimension, and media options require an EmbeddingGemma 2 model.")
        }
    }

    struct ModelConfigProbe: Decodable {
        let modelType: String
        enum CodingKeys: String, CodingKey { case modelType = "model_type" }
    }

    static func isEmbeddingGemma2(root: URL) throws -> Bool {
        let probe = try JSONDecoder().decode(ModelConfigProbe.self, from: Data(contentsOf: root.appending(path: "config.json")))
        guard ["embedding_gemma2", "qwen3"].contains(probe.modelType) else {
            throw ValidationError("Unsupported text embedding model type: \(probe.modelType).")
        }
        return probe.modelType == "embedding_gemma2"
    }

    func run() async throws {
        let inputs = try loadMediaInputs()
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let resolvedModelRoot = try await resolveModelRoot()
        let result: (embeddings: [[Float]], tokenCounts: [Int])
        let modelID: String
        if try Self.isEmbeddingGemma2(root: resolvedModelRoot) {
            let encoder = try await EmbeddingGemma2Model(resources: EmbeddingGemma2Resources(rootURL: resolvedModelRoot))
            if let inputs {
                result = try encoder.embed(inputs: inputs, task: EmbeddingGemma2Task(rawValue: task)!, title: title,
                                           dimensions: dimensions ?? 768, maxTokens: maxTokens)
            } else {
                result = try encoder.embed(texts: texts, task: EmbeddingGemma2Task(rawValue: task)!, title: title,
                                           dimensions: dimensions ?? 768, maxTokens: maxTokens)
            }
            modelID = EmbeddingGemma2Catalog.modelID
        } else {
            try validateQwenOptions()
            let encoder = try Qwen3EmbeddingModel(resources: Qwen3EmbeddingResources(rootURL: resolvedModelRoot))
            result = try encoder.embed(texts: texts, maxTokens: maxTokens)
            modelID = Qwen3EmbeddingCatalog.modelId
        }

        let promptTokens = result.tokenCounts.reduce(0, +)
        let payload = OpenAIEmbeddingResponse(
            model: modelID,
            data: result.embeddings.enumerated().map { index, vector in
                OpenAIEmbeddingDatum(index: index, embedding: vector)
            },
            usage: OpenAIEmbeddingUsage(
                prompt_tokens: promptTokens,
                total_tokens: promptTokens
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

        if let text = String(data: try encoder.encode(GateWarned(payload)), encoding: .utf8) {
            print(text)
        } else {
            throw ValidationError("Failed to encode embedding output as UTF-8.")
        }
    }

    func loadMediaInputs() throws -> [EmbeddingGemma2Input]? {
        if let inputJSON {
            let url = URL(fileURLWithPath: inputJSON).standardizedFileURL
            let data = inputJSON == "-" ? FileHandle.standardInput.readDataToEndOfFile() : try Data(contentsOf: url)
            let root = inputJSON == "-" ? URL(fileURLWithPath: FileManager.default.currentDirectoryPath) : url.deletingLastPathComponent()
            return try JSONDecoder().decode(EmbeddingGemma2InputDocument.self, from: data).resolved(relativeTo: root)
        }
        guard !image.isEmpty || !audio.isEmpty || !video.isEmpty else { return nil }
        return texts.map { EmbeddingGemma2Input(content: [.text($0)]) }
            + image.map { EmbeddingGemma2Input(content: [.image(URL(fileURLWithPath: $0).standardizedFileURL)]) }
            + audio.map { EmbeddingGemma2Input(content: [.audio(URL(fileURLWithPath: $0).standardizedFileURL)]) }
            + video.map { EmbeddingGemma2Input(content: [.video(URL(fileURLWithPath: $0).standardizedFileURL)]) }
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
