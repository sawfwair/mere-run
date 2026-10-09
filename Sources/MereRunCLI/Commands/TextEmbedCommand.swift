import ArgumentParser
import Foundation
import MereRunCore

struct TextEmbed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Generate native text, multimodal, or PPLX multi-vector embeddings.",
        discussion: """
        Uses native MLX encoders. EmbeddingGemma 2 supports text, code, images, audio, and video,
        task prefixes, and 128/256/512/768-dimensional normalized vectors.
        PPLX v2 late returns normalized 128-dimensional token vectors; context returns
        int8 chunk vectors (1024/2048 dimensions). PPLX output uses data[].embeddings.
        Context JSON uses {"documents":[["first chunk","second chunk"]]} and never truncates chunks.

        Example:
          mere.run text embed "find this" --model text-embed-pplx-v2-late-0.6b --task query
          mere.run text embed --chunks-json chunks.json --model text-embed-pplx-v2-context-9b-preview --normalize
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

    @Option(name: .long, parsing: .upToNextOption, help: "Independent local images (EmbeddingGemma 2 or PPLX late documents).")
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

    @Option(name: .long, help: "Task (PPLX: query/document; default document): raw, query, document, code-retrieval, question-answering, fact-checking, classification, clustering, similarity.")
    var task: String = "raw"

    @Option(name: .long, help: "EmbeddingGemma 2 document title (requires --task document).")
    var title: String?

    @Option(name: .long, help: "Output dimensions: Gemma 128/256/512/768; PPLX late 128; PPLX context 1024/2048.")
    var dimensions: Int?

    @Option(name: .long, help: "PPLX contextual JSON: {documents:[[chunk,...],...]}, or - for stdin.")
    var chunksJSON: String?

    @Flag(name: .long, help: "Normalize PPLX contextual int8 vectors after dimension truncation.")
    var normalize: Bool = false

    @Option(name: [.customShort("o"), .long], help: "Optional output JSON path.")
    var output: String?

    @Flag(name: [.long], help: "Pretty-print JSON output.")
    var pretty: Bool = false

    func validate() throws {
        guard !texts.isEmpty || !image.isEmpty || !audio.isEmpty || !video.isEmpty || inputJSON != nil || chunksJSON != nil else {
            throw ValidationError("Provide text, --image, --audio, --video, --input-json, or --chunks-json.")
        }
        guard chunksJSON == nil || texts.isEmpty && image.isEmpty && audio.isEmpty && video.isEmpty && inputJSON == nil else {
            throw ValidationError("--chunks-json cannot be combined with direct or mixed inputs.")
        }
        guard inputJSON == nil || texts.isEmpty && image.isEmpty && audio.isEmpty && video.isEmpty else {
            throw ValidationError("--input-json cannot be combined with direct inputs.")
        }
        guard (image + audio + video).allSatisfy({ !$0.isEmpty && !$0.contains("://") }) else {
            throw ValidationError("Media paths must refer to local files.")
        }
        guard EmbeddingGemma2Task(rawValue: task) != nil else { throw ValidationError("Unknown embedding task: \(task).") }
        guard title == nil || task == "document" else { throw ValidationError("--title requires --task document.") }
        if let dimensions, ![128, 256, 512, 768, 1024, 2048].contains(dimensions) {
            throw ValidationError("--dimensions must be 128, 256, 512, 768, 1024, or 2048.")
        }
        if let maxTokens, maxTokens <= 0 { throw ValidationError("--max-tokens must be positive.") }
        if let model, PPLXEmbedV2Catalog.modelIDs.contains(model) {
            try validatePPLXOptions(contextual: PPLXEmbedV2Catalog.isContextual(model))
        } else if model == EmbeddingGemma2Catalog.modelID {
            try validateGemmaOptions()
            if let maxTokens, maxTokens < 2 { throw ValidationError("EmbeddingGemma 2 requires at least two tokens (BOS and EOS).") }
        } else if model == nil || model == Qwen3EmbeddingCatalog.modelId {
            try validateQwenOptions()
        }
    }

    private func validateQwenOptions() throws {
        guard dimensions == nil, task == "raw", title == nil,
              image.isEmpty, audio.isEmpty, video.isEmpty, inputJSON == nil, chunksJSON == nil, !normalize else {
            throw ValidationError("Embedding task, dimension, and media options require a compatible EmbeddingGemma 2 or PPLX model.")
        }
    }

    private func validateGemmaOptions() throws {
        guard chunksJSON == nil, !normalize, dimensions == nil || EmbeddingGemma2Config.outputDimensions.contains(dimensions!) else {
            throw ValidationError("--chunks-json and --normalize require PPLX context; Gemma dimensions must be 128/256/512/768.")
        }
    }

    func validatePPLXOptions(contextual: Bool) throws {
        guard ["raw", "query", "document"].contains(task), title == nil,
              audio.isEmpty, video.isEmpty, inputJSON == nil else {
            throw ValidationError("PPLX supports query/document tasks, text, late-model images, and contextual --chunks-json.")
        }
        guard image.isEmpty || !contextual && texts.isEmpty && task != "query" && chunksJSON == nil else {
            throw ValidationError("PPLX images require a separate document batch with a late model.")
        }
        guard contextual || chunksJSON == nil && !normalize else {
            throw ValidationError("--chunks-json and --normalize require PPLX context.")
        }
        if let dimensions, contextual ? ![1024, 2048].contains(dimensions) : dimensions != 128 {
            throw ValidationError("PPLX late dimensions are 128; context dimensions are 1024 or 2048.")
        }
    }

    struct ModelConfigProbe: Decodable {
        let modelType: String
        enum CodingKeys: String, CodingKey { case modelType = "model_type" }
    }

    static func embeddingModelType(root: URL) throws -> String {
        let probe = try JSONDecoder().decode(ModelConfigProbe.self, from: Data(contentsOf: root.appending(path: "config.json")))
        guard ["embedding_gemma2", "qwen3", "qwen3_5", "pplx_contextual_qwen3_5"].contains(probe.modelType) else {
            throw ValidationError("Unsupported text embedding model type: \(probe.modelType).")
        }
        return probe.modelType
    }

    static func isEmbeddingGemma2(root: URL) throws -> Bool {
        try embeddingModelType(root: root) == "embedding_gemma2"
    }

    func run() async throws {
        let inputs = try loadMediaInputs()
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let resolvedModelRoot = try await resolveModelRoot()
        let type = try Self.embeddingModelType(root: resolvedModelRoot)
        if ["qwen3_5", "pplx_contextual_qwen3_5"].contains(type) {
            try validatePPLXOptions(contextual: type == "pplx_contextual_qwen3_5")
            let encoder = try PPLXEmbedV2Model(resources: PPLXEmbedV2Resources(rootURL: resolvedModelRoot))
            let result: PPLXEmbedV2Result
            if !image.isEmpty {
                result = try encoder.embed(images: image.map { URL(fileURLWithPath: $0) }, maxTokens: maxTokens)
            } else {
                let documents: [[String]]
                if let chunksJSON {
                    struct ChunkDocument: Decodable { let documents: [[String]] }
                    let data = chunksJSON == "-" ? FileHandle.standardInput.readDataToEndOfFile()
                        : try Data(contentsOf: URL(fileURLWithPath: chunksJSON))
                    documents = try JSONDecoder().decode(ChunkDocument.self, from: data).documents
                } else { documents = texts.map { [$0] } }
                result = try encoder.embed(documents: documents, task: task == "query" ? .query : .document,
                                           dimensions: dimensions, normalize: normalize, maxTokens: maxTokens)
            }
            try emit(result)
            return
        }
        let result: (embeddings: [[Float]], tokenCounts: [Int])
        let modelID: String
        if type == "embedding_gemma2" {
            try validateGemmaOptions()
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

        try emit(payload)
    }

    private func emit<Value: Encodable>(_ payload: Value) throws {
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
