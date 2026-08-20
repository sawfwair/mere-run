import ArgumentParser
import Foundation
import MereRunCore

struct VisionEmbed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Generate shared text/image embeddings with native Qwen3-VL.",
        discussion: """
        Each --text and --image value becomes an independent embedding item.
        Use --input-json for mixed text+image records or stable caller IDs.

        Examples:
          mere.run vision embed --text "a white SUV"
          mere.run vision embed --image ./vehicle.jpg --dimensions 1024
          mere.run vision embed --input-json ./retrieval-inputs.json --pretty

        JSON input may be an array or {"inputs":[...]}. Each record accepts:
          {"id":"query","text":"a white SUV","instruction":"Retrieve similar vehicles"}
          {"id":"candidate-1","images":["./crop.jpg"]}
          {"id":"mixed","text":"rear view","image":"./vehicle.jpg"}
        """
    )

    @Option(
        name: [.long],
        parsing: .upToNextOption,
        help: "One or more independent text inputs."
    )
    var text: [String] = []

    @Option(
        name: [.long],
        parsing: .upToNextOption,
        help: "One or more independent local image paths."
    )
    var image: [String] = []

    @Option(name: [.long], help: "JSON batch path, or - to read JSON from stdin.")
    var inputJSON: String?

    @Option(name: [.long], help: "Retrieval task instruction applied to direct inputs.")
    var instruction: String?

    @Option(
        name: [.customShort("m"), .long],
        help: "Model path or model id (default: vision-embed-qwen3-vl-2b)."
    )
    var model: String?

    @Option(name: [.long], help: "Output dimensions (1...2048; truncated vectors are renormalized).")
    var dimensions: Int?

    @Option(name: [.long], help: "Maximum tokens per item.")
    var maxTokens: Int = Qwen3VLEmbeddingCatalog.defaultMaxTokens

    @Option(name: [.long], help: "Minimum pixels used when resizing each image.")
    var minPixels: Int = Qwen3VLEmbeddingCatalog.defaultMinPixels

    @Option(name: [.long], help: "Maximum pixels used when resizing each image.")
    var maxPixels: Int = Qwen3VLEmbeddingCatalog.defaultMaxPixels

    @Option(name: [.customShort("o"), .long], help: "Optional output JSON path.")
    var output: String?

    @Flag(name: [.long], help: "Pretty-print JSON output.")
    var pretty: Bool = false

    func validate() throws {
        if inputJSON != nil, !text.isEmpty || !image.isEmpty {
            throw ValidationError("--input-json cannot be combined with --text or --image.")
        }
        if inputJSON == nil, text.isEmpty, image.isEmpty {
            throw ValidationError("Provide --text, --image, or --input-json.")
        }
        if let dimensions,
           !(1...Qwen3VLEmbeddingCatalog.nativeDimensions).contains(dimensions) {
            throw ValidationError("--dimensions must be between 1 and 2048.")
        }
        guard maxTokens > 0 else {
            throw ValidationError("--max-tokens must be positive.")
        }
        guard minPixels > 0, maxPixels >= minPixels else {
            throw ValidationError("--min-pixels must be positive and no greater than --max-pixels.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let prepared = try loadInputs()
        let resolution: ManagedModelResolver.RuntimeResolution
        do {
            resolution = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: model,
                defaultModelID: Qwen3VLEmbeddingCatalog.modelID,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        CLIStderr.write("\rDownloading Qwen3-VL embedding model (\(percent)%)")
                    case .extracting:
                        CLIStderr.write("\rPreparing Qwen3-VL embedding model")
                    }
                }
            )
            CLIStderr.write("\n")
        } catch let error as ManagedModelResolver.ResolverError {
            throw ValidationError(error.localizedDescription)
        }

        let embedder = try Qwen3VLEmbeddingModel(
            resources: Qwen3VLEmbeddingResources(rootURL: resolution.url)
        )
        let results = try embedder.embed(
            inputs: prepared.map(\.input),
            dimensions: dimensions,
            maxTokens: maxTokens,
            minPixels: minPixels,
            maxPixels: maxPixels
        )
        let outputDimensions = dimensions ?? embedder.dimensions
        let payload = VisionEmbeddingResponse(
            model: resolution.spec.id,
            dimensions: outputDimensions,
            data: zip(prepared, results).enumerated().map { index, pair in
                VisionEmbeddingDatum(
                    index: index,
                    id: pair.1.id ?? pair.0.input.id,
                    input: pair.0.summary,
                    embedding: pair.1.embedding
                )
            },
            usage: VisionEmbeddingUsage(
                prompt_tokens: results.reduce(0) { $0 + $1.tokenCount },
                total_tokens: results.reduce(0) { $0 + $1.tokenCount }
            )
        )

        let encoder = JSONEncoder()
        if pretty { encoder.outputFormatting = [.prettyPrinted, .sortedKeys] }
        let data = try encoder.encode(payload)
        if let output {
            let outputURL = URL(fileURLWithPath: output).standardizedFileURL
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: outputURL, options: .atomic)
        }
        guard let json = String(data: data, encoding: .utf8) else {
            throw ValidationError("Failed to encode embedding output as UTF-8.")
        }
        print(json)
    }

    private func loadInputs() throws -> [PreparedVisionEmbeddingInput] {
        if let inputJSON {
            return try loadJSONInputs(from: inputJSON)
        }
        let textInputs = text.enumerated().map { index, value in
            let id = "text-\(index)"
            return PreparedVisionEmbeddingInput(
                input: Qwen3VLEmbeddingInput(id: id, text: value, instruction: instruction),
                summary: .init(type: "text", text: value, images: [])
            )
        }
        let imageInputs = image.enumerated().map { index, path in
            let url = URL(fileURLWithPath: path).standardizedFileURL
            let id = "image-\(index)"
            return PreparedVisionEmbeddingInput(
                input: Qwen3VLEmbeddingInput(id: id, imageURLs: [url], instruction: instruction),
                summary: .init(type: "image", text: nil, images: [url.path])
            )
        }
        return textInputs + imageInputs
    }

    private func loadJSONInputs(from path: String) throws -> [PreparedVisionEmbeddingInput] {
        let data: Data
        let baseURL: URL
        if path == "-" {
            data = FileHandle.standardInput.readDataToEndOfFile()
            baseURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        } else {
            let inputURL = URL(fileURLWithPath: path).standardizedFileURL
            data = try Data(contentsOf: inputURL)
            baseURL = inputURL.deletingLastPathComponent()
        }

        let decoder = JSONDecoder()
        let records: [VisionEmbeddingInputRecord]
        if let array = try? decoder.decode([VisionEmbeddingInputRecord].self, from: data) {
            records = array
        } else {
            records = try decoder.decode(VisionEmbeddingInputDocument.self, from: data).inputs
        }
        guard !records.isEmpty else {
            throw ValidationError("--input-json contains no input records.")
        }

        return try records.enumerated().map { index, record in
            let rawImages = (record.image.map { [$0] } ?? []) + (record.images ?? [])
            let imageURLs = rawImages.map { rawPath -> URL in
                if rawPath.hasPrefix("/") {
                    return URL(fileURLWithPath: rawPath).standardizedFileURL
                }
                return baseURL.appendingPathComponent(rawPath).standardizedFileURL
            }
            let trimmedText = record.text?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmedText?.isEmpty == false || !imageURLs.isEmpty else {
                throw ValidationError("Input record \(index) has neither text nor images.")
            }
            let id = record.id ?? "input-\(index)"
            let type = imageURLs.isEmpty ? "text" : (trimmedText == nil ? "image" : "multimodal")
            return PreparedVisionEmbeddingInput(
                input: Qwen3VLEmbeddingInput(
                    id: id,
                    text: trimmedText,
                    imageURLs: imageURLs,
                    instruction: record.instruction ?? instruction
                ),
                summary: .init(type: type, text: trimmedText, images: imageURLs.map(\.path))
            )
        }
    }
}

private struct VisionEmbeddingInputDocument: Decodable {
    let inputs: [VisionEmbeddingInputRecord]
}

private struct VisionEmbeddingInputRecord: Decodable {
    let id: String?
    let text: String?
    let image: String?
    let images: [String]?
    let instruction: String?
}

private struct PreparedVisionEmbeddingInput {
    let input: Qwen3VLEmbeddingInput
    let summary: VisionEmbeddingInputSummary
}

private struct VisionEmbeddingResponse: Encodable {
    let object = "list"
    let model: String
    let dimensions: Int
    let data: [VisionEmbeddingDatum]
    let usage: VisionEmbeddingUsage
}

private struct VisionEmbeddingDatum: Encodable {
    let object = "embedding"
    let index: Int
    let id: String?
    let input: VisionEmbeddingInputSummary
    let embedding: [Float]
}

private struct VisionEmbeddingInputSummary: Encodable {
    let type: String
    let text: String?
    let images: [String]
}

private struct VisionEmbeddingUsage: Encodable {
    let prompt_tokens: Int
    let total_tokens: Int
}
