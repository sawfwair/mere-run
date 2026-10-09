import Foundation
import MereRunQwenModel
import MLX
import MLXNN

public final class PPLXEmbedV2Model {
    public let resources: PPLXEmbedV2Resources
    public let config: PPLXEmbedV2Config
    public let modelID: String
    let encoder: PPLXEmbedV2Encoder
    let tokenizer: PPLXEmbedV2Tokenizer
    let projection: MLXArray
    var vision: Q35VisionTower?

    public init(resources: PPLXEmbedV2Resources) throws {
        self.resources = resources
        config = try JSONDecoder().decode(PPLXEmbedV2Config.self, from: Data(contentsOf: resources.configURL))
        let manifest = try MereRunModelManifest.loadIfPresent(from: resources.rootURL)
        guard manifest == nil || manifest?.engine == .pplxEmbedV2 else {
            throw PPLXEmbedV2Error.invalidConfiguration("The installed manifest must identify the PPLX embedding runtime.")
        }
        modelID = manifest?.id ?? (config.isContextual ? PPLXEmbedV2Catalog.contextID
            : config.backbone.textConfig.hiddenSize == 1_024 ? PPLXEmbedV2Catalog.lateSmallID : PPLXEmbedV2Catalog.lateLargeID)
        let missing = resources.validate()
        guard missing.isEmpty else { throw PPLXEmbedV2Error.invalidConfiguration("Missing PPLX Embed v2 files: " + missing.map(\.path).joined(separator: ", ")) }
        tokenizer = try PPLXEmbedV2Tokenizer.load(root: resources.rootURL, contextual: config.isContextual)
        encoder = PPLXEmbedV2Encoder(config: config.backbone)
        let required = encoder.checkpointParameterNames
        var loaded: Set<String> = []
        var contextualProjection: MLXArray?
        if let quantization = config.quantization {
            let arrays = try FileManager.default.fileExists(atPath: resources.indexURL.path)
                ? HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: resources.indexURL)
                : MLX.loadArrays(url: resources.weightsURL)
            contextualProjection = arrays["contextual_projection.weight"]
            let mapped = Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value -> (String, MLXArray)? in
                guard let name = Self.textKey(key) else { return nil }
                return (name, name.hasSuffix(".conv1d.weight") ? value.transposed(0, 2, 1) : value)
            })
            try encoder.installQuantizedWeights(mapped, config: quantization)
        } else {
            try ModelWeightsLoader.applyHFSafetensors(
                indexURL: resources.indexURL, singleURL: resources.weightsURL, to: encoder, dtype: .float32,
                verify: [.noUnusedKeys, .shapeMismatch], mapper: { key, value in
                    if key == "contextual_projection.weight" { contextualProjection = value; return [] }
                    guard let name = Self.textKey(key) else { return [] }
                    loaded.insert(name)
                    return [(name, name.hasSuffix(".conv1d.weight") ? value.transposed(0, 2, 1) : value)]
                }
            )
            let absent = required.subtracting(loaded).sorted()
            guard absent.isEmpty else { throw PPLXEmbedV2Error.invalidConfiguration("Missing PPLX text tensors: " + absent.joined(separator: ", ")) }
        }
        if config.isContextual {
            guard let contextualProjection else { throw PPLXEmbedV2Error.invalidConfiguration("Missing contextual_projection.weight.") }
            projection = contextualProjection
        } else {
            struct DenseConfig: Decodable {
                let inFeatures: Int, outFeatures: Int, bias: Bool
                enum CodingKeys: String, CodingKey { case inFeatures = "in_features", outFeatures = "out_features", bias }
            }
            let dense = try JSONDecoder().decode(DenseConfig.self, from: Data(contentsOf: resources.rootURL.appending(path: "1_Dense/config.json")))
            guard dense.inFeatures == config.backbone.textConfig.hiddenSize, dense.outFeatures == 128, !dense.bias else {
                throw PPLXEmbedV2Error.invalidConfiguration("Unsupported PPLX late projection config.")
            }
            let arrays = try MLX.loadArrays(url: resources.rootURL.appending(path: "1_Dense/model.safetensors"))
            guard let weight = arrays["linear.weight"], arrays.count == 1 else {
                throw PPLXEmbedV2Error.invalidConfiguration("Missing or unsupported PPLX late projection tensors.")
            }
            projection = weight
        }
        guard projection.dtype == .float32, projection.shape == [config.embeddingDim, config.backbone.textConfig.hiddenSize] else {
            throw PPLXEmbedV2Error.invalidConfiguration("PPLX projection must retain FP32 and match the configured shape.")
        }
    }

    static func textKey(_ key: String) -> String? {
        let prefix = "language_model."
        guard key.hasPrefix(prefix) else { return nil }
        return String(key.dropFirst(prefix.count))
    }

    public func embed(texts: [String], task: PPLXEmbedV2Task = .document, dimensions: Int? = nil,
                      normalize: Bool = false, maxTokens: Int? = nil) throws -> PPLXEmbedV2Result {
        try embed(documents: texts.map { [$0] }, task: task, dimensions: dimensions, normalize: normalize, maxTokens: maxTokens)
    }

    public func embed(documents: [[String]], task: PPLXEmbedV2Task = .document, dimensions: Int? = nil,
                      normalize: Bool = false, maxTokens: Int? = nil) throws -> PPLXEmbedV2Result {
        guard !documents.isEmpty, documents.allSatisfy({ !$0.isEmpty }),
              config.isContextual || documents.allSatisfy({ $0.count == 1 }) else {
            throw PPLXEmbedV2Error.invalidInput("Provide nonempty document rows; late models accept one text per row.")
        }
        let size = try outputDimensions(dimensions)
        let limit = try tokenLimit(task: task, maxTokens: maxTokens)
        // Prepare all rows before executing so an oversized later document
        // cannot produce a partial result or silently truncate its chunks.
        let sequences = try documents.map { chunks in
            try config.isContextual ? tokenizer.contextual(chunks, task: task, config: config, limit: limit)
                : tokenizer.late(chunks[0], task: task, limit: limit)
        }
        let rows = sequences.enumerated().map { index, sequence in
            let hidden = encoder(inputIDs: MLXArray(sequence.ids.map(Int32.init), [1, sequence.ids.count]))
            let vectors = config.isContextual
                ? PPLXEmbedV2Encoder.contextualVectors(hidden: hidden, spans: sequence.spans, projection: projection,
                                                       dimensions: size, normalize: normalize, query: task == .query)
                : lateVectors(hidden: hidden, ids: sequence.ids, task: task)
            return PPLXEmbedV2Result.Row(index: index, embeddings: Self.readVectors(vectors), tokenCount: sequence.ids.count)
        }
        return result(rows: rows, dimensions: size, normalize: normalize)
    }

    func outputDimensions(_ dimensions: Int?) throws -> Int {
        let size = dimensions ?? config.embeddingDim
        guard config.isContextual ? [1_024, 2_048].contains(size) : size == 128 else {
            throw PPLXEmbedV2Error.invalidInput("PPLX late dimensions are 128; contextual dimensions are 1024 or 2048.")
        }
        return size
    }

    func tokenLimit(task: PPLXEmbedV2Task, maxTokens: Int?) throws -> Int {
        let modelLimit = task == .query ? config.queryLength : config.documentLength
        guard maxTokens == nil || maxTokens! > 0 else { throw PPLXEmbedV2Error.invalidInput("maxTokens must be positive.") }
        return min(modelLimit, maxTokens ?? modelLimit)
    }

    func lateVectors(hidden: MLXArray, ids: [Int], task: PPLXEmbedV2Task) -> MLXArray {
        let keep = ids.indices.filter { task == .query || !tokenizer.skipIDs.contains(ids[$0]) }
        return PPLXEmbedV2Encoder.lateVectors(hidden: hidden, projection: projection, retainedIndices: keep)
    }

    static func readVectors(_ vectors: MLXArray) -> [[Float]] {
        MLX.eval(vectors)
        let width = vectors.dim(1), flat = vectors.asArray(Float.self)
        return (0..<vectors.dim(0)).map { Array(flat[($0 * width)..<(($0 + 1) * width)]) }
    }

    func result(rows: [PPLXEmbedV2Result.Row], dimensions: Int, normalize: Bool) -> PPLXEmbedV2Result {
        return .init(model: modelID, representation: config.isContextual ? "contextual-int8" : "late-interaction",
                     dimensions: dimensions, normalized: !config.isContextual || normalize, data: rows)
    }
}
