import Foundation
import MLX
import MLXNN
@preconcurrency import Hub
@preconcurrency import Tokenizers

/// Text and ordered media embedding orchestration. Media towers are not instantiated for text-only inference.
public final class EmbeddingGemma2Model {
    public let config: EmbeddingGemma2Config
    let tokenizer: any Tokenizer
    let encoder: EmbeddingGemma2TextModel
    let resources: EmbeddingGemma2Resources
    let dtype: DType
    var vision: EmbeddingGemma2VisionModel?
    var audio: EmbeddingGemma2AudioModel?

    public init(resources: EmbeddingGemma2Resources, dtype: DType = .bfloat16) async throws {
        guard dtype == .bfloat16 || dtype == .float32 else {
            throw EmbeddingGemma2Error.invalidConfiguration("EmbeddingGemma 2 requires bfloat16 or float32 precision.")
        }
        self.resources = resources
        self.dtype = dtype
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw EmbeddingGemma2Error.invalidConfiguration("Missing EmbeddingGemma 2 resources:\n" + missing.map(\.path).joined(separator: "\n"))
        }
        config = try JSONDecoder().decode(EmbeddingGemma2Config.self, from: Data(contentsOf: resources.configURL))
        try config.validate()
        guard config.textConfig.embeddingDim == 768 else {
            throw EmbeddingGemma2Error.invalidConfiguration("EmbeddingGemma 2 requires a 768-dimensional projection.")
        }
        tokenizer = try await AutoTokenizer.from(modelFolder: resources.rootURL)
        encoder = try EmbeddingGemma2TextModel(config: config)
        try Self.loadWeights(into: encoder, resources: resources, dtype: dtype)
    }

    static func loadWeights(into encoder: EmbeddingGemma2TextModel, resources: EmbeddingGemma2Resources, dtype: DType) throws {
        let requiredKeys = Set(encoder.parameters().flattened().map(\.0))
        var loadedKeys: Set<String> = []
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.indexURL, singleURL: resources.weightsURL,
            to: encoder, dtype: dtype, verify: [.noUnusedKeys, .shapeMismatch],
            mapper: { key, value in
                let mapped = Self.mapWeight(key, value)
                loadedKeys.formUnion(mapped.map(\.0))
                return mapped
            }
        )
        let missing = requiredKeys.subtracting(loadedKeys).sorted()
        guard missing.isEmpty else {
            throw EmbeddingGemma2Error.invalidConfiguration("Missing EmbeddingGemma 2 text tensors: " + missing.joined(separator: ", "))
        }
    }

    static func mapWeight(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        guard key.hasPrefix("language_model.") else { return [] }
        return [(String(key.dropFirst("language_model.".count)), value)]
    }

    public func embed(
        texts: [String], task: EmbeddingGemma2Task = .raw, title: String? = nil,
        dimensions: Int = 768, maxTokens: Int? = nil, maxPaddedTokensPerBatch: Int = 8_192
    ) throws -> (embeddings: [[Float]], tokenCounts: [Int]) {
        guard !texts.isEmpty else { throw EmbeddingGemma2Error.invalidInput("At least one input text is required.") }
        guard EmbeddingGemma2Config.outputDimensions.contains(dimensions) else {
            throw EmbeddingGemma2Error.invalidInput("EmbeddingGemma 2 dimensions must be 128, 256, 512, or 768.")
        }
        guard title == nil || task == .document else {
            throw EmbeddingGemma2Error.invalidInput("A title requires the document task.")
        }
        let limit = min(maxTokens ?? EmbeddingGemma2Config.contextLength,
                        EmbeddingGemma2Config.contextLength, config.textConfig.maxPositionEmbeddings)
        guard limit >= 2, maxPaddedTokensPerBatch > 0 else {
            throw EmbeddingGemma2Error.invalidInput("maxTokens must be at least 2 and the padded-token budget must be positive.")
        }
        let inputs = texts.enumerated().map { index, text in
            let body = tokenizer.encode(text: task.format(text, title: title), addSpecialTokens: false)
            let ids = Self.tokenIDs(body: body, config: config.textConfig, limit: limit)
            return Qwen3EmbeddingTokenizedInput(originalIndex: index, tokenIDs: ids)
        }
        let embeddings = Qwen3EmbeddingBatcher.mapInInputOrder(inputs: inputs, maxPaddedTokens: maxPaddedTokensPerBatch) { batch in
            let length = batch.map(\.tokenIDs.count).max()!
            var ids: [Int32] = [], mask: [Int32] = []
            for row in batch {
                ids += row.tokenIDs + Array(repeating: Int32(config.textConfig.padTokenID), count: length - row.tokenIDs.count)
                mask += Array(repeating: 1, count: row.tokenIDs.count) + Array(repeating: 0, count: length - row.tokenIDs.count)
            }
            let attentionMask = MLXArray(mask, [batch.count, length])
            let tokenEmbeddings = encoder(inputIDs: MLXArray(ids, [batch.count, length]), attentionMask: attentionMask)
            let normalized = EmbeddingGemma2TextModel.pool(tokenEmbeddings: tokenEmbeddings, attentionMask: attentionMask, dimensions: dimensions)
            MLX.eval(normalized)
            let flat = normalized.asArray(Float.self)
            return (0..<batch.count).map { Array(flat[($0 * dimensions)..<(($0 + 1) * dimensions)]) }
        }
        return (embeddings, inputs.map(\.tokenIDs.count))
    }

    static func tokenIDs(body: [Int], config: EmbeddingGemma2Config.TextConfig, limit: Int) -> [Int32] {
        [Int32(config.bosTokenID)] + body.prefix(limit - 2).map(Int32.init) + [Int32(config.eosTokenID)]
    }
}
