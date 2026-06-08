import Foundation
import MLX

public final class Qwen3EmbeddingModel {
    public enum EmbeddingError: LocalizedError {
        case missingFiles([URL])
        case noInputTexts
        case invalidMaxTokens(Int)

        public var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                let list = urls.map(\.path).joined(separator: "\n")
                return "Missing Qwen3 embedding resources:\n\(list)"
            case .noInputTexts:
                return "At least one input text is required."
            case .invalidMaxTokens(let value):
                return "maxTokens must be positive (received \(value))."
            }
        }
    }

    public let resources: Qwen3EmbeddingResources
    public let config: ACEStep5HzLMConfig

    private let tokenizer: ACEStep5HzLMTokenizer
    private let encoder: QwenEncoder

    public init(
        resources: Qwen3EmbeddingResources,
        dtype: DType? = .bfloat16,
        fileManager: FileManager = .default
    ) throws {
        let missing = resources.validate(fileManager: fileManager)
        if !missing.isEmpty {
            throw EmbeddingError.missingFiles(missing)
        }

        self.resources = resources
        self.config = try JSONDecoder().decode(
            ACEStep5HzLMConfig.self,
            from: Data(contentsOf: resources.configURL)
        )

        let qwenConfig = QwenTextEncoderConfiguration(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            numHiddenLayers: config.numHiddenLayers,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            intermediateSize: config.intermediateSize,
            ropeTheta: config.ropeTheta,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            rmsNormEps: config.rmsNormEps,
            promptDropIndex: 0,
            headDim: config.headDim,
            mropeSection: nil,
            mropeInterleaved: false
        )

        let qwen = QwenEncoder(configuration: qwenConfig)
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.weightsURL,
            to: qwen,
            dtype: dtype,
            mapper: QwenEncoder.mapHFSafetensorWeight,
            fileManager: fileManager
        )

        self.tokenizer = try ACEStep5HzLMTokenizer.load(
            from: resources.rootURL,
            requireAudioCodeTokens: false
        )
        self.encoder = qwen
    }

    /// Returns one normalized embedding vector per input text.
    /// Pooling strategy matches Qwen3 embedding reference behavior: take the final valid token.
    public func embed(
        texts: [String],
        maxTokens: Int? = nil
    ) throws -> (embeddings: [[Float]], tokenCounts: [Int]) {
        guard !texts.isEmpty else {
            throw EmbeddingError.noInputTexts
        }

        let modelMax = config.maxPositionEmbeddings
        let effectiveMaxTokens = min(maxTokens ?? modelMax, modelMax)
        guard effectiveMaxTokens > 0 else {
            throw EmbeddingError.invalidMaxTokens(effectiveMaxTokens)
        }

        var tokenLists: [[Int32]] = []
        var tokenCounts: [Int] = []
        tokenLists.reserveCapacity(texts.count)
        tokenCounts.reserveCapacity(texts.count)

        for text in texts {
            var ids = tokenizer.encode(text, addSpecialTokens: true)
            if ids.count > effectiveMaxTokens {
                ids = Array(ids.prefix(effectiveMaxTokens))
            }
            if ids.isEmpty {
                ids = [tokenizer.eosTokenId ?? tokenizer.padTokenId]
            }
            tokenLists.append(ids.map(Int32.init))
            tokenCounts.append(ids.count)
        }

        let batchSize = tokenLists.count
        let sequenceLength = max(1, tokenCounts.max() ?? 1)

        var flatIds: [Int32] = []
        var flatMask: [Int32] = []
        flatIds.reserveCapacity(batchSize * sequenceLength)
        flatMask.reserveCapacity(batchSize * sequenceLength)

        for ids in tokenLists {
            flatIds.append(contentsOf: ids)
            flatMask.append(contentsOf: Array(repeating: 1, count: ids.count))

            let padCount = sequenceLength - ids.count
            if padCount > 0 {
                flatIds.append(contentsOf: Array(repeating: Int32(tokenizer.padTokenId), count: padCount))
                flatMask.append(contentsOf: Array(repeating: 0, count: padCount))
            }
        }

        let shape = [batchSize, sequenceLength]
        let inputIds = MLXArray(flatIds.map(Float32.init), shape).asType(.int32)
        let attentionMask = MLXArray(flatMask.map(Float32.init), shape).asType(.int32)

        let hiddenStates = encoder.forward(
            inputIds: inputIds,
            attentionMask: attentionMask,
            outputHiddenStates: false
        ).lastHiddenState

        var output: [[Float]] = []
        output.reserveCapacity(batchSize)

        for row in 0..<batchSize {
            let tokenCount = tokenCounts[row]
            let lastIndex = max(0, tokenCount - 1)
            let pooled = hiddenStates[row, lastIndex, 0...].asType(.float32)
            let normalized = l2Normalize(pooled)
            MLX.eval(normalized)
            output.append(normalized.asArray(Float.self))
        }

        return (output, tokenCounts)
    }

    private func l2Normalize(_ vector: MLXArray) -> MLXArray {
        let eps = MLXArray(Float32(1e-12))
        let norm = MLX.sqrt(MLX.sum(vector * vector) + eps)
        return vector / norm
    }
}
