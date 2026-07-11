import Foundation
import MLX

struct Qwen3EmbeddingTokenizedInput: Equatable {
    let originalIndex: Int
    let tokenIDs: [Int32]
}

enum Qwen3EmbeddingBatcher {
    /// Packs similarly sized rows together so padding never exceeds the budget.
    /// A single row longer than the budget remains intact and runs by itself.
    static func subbatches(
        inputs: [Qwen3EmbeddingTokenizedInput],
        maxPaddedTokens: Int
    ) -> [[Qwen3EmbeddingTokenizedInput]] {
        precondition(maxPaddedTokens > 0, "maxPaddedTokens must be positive")

        let lengthSorted = inputs.sorted { lhs, rhs in
            if lhs.tokenIDs.count == rhs.tokenIDs.count {
                return lhs.originalIndex < rhs.originalIndex
            }
            return lhs.tokenIDs.count > rhs.tokenIDs.count
        }
        var batches: [[Qwen3EmbeddingTokenizedInput]] = []
        var start = 0
        while start < lengthSorted.count {
            let longestCount = max(1, lengthSorted[start].tokenIDs.count)
            let batchCapacity = max(1, maxPaddedTokens / longestCount)
            let batchCount = min(lengthSorted.count - start, batchCapacity)
            let end = start + batchCount
            batches.append(Array(lengthSorted[start..<end]))
            start = end
        }
        return batches
    }

    static func mapInInputOrder<Output>(
        inputs: [Qwen3EmbeddingTokenizedInput],
        maxPaddedTokens: Int,
        operation: ([Qwen3EmbeddingTokenizedInput]) throws -> [Output]
    ) rethrows -> [Output] {
        var indexedOutputs: [(index: Int, output: Output)] = []
        indexedOutputs.reserveCapacity(inputs.count)
        for batch in subbatches(inputs: inputs, maxPaddedTokens: maxPaddedTokens) {
            let outputs = try operation(batch)
            precondition(outputs.count == batch.count, "Embedding batch output count must match input count")
            indexedOutputs.append(contentsOf: zip(batch, outputs).map { input, output in
                (index: input.originalIndex, output: output)
            })
        }
        return indexedOutputs.sorted { $0.index < $1.index }.map(\.output)
    }
}

public final class Qwen3EmbeddingModel {
    public static let defaultMaxPaddedTokensPerBatch = 8_192

    public enum EmbeddingError: LocalizedError {
        case missingFiles([URL])
        case noInputTexts
        case invalidMaxTokens(Int)
        case invalidMaxPaddedTokensPerBatch(Int)

        public var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                let list = urls.map(\.path).joined(separator: "\n")
                return "Missing Qwen3 embedding resources:\n\(list)"
            case .noInputTexts:
                return "At least one input text is required."
            case .invalidMaxTokens(let value):
                return "maxTokens must be positive (received \(value))."
            case .invalidMaxPaddedTokensPerBatch(let value):
                return "maxPaddedTokensPerBatch must be positive (received \(value))."
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
    ///
    /// Pooling strategy matches Qwen3 embedding reference behavior: take the
    /// final valid token. Inputs are grouped by length and evaluated in bounded
    /// padded-token sub-batches, then restored to caller order. A single input
    /// longer than the padded-token budget is evaluated alone without changing
    /// its existing `maxTokens` truncation semantics.
    public func embed(
        texts: [String],
        maxTokens: Int? = nil,
        maxPaddedTokensPerBatch: Int = Qwen3EmbeddingModel.defaultMaxPaddedTokensPerBatch
    ) throws -> (embeddings: [[Float]], tokenCounts: [Int]) {
        guard !texts.isEmpty else {
            throw EmbeddingError.noInputTexts
        }

        let modelMax = config.maxPositionEmbeddings
        let effectiveMaxTokens = min(maxTokens ?? modelMax, modelMax)
        guard effectiveMaxTokens > 0 else {
            throw EmbeddingError.invalidMaxTokens(effectiveMaxTokens)
        }
        guard maxPaddedTokensPerBatch > 0 else {
            throw EmbeddingError.invalidMaxPaddedTokensPerBatch(maxPaddedTokensPerBatch)
        }

        var inputs: [Qwen3EmbeddingTokenizedInput] = []
        var tokenCounts: [Int] = []
        inputs.reserveCapacity(texts.count)
        tokenCounts.reserveCapacity(texts.count)

        for (index, text) in texts.enumerated() {
            var ids = tokenizer.encode(text, addSpecialTokens: true)
            if ids.count > effectiveMaxTokens {
                ids = Array(ids.prefix(effectiveMaxTokens))
            }
            if ids.isEmpty {
                ids = [tokenizer.eosTokenId ?? tokenizer.padTokenId]
            }
            inputs.append(
                Qwen3EmbeddingTokenizedInput(
                    originalIndex: index,
                    tokenIDs: ids.map(Int32.init)
                )
            )
            tokenCounts.append(ids.count)
        }

        let embeddings = Qwen3EmbeddingBatcher.mapInInputOrder(
            inputs: inputs,
            maxPaddedTokens: maxPaddedTokensPerBatch,
            operation: embedSubbatch
        )
        return (embeddings, tokenCounts)
    }

    private func embedSubbatch(
        _ inputs: [Qwen3EmbeddingTokenizedInput]
    ) -> [[Float]] {
        let batchSize = inputs.count
        let sequenceLength = max(1, inputs.map(\.tokenIDs.count).max() ?? 1)

        var flatIds: [Int32] = []
        var flatMask: [Int32] = []
        flatIds.reserveCapacity(batchSize * sequenceLength)
        flatMask.reserveCapacity(batchSize * sequenceLength)

        for input in inputs {
            let ids = input.tokenIDs
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

        // Pool every row's final valid token in one gather and normalize the
        // whole batch before a single readback. The previous loop evaluated
        // and read back each embedding separately — N GPU→CPU syncs per
        // batch.
        let hiddenSize = hiddenStates.dim(2)
        let lastIndices = MLXArray(inputs.map { Int32(max(0, $0.tokenIDs.count - 1)) })
        let flatHidden = hiddenStates.reshaped([batchSize * sequenceLength, hiddenSize])
        let rowBases = MLXArray((0..<batchSize).map { Int32($0 * sequenceLength) })
        let pooled = MLX.take(flatHidden, rowBases + lastIndices, axis: 0).asType(.float32)
        let eps = MLXArray(Float32(1e-12))
        let norms = MLX.sqrt(MLX.sum(pooled * pooled, axis: -1, keepDims: true) + eps)
        let normalized = pooled / norms
        MLX.eval(normalized)
        let flat = normalized.asArray(Float.self)

        var output: [[Float]] = []
        output.reserveCapacity(batchSize)
        for row in 0..<batchSize {
            output.append(Array(flat[(row * hiddenSize)..<((row + 1) * hiddenSize)]))
        }

        return output
    }

    private func l2Normalize(_ vector: MLXArray) -> MLXArray {
        let eps = MLXArray(Float32(1e-12))
        let norm = MLX.sqrt(MLX.sum(vector * vector) + eps)
        return vector / norm
    }
}
