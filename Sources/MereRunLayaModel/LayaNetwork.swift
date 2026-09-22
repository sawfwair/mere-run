import MLX
import MLXNN

/// Native bidirectional ModernBERT and Laya option/action heads.
public final class LayaNetwork {
    public let configuration: LayaEncoderConfiguration
    private let embeddings: MLXArray
    private let embeddingNorm: LayaNorm
    private let encoder: [LayaEncoderLayer]
    private let finalNorm: LayaNorm
    private let typeEmbedding: MLXArray
    private let head: [LayaDecisionLayer]
    private let scorerNorm: LayaNorm
    private let scorerInput: LayaLinear
    private let scorerOutput: LayaLinear
    private let actionInput: LayaLinear
    private let actionOutput: LayaLinear

    public init(configuration: LayaEncoderConfiguration, agent: LayaAgentConfiguration, arrays: [String: MLXArray]) throws {
        try agent.validate(encoder: configuration)
        self.configuration = configuration
        let size = configuration.hiddenSize
        var weights = LayaWeights(arrays: arrays)
        embeddings = try weights.take("encoder.embeddings.tok_embeddings.weight", [configuration.vocabSize, size])
        embeddingNorm = try weights.norm("encoder.embeddings.norm", size, bias: configuration.normBias, epsilon: configuration.normEps)
        encoder = try (0..<configuration.numHiddenLayers).map {
            try LayaEncoderLayer(configuration: configuration, index: $0, weights: &weights)
        }
        finalNorm = try weights.norm("encoder.final_norm", size, bias: configuration.normBias, epsilon: configuration.normEps)
        typeEmbedding = try weights.take("type_emb.weight", [3, size])
        head = try (0..<agent.headLayers).map { try LayaDecisionLayer(size: size, index: $0, weights: &weights) }
        scorerNorm = try weights.norm("scorer.0", size)
        scorerInput = try weights.linear("scorer.1", size, size)
        scorerOutput = try weights.linear("scorer.3", size, 1)
        actionInput = try weights.linear("act_head.0", size + 4, 256)
        actionOutput = try weights.linear("act_head.2", 256, agent.actCosts.count + 1)
        _ = try weights.take("temperature", [3])
        guard weights.arrays.isEmpty else {
            throw LayaModelError.invalidWeights("Unexpected Laya tensors: \(weights.arrays.keys.sorted().joined(separator: ", ")).")
        }
    }

    /// Inputs have shapes [batch, tokens], [batch, options], and [batch].
    /// The caller validates token, marker, and question-type indices before tensor creation.
    public func callAsFunction(
        inputIDs: MLXArray, attentionMask: MLXArray, markerPositions: MLXArray,
        markerMask: MLXArray, questionTypes: MLXArray
    ) -> (logits: MLXArray, actionLogits: MLXArray) {
        let length = inputIDs.dim(1)
        let globalMask = MLX.where(attentionMask.expandedDimensions(axes: [1, 2]), Float(0), -Float.infinity)
        let positions = MLXArray(0..<length)
        let window = abs(positions.expandedDimensions(axis: 0) - positions.expandedDimensions(axis: 1))
            .<= configuration.localAttention / 2
        let localMask = MLX.where(window, globalMask, -Float.infinity)
        var hidden = embeddingNorm(embeddings[inputIDs])
        for layer in encoder {
            hidden = layer(hidden, globalMask: globalMask, localMask: localMask)
            eval(hidden)
        }
        hidden = finalNorm(hidden) + typeEmbedding[questionTypes].expandedDimensions(axis: 1)
        for layer in head {
            hidden = layer(hidden, mask: globalMask)
            eval(hidden)
        }
        let indices = broadcast(markerPositions.expandedDimensions(axis: -1), to: [inputIDs.dim(0), markerPositions.dim(1), hidden.dim(-1)])
        let markers = takeAlong(hidden, indices, axis: 1)
        let scores = scorerOutput(gelu(scorerInput(scorerNorm(markers)))).squeezed(axis: -1)
        let logits = MLX.where(markerMask, scores, Float(-1e4))
        let probabilities = softmax(logits, axis: -1)
        let counts = maximum(sum(markerMask.asType(.float32), axis: -1), Float(2))
        let entropy = -sum(probabilities * log(maximum(probabilities, Float(1e-9))), axis: -1) / log(counts)
        let ordered = sorted(probabilities, axis: -1)
        let top1 = ordered[0..., -1]
        let top2 = markerPositions.dim(1) > 1 ? ordered[0..., -2] : zeros(like: top1)
        let features = stacked([top1, top1 - top2, entropy, counts / 255], axis: -1)
        let pooled = concatenated([hidden[0..., 0, 0...], features], axis: -1)
        return (logits, actionOutput(gelu(actionInput(pooled))))
    }
}
