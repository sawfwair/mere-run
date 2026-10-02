import Foundation
import MLX
import MLXNN

public enum ClefError: Error, LocalizedError {
    case invalidInput(String)
    case invalidConfiguration(String)
    case invalidWeights(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message), .invalidConfiguration(let message), .invalidWeights(let message): message
        }
    }
}

public struct ClefHeadConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let width: Int
    public let routingLayers: Int
    public let layers: Int
    public let heads: Int
    public let feedforward: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", width, routingLayers = "routing_layers", layers, heads, feedforward
    }

    public func validate(backboneHiddenSize: Int) throws {
        guard hiddenSize == backboneHiddenSize, hiddenSize > 0, hiddenSize <= 16_384,
              width > 0, width <= 8_192, heads > 0, width.isMultiple(of: heads),
              (0...16).contains(routingLayers), (0...16).contains(layers),
              feedforward > 0, feedforward <= 65_536 else {
            throw ClefError.invalidConfiguration("Invalid Clef joint head geometry or backbone hidden size.")
        }
    }
}

package struct ClefHeadField {
    package let type: Int
    package let questionSpan: Range<Int>
    package let optionSpans: [Range<Int>]

    package init(type: Int, questionSpan: Range<Int>, optionSpans: [Range<Int>]) {
        self.type = type
        self.questionSpan = questionSpan
        self.optionSpans = optionSpans
    }
}

/// Clef's bidirectional evidence router and joint field decoder. Core owns token spans and I/O.
package final class ClefJointHead: Module {
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: LayerNorm
    @ModuleInfo(key: "memory_projection") var memoryProjection: Linear
    @ModuleInfo(key: "question_projection") var questionProjection: Linear
    @ModuleInfo(key: "option_question_projection") var optionQuestionProjection: Linear
    @ModuleInfo(key: "global_projection") var globalProjection: Linear
    @ModuleInfo(key: "option_context_projection") var optionContextProjection: Linear
    @ModuleInfo(key: "option_lexical_projection") var optionLexicalProjection: Linear
    @ModuleInfo(key: "type_embedding") var typeEmbedding: Embedding
    @ModuleInfo(key: "evidence_layers") var evidenceLayers: [ClefEvidenceLayer]
    @ModuleInfo(key: "option_summary_norm") var optionSummaryNorm: LayerNorm
    @ModuleInfo var layers: [ClefDecoderLayer]
    @ModuleInfo(key: "field_norm") var fieldNorm: LayerNorm
    @ModuleInfo(key: "option_norm") var optionNorm: LayerNorm
    @ModuleInfo var scorer1: Linear
    @ModuleInfo var scorer2: Linear
    @ParameterInfo(key: "prior_logit_scale") var priorLogitScale: MLXArray
    @ParameterInfo(key: "joint_logit_scale") var jointLogitScale: MLXArray
    @ParameterInfo(key: "residual_gate") var residualGate: MLXArray
    let configuration: ClefHeadConfiguration

    package init(configuration config: ClefHeadConfiguration) throws {
        try config.validate(backboneHiddenSize: config.hiddenSize)
        configuration = config
        self._hiddenNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._memoryProjection.wrappedValue = Linear(config.hiddenSize, config.width, bias: false)
        self._questionProjection.wrappedValue = Linear(config.hiddenSize, config.width, bias: false)
        self._optionQuestionProjection.wrappedValue = Linear(config.hiddenSize, config.width, bias: false)
        self._globalProjection.wrappedValue = Linear(config.hiddenSize, config.width, bias: false)
        self._optionContextProjection.wrappedValue = Linear(config.hiddenSize, config.width, bias: false)
        self._optionLexicalProjection.wrappedValue = Linear(config.hiddenSize, config.width, bias: false)
        self._typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: config.width)
        self._evidenceLayers.wrappedValue = (0..<config.routingLayers).map { _ in
            ClefEvidenceLayer(width: config.width, heads: config.heads, feedforward: config.feedforward)
        }
        self._optionSummaryNorm.wrappedValue = LayerNorm(dimensions: config.width)
        self._layers.wrappedValue = (0..<config.layers).map { _ in
            ClefDecoderLayer(width: config.width, heads: config.heads, feedforward: config.feedforward)
        }
        self._fieldNorm.wrappedValue = LayerNorm(dimensions: config.width)
        self._optionNorm.wrappedValue = LayerNorm(dimensions: config.width)
        self._scorer1.wrappedValue = Linear(4 * config.width, config.width)
        self._scorer2.wrappedValue = Linear(config.width, 1)
        self._priorLogitScale.wrappedValue = MLXArray(Float(0))
        self._jointLogitScale.wrappedValue = MLXArray(Float(0))
        self._residualGate.wrappedValue = MLXArray(Float(0))
    }

    package func load(_ arrays: [String: MLXArray], dtype: DType = .bfloat16) throws {
        let mapped = arrays.map { key, value in
            (key.replacingOccurrences(of: ".feedforward.0.", with: ".feedforward.fc1.")
                .replacingOccurrences(of: ".feedforward.3.", with: ".feedforward.fc2.")
                .replacingOccurrences(of: "residual_scorer.0.", with: "scorer1.")
                .replacingOccurrences(of: "residual_scorer.3.", with: "scorer2."), value.asType(dtype))
        }
        let expected = Set(parameters().flattened().map(\.0))
        guard Set(mapped.map(\.0)) == expected, mapped.count == expected.count else {
            throw ClefError.invalidWeights("Clef joint head parameter keys do not match its configuration.")
        }
        try update(parameters: ModuleParameters.unflattened(mapped), verify: [.all])
        eval(parameters())
    }

    package func callAsFunction(
        hidden: MLXArray, fields: [ClefHeadField], lexical: [[MLXArray]]
    ) throws -> [MLXArray] {
        guard hidden.ndim == 2, hidden.dim(1) == configuration.hiddenSize,
              !fields.isEmpty, fields.count == lexical.count else {
            throw ClefError.invalidInput("Clef hidden states and schema fields do not match.")
        }
        for (field, embeddings) in zip(fields, lexical) {
            guard (0..<3).contains(field.type), !field.optionSpans.isEmpty,
                  field.optionSpans.count == embeddings.count,
                  ([field.questionSpan] + field.optionSpans).allSatisfy({
                      !$0.isEmpty && $0.lowerBound >= 0 && $0.upperBound <= hidden.dim(0)
                  }), zip(field.optionSpans, embeddings).allSatisfy({ span, embedding in
                      embedding.shape == [span.count, configuration.hiddenSize]
                  }) else {
                throw ClefError.invalidInput("Invalid Clef schema token span or lexical embedding shape.")
            }
        }
        let normalized = hiddenNorm(hidden)
        let memory = memoryProjection(normalized).expandedDimensions(axis: 0)
        let global = normalized[normalized.dim(0) - 1]
        let questions = stacked(fields.map { normalized[$0.questionSpan].mean(axis: 0) })
        let lexicalOptions = lexical.map { stacked($0.map { $0.mean(axis: 0) }) }
        let optionQueries = fields.enumerated().map { index, field in
            optionContextProjection(stacked(field.optionSpans.map { normalized[$0].mean(axis: 0) }))
                + optionLexicalProjection(lexicalOptions[index])
                + optionQuestionProjection(questions[index]).expandedDimensions(axis: 0)
        }
        var routed = concatenated(optionQueries, axis: 0).expandedDimensions(axis: 0)
        for layer in evidenceLayers { routed = layer(routed, memory: memory) }
        var offset = 0
        let options = fields.map { field -> MLXArray in
            defer { offset += field.optionSpans.count }
            return routed[0, offset..<(offset + field.optionSpans.count)]
        }
        let baseFields = questionProjection(questions)
        let summaries = options.enumerated().map { index, option in
            let weights = softmax(matmul(option, baseFields[index]) / sqrt(Float(configuration.width)), axis: 0)
            return (weights.expandedDimensions(axis: 1) * option).sum(axis: 0)
        }
        var joint = (baseFields + optionSummaryNorm(stacked(summaries))
            + globalProjection(global).expandedDimensions(axis: 0)
            + typeEmbedding(MLXArray(fields.map { Int32($0.type) }))).expandedDimensions(axis: 0)
        for layer in layers { joint = layer(joint, memory: memory) }
        let resultFields = fieldNorm(joint[0])
        let priorScale = exp(minimum(priorLogitScale, Float(log(100.0))))
        let jointScale = exp(minimum(jointLogitScale, Float(log(100.0))))
        let gate = sigmoid(residualGate)
        func norm(_ value: MLXArray) -> MLXArray { sqrt((value * value).sum(axis: -1, keepDims: true)) }
        func unit(_ value: MLXArray) -> MLXArray { value / maximum(norm(value), Float(1e-12)) }
        return options.enumerated().map { index, routedOptions in
            let prior = priorScale * matmul(unit(lexicalOptions[index]), unit(questions[index] + global))
            let option = optionNorm(routedOptions)
            let field = broadcast(resultFields[index].expandedDimensions(axis: 0), to: option.shape)
            let cosine = (field * option).sum(axis: -1)
                / maximum((norm(field) * norm(option)).squeezed(axis: -1), Float(1e-8))
            let features = concatenated([field, option, field * option, abs(field - option)], axis: -1)
            let residual = scorer2(gelu(scorer1(features)))[0..., 0]
            return prior + gate * (jointScale * cosine + residual)
        }
    }
}
