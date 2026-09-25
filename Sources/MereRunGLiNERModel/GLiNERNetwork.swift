import Darwin
import MLX
import MLXFast
import MLXNN

private struct Weights {
    let arrays: [String: MLXArray]

    func tensor(_ name: String, _ shape: [Int]) throws -> MLXArray {
        guard let value = arrays[name], value.shape == shape,
              [.float16, .bfloat16, .float32].contains(value.dtype) else {
            throw GLiNERModelError.invalidWeights("\(name) must have shape \(shape).")
        }
        return value.asType(.float32)
    }

    func linear(_ prefix: String, _ input: Int, _ output: Int) throws -> Linear {
        try Linear(weight: tensor("\(prefix).weight", [output, input]),
                   bias: tensor("\(prefix).bias", [output]))
    }

    func norm(_ prefix: String, _ size: Int, epsilon: Float) throws -> Norm {
        try Norm(weight: tensor("\(prefix).weight", [size]),
                 bias: tensor("\(prefix).bias", [size]), epsilon: epsilon)
    }
}

private struct Linear {
    let weight: MLXArray
    let bias: MLXArray

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        matmul(input, weight.T) + bias
    }
}

private struct Norm {
    let weight: MLXArray
    let bias: MLXArray
    let epsilon: Float

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.layerNorm(input, weight: weight, bias: bias, eps: epsilon)
    }
}

private struct EncoderLayer {
    let query: Linear
    let key: Linear
    let value: Linear
    let attentionOutput: Linear
    let attentionNorm: Norm
    let intermediate: Linear
    let output: Linear
    let outputNorm: Norm

    init(index: Int, weights: Weights, epsilon: Float) throws {
        let base = "encoder.encoder.layer.\(index)"
        query = try weights.linear("\(base).attention.self.query_proj", 1024, 1024)
        key = try weights.linear("\(base).attention.self.key_proj", 1024, 1024)
        value = try weights.linear("\(base).attention.self.value_proj", 1024, 1024)
        attentionOutput = try weights.linear("\(base).attention.output.dense", 1024, 1024)
        attentionNorm = try weights.norm("\(base).attention.output.LayerNorm", 1024, epsilon: epsilon)
        intermediate = try weights.linear("\(base).intermediate.dense", 1024, 4096)
        output = try weights.linear("\(base).output.dense", 4096, 1024)
        outputNorm = try weights.norm("\(base).output.LayerNorm", 1024, epsilon: epsilon)
    }

    func callAsFunction(_ hidden: MLXArray, relative: MLXArray, positions: MLXArray,
                        attentionMask: MLXArray) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        func heads(_ value: MLXArray) -> MLXArray {
            value.reshaped([batch, length, 16, 64]).transposed(0, 2, 1, 3)
        }
        let queryStates = heads(query(hidden))
        let keyStates = heads(key(hidden))
        let valueStates = heads(value(hidden))
        let relativeKey = headsRelative(key(relative), length: 512)
        let relativeQuery = headsRelative(query(relative), length: 512)
        let scale = Float(192).squareRoot()
        var scores = matmul(queryStates, keyStates.transposed(0, 1, 3, 2)) / scale
        let gatherPositions = broadcast(positions.reshaped([1, 1, length, length]),
                                        to: [batch, 16, length, length])
        let contentPosition = matmul(queryStates, relativeKey.transposed(0, 1, 3, 2))
        scores += takeAlong(contentPosition, gatherPositions, axis: -1) / scale
        let positionContent = matmul(keyStates, relativeQuery.transposed(0, 1, 3, 2))
        let reversePositions = gatherPositions.transposed(0, 1, 3, 2)
        scores += takeAlong(positionContent, reversePositions, axis: -1)
            .transposed(0, 1, 3, 2) / scale
        let masked = MLX.where(attentionMask.reshaped([batch, 1, 1, length]), scores, Float(-1e9))
        let attended = matmul(softmax(masked, axis: -1), valueStates)
            .transposed(0, 2, 1, 3).reshaped([batch, length, 1024])
        let attention = attentionNorm(attentionOutput(attended) + hidden)
        return outputNorm(output(gelu(intermediate(attention))) + attention)
    }

    private func headsRelative(_ value: MLXArray, length: Int) -> MLXArray {
        value.reshaped([1, length, 16, 64]).transposed(0, 2, 1, 3)
    }
}

private struct SpanProjection {
    let first: Linear
    let second: Linear

    init(_ prefix: String, input: Int, output: Int, weights: Weights) throws {
        first = try weights.linear("\(prefix).0", input, output * 4)
        second = try weights.linear("\(prefix).3", output * 4, output)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        second(relu(first(value)))
    }
}

public struct GLiNERSpanScores {
    public let count: Int
    /// Row-major `[count, field, startWord, width]`, with eight width slots.
    public let values: [Float]
    public let fieldCount: Int
    public let wordCount: Int

    public init(count: Int, values: [Float], fieldCount: Int, wordCount: Int) {
        self.count = count
        self.values = values
        self.fieldCount = fieldCount
        self.wordCount = wordCount
    }

    public func probability(instance: Int, field: Int, start: Int, width: Int) -> Float {
        let index = (((instance * fieldCount + field) * wordCount + start) * 8) + width
        return values[index]
    }
}

/// DeBERTa v3 encoder with the GLiNER2 classification, span, and count heads.
public final class GLiNERNetwork {
    private let embeddings: MLXArray
    private let embeddingNorm: Norm
    private let relativeEmbeddings: MLXArray
    private let relativeNorm: Norm
    private let layers: [EncoderLayer]
    private let classifierInput: Linear
    private let classifierOutput: Linear
    private let countInput: Linear
    private let countOutput: Linear
    private let countPositions: MLXArray
    private let gruInputWeight: MLXArray
    private let gruHiddenWeight: MLXArray
    private let gruInputBias: MLXArray
    private let gruHiddenBias: MLXArray
    private let countProjectInput: Linear
    private let countProjectOutput: Linear
    private let spanStart: SpanProjection
    private let spanEnd: SpanProjection
    private let spanOutput: SpanProjection

    public init(configuration: GLiNEREncoderConfiguration, arrays: [String: MLXArray]) throws {
        try configuration.validate()
        let weights = Weights(arrays: arrays)
        embeddings = try weights.tensor("encoder.embeddings.word_embeddings.weight", [128_011, 1024])
        embeddingNorm = try weights.norm("encoder.embeddings.LayerNorm", 1024, epsilon: configuration.layerNormEps)
        relativeEmbeddings = try weights.tensor("encoder.encoder.rel_embeddings.weight", [512, 1024])
        relativeNorm = try weights.norm("encoder.encoder.LayerNorm", 1024, epsilon: configuration.layerNormEps)
        layers = try (0..<24).map { try EncoderLayer(index: $0, weights: weights, epsilon: configuration.layerNormEps) }
        classifierInput = try weights.linear("classifier.0", 1024, 2048)
        classifierOutput = try weights.linear("classifier.2", 2048, 1)
        countInput = try weights.linear("count_pred.0", 1024, 2048)
        countOutput = try weights.linear("count_pred.2", 2048, 20)
        countPositions = try weights.tensor("count_embed.pos_embedding.weight", [20, 1024])
        gruInputWeight = try weights.tensor("count_embed.gru.weight_ih_l0", [3072, 1024])
        gruHiddenWeight = try weights.tensor("count_embed.gru.weight_hh_l0", [3072, 1024])
        gruInputBias = try weights.tensor("count_embed.gru.bias_ih_l0", [3072])
        gruHiddenBias = try weights.tensor("count_embed.gru.bias_hh_l0", [3072])
        countProjectInput = try weights.linear("count_embed.projector.0", 2048, 4096)
        countProjectOutput = try weights.linear("count_embed.projector.2", 4096, 1024)
        let spanBase = "span_rep.span_rep_layer"
        spanStart = try SpanProjection("\(spanBase).project_start", input: 1024, output: 1024, weights: weights)
        spanEnd = try SpanProjection("\(spanBase).project_end", input: 1024, output: 1024, weights: weights)
        spanOutput = try SpanProjection("\(spanBase).out_project", input: 2048, output: 1024, weights: weights)
    }

    /// `markers` are the tokenizer positions of each classification `[L]` marker.
    public func callAsFunction(inputIDs: MLXArray, markers: MLXArray) -> MLXArray {
        classify(hidden: encode(inputIDs: inputIDs), markers: markers)
    }

    public func encode(inputIDs: MLXArray) -> MLXArray {
        let length = inputIDs.dim(1)
        let positions = Self.relativePositions(length: length)
        let relative = relativeNorm(relativeEmbeddings)
        let mask = inputIDs .!= 0
        var hidden = embeddingNorm(embeddings[inputIDs]) * mask.expandedDimensions(axis: -1)
        for layer in layers {
            hidden = layer(hidden, relative: relative, positions: positions, attentionMask: mask)
            eval(hidden)
        }
        return hidden
    }

    public func classify(hidden: MLXArray, markers: MLXArray) -> MLXArray {
        let markerIndices = broadcast(markers.expandedDimensions(axis: -1),
                                      to: [hidden.dim(0), markers.dim(1), 1024])
        let markerStates = takeAlong(hidden, markerIndices, axis: 1)
        return classifierOutput(relu(classifierInput(markerStates))).squeezed(axis: -1)
    }

    /// Score word spans for one extractive schema. `queryPositions` starts with
    /// the schema's `[P]` marker, followed by `[E]`, `[R]`, or `[C]` fields.
    public func scoreSpans(hidden: MLXArray, wordPositions: [Int], queryPositions: [Int]) -> GLiNERSpanScores {
        let states = hidden[0]
        let queries = take(states, MLXArray(queryPositions), axis: 0)
        let fields = queries[1..., 0...]
        let countLogits = countOutput(relu(countInput(queries[0])))
        let count = Int(argMax(countLogits).item(Int32.self))
        guard count > 0, !wordPositions.isEmpty else {
            return GLiNERSpanScores(count: count, values: [], fieldCount: queryPositions.count - 1,
                                    wordCount: wordPositions.count)
        }

        let words = take(states, MLXArray(wordPositions), axis: 0)
        let starts = spanStart(words)
        let ends = spanEnd(words)
        let wordCount = wordPositions.count
        let startIndices = (0..<wordCount).flatMap { start in Array(repeating: start, count: 8) }
        let endIndices = (0..<wordCount).flatMap { start in (0..<8).map { min(start + $0, wordCount - 1) } }
        let startStates = take(starts, MLXArray(startIndices), axis: 0)
        let endStates = take(ends, MLXArray(endIndices), axis: 0)
        let spanStates = spanOutput(relu(concatenated([startStates, endStates], axis: -1)))

        let projections = countProjections(fields: fields, count: count)
        let scores = matmul(projections.reshaped([count * fields.dim(0), 1024]), spanStates.T)
            .reshaped([count, fields.dim(0), wordCount, 8])
        let probabilities = 1 / (1 + exp(-scores))
        eval(probabilities)
        return GLiNERSpanScores(count: count, values: probabilities.asArray(Float.self),
                                fieldCount: fields.dim(0), wordCount: wordCount)
    }

    private func countProjections(fields: MLXArray, count: Int) -> MLXArray {
        var state = fields
        var projected: [MLXArray] = []
        for index in 0..<count {
            let position = countPositions[index]
            let input = position + zeros(like: fields)
            let inputGates = matmul(input, gruInputWeight.T) + gruInputBias
            let hiddenGates = matmul(state, gruHiddenWeight.T) + gruHiddenBias
            let reset = 1 / (1 + exp(-(inputGates[0..., 0..<1024] + hiddenGates[0..., 0..<1024])))
            let update = 1 / (1 + exp(-(inputGates[0..., 1024..<2048] + hiddenGates[0..., 1024..<2048])))
            let candidate = tanh(inputGates[0..., 2048..<3072] + reset * hiddenGates[0..., 2048..<3072])
            state = (1 - update) * candidate + update * state
            let combined = concatenated([state, fields], axis: -1)
            projected.append(countProjectOutput(relu(countProjectInput(combined))))
        }
        return stacked(projected, axis: 0)
    }

    private static func relativePositions(length: Int) -> MLXArray {
        let middle = 128.0
        let denominator = Darwin.log(511.0 / middle)
        let values = (0..<length).flatMap { query in
            (0..<length).map { key -> Int in
                let distance = query - key
                let magnitude = abs(distance)
                let bucket: Int
                if magnitude <= 128 {
                    bucket = distance
                } else {
                    let logarithmic = Darwin.ceil(Darwin.log(Double(magnitude) / middle) / denominator * 127.0) + middle
                    bucket = Int(logarithmic) * (distance < 0 ? -1 : 1)
                }
                return min(511, max(0, bucket + 256))
            }
        }
        return MLXArray(values).reshaped([length, length])
    }
}
