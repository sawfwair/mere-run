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

/// DeBERTa v3 encoder with the GLiNER2 classification head. Span extraction
/// tensors are intentionally unused for classification-only inference.
public final class GLiNERNetwork {
    private let embeddings: MLXArray
    private let embeddingNorm: Norm
    private let relativeEmbeddings: MLXArray
    private let relativeNorm: Norm
    private let layers: [EncoderLayer]
    private let classifierInput: Linear
    private let classifierOutput: Linear

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
    }

    /// `markers` are the tokenizer positions of each classification `[L]` marker.
    public func callAsFunction(inputIDs: MLXArray, markers: MLXArray) -> MLXArray {
        let length = inputIDs.dim(1)
        let positions = Self.relativePositions(length: length)
        let relative = relativeNorm(relativeEmbeddings)
        let mask = inputIDs .!= 0
        var hidden = embeddingNorm(embeddings[inputIDs]) * mask.expandedDimensions(axis: -1)
        for layer in layers {
            hidden = layer(hidden, relative: relative, positions: positions, attentionMask: mask)
            eval(hidden)
        }
        let markerIndices = broadcast(markers.expandedDimensions(axis: -1),
                                      to: [inputIDs.dim(0), markers.dim(1), 1024])
        let markerStates = takeAlong(hidden, markerIndices, axis: 1)
        return classifierOutput(relu(classifierInput(markerStates))).squeezed(axis: -1)
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
