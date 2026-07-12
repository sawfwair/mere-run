import Foundation
@preconcurrency import Hub
import MLX
import MLXFast
import MLXNN
@preconcurrency import Tokenizers

public struct Wan2TextEncoderConfiguration: Hashable, Sendable {
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let attentionSize: Int
    public let feedForwardSize: Int
    public let headCount: Int
    public let layerCount: Int
    public let relativePositionBuckets: Int

    public init(
        vocabularySize: Int = 256_384,
        hiddenSize: Int = 4_096,
        attentionSize: Int = 4_096,
        feedForwardSize: Int = 10_240,
        headCount: Int = 64,
        layerCount: Int = 24,
        relativePositionBuckets: Int = 32
    ) {
        precondition(attentionSize % headCount == 0)
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.attentionSize = attentionSize
        self.feedForwardSize = feedForwardSize
        self.headCount = headCount
        self.layerCount = layerCount
        self.relativePositionBuckets = relativePositionBuckets
    }
}

final class Wan2T5LayerNorm: Module {
    let epsilon: Float
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, epsilon: Float = 1e-6) {
        self.epsilon = epsilon
        self._weight.wrappedValue = MLX.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: epsilon)
    }
}

final class Wan2T5RelativeEmbedding: Module {
    let bucketCount: Int
    let headCount: Int
    let maxDistance: Int
    @ModuleInfo(key: "embedding") var embedding: Embedding

    init(bucketCount: Int, headCount: Int, maxDistance: Int = 128) {
        self.bucketCount = bucketCount
        self.headCount = headCount
        self.maxDistance = maxDistance
        self._embedding.wrappedValue = Embedding(embeddingCount: bucketCount, dimensions: headCount)
    }

    func callAsFunction(queryLength: Int, keyLength: Int) -> MLXArray {
        let keyPositions = MLX.arange(keyLength).expandedDimensions(axis: 0)
        let queryPositions = MLX.arange(queryLength).expandedDimensions(axis: 1)
        let relative = keyPositions - queryPositions
        let halfBuckets = bucketCount / 2
        var buckets = (relative .> 0).asType(.int32) * halfBuckets
        let distance = MLX.abs(relative)
        let maxExact = halfBuckets / 2
        let isSmall = distance .< maxExact
        let distanceFloat = distance.asType(.float32)
        let large = maxExact + (
            MLX.log(distanceFloat / Float(maxExact))
                / log(Float(maxDistance) / Float(maxExact))
                * Float(halfBuckets - maxExact)
        ).asType(.int32)
        let capped = MLX.minimum(large, MLX.ones(large.shape, dtype: .int32) * (halfBuckets - 1))
        buckets = buckets + MLX.where(isSmall, distance.asType(.int32), capped)
        return embedding(buckets).transposed(2, 0, 1).expandedDimensions(axis: 0)
    }
}

final class Wan2T5Attention: Module {
    let heads: Int
    let headDimension: Int
    @ModuleInfo(key: "q") var query: Linear
    @ModuleInfo(key: "k") var key: Linear
    @ModuleInfo(key: "v") var value: Linear
    @ModuleInfo(key: "o") var output: Linear

    init(dimensions: Int, attentionDimensions: Int, heads: Int) {
        self.heads = heads
        self.headDimension = attentionDimensions / heads
        self._query.wrappedValue = Linear(dimensions, attentionDimensions, bias: false)
        self._key.wrappedValue = Linear(dimensions, attentionDimensions, bias: false)
        self._value.wrappedValue = Linear(dimensions, attentionDimensions, bias: false)
        self._output.wrappedValue = Linear(attentionDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray?, positionBias: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let q = query(input).reshaped(batch, -1, heads, headDimension).transposed(0, 2, 1, 3)
        let k = key(input).reshaped(batch, -1, heads, headDimension).transposed(0, 2, 1, 3)
        let v = value(input).reshaped(batch, -1, heads, headDimension).transposed(0, 2, 1, 3)
        var scores = matmul(q.asType(.float32), k.asType(.float32).transposed(0, 1, 3, 2))
        scores = scores + positionBias.asType(.float32)
        if let mask {
            let expanded = mask.ndim == 2
                ? mask.expandedDimensions(axes: [1, 2])
                : mask.expandedDimensions(axis: 1)
            let additive = MLX.where(expanded .== 0, MLXArray(-3.389e38), MLXArray(0)).asType(.float32)
            scores = scores + additive
        }
        let probabilities = MLX.softmax(scores, axis: -1).asType(q.dtype)
        let attended = matmul(probabilities, v)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, -1, heads * headDimension)
        return output(attended)
    }
}

final class Wan2T5FeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._input.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._output.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(input(value) * MLXNN.geluApproximate(gate(value)))
    }
}

final class Wan2T5Block: Module {
    @ModuleInfo(key: "norm1") var attentionNorm: Wan2T5LayerNorm
    @ModuleInfo(key: "attn") var attention: Wan2T5Attention
    @ModuleInfo(key: "norm2") var feedForwardNorm: Wan2T5LayerNorm
    @ModuleInfo(key: "ffn") var feedForward: Wan2T5FeedForward
    @ModuleInfo(key: "pos_embedding") var positionEmbedding: Wan2T5RelativeEmbedding

    init(configuration: Wan2TextEncoderConfiguration) {
        self._attentionNorm.wrappedValue = Wan2T5LayerNorm(dimensions: configuration.hiddenSize)
        self._attention.wrappedValue = Wan2T5Attention(
            dimensions: configuration.hiddenSize,
            attentionDimensions: configuration.attentionSize,
            heads: configuration.headCount
        )
        self._feedForwardNorm.wrappedValue = Wan2T5LayerNorm(dimensions: configuration.hiddenSize)
        self._feedForward.wrappedValue = Wan2T5FeedForward(
            dimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.feedForwardSize
        )
        self._positionEmbedding.wrappedValue = Wan2T5RelativeEmbedding(
            bucketCount: configuration.relativePositionBuckets,
            headCount: configuration.headCount
        )
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray?) -> MLXArray {
        let bias = positionEmbedding(queryLength: input.dim(1), keyLength: input.dim(1))
        let attended = input + attention(attentionNorm(input), mask: mask, positionBias: bias)
        return attended + feedForward(feedForwardNorm(attended))
    }
}

public final class Wan2TextEncoderModel: Module {
    public let configuration: Wan2TextEncoderConfiguration
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "blocks") var blocks: [Wan2T5Block]
    @ModuleInfo(key: "norm") var norm: Wan2T5LayerNorm

    public init(configuration: Wan2TextEncoderConfiguration = Wan2TextEncoderConfiguration()) {
        self.configuration = configuration
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            Wan2T5Block(configuration: configuration)
        }
        self._norm.wrappedValue = Wan2T5LayerNorm(dimensions: configuration.hiddenSize)
    }

    public func callAsFunction(tokenIDs: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = tokenEmbedding(tokenIDs)
        for block in blocks {
            hidden = block(hidden, mask: mask)
        }
        return norm(hidden)
    }
}

public final class Wan2Tokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer
    public let maxLength: Int

    init(tokenizer: any Tokenizer, maxLength: Int = 512) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
    }

    public static func load(from tokenizerURL: URL, hubApi: HubApi = .shared) throws -> Wan2Tokenizer {
        let tokenizerData = try hubApi.configuration(fileURL: tokenizerURL)
        let tokenizerConfig: Config = [
            "tokenizer_class": "T5Tokenizer",
            "model_max_length": 512,
            "eos_token": "</s>",
            "pad_token": "<pad>",
        ]
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            strict: false
        )
        return Wan2Tokenizer(tokenizer: tokenizer)
    }

    public func encode(_ text: String) -> (tokenIDs: [Int], mask: [Int]) {
        var ids = tokenizer.encode(text: text, addSpecialTokens: true)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        let count = ids.count
        ids.append(contentsOf: repeatElement(0, count: maxLength - ids.count))
        return (ids, Array(repeating: 1, count: count) + Array(repeating: 0, count: maxLength - count))
    }
}
