import MLX
import MLXNN

/// PPLX keeps Qwen3.5's causal convolution and delta recurrence. Only full
/// attention is bidirectional. Rows run without padding or generation caches.
private final class PPLXEmbedV2Layer: Module {
    @ModuleInfo(key: "linear_attn") var linearAttention: Q35LinearAttention?
    @ModuleInfo(key: "self_attn") var selfAttention: Q35FullAttention?
    @ModuleInfo(key: "mlp") var mlp: Q35FeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: Q35RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: Q35RMSNorm

    init(config: Q35Config, index: Int) {
        let text = config.textConfig
        _linearAttention.wrappedValue = text.layerTypes[index] == "linear_attention" ? Q35LinearAttention(config: config) : nil
        _selfAttention.wrappedValue = text.layerTypes[index] == "full_attention" ? Q35FullAttention(config: config) : nil
        _mlp.wrappedValue = Q35FeedForward(config: config)
        _inputNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _postNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, positionIDs: MLXArray?) -> MLXArray {
        let normalized = inputNorm(x)
        let attended = linearAttention?(normalized, cache: nil, normalizeQKInFloat32: true)
            ?? selfAttention!(normalized, mask: .none, cache: nil, positionIds: positionIDs)
        let residual = x + attended
        return residual + mlp(postNorm(residual))
    }
}

public final class PPLXEmbedV2Encoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    @ModuleInfo(key: "layers") private var layers: [PPLXEmbedV2Layer]
    @ModuleInfo(key: "norm") private var norm: Q35RMSNorm

    public init(config: Q35Config) {
        let text = config.textConfig
        _embedding.wrappedValue = Embedding(embeddingCount: text.vocabSize, dimensions: text.hiddenSize)
        _layers.wrappedValue = (0..<text.numHiddenLayers).map { PPLXEmbedV2Layer(config: config, index: $0) }
        _norm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
    }

    public var checkpointParameterNames: Set<String> {
        // Q35's constant BF16 normalization buffer is not a checkpoint tensor.
        Set(parameters().flattened().map(\.0).filter { !$0.hasSuffix(".qkNormWeightBF16") })
    }

    public func embedTokens(_ inputIDs: MLXArray) -> MLXArray { embedding(inputIDs) }

    public func callAsFunction(inputIDs: MLXArray, embeddings: MLXArray? = nil, positionIDs: MLXArray? = nil) -> MLXArray {
        var hidden = embeddings ?? embedding(inputIDs)
        for layer in layers {
            hidden = layer(hidden, positionIDs: positionIDs)
            // Bound the lazy graph, especially the 9B FP32 checkpoint.
            MLX.eval(hidden)
        }
        return norm(hidden)
    }

    public static func normalize(_ vectors: MLXArray) -> MLXArray {
        let values = vectors.asType(.float32)
        let lengths = MLX.maximum(MLX.sqrt(MLX.sum(values * values, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return values / lengths
    }

    public static func lateVectors(hidden: MLXArray, projection: MLXArray, retainedIndices: [Int]) -> MLXArray {
        let projected = MLX.matmul(hidden[0, 0..., 0...].asType(.float32), projection.transposed())
        let selected = MLX.take(projected, MLXArray(retainedIndices.map(Int32.init)), axis: 0)
        return normalize(selected)
    }

    public static func contextualVectors(hidden: MLXArray, spans: [Range<Int>], projection: MLXArray,
                                         dimensions: Int, normalize: Bool, query: Bool = false) -> MLXArray {
        let values = hidden[0, 0..., 0...].asType(.float32)
        let prefix = MLX.concatenated([MLXArray.zeros([1, hidden.dim(-1)], dtype: .float32),
                                      MLX.cumsum(values, axis: 0)], axis: 0)
        let rows = spans.map { span in
            span.isEmpty ? MLXArray.zeros([hidden.dim(-1)], dtype: .float32)
                : query ? MLX.mean(values[span, 0...], axis: 0)
                : (prefix[span.upperBound, 0...] - prefix[span.lowerBound, 0...]) / MLXArray(Float(span.count))
        }
        let pooled = MLX.stacked(rows)
        let projected = MLX.matmul(pooled, projection.asType(.float32).transposed())
        let quantized = MLX.clip(MLX.round(MLX.tanh(projected) * MLXArray(Float(127))), min: -128, max: 127)
        let truncated = quantized[0..., 0..<dimensions]
        return normalize ? Self.normalize(truncated) : truncated
    }
}
