#if !os(iOS)
import MLX
import MLXFast
import MLXNN

/// Packed PyTorch MultiheadAttention parameters, without causal masking.
final class ClefAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var weight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var bias: MLXArray
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        self._weight.wrappedValue = MLX.zeros([3 * width, width])
        self._bias.wrappedValue = MLX.zeros([3 * width])
        self._output.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ query: MLXArray, _ key: MLXArray, _ value: MLXArray) -> MLXArray {
        let width = weight.dim(1)
        let headWidth = width / heads
        func project(_ input: MLXArray, _ offset: Int) -> MLXArray {
            let projected = matmul(input, weight[offset..<(offset + width)].T) + bias[offset..<(offset + width)]
            return projected.reshaped(input.dim(0), input.dim(1), heads, headWidth).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: project(query, 0), keys: project(key, width), values: project(value, 2 * width),
            scale: 1 / Float(headWidth).squareRoot(), mask: .none)
        return output(attended.transposed(0, 2, 1, 3).reshaped(query.dim(0), query.dim(1), width))
    }
}

final class ClefFeedForward: Module {
    @ModuleInfo(key: "fc1") var first: Linear
    @ModuleInfo(key: "fc2") var second: Linear

    init(width: Int, feedforward: Int) {
        self._first.wrappedValue = Linear(width, feedforward)
        self._second.wrappedValue = Linear(feedforward, width)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { second(gelu(first(input))) }
}

final class ClefEvidenceLayer: Module {
    @ModuleInfo(key: "query_norm") var queryNorm: LayerNorm
    @ModuleInfo(key: "memory_norm") var memoryNorm: LayerNorm
    @ModuleInfo var attention: ClefAttention
    @ModuleInfo(key: "feedforward_norm") var feedforwardNorm: LayerNorm
    @ModuleInfo var feedforward: ClefFeedForward

    init(width: Int, heads: Int, feedforward: Int) {
        self._queryNorm.wrappedValue = LayerNorm(dimensions: width)
        self._memoryNorm.wrappedValue = LayerNorm(dimensions: width)
        self._attention.wrappedValue = ClefAttention(width: width, heads: heads)
        self._feedforwardNorm.wrappedValue = LayerNorm(dimensions: width)
        self._feedforward.wrappedValue = ClefFeedForward(width: width, feedforward: feedforward)
    }

    func callAsFunction(_ queries: MLXArray, memory: MLXArray) -> MLXArray {
        let normalized = memoryNorm(memory)
        let routed = queries + attention(queryNorm(queries), normalized, normalized)
        return routed + feedforward(feedforwardNorm(routed))
    }
}

final class ClefDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: ClefAttention
    @ModuleInfo(key: "multihead_attn") var crossAttention: ClefAttention
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var norm3: LayerNorm

    init(width: Int, heads: Int, feedforward: Int) {
        self._selfAttention.wrappedValue = ClefAttention(width: width, heads: heads)
        self._crossAttention.wrappedValue = ClefAttention(width: width, heads: heads)
        self._linear1.wrappedValue = Linear(width, feedforward)
        self._linear2.wrappedValue = Linear(feedforward, width)
        self._norm1.wrappedValue = LayerNorm(dimensions: width)
        self._norm2.wrappedValue = LayerNorm(dimensions: width)
        self._norm3.wrappedValue = LayerNorm(dimensions: width)
    }

    func callAsFunction(_ fields: MLXArray, memory: MLXArray) -> MLXArray {
        let normalized = norm1(fields)
        let attended = fields + selfAttention(normalized, normalized, normalized)
        let joint = attended + crossAttention(norm2(attended), memory, memory)
        return joint + linear2(gelu(linear1(norm3(joint))))
    }
}
#endif
