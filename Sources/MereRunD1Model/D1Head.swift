#if !os(iOS)
// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MLX
import MLXFast
import MLXNN

final class D1HeadAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var weight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var bias: MLXArray
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    let width: Int
    init(width: Int) {
        self.width = width; heads = width / 64
        _weight.wrappedValue = MLXArray.zeros([3 * width, width])
        _bias.wrappedValue = MLXArray.zeros([3 * width])
        _output.wrappedValue = Linear(width, width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = matmul(x, weight.T) + bias
        let pieces = (0..<3).map {
            projected[.ellipsis, ($0 * width)..<(($0 + 1) * width)].reshaped(x.dim(0), x.dim(1), heads, 64).transposed(0, 2, 1, 3)
        }
        return output(MLXFast.scaledDotProductAttention(queries: pieces[0], keys: pieces[1], values: pieces[2],
            scale: 0.125, mask: .none).transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), width))
    }
}

final class D1HeadLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: D1HeadAttention
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    init(width: Int) {
        _attention.wrappedValue = D1HeadAttention(width: width)
        _linear1.wrappedValue = Linear(width, 4 * width)
        _linear2.wrappedValue = Linear(4 * width, width)
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _norm2.wrappedValue = LayerNorm(dimensions: width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        return h + linear2(relu(linear1(norm2(h))))
    }
}

final class D1Scorer: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var input: Linear
    @ModuleInfo var output: Linear
    init(width: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: width)
        _input.wrappedValue = Linear(width, width)
        _output.wrappedValue = Linear(width, 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(gelu(input(norm(x)))) }
}

public final class D1Head: Module {
    @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
    @ModuleInfo var layers: [D1HeadLayer]
    @ModuleInfo var scorer: D1Scorer
    public init(width: Int, layerCount: Int) {
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
        _layers.wrappedValue = (0..<layerCount).map { _ in D1HeadLayer(width: width) }
        _scorer.wrappedValue = D1Scorer(width: width)
        super.init()
    }
    public func callAsFunction(_ text: MLXArray, markers: [Int], type: Int) -> MLXArray {
        let h = text + typeEmbedding(MLXArray([type])).expandedDimensions(axis: 1)
        let encoded = layers.reduce(h) { $1($0) }
        return scorer(take(encoded, MLXArray(markers), axis: 1)).reshaped(-1).asType(.float32)
    }
}
#endif
