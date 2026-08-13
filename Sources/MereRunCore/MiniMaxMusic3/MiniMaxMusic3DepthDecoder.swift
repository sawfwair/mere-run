import MLX
import MLXFast
import MLXNN

final class MiniMaxMusic3DepthAttention: Module {
    let heads: Int
    let headDimension: Int

    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "to_out") var output: Linear

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        self.headDimension = dimensions / heads
        self._query.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._key.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._value.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._output.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let q = query(hidden).reshaped(batch, length, heads, headDimension).transposed(0, 2, 1, 3)
        let k = key(hidden).reshaped(batch, length, heads, headDimension).transposed(0, 2, 1, 3)
        let v = value(hidden).reshaped(batch, length, heads, headDimension).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32),
            keys: k.asType(.float32),
            values: v.asType(.float32),
            scale: 1 / Float(headDimension).squareRoot(),
            mask: .causal
        )
        return output(attended.asType(q.dtype).transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class MiniMaxMusic3DepthBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxMusic3DepthAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(dimensions: Int, heads: Int, intermediateSize: Int) {
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: dimensions, eps: 1e-6)
        self._attention.wrappedValue = MiniMaxMusic3DepthAttention(dimensions: dimensions, heads: heads)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: dimensions, eps: 1e-6)
        self._gateProjection.wrappedValue = Linear(dimensions, intermediateSize, bias: false)
        self._upProjection.wrappedValue = Linear(dimensions, intermediateSize, bias: false)
        self._downProjection.wrappedValue = Linear(intermediateSize, dimensions, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(inputLayerNorm(input))
        let normalized = postAttentionLayerNorm(attended)
        return attended + downProjection(MLXNN.silu(gateProjection(normalized)) * upProjection(normalized))
    }
}

public final class MiniMaxMusic3DepthDecoder: Module {
    public let configuration: MiniMaxMusic3DepthConfiguration

    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
    @ModuleInfo(key: "projection") var projection: Linear
    @ModuleInfo(key: "pos_embedding") var positionEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [MiniMaxMusic3DepthBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "audio_heads") var audioHeads: [Linear]

    public init(configuration: MiniMaxMusic3DepthConfiguration) {
        self.configuration = configuration
        self._audioEmbeddings.wrappedValue = Embedding(
            embeddingCount: configuration.audioVocabSize * (configuration.numCodebooks - 1),
            dimensions: configuration.hiddenSize
        )
        self._projection.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false)
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.maxPositionEmbeddings,
            dimensions: configuration.hiddenSize
        )
        self._layers.wrappedValue = (0..<configuration.numLayers).map { _ in
            MiniMaxMusic3DepthBlock(
                dimensions: configuration.hiddenSize,
                heads: configuration.numAttentionHeads,
                intermediateSize: configuration.intermediateSize
            )
        }
        self._norm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-6)
        self._audioHeads.wrappedValue = (1..<configuration.numCodebooks).map { _ in
            Linear(configuration.hiddenSize, configuration.audioVocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputEmbeddings: MLXArray) -> MLXArray {
        let positions = MLXArray((0..<inputEmbeddings.dim(1)).map(Int32.init))
        var hidden = inputEmbeddings + positionEmbedding(positions).expandedDimensions(axis: 0)
        for layer in layers {
            hidden = layer(hidden)
        }
        return norm(hidden)
    }

    public func embedResidualCodes(_ codes: MLXArray, codebookIndex: Int) -> MLXArray {
        audioEmbeddings(codes.asType(.int32) + MLXArray(Int32(codebookIndex * configuration.audioVocabSize)))
    }

    public func logits(_ hidden: MLXArray, codebookIndex: Int) -> MLXArray {
        audioHeads[codebookIndex](hidden)
    }
}
