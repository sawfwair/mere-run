import Foundation
import MLX
import MLXFast
import MLXNN

final class MuseGlimmerAssistantMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: MuseGlimmerAssistantConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

final class MuseGlimmerAssistantAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MuseGlimmerRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MuseGlimmerRMSNorm

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDimension: Int
    private let scale: Float
    private let slidingWindow: Int
    private let rope: RoPE

    init(config: MuseGlimmerAssistantConfig) {
        headCount = config.numAttentionHeads
        keyValueHeadCount = config.numKeyValueHeads
        headDimension = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        slidingWindow = config.slidingWindow
        rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeParameters.ropeTheta
        )
        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: false
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._oProj.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim,
            config.hiddenSize,
            bias: false
        )
        self._qNorm.wrappedValue = MuseGlimmerRMSNorm(
            dimensions: config.headDim,
            eps: config.rmsNormEps
        )
        self._kNorm.wrappedValue = MuseGlimmerRMSNorm(
            dimensions: config.headDim,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let offset = cache.offset
        var queries = qProj(x)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        var keys = kProj(x)
            .reshaped(batch, sequence, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = vProj(x)
            .reshaped(batch, sequence, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        queries = rope(qNorm(queries), offset: offset)
        keys = rope(kNorm(keys), offset: offset)
        let state = cache.attentionState(appending: keys, values: values)!
        let mask = bidirectionalSlidingMask(
            queryLength: sequence,
            queryOffset: offset,
            keyLength: state.0.dim(2),
            dtype: x.dtype
        )
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: state.0,
            values: state.1,
            scale: scale,
            mask: mask
        )
        return oProj(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, headCount * headDimension)
        )
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        let batch = context.dim(0)
        let sequence = context.dim(1)
        let offset = cache.offset
        var keys = kProj(context)
            .reshaped(batch, sequence, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = vProj(context)
            .reshaped(batch, sequence, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        keys = rope(kNorm(keys), offset: offset)
        cache.append(keys: keys, values: values)
    }

    private func bidirectionalSlidingMask(
        queryLength: Int,
        queryOffset: Int,
        keyLength: Int,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let keyStart = max(0, queryOffset + queryLength - keyLength)
        let queryPositions = MLXArray(
            Int32(queryOffset)..<Int32(queryOffset + queryLength)
        ).reshaped(queryLength, 1)
        let keyPositions = MLXArray(
            Int32(keyStart)..<Int32(keyStart + keyLength)
        ).reshaped(1, keyLength)
        let allowed = (keyPositions .>= (queryPositions - Int32(slidingWindow)))
            .&& (keyPositions .<= (queryPositions + Int32(slidingWindow)))
        let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
        return .array(MLX.where(
            allowed.reshaped(1, 1, queryLength, keyLength),
            zeros,
            zeros + MLXArray(-1e9).asType(dtype)
        ))
    }
}

final class MuseGlimmerAssistantDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MuseGlimmerAssistantAttention
    @ModuleInfo(key: "mlp") var mlp: MuseGlimmerAssistantMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: MuseGlimmerRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: MuseGlimmerRMSNorm

    init(config: MuseGlimmerAssistantConfig) {
        self._attention.wrappedValue = MuseGlimmerAssistantAttention(config: config)
        self._mlp.wrappedValue = MuseGlimmerAssistantMLP(config: config)
        self._inputNorm.wrappedValue = MuseGlimmerRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionNorm.wrappedValue = MuseGlimmerRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let attended = x + attention(inputNorm(x), cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        attention.appendContext(context, cache: cache)
    }
}

final class MuseGlimmerAssistantContextProjection: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "output_norm_enc") var outputNorm: MuseGlimmerRMSNorm

    init(config: MuseGlimmerAssistantConfig) {
        self._fc.wrappedValue = Linear(
            config.hiddenSize * config.targetLayerIds.count,
            config.hiddenSize,
            bias: false
        )
        self._outputNorm.wrappedValue = MuseGlimmerRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        outputNorm(fc(hidden))
    }
}

final class MuseGlimmerAssistantModel: Module {
    @ModuleInfo(key: "encoder") var encoder: MuseGlimmerAssistantContextProjection
    @ModuleInfo(key: "layers") var layers: [MuseGlimmerAssistantDecoderLayer]
    @ModuleInfo(key: "norm") var norm: MuseGlimmerRMSNorm

    let config: MuseGlimmerAssistantConfig

    init(config: MuseGlimmerAssistantConfig) {
        self.config = config
        self._encoder.wrappedValue = MuseGlimmerAssistantContextProjection(config: config)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            MuseGlimmerAssistantDecoderLayer(config: config)
        }
        self._norm.wrappedValue = MuseGlimmerRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func appendTargetContext(
        _ states: [Int: MLXArray],
        cache: [Gemma4AttentionCache]
    ) {
        precondition(cache.count == layers.count)
        let context = encoder(concatenated(
            config.targetLayerIds.map { states[$0]! },
            axis: -1
        ))
        for (index, layer) in layers.enumerated() {
            layer.appendContext(context, cache: cache[index])
        }
    }

    func draftLogits(
        anchorTokens: MLXArray,
        speculativeTokenCount: Int,
        cache: [Gemma4AttentionCache],
        target: MuseGlimmerModel
    ) -> MLXArray {
        precondition(speculativeTokenCount > 0 && speculativeTokenCount < config.blockSize)
        let batch = anchorTokens.dim(0)
        let masks = MLX.full(
            [batch, speculativeTokenCount],
            values: MLXArray(Int32(config.maskTokenId))
        )
        let inputIds = concatenated(
            [anchorTokens.reshaped(batch, 1).asType(.int32), masks],
            axis: 1
        )
        var hidden = target.rawInputEmbeddings(for: inputIds)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache[index])
        }
        return target.rawOutputLogits(from: norm(hidden))[0..., 1..., 0...]
    }

    func makeCache(initialOffset: Int = 0) -> [Gemma4AttentionCache] {
        layers.map { _ in
            Gemma4SlidingKVCache(
                maxSize: config.slidingWindow,
                initialOffset: initialOffset
            )
        }
    }
}
