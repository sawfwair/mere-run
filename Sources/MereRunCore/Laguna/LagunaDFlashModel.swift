import Foundation
import MLX
import MLXFast
import MLXNN

final class LagunaDFlashAttention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDim: Int
    private let querySize: Int
    private let keyValueSize: Int
    private let scale: Float
    private let slidingWindow: Int
    private let rope: LagunaRoPE

    init(config: LagunaDFlashConfig) {
        self.headCount = config.numAttentionHeads
        self.keyValueHeadCount = config.numKeyValueHeads
        self.headDim = config.headDim
        self.querySize = config.numAttentionHeads * config.headDim
        self.keyValueSize = config.numKeyValueHeads * config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.slidingWindow = config.slidingWindow
        self.rope = LagunaRoPE(
            headDim: config.headDim,
            parameters: config.ropeParameters
        )
        self._qkvProj.wrappedValue = Linear(
            config.hiddenSize,
            querySize + 2 * keyValueSize,
            bias: false
        )
        self._oProj.wrappedValue = Linear(querySize, config.hiddenSize, bias: false)
        self._gProj.wrappedValue = Linear(config.hiddenSize, headCount, bias: false)
        self._qNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim,
            eps: config.rmsNormEps
        )
        self._kNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let positionOffsets = (cache as? LagunaRaggedKVCache)?.positionOffsets
            ?? Array(repeating: cache.offset, count: batch)
        let qkv = qkvProj(x)
        var queries = qkv[0..., 0..., ..<querySize]
            .reshaped(batch, sequenceLength, headCount, headDim)
        var keys = qkv[0..., 0..., querySize..<(querySize + keyValueSize)]
            .reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        var values = qkv[0..., 0..., (querySize + keyValueSize)...]
            .reshaped(batch, sequenceLength, keyValueHeadCount, headDim)

        queries = rope(
            qNorm(queries).transposed(0, 2, 1, 3),
            offsets: positionOffsets
        )
        keys = rope(
            kNorm(keys).transposed(0, 2, 1, 3),
            offsets: positionOffsets
        )
        values = values.transposed(0, 2, 1, 3)

        let state = cache.attentionState(appending: keys, values: values)!
        let keyLengths = (cache as? LagunaRaggedKVCache)?.lastAttentionKeyLengths
            ?? Array(repeating: state.0.dim(2), count: batch)
        let mask = attentionMask(
            queryLength: sequenceLength,
            queryOffsets: positionOffsets,
            keyLengths: keyLengths,
            keyLength: state.0.dim(2),
            dtype: x.dtype
        )
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: state.0,
            values: state.1,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3)
        let gate = MLXNN.softplus(gProj(x).asType(.float32)).asType(output.dtype)
        output = output * MLX.expandedDimensions(gate, axis: gate.ndim)
        return oProj(output.reshaped(batch, sequenceLength, querySize))
    }

    func appendContext(
        _ context: MLXArray,
        inputLayerNorm: RMSNorm,
        cache: Gemma4AttentionCache
    ) {
        let batch = context.dim(0)
        let sequenceLength = context.dim(1)
        let positionOffsets = (cache as? LagunaRaggedKVCache)?.positionOffsets
            ?? Array(repeating: cache.offset, count: batch)
        let normalized = inputLayerNorm(context)
        let keyValueWeight = qkvProj.weight[querySize..., 0...]
        let projected = normalized.matmul(keyValueWeight.T)
        var keys = projected[0..., 0..., ..<keyValueSize]
            .reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        let values = projected[0..., 0..., keyValueSize...]
            .reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
            .transposed(0, 2, 1, 3)
        keys = rope(
            kNorm(keys).transposed(0, 2, 1, 3),
            offsets: positionOffsets
        )
        cache.append(keys: keys, values: values)
    }

    private func attentionMask(
        queryLength: Int,
        queryOffsets: [Int],
        keyLengths: [Int],
        keyLength: Int,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let rowMasks = zip(queryOffsets, keyLengths).map { queryOffset, validKeyLength in
            let keyStart = max(0, queryOffset + queryLength - validKeyLength)
            let queryPositions = MLXArray(
                Int32(queryOffset)..<Int32(queryOffset + queryLength)
            ).reshaped(queryLength, 1)
            let keyIndices = MLXArray(Int32(0)..<Int32(keyLength)).reshaped(1, keyLength)
            let keyPositions = keyIndices + Int32(keyStart)
            return (keyIndices .< Int32(validKeyLength))
                .&& (keyPositions .<= queryPositions)
                .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
        }
        let allowed = stacked(rowMasks).reshaped(
            queryOffsets.count,
            1,
            queryLength,
            keyLength
        )
        let zeros = MLXArray.zeros(
            [queryOffsets.count, 1, queryLength, keyLength],
            dtype: dtype
        )
        return .array(MLX.where(
            allowed,
            zeros,
            zeros + MLXArray(-1e9).asType(dtype)
        ))
    }
}

final class LagunaDFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: LagunaDFlashAttention
    @ModuleInfo(key: "mlp") var mlp: LagunaDenseMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: LagunaDFlashConfig) {
        self._selfAttention.wrappedValue = LagunaDFlashAttention(config: config)
        self._mlp.wrappedValue = LagunaDenseMLP(
            inputDimensions: config.hiddenSize,
            hiddenDimensions: config.intermediateSize
        )
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let attended = x + selfAttention(inputLayerNorm(x), cache: cache)
        return attended + mlp(postAttentionLayerNorm(attended))
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        selfAttention.appendContext(
            context,
            inputLayerNorm: inputLayerNorm,
            cache: cache
        )
    }
}

final class LagunaDFlashModel: Module {
    @ModuleInfo(key: "aux_hidden_norms") var auxHiddenNorms: [RMSNorm]
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [LagunaDFlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: LagunaDFlashConfig

    init(config: LagunaDFlashConfig) {
        self.config = config
        self._auxHiddenNorms.wrappedValue = config.dflash.targetLayerIDs.map { _ in
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self._fc.wrappedValue = Linear(
            config.hiddenSize * config.dflash.targetLayerIDs.count,
            config.hiddenSize,
            bias: false
        )
        self._hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            LagunaDFlashDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func combineTargetHiddenStates(_ states: [Int: MLXArray]) -> MLXArray {
        let normalized = zip(config.dflash.targetLayerIDs, auxHiddenNorms).map {
            layerID, layerNorm in
            layerNorm(states[layerID]!)
        }
        return hiddenNorm(fc(concatenated(normalized, axis: -1)))
    }

    func appendTargetContext(
        _ combinedContext: MLXArray,
        cache: [Gemma4AttentionCache]
    ) {
        precondition(cache.count == layers.count)
        for (index, layer) in layers.enumerated() {
            layer.appendContext(combinedContext, cache: cache[index])
        }
    }

    func draftLogits(
        anchorTokens: MLXArray,
        speculativeTokenCount: Int,
        cache: [Gemma4AttentionCache],
        target: LagunaCausalLM
    ) -> MLXArray {
        precondition(
            speculativeTokenCount > 0
                && speculativeTokenCount < config.dflash.blockSize
        )
        let batch = anchorTokens.dim(0)
        let masks = MLX.full(
            [batch, speculativeTokenCount],
            values: MLXArray(Int32(config.dflash.maskTokenID))
        )
        let inputIDs = concatenated(
            [anchorTokens.reshaped(batch, 1).asType(.int32), masks],
            axis: 1
        )
        var hidden = target.inputEmbeddings(for: inputIDs)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache[index])
        }
        hidden = norm(hidden)
        return target.logits(from: hidden)[0..., 1..., 0...]
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
