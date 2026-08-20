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
    private let usesDFlash2BlockMask: Bool
    private let rope: RoPE

    init(config: MuseGlimmerAssistantConfig) {
        headCount = config.numAttentionHeads
        keyValueHeadCount = config.numKeyValueHeads
        headDimension = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        slidingWindow = config.slidingWindow
        usesDFlash2BlockMask = config.isDFlash2
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
        let mask = usesDFlash2BlockMask
            ? blockSlidingMask(
                queryLength: sequence,
                queryOffset: offset,
                keyLength: state.0.dim(2),
                dtype: x.dtype
            )
            : bidirectionalSlidingMask(
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

    private func blockSlidingMask(
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
        let context = (keyPositions .< Int32(queryOffset))
            .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
        let block = keyPositions .>= Int32(queryOffset)
        let allowed = context .|| block
        let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
        return .array(MLX.where(
            allowed.reshaped(1, 1, queryLength, keyLength),
            zeros,
            zeros + MLXArray(-1e9).asType(dtype)
        ))
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
    @ModuleInfo(key: "attention_conv") var attentionConv: MuseGlimmerGroupedDynamicCausalConv?
    @ModuleInfo(key: "mlp_conv") var mlpConv: MuseGlimmerGroupedDynamicCausalConv?

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
        if let dflash2 = config.dflash2, config.isDFlash2 {
            self._attentionConv.wrappedValue = MuseGlimmerGroupedDynamicCausalConv(
                hiddenSize: config.hiddenSize,
                kernelSize: dflash2.convKernelSize,
                groupSize: dflash2.convGroupSize
            )
            self._mlpConv.wrappedValue = MuseGlimmerGroupedDynamicCausalConv(
                hiddenSize: config.hiddenSize,
                kernelSize: dflash2.convKernelSize,
                groupSize: dflash2.convGroupSize
            )
        } else {
            self._attentionConv.wrappedValue = nil
            self._mlpConv.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        if let attentionConv, let mlpConv {
            let attentionInput = attentionConv.prepare(inputNorm(x))
            let attended = x + attentionConv.finish(
                attention(attentionInput.hidden, cache: cache),
                dynamic: attentionInput.dynamic
            )
            let mlpInput = mlpConv.prepare(postAttentionNorm(attended))
            return attended + mlpConv.finish(
                mlp(mlpInput.hidden),
                dynamic: mlpInput.dynamic
            )
        }
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
    @ModuleInfo(key: "candidate_selector") var candidateSelector: MuseGlimmerDFlash2CandidateSelector?

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
        if let dflash2 = config.dflash2,
           let vocabularySize = config.vocabSize,
           config.isDFlash2 {
            self._candidateSelector.wrappedValue = MuseGlimmerDFlash2CandidateSelector(
                vocabularySize: vocabularySize,
                hiddenSize: config.hiddenSize,
                rank: dflash2.selectorRank,
                topK: dflash2.selectorTopK
            )
        } else {
            self._candidateSelector.wrappedValue = nil
        }
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

    func draft(
        anchorTokens: MLXArray,
        speculativeTokenCount: Int,
        cache: [Gemma4AttentionCache],
        target: MuseGlimmerModel
    ) -> MuseGlimmerAssistantDraftOutput {
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
        var hidden = config.isDFlash2
            ? target.inputEmbeddings(for: inputIds) * (config.dflash2?.inputEmbeddingScale ?? 1)
            : target.rawInputEmbeddings(for: inputIds)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache[index])
        }
        hidden = norm(hidden)[0..., 1..., 0...]
        let logits = config.isDFlash2
            ? target.outputLogits(from: hidden)
            : target.rawOutputLogits(from: hidden)
        return MuseGlimmerAssistantDraftOutput(hidden: hidden, logits: logits)
    }

    func draftLogits(
        anchorTokens: MLXArray,
        speculativeTokenCount: Int,
        cache: [Gemma4AttentionCache],
        target: MuseGlimmerModel
    ) -> MLXArray {
        draft(
            anchorTokens: anchorTokens,
            speculativeTokenCount: speculativeTokenCount,
            cache: cache,
            target: target
        ).logits
    }

    func selectDFlash2Candidates(
        draft: MuseGlimmerAssistantDraftOutput,
        anchorTokens: MLXArray,
        temperature: Float
    ) -> MuseGlimmerDFlash2Selection? {
        candidateSelector?.select(
            hidden: draft.hidden,
            logits: draft.logits,
            anchorTokenIds: anchorTokens,
            temperature: temperature
        )
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
