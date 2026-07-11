import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunCore

// MARK: - Rotary helpers

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half...]
    return MLX.concatenated([-x2, x1], axis: -1)
}

private func applyRotaryPosEmb(
    _ q: MLXArray,
    _ k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    let cosExp = cos[0..., .newAxis, 0..., 0...]
    let sinExp = sin[0..., .newAxis, 0..., 0...]
    let qEmbed = (q * cosExp) + (rotateHalf(q) * sinExp)
    let kEmbed = (k * cosExp) + (rotateHalf(k) * sinExp)
    return (qEmbed, kEmbed)
}

private func repeatKVHeads(_ x: MLXArray, groups: Int) -> MLXArray {
    guard groups > 1 else { return x }
    let batch = x.dim(0)
    let heads = x.dim(1)
    let seqLen = x.dim(2)
    let headDim = x.dim(3)
    let expanded = x.expandedDimensions(axis: 2)
    let tiled = broadcast(expanded, to: [batch, heads, groups, seqLen, headDim])
    return tiled.reshaped(batch, heads * groups, seqLen, headDim)
}

// MARK: - Rotary Embeddings

final class Qwen3TTSTalkerRotaryEmbedding {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let mropeSection: [Int]
    let invFreq: MLXArray
    let hMask: MLXArray
    let wMask: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int, base: Float, mropeSection: [Int]) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        self.mropeSection = mropeSection

        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self.invFreq = Float(1.0) / MLX.pow(MLXArray(base), indices / Float(dim))

        // Precompute interleaved masks
        let headDimHalf = max(1, dim / 2)
        var hMaskVals = [Bool](repeating: false, count: headDimHalf)
        var wMaskVals = [Bool](repeating: false, count: headDimHalf)

        if mropeSection.count >= 3 {
            let hLength = min(headDimHalf, mropeSection[1] * 3)
            if hLength > 1 {
                for idx in stride(from: 1, to: hLength, by: 3) {
                    hMaskVals[idx] = true
                }
            }

            let wLength = min(headDimHalf, mropeSection[2] * 3)
            if wLength > 2 {
                for idx in stride(from: 2, to: wLength, by: 3) {
                    wMaskVals[idx] = true
                }
            }
        }

        let hMaskArray = MLXArray(hMaskVals).asType(.bool)
        let wMaskArray = MLXArray(wMaskVals).asType(.bool)
        self.hMask = hMaskArray.reshaped(1, 1, headDimHalf)
        self.wMask = wMaskArray.reshaped(1, 1, headDimHalf)
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var posIds = positionIds
        if posIds.ndim == 2 {
            posIds = broadcast(posIds[.newAxis, 0..., 0...], to: [3, posIds.dim(0), posIds.dim(1)])
        }

        let invFreqExpanded = broadcast(
            invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32),
            to: [3, posIds.dim(1), invFreq.count, 1]
        )
        let posExpanded = posIds[0..., 0..., .newAxis, 0...].asType(.float32)

        let freqs = MLX.matmul(invFreqExpanded, posExpanded).transposed(0, 1, 3, 2) // [3, B, T, dim/2]

        let freqsT = freqs[0]
        let freqsH = freqs[1]
        let freqsW = freqs[2]

        var combined = MLX.where(hMask, freqsH, freqsT)
        combined = MLX.where(wMask, freqsW, combined)

        let emb = MLX.concatenated([combined, combined], axis: -1)
        return (MLX.cos(emb).asType(x.dtype), MLX.sin(emb).asType(x.dtype))
    }
}

final class Qwen3TTSRotaryEmbedding {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let invFreq: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int, base: Float) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self.invFreq = Float(1.0) / MLX.pow(MLXArray(base), indices / Float(dim))
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let invFreqExpanded = invFreq[.newAxis, 0..., .newAxis].asType(.float32)
        let posExpanded = positionIds[0..., .newAxis, 0...].asType(.float32)
        let freqs = (invFreqExpanded * posExpanded).transposed(0, 2, 1)
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(x.dtype), MLX.sin(emb).asType(x.dtype))
    }
}

// MARK: - Talker Attention / MLP

final class Qwen3TTSTalkerAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: Qwen3TTSTalkerConfig, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = Float(self.headDim).squareRoot().reciprocal

        self._qProj.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKVHeads, headDim)
        var v = vProj(x).reshaped(batch, seqLen, numKVHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        (q, k) = applyRotaryPosEmb(q, k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        let kvGroups = numHeads / max(1, numKVHeads)
        if kvGroups > 1 {
            k = repeatKVHeads(k, groups: kvGroups)
            v = repeatKVHeads(v, groups: kvGroups)
        }

        if let cache {
            let updated = cache.update(keys: k, values: v)
            k = updated.0
            v = updated.1
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        return oProj(output)
    }
}

final class Qwen3TTSTalkerMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTalkerConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

final class Qwen3TTSTalkerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3TTSTalkerAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3TTSTalkerMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: Qwen3TTSTalkerConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = Qwen3TTSTalkerAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Qwen3TTSTalkerMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hidden = x
        let residual1 = hidden
        hidden = inputLayerNorm(hidden)
        hidden = selfAttn(hidden, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        hidden = residual1 + hidden

        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + hidden
        return hidden
    }
}

final class Qwen3TTSTalkerModel: Module {
    let config: Qwen3TTSTalkerConfig

    @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3TTSTalkerDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let rotaryEmb: Qwen3TTSTalkerRotaryEmbedding

    init(config: Qwen3TTSTalkerConfig) {
        self.config = config
        self._codecEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._textEmbedding.wrappedValue = Embedding(embeddingCount: config.textVocabSize, dimensions: config.textHiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { Qwen3TTSTalkerDecoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        let mropeSection = config.ropeScaling?.mropeSection ?? [24, 20, 20]
        self.rotaryEmb = Qwen3TTSTalkerRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta,
            mropeSection: mropeSection
        )
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let batch = inputsEmbeds.dim(0)
        let seqLen = inputsEmbeds.dim(1)

        let offset = cache?.first?.offset ?? 0
        let pos: MLXArray
        if let positionIds {
            pos = positionIds
        } else {
            let base = MLXArray(Int32(offset)..<Int32(offset + seqLen)).reshaped(1, seqLen)
            let broadcasted = broadcast(base, to: [batch, seqLen])
            pos = MLX.stacked([broadcasted, broadcasted, broadcasted], axis: 0)
        }

        let positionEmbeddings = rotaryEmb(inputsEmbeds, positionIds: pos)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = {
            if let mask { return mask }
            if let cache, let first = cache.first {
                return first.makeMask(n: seqLen)
            }
            return seqLen == 1 ? .none : .causal
        }()

        var hidden = inputsEmbeds
        for (idx, layer) in layers.enumerated() {
            let layerCache = cache?[idx]
            hidden = layer(hidden, positionEmbeddings: positionEmbeddings, mask: maskMode, cache: layerCache)
        }

        return norm(hidden)
    }

    func makeCache() -> [KVCache] {
        (0..<config.numHiddenLayers).map { _ in KVCacheSimple(step: 256) }
    }
}

// MARK: - Code Predictor

final class Qwen3TTSCodePredictorAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = Float(self.headDim).squareRoot().reciprocal

        self._qProj.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKVHeads, headDim)
        var v = vProj(x).reshaped(batch, seqLen, numKVHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        (q, k) = applyRotaryPosEmb(q, k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        let kvGroups = numHeads / max(1, numKVHeads)
        if kvGroups > 1 {
            k = repeatKVHeads(k, groups: kvGroups)
            v = repeatKVHeads(v, groups: kvGroups)
        }

        if let cache {
            let updated = cache.update(keys: k, values: v)
            k = updated.0
            v = updated.1
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        return oProj(output)
    }
}

final class Qwen3TTSCodePredictorMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTalkerCodePredictorConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

final class Qwen3TTSCodePredictorDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3TTSCodePredictorAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3TTSCodePredictorMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = Qwen3TTSCodePredictorAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Qwen3TTSCodePredictorMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hidden = x
        let residual1 = hidden
        hidden = inputLayerNorm(hidden)
        hidden = selfAttn(hidden, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        hidden = residual1 + hidden

        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + hidden
        return hidden
    }
}

final class Qwen3TTSCodePredictorModel: Module {
    let config: Qwen3TTSTalkerCodePredictorConfig

    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]
    @ModuleInfo(key: "layers") var layers: [Qwen3TTSCodePredictorDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let rotaryEmb: Qwen3TTSRotaryEmbedding

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self._codecEmbedding.wrappedValue = (0..<(config.numCodeGroups - 1)).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize)
        }
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { Qwen3TTSCodePredictorDecoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmb = Qwen3TTSRotaryEmbedding(dim: config.headDim, maxPositionEmbeddings: config.maxPositionEmbeddings, base: config.ropeTheta)
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let batch = inputsEmbeds.dim(0)
        let seqLen = inputsEmbeds.dim(1)

        let offset = cache?.first?.offset ?? 0
        let pos: MLXArray
        if let positionIds {
            pos = positionIds
        } else {
            let base = MLXArray(Int32(offset)..<Int32(offset + seqLen)).reshaped(1, seqLen)
            pos = broadcast(base, to: [batch, seqLen])
        }

        let positionEmbeddings = rotaryEmb(inputsEmbeds, positionIds: pos)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = {
            if let mask { return mask }
            if let cache, let first = cache.first {
                return first.makeMask(n: seqLen)
            }
            return seqLen == 1 ? .none : .causal
        }()

        var hidden = inputsEmbeds
        for (idx, layer) in layers.enumerated() {
            let layerCache = cache?[idx]
            hidden = layer(hidden, positionEmbeddings: positionEmbeddings, mask: maskMode, cache: layerCache)
        }

        return norm(hidden)
    }

    func makeCache() -> [KVCache] {
        (0..<config.numHiddenLayers).map { _ in KVCacheSimple(step: 256) }
    }
}

final class Qwen3TTSTalkerCodePredictor: Module {
    let config: Qwen3TTSTalkerCodePredictorConfig

    @ModuleInfo(key: "small_to_mtp_projection") var smallToMtpProjection: Linear?
    @ModuleInfo(key: "model") var model: Qwen3TTSCodePredictorModel
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config

        if config.hiddenSize != talkerHiddenSize {
            self._smallToMtpProjection.wrappedValue = Linear(talkerHiddenSize, config.hiddenSize, bias: true)
        } else {
            self._smallToMtpProjection.wrappedValue = nil
        }

        self._model.wrappedValue = Qwen3TTSCodePredictorModel(config: config, talkerHiddenSize: talkerHiddenSize)
        self._lmHead.wrappedValue = (0..<(config.numCodeGroups - 1)).map { _ in
            Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    var codecEmbedding: [Embedding] { model.codecEmbedding }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil,
        generationStep: Int = 0
    ) -> (logits: MLXArray, cache: [KVCache]?, nextStep: Int) {
        var embeds = inputsEmbeds
        if let projection = smallToMtpProjection {
            embeds = projection(embeds)
        }

        let hidden = model(embeds, positionIds: positionIds, mask: mask, cache: cache)
        let lastHidden = hidden.dim(1) > 1
            ? hidden[0..., (hidden.dim(1) - 1)..., 0...]
            : hidden
        let logits = lmHead[generationStep](lastHidden)
        return (logits, cache, generationStep + 1)
    }

    func makeCache() -> [KVCache] {
        model.makeCache()
    }
}

// MARK: - Talker wrapper

final class Qwen3TTSTalkerForConditionalGeneration: Module {
    let config: Qwen3TTSTalkerConfig

    @ModuleInfo(key: "model") var model: Qwen3TTSTalkerModel
    @ModuleInfo(key: "text_projection") var textProjection: ResizeMLP
    @ModuleInfo(key: "codec_head") var codecHead: Linear
    @ModuleInfo(key: "code_predictor") var codePredictor: Qwen3TTSTalkerCodePredictor

    init(config: Qwen3TTSTalkerConfig) {
        self.config = config
        self._model.wrappedValue = Qwen3TTSTalkerModel(config: config)
        self._textProjection.wrappedValue = ResizeMLP(
            inputSize: config.textHiddenSize,
            intermediateSize: config.textHiddenSize,
            outputSize: config.hiddenSize,
            hiddenAct: config.hiddenAct,
            bias: true
        )
        self._codecHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        self._codePredictor.wrappedValue = Qwen3TTSTalkerCodePredictor(config: config.codePredictorConfig, talkerHiddenSize: config.hiddenSize)
    }

    func getInputEmbeddings() -> Embedding {
        model.codecEmbedding
    }

    func getTextEmbeddings() -> Embedding {
        model.textEmbedding
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> (logits: MLXArray, hidden: MLXArray) {
        var hidden = model(inputsEmbeds, positionIds: positionIds, mask: mask, cache: cache)
        if hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        let logits = codecHead(hidden)
        return (logits, hidden)
    }

    func makeCache() -> [KVCache] {
        model.makeCache()
    }
}

// MARK: - Resize MLP

final class ResizeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "linear_fc1") var linearFc1: Linear
    @ModuleInfo(key: "linear_fc2") var linearFc2: Linear
    private let activation: (MLXArray) -> MLXArray

    init(
        inputSize: Int,
        intermediateSize: Int,
        outputSize: Int,
        hiddenAct: String,
        bias: Bool
    ) {
        self._linearFc1.wrappedValue = Linear(inputSize, intermediateSize, bias: bias)
        self._linearFc2.wrappedValue = Linear(intermediateSize, outputSize, bias: bias)

        switch hiddenAct.lowercased() {
        case "gelu":
            self.activation = MLXNN.gelu
        case "relu":
            self.activation = MLXNN.relu
        default:
            self.activation = MLXNN.silu
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearFc2(activation(linearFc1(x)))
    }
}

private extension Float {
    var reciprocal: Float { 1.0 / self }
}
