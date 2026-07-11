import Foundation
import MLX
import MLXNN
import MLXFast
import MereRunCore

// MARK: - Qwen3 ASR Full Model

// MARK: - Multimodal RoPE (Qwen3)

/// Qwen2-style rotary embeddings (used for non-interleaved MRoPE).
private final class ASRQwen2VLRotaryEmbedding {
    let dim: Int
    let base: Float
    let invFreq: MLXArray

    init(dim: Int, base: Float = 1_000_000.0) {
        self.dim = dim
        self.base = base

        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self.invFreq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))
    }

    func callAsFunction(positionIds: MLXArray, dtype: DType = .bfloat16) -> (cos: MLXArray, sin: MLXArray) {
        var posIds = positionIds

        if posIds.ndim == 2 {
            posIds = broadcast(posIds[.newAxis, 0..., 0...], to: [3, posIds.dim(0), posIds.dim(1)])
        }

        let invFreqExpanded = broadcast(
            invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32),
            to: [3, posIds.dim(1), invFreq.count, 1]
        )

        let posIdsExpanded = posIds[0..., 0..., .newAxis, 0...].asType(.float32)

        let freqs = MLX.matmul(invFreqExpanded, posIdsExpanded).transposed(0, 1, 3, 2)
        let emb = MLX.concatenated([freqs, freqs], axis: -1)

        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }
}

/// Qwen3 interleaved MRoPE.
private final class ASRQwen3VLRotaryEmbedding {
    let dim: Int
    let base: Float
    let invFreq: MLXArray
    let mropeSection: [Int]
    let interleavedSelector: MLXArray

    init(dim: Int, base: Float = 1_000_000.0, mropeSection: [Int]) {
        self.dim = dim
        self.base = base
        self.mropeSection = mropeSection

        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self.invFreq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))
        self.interleavedSelector = ASRQwen3VLRotaryEmbedding.buildInterleavedSelector(
            dim: max(1, dim / 2),
            mropeSection: mropeSection
        )
    }

    private static func buildInterleavedSelector(dim: Int, mropeSection: [Int]) -> MLXArray {
        var selector = [Int32](repeating: 0, count: dim)
        guard mropeSection.count >= 3 else {
            return MLXArray(selector).reshaped(1, 1, 1, dim)
        }

        let limits = [(1, 1), (2, 2)]
        for (dimIdx, offset) in limits {
            let length = min(dim, mropeSection[dimIdx] * 3)
            guard offset < length else { continue }
            for i in stride(from: offset, to: length, by: 3) {
                selector[i] = Int32(dimIdx)
            }
        }

        return MLXArray(selector).reshaped(1, 1, 1, dim)
    }

    private func applyInterleavedMRoPE(_ freqs: MLXArray) -> MLXArray {
        let selector = broadcast(
            interleavedSelector,
            to: [1, freqs.dim(1), freqs.dim(2), freqs.dim(3)]
        )
        return takeAlong(freqs, selector, axis: 0).squeezed(axis: 0)
    }

    func callAsFunction(positionIds: MLXArray, dtype: DType = .bfloat16) -> (cos: MLXArray, sin: MLXArray) {
        var posIds = positionIds

        if posIds.ndim == 2 {
            posIds = broadcast(posIds[.newAxis, 0..., 0...], to: [3, posIds.dim(0), posIds.dim(1)])
        }

        let invFreqExpanded = broadcast(
            invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32),
            to: [3, posIds.dim(1), invFreq.count, 1]
        )

        let posIdsExpanded = posIds[0..., 0..., .newAxis, 0...].asType(.float32)

        let freqs = MLX.matmul(invFreqExpanded, posIdsExpanded).transposed(0, 1, 3, 2)
        let freqsInterleaved = applyInterleavedMRoPE(freqs)
        let emb = MLX.concatenated([freqsInterleaved, freqsInterleaved], axis: -1)

        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[0..., 0..., 0..., 0..<halfDim]
    let x2 = x[0..., 0..., 0..., halfDim...]
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

private func applyMultimodalRoPE(
    _ q: MLXArray,
    _ k: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    mropeSection: [Int]
) -> (MLXArray, MLXArray) {
    let sectionSizes = mropeSection.map { $0 * 2 }
    var cosSlices: [MLXArray] = []
    var sinSlices: [MLXArray] = []
    cosSlices.reserveCapacity(sectionSizes.count)
    sinSlices.reserveCapacity(sectionSizes.count)

    var start = 0
    for (index, size) in sectionSizes.enumerated() {
        let end = start + size
        let cosChunk = cos[0..., 0..., 0..., start..<end]
        let sinChunk = sin[0..., 0..., 0..., start..<end]
        cosSlices.append(cosChunk[index % 3])
        sinSlices.append(sinChunk[index % 3])
        start = end
    }

    let cosMerged = MLX.concatenated(cosSlices, axis: -1)
    let sinMerged = MLX.concatenated(sinSlices, axis: -1)

    let cosExp = cosMerged[0..., .newAxis, 0..., 0...]
    let sinExp = sinMerged[0..., .newAxis, 0..., 0...]

    let qEmbed = (q * cosExp) + (rotateHalf(q) * sinExp)
    let kEmbed = (k * cosExp) + (rotateHalf(k) * sinExp)

    return (qEmbed, kEmbed)
}

/// Qwen3-ASR model for speech recognition
/// Structure: audio_tower (encoder) + model (Qwen3 decoder) + lm_head
public final class Qwen3ASRThinker: Module {
    let config: Qwen3ASRModelConfig

    @ModuleInfo(key: "audio_tower") var audioTower: Qwen3ASRAudioTower
    @ModuleInfo(key: "model") var model: Qwen3ASRDecoderModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(config: Qwen3ASRModelConfig) {
        self.config = config

        self._audioTower.wrappedValue = Qwen3ASRAudioTower(config: config.audioConfig)
        self._model.wrappedValue = Qwen3ASRDecoderModel(config: config.textConfig)
        if config.textConfig.tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(
                config.textConfig.hiddenSize,
                config.textConfig.vocabSize,
                bias: false
            )
        }
    }

    /// Encode audio to features
    /// - Parameter melSpec: Mel spectrogram [B, nMels, T]
    /// - Returns: Audio features [B, T, hiddenSize]
    public func encodeAudio(_ melSpec: MLXArray) -> MLXArray {
        let audioFeatures = audioTower(melSpec)  // [B, T, 896]

        // Project to decoder hidden size (896 -> 1024)
        // Note: This projection is implicit in how embeddings are merged
        return audioFeatures
    }

    /// Forward pass
    public func callAsFunction(
        inputIds: MLXArray,
        audioFeatures: MLXArray? = nil,
        cache: [KVCache]? = nil,
        positionIds: MLXArray? = nil,
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        var hidden = model(
            inputIds: inputIds,
            audioFeatures: audioFeatures,
            audioTokenId: config.audioTokenId,
            cache: cache,
            positionIds: positionIds
        )
        if lastPositionOnly && hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    /// Create fresh KV cache
    public func makeCache() -> [KVCache] {
        (0..<config.textConfig.numHiddenLayers).map { _ in
            KVCacheSimple(step: 256)
        }
    }
}

// MARK: - Decoder Model (Qwen3)

final class Qwen3ASRDecoderModel: Module {
    let config: Qwen3ASRDecoderConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3ASRDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: Qwen3ASRDecoderConfig) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            Qwen3ASRDecoderLayer(config: config)
        }

        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )

    }

    func callAsFunction(
        inputIds: MLXArray,
        audioFeatures: MLXArray? = nil,
        audioTokenId: Int = 151676,
        cache: [KVCache]? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        // Get token embeddings
        var h = embedTokens(tokenIds)

        // If we have audio features, replace embeddings at audio token positions
        if let audioFeatures {
            h = injectAudioFeatures(
                embeddings: h,
                audioFeatures: audioFeatures,
                tokenIds: tokenIds,
                audioTokenId: audioTokenId
            )
        }

        // Create attention mask
        let mask = createMask(h: h, cache: cache?.first)

        // Apply transformer layers
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], positionIds: positionIds)
        }

        // Final norm
        h = norm(h)
        return h
    }

    /// Inject audio features into embeddings at audio token positions
    private func injectAudioFeatures(
        embeddings: MLXArray,
        audioFeatures: MLXArray,
        tokenIds: MLXArray,
        audioTokenId: Int
    ) -> MLXArray {
        let debugEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_DEBUG"]?.lowercased() {
            debugEnabled = raw == "1" || raw == "true" || raw == "yes"
        } else {
            debugEnabled = false
        }

        // Audio features are now projected to 1024 (same as text embeddings)
        let audioLen = audioFeatures.dim(1)
        let seqLen = embeddings.dim(1)

        // Find the start of audio tokens (first position where token == audioTokenId)
        let tokenList = tokenIds.reshaped(-1).asArray(Int32.self)
        var audioStart = -1
        for (i, tok) in tokenList.enumerated() {
            if tok == Int32(audioTokenId) {
                audioStart = i
                break
            }
        }

        if debugEnabled {
            let message = "[ASR DEBUG] injectAudio: start=\(audioStart) len=\(audioLen) seq=\(seqLen)\n"
            FileHandle.standardError.write(Data(message.utf8))
        }

        guard audioStart >= 0 && audioStart + audioLen <= seqLen else {
            return embeddings
        }

        // Replace embeddings at audio positions with audio features
        let features = (audioFeatures.dtype == embeddings.dtype)
            ? audioFeatures
            : audioFeatures.asType(embeddings.dtype)

        if debugEnabled {
            let embStd = MLX.std(embeddings.asType(.float32)).item(Float.self)
            let featStd = MLX.std(features.asType(.float32)).item(Float.self)
            let embMean = MLX.mean(embeddings.asType(.float32)).item(Float.self)
            let featMean = MLX.mean(features.asType(.float32)).item(Float.self)
            let message = String(
                format: "[ASR DEBUG] embed mean=%.6f std=%.6f | audio mean=%.6f std=%.6f\n",
                embMean, embStd, featMean, featStd
            )
            FileHandle.standardError.write(Data(message.utf8))
        }
        let before = embeddings[0..., 0..<audioStart, 0...]
        let after = embeddings[0..., (audioStart + audioLen)..., 0...]
        return MLX.concatenated([before, features, after], axis: 1)
    }

    private func createMask(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let n = h.dim(1)
        if let cache = cache {
            return cache.makeMask(n: n)
        }
        if n == 1 {
            return .none
        }
        return .causal
    }
}

// MARK: - Decoder Layer

final class Qwen3ASRDecoderLayer: Module {
    let config: Qwen3ASRDecoderConfig

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3ASRDecoderAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3ASRMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: Qwen3ASRDecoderConfig) {
        self.config = config

        self._selfAttn.wrappedValue = Qwen3ASRDecoderAttention(config: config)
        self._mlp.wrappedValue = Qwen3ASRMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attnOut = selfAttn(normed, mask: mask, cache: cache, positionIds: positionIds)
        let h = x + attnOut

        let postNormed = postAttentionLayerNorm(h)
        let mlpOut = mlp(postNormed)

        return h + mlpOut
    }
}

// MARK: - Decoder Attention

final class Qwen3ASRDecoderAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let numKVGroups: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let rotaryEmb: ASRQwen3VLRotaryEmbedding?
    private let legacyRotaryEmb: ASRQwen2VLRotaryEmbedding?
    private let mropeSection: [Int]?
    private let rope: RoPE?

    init(config: Qwen3ASRDecoderConfig) {
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.numKVGroups = config.numAttentionHeads / config.numKeyValueHeads
        self.scale = pow(Float(config.headDim), -0.5)

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        let useMRoPE: Bool = {
            guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_USE_MROPE"]?.lowercased() else {
                return false
            }
            return raw == "1" || raw == "true" || raw == "yes"
        }()

        if useMRoPE,
           let ropeScaling = config.ropeScaling,
           let mropeSection = ropeScaling.mropeSection,
           !mropeSection.isEmpty {
            self.mropeSection = mropeSection
            if ropeScaling.mropeInterleaved ?? false {
                self.rotaryEmb = ASRQwen3VLRotaryEmbedding(dim: headDim, base: config.ropeTheta, mropeSection: mropeSection)
                self.legacyRotaryEmb = nil
            } else {
                self.rotaryEmb = nil
                self.legacyRotaryEmb = ASRQwen2VLRotaryEmbedding(dim: headDim, base: config.ropeTheta)
            }
            self.rope = nil
        } else {
            self.mropeSection = nil
            self.rotaryEmb = nil
            self.legacyRotaryEmb = nil
            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta,
                scale: 1.0
            )
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape and normalize
        queries = qNorm(queries.reshaped(B, L, numHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Apply RoPE / MRoPE
        let offset = cache?.offset ?? 0
        if let rotaryEmb {
            let posIds: MLXArray
            if let positionIds {
                posIds = positionIds
            } else {
                let positions = MLXArray(Array(offset..<(offset + L)).map { Int32($0) })
                posIds = broadcast(positions.reshaped(1, L), to: [B, L])
            }
            let (cos, sin) = rotaryEmb.callAsFunction(positionIds: posIds, dtype: queries.dtype)
            (queries, keys) = applyRotaryPosEmb(queries, keys, cos: cos, sin: sin)
        } else if let legacyRotaryEmb, let mropeSection {
            let posIds: MLXArray
            if let positionIds {
                posIds = positionIds
            } else {
                let positions = MLXArray(Array(offset..<(offset + L)).map { Int32($0) })
                posIds = broadcast(positions.reshaped(1, L), to: [B, L])
            }
            let (cos, sin) = legacyRotaryEmb.callAsFunction(positionIds: posIds, dtype: queries.dtype)
            (queries, keys) = applyMultimodalRoPE(queries, keys, cos: cos, sin: sin, mropeSection: mropeSection)
        } else if let rope {
            queries = rope(queries.asType(.bfloat16), offset: offset)
            keys = rope(keys.asType(.bfloat16), offset: offset)
        }

        // Update cache
        if let cache = cache {
            let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
            keys = cachedKeys
            values = cachedValues
        }

        // Expand KV heads if needed
        if numKVHeads != numHeads {
            keys = expandKeyValue(keys, repeats: numKVGroups)
            values = expandKeyValue(values, repeats: numKVGroups)
        }

        // Attention (use float32 for stability, matches Qwen text encoder)
        let queriesF32 = queries.asType(.float32)
        let keysF32 = keys.asType(.float32)
        let valuesF32 = values.asType(.float32)
        var output = MLXFast.scaledDotProductAttention(
            queries: queriesF32,
            keys: keysF32,
            values: valuesF32,
            scale: scale,
            mask: mask
        )
        output = output.asType(queries.dtype)

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(output)
    }

    private func expandKeyValue(_ x: MLXArray, repeats: Int) -> MLXArray {
        guard repeats > 1 else { return x }
        var expanded = MLX.expandedDimensions(x, axis: 2)
        expanded = MLX.repeated(expanded, count: repeats, axis: 2)
        let shape = x.shape
        return expanded.reshaped(shape[0], shape[1] * repeats, shape[2], shape[3])
    }
}

// MARK: - MLP

final class Qwen3ASRMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3ASRDecoderConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
