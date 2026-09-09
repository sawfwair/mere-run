import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor

final class Gemma4Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let layerType: String
    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let windowSize: Int
    private let rope: any OffsetLayer
    private let rmsNormEps: Float
    private let isKVSharedLayer: Bool
    private let useKeyEqualsValue: Bool
    private var fusedQKV: FusedQuantizedProjection?
    private var fusedQKVAttempted = false
    private var compiledQKVSegment: Gemma4CompiledSegment?
    private var compiledQKVAttempted = false

    init(config: Gemma4TextConfig, layerIndex: Int, forceKVShared: Bool = false) {
        self.layerType = config.layerTypes[layerIndex]
        self.numHeads = config.numAttentionHeads
        self.windowSize = config.slidingWindow
        self.rmsNormEps = config.rmsNormEps

        let isFullAttention = layerType == "full_attention"
        self.headDim = isFullAttention ? (config.globalHeadDim ?? config.headDim) : config.headDim
        self.numKVHeads = isFullAttention ? (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads) : config.numKeyValueHeads
        self.isKVSharedLayer = forceKVShared || (config.numKVSharedLayers > 0 && layerIndex >= (config.numHiddenLayers - config.numKVSharedLayers))
        self.useKeyEqualsValue = config.attentionKEqV && isFullAttention

        let ropeConfig = config.ropeParameters[layerType] ?? config.ropeParameters["sliding_attention"]
        let ropeBase = ropeConfig?.ropeTheta ?? 10_000
        let partial = max(0, min(1, ropeConfig?.partialRotaryFactor ?? 1))
        let ropeType = ropeConfig?.ropeType ?? "default"
        if ropeType == "proportional" {
            self.rope = Gemma4ProportionalRoPE(
                dims: headDim,
                traditional: false,
                base: ropeBase,
                partialRotaryFactor: partial
            )
        } else {
            let ropeDims = max(1, Int(Float(headDim) * partial))
            self.rope = RoPE(
                dimensions: ropeDims,
                traditional: false,
                base: ropeBase
            )
        }

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = useKeyEqualsValue
            ? nil
            : Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        visionBlockIDs: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let sequenceLength = x.dim(1)
        let offset = cache?.offset ?? 0

        func preRopeQueryHeads(_ rawQueries: MLXArray) -> MLXArray {
            let shaped = rawQueries.reshaped(batchSize, sequenceLength, numHeads, headDim)
            return qNorm(shaped).transposed(0, 2, 1, 3)
        }

        func preRopeKeyValueHeads(rawKeys: MLXArray, rawValues: MLXArray) -> (MLXArray, MLXArray) {
            let normedKeys = kNorm(rawKeys.reshaped(batchSize, sequenceLength, numKVHeads, headDim))
            let normedValues = gemma4RMSNormNoScale(
                rawValues.reshaped(batchSize, sequenceLength, numKVHeads, headDim),
                eps: rmsNormEps
            )
            return (normedKeys.transposed(0, 2, 1, 3), normedValues.transposed(0, 2, 1, 3))
        }

        func finishedQueries(_ rawQueries: MLXArray) -> MLXArray {
            rope(preRopeQueryHeads(rawQueries), offset: offset)
        }

        func finishedKeyValues(rawKeys: MLXArray, rawValues: MLXArray) -> (MLXArray, MLXArray) {
            let (preRopeKeys, computedValues) = preRopeKeyValueHeads(rawKeys: rawKeys, rawValues: rawValues)
            return (rope(preRopeKeys, offset: offset), computedValues)
        }

        let scale: Float = 1.0
        let repeats = max(1, numHeads / max(1, numKVHeads))

        let queries: MLXArray
        let keys: MLXArray
        let values: MLXArray
        if isKVSharedLayer, let cache {
            queries = finishedQueries(qProj(x))
            if sequenceLength == 1,
               let attended = cache.specializedAttention(queries: queries, repeats: repeats, scale: scale) {
                let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
                return oProj(reshaped)
            }
            if let shared = sequenceLength == 1 ? cache.decodeState() : cache.currentState() {
                keys = shared.0
                values = shared.1
            } else {
                let rawKeys = kProj(x)
                let rawValues = useKeyEqualsValue ? rawKeys : vProj!(x)
                let (computedKeys, computedValues) = finishedKeyValues(rawKeys: rawKeys, rawValues: rawValues)

                cache.append(keys: computedKeys, values: computedValues)
                let updated = cache.currentState()!
                keys = updated.0
                values = updated.1
            }
        } else {
            let computedKeys: MLXArray
            let computedValues: MLXArray
            if let fusedHeads = fusedDecodeQKV(x, sequenceLength: sequenceLength) {
                queries = rope(fusedHeads.0, offset: offset)
                computedKeys = rope(fusedHeads.1, offset: offset)
                computedValues = fusedHeads.2
            } else if let compiled = resolvedCompiledQKVSegment(sequenceLength: sequenceLength) {
                let parts = compiled.function([x])
                queries = rope(parts[0], offset: offset)
                computedKeys = rope(parts[1], offset: offset)
                computedValues = parts[2]
            } else {
                let rawQueries: MLXArray
                let rawKeys: MLXArray
                let rawValues: MLXArray
                if let fused = resolvedFusedQKV() {
                    let parts = fused.callSplit(x)
                    rawQueries = parts[0]
                    rawKeys = parts[1]
                    rawValues = useKeyEqualsValue ? parts[1] : parts[2]
                } else {
                    rawQueries = qProj(x)
                    rawKeys = kProj(x)
                    rawValues = useKeyEqualsValue ? rawKeys : vProj!(x)
                }
                queries = finishedQueries(rawQueries)
                let keyValues = finishedKeyValues(rawKeys: rawKeys, rawValues: rawValues)
                computedKeys = keyValues.0
                computedValues = keyValues.1
            }

            if let cache {
                cache.append(keys: computedKeys, values: computedValues)
                if sequenceLength == 1,
                   let attended = cache.specializedAttention(queries: queries, repeats: repeats, scale: scale) {
                    let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
                    return oProj(reshaped)
                }
                let updated = (sequenceLength == 1 ? cache.decodeState() : cache.currentState())!
                keys = updated.0
                values = updated.1
            } else {
                keys = computedKeys
                values = computedValues
            }
        }

        let mask = makeAttentionMask(
            queryLength: sequenceLength,
            queryOffset: offset,
            keyLength: keys.dim(2),
            windowSize: layerType == "sliding_attention" ? windowSize : nil,
            dtype: x.dtype,
            visionBlockIDs: visionBlockIDs,
            keyPaddingMask: attentionMask
        )

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
        return oProj(reshaped)
    }

    private func resolvedFusedQKV() -> FusedQuantizedProjection? {
        guard Gemma4FusedProjectionPolicy.enabled else { return nil }
        let sources: [Linear?] = useKeyEqualsValue ? [qProj, kProj] : [qProj, kProj, vProj]
        if let fused = fusedQKV {
            if fused.matches(sources) { return fused }
            fusedQKV = nil
            fusedQKVAttempted = false
            compiledQKVSegment = nil
            compiledQKVAttempted = false
        }
        if !fusedQKVAttempted {
            fusedQKVAttempted = true
            fusedQKV = FusedQuantizedProjection.fuse(sources)
        }
        return fusedQKV
    }

    /// Fused-kernel decode path: one quantized matmul over the concatenated
    /// QKV weights, then a single Metal kernel that splits heads and applies
    /// qNorm, kNorm, and the value no-scale norm in transposed layout.
    private func fusedDecodeQKV(
        _ x: MLXArray,
        sequenceLength: Int
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard Gemma4FusedProjectionPolicy.fusedDecodeKernelsEnabled,
              sequenceLength == 1,
              !useKeyEqualsValue else {
            return nil
        }
        guard let fused = resolvedFusedQKV() else { return nil }
        return Gemma4DecodeFusedKernels.qkvNorms(
            qkv: fused.callFused(x),
            qNormWeight: qNorm.weight,
            kNormWeight: kNorm.weight,
            eps: Gemma4DecodeScalarCache.epsilon(rmsNormEps),
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim
        )
    }

    /// Compiled decode segment: fused QKV matmul, head reshape, q/k norms, and
    /// the value no-scale norm — everything between the layer input and RoPE.
    /// RoPE stays outside because its integer offset changes every token, which
    /// would force a retrace per position.
    private func resolvedCompiledQKVSegment(sequenceLength: Int) -> Gemma4CompiledSegment? {
        guard Gemma4FusedProjectionPolicy.compiledSegmentsEnabled,
              sequenceLength == 1,
              !useKeyEqualsValue else {
            return nil
        }
        guard let fused = resolvedFusedQKV() else { return nil }
        let fingerprint = [
            ObjectIdentifier(fused),
            ObjectIdentifier(qNorm),
            ObjectIdentifier(kNorm),
        ]
        if let segment = compiledQKVSegment {
            if segment.matches(fingerprint) { return segment }
            compiledQKVSegment = nil
            compiledQKVAttempted = false
        }
        if !compiledQKVAttempted {
            compiledQKVAttempted = true
            let numHeads = self.numHeads
            let numKVHeads = self.numKVHeads
            let headDim = self.headDim
            let eps = self.rmsNormEps
            let qNorm = self.qNorm
            let kNorm = self.kNorm
            let function = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
                let x = inputs[0]
                let batch = x.dim(0)
                let sequence = x.dim(1)
                let parts = fused.callSplit(x)
                let queries = qNorm(parts[0].reshaped(batch, sequence, numHeads, headDim))
                    .transposed(0, 2, 1, 3)
                let keys = kNorm(parts[1].reshaped(batch, sequence, numKVHeads, headDim))
                    .transposed(0, 2, 1, 3)
                let values = gemma4RMSNormNoScale(
                    parts[2].reshaped(batch, sequence, numKVHeads, headDim),
                    eps: eps
                ).transposed(0, 2, 1, 3)
                return [queries, keys, values]
            }
            compiledQKVSegment = Gemma4CompiledSegment(function: function, fingerprint: fingerprint)
        }
        return compiledQKVSegment
    }

    private func makeAttentionMask(
        queryLength: Int,
        queryOffset: Int,
        keyLength: Int,
        windowSize: Int?,
        dtype: DType,
        visionBlockIDs: MLXArray?,
        keyPaddingMask: MLXArray?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard queryLength > 1 || keyPaddingMask != nil else {
            return .none
        }

        let keyStart = max(0, queryOffset + queryLength - keyLength)
        let queryPositions = MLXArray(Int32(queryOffset)..<Int32(queryOffset + queryLength)).reshaped(queryLength, 1)
        let keyPositions = MLXArray(Int32(keyStart)..<Int32(keyStart + keyLength)).reshaped(1, keyLength)

        var allowed = keyPositions .<= queryPositions
        if let windowSize {
            allowed = allowed .&& (keyPositions .> (queryPositions - Int32(windowSize)))
        }
        if let keyPaddingMask,
           queryOffset == 0,
           keyStart == 0,
           keyPaddingMask.dim(-1) == keyLength {
            let padding = keyPaddingMask
                .asType(.int32)
                .reshaped(-1, 1, 1, keyLength)
            allowed = allowed.reshaped(1, 1, queryLength, keyLength)
                .&& (padding .> MLXArray(Int32(0)))
        }
        if let visionBlockIDs,
           queryOffset == 0,
           keyStart == 0,
           visionBlockIDs.dim(0) == queryLength,
           keyLength == queryLength {
            let queryBlocks = visionBlockIDs.reshaped(queryLength, 1)
            let keyBlocks = visionBlockIDs.reshaped(1, keyLength)
            let sameVisionBlock = (queryBlocks .>= MLXArray(Int32(0))) .&& (queryBlocks .== keyBlocks)
            allowed = (allowed.asType(.int32) + sameVisionBlock.asType(.int32)) .> MLXArray(Int32(0))
        }

        let allowedTyped = allowed.asType(dtype).reshaped(-1, 1, queryLength, keyLength)
        let zeros = MLXArray.zeros(allowedTyped.shape, dtype: dtype)
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(allowedTyped .> MLXArray(0).asType(dtype), zeros, negative))
    }
}
