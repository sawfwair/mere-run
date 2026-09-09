import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package final class LagunaAttention: Module {
    @ModuleInfo(key: "q_proj") package var qProj: Linear
    @ModuleInfo(key: "k_proj") package var kProj: Linear
    @ModuleInfo(key: "v_proj") package var vProj: Linear
    @ModuleInfo(key: "o_proj") package var oProj: Linear
    @ModuleInfo(key: "g_proj") package var gProj: Linear?
    @ModuleInfo(key: "q_norm") package var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") package var kNorm: RMSNorm

    let headCount: Int
    let keyValueHeadCount: Int
    let headDim: Int
    let scale: Float
    let slidingWindow: Int?
    let gatePerHead: Bool
    let rope: LagunaRoPE
    let layerIndex: Int
    let isTerminalLayer: Bool
    var _nativeAffineQKV: LagunaNativeAffineWeight?
    var _nativeAffineOProj: LagunaNativeAffineWeight?
    var _nativeAffineGProj: LagunaNativeAffineWeight?
    var _nativeAffineQKVGateRows = 0
    var _terminalPrefillQGateWeight: MLXArray?
    var _terminalPrefillKVWeight: MLXArray?

    package init(config: LagunaConfig, layerIndex: Int) {
        self.layerIndex = layerIndex
        self.headCount = config.attentionHeads(layerIndex: layerIndex)
        self.keyValueHeadCount = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.slidingWindow = config.layerTypes[layerIndex] == "sliding_attention"
            ? config.slidingWindow
            : nil
        self.gatePerHead = config.gating == "per-head"
        self.isTerminalLayer = layerIndex == config.numHiddenLayers - 1
        self.rope = LagunaRoPE(
            headDim: config.headDim,
            parameters: config.ropeParameters(layerIndex: layerIndex)
        )

        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            headCount * config.headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            keyValueHeadCount * config.headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            keyValueHeadCount * config.headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(headCount * config.headDim, config.hiddenSize, bias: false)
        if config.gating == "none" || config.gating == "false" {
            self._gProj.wrappedValue = nil
        } else {
            let gateDimensions = gatePerHead ? headCount : headCount * config.headDim
            self._gProj.wrappedValue = Linear(config.hiddenSize, gateDimensions, bias: false)
        }
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    func projectOutput(_ input: MLXArray, useCustomKernels: Bool) -> MLXArray {
        if useCustomKernels,
           input.shape == [1, 1, headCount * headDim],
           input.dtype == .bfloat16,
           let affine = _nativeAffineOProj,
           affine.originalShape == [2_048, headCount * headDim] {
            return MLX.quantizedMM(
                input,
                affine.packedCodes,
                scales: affine.scales,
                biases: affine.biases,
                transpose: true,
                groupSize: 32,
                bits: 8,
                mode: .affine
            )
        }
        return oProj(input)
    }

    package func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        precomputedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        precomputedRoPEAtlas: MLXArray? = nil,
        residualForFusedQKV: MLXArray? = nil,
        rmsNormWeight: MLXArray? = nil,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let offset = cache?.offset ?? 0
        let positionOffsets = (cache as? LagunaRaggedKVCache)?.positionOffsets
            ?? Array(repeating: offset, count: batch)

        let rawQueries: MLXArray
        let rawKeys: MLXArray
        var values: MLXArray
        let bankedGate: MLXArray?
        let queryDimensions = headCount * headDim
        let keyValueDimensions = keyValueHeadCount * headDim
        let bankedQKV: MLXArray? = {
            guard useCustomKernels,
                  batch == 1,
                  sequenceLength == 1,
                  x.dtype == .bfloat16,
                  x.shape == [1, 1, 2_048],
                  let affine = _nativeAffineQKV,
                  affine.originalShape == [
                      queryDimensions + 2 * keyValueDimensions
                          + _nativeAffineQKVGateRows,
                      2_048,
                  ] else {
                return nil
            }
            if LagunaGraphAccelerationPolicy.fusedNormAffineQKVEnabled,
               let residualForFusedQKV,
               let rmsNormWeight,
               let fused = LagunaNormAffineQKV.call(
                   residual: residualForFusedQKV,
                   normWeight: rmsNormWeight,
                   codes: affine.packedCodes,
                   scales: affine.scales,
                   biases: affine.biases,
                   heads: headCount,
                   gateRows: _nativeAffineQKVGateRows
               ) {
                return fused
            }
            return MLX.quantizedMM(
                x,
                affine.packedCodes,
                scales: affine.scales,
                biases: affine.biases,
                transpose: true,
                groupSize: 32,
                bits: 8,
                mode: .affine
            )
        }()
        if let qkv = bankedQKV {
            rawQueries = qkv[.ellipsis, 0..<queryDimensions]
            rawKeys = qkv[
                .ellipsis,
                queryDimensions..<(queryDimensions + keyValueDimensions)
            ]
            values = qkv[
                .ellipsis,
                (queryDimensions + keyValueDimensions)..<(queryDimensions + 2 * keyValueDimensions)
            ].reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
            if _nativeAffineQKVGateRows == headCount {
                let gateStart = queryDimensions + 2 * keyValueDimensions
                bankedGate = qkv[.ellipsis, gateStart..<(gateStart + headCount)]
            } else {
                bankedGate = nil
            }
        } else {
            rawQueries = qProj(x)
            rawKeys = kProj(x)
            values = vProj(x).reshaped(
                batch,
                sequenceLength,
                keyValueHeadCount,
                headDim
            )
            bankedGate = nil
        }
        var queries: MLXArray
        var keys: MLXArray
        let fusedQK: (queries: MLXArray, keys: MLXArray)? =
            useCustomKernels
                && LagunaGraphAccelerationPolicy.prefillQKNormRoPEEnabled
            ? rope.prefillFusionKind.flatMap { kind in
                guard let precomputedRoPEAtlas,
                      batch == 1,
                      positionOffsets.allSatisfy({ $0 == offset }) else {
                    return nil
                }
                return LagunaFusedPrefill.qkNormRoPE(
                    kind: kind,
                    rawQueries: rawQueries,
                    rawKeys: rawKeys,
                    queryWeight: qNorm.weight,
                    keyWeight: kNorm.weight,
                    angleAtlas: precomputedRoPEAtlas,
                    offset: offset,
                    length: sequenceLength
                )
            } : nil
        if let fusedQK {
            queries = fusedQK.queries
            keys = fusedQK.keys
        } else {
            queries = rawQueries.reshaped(batch, sequenceLength, headCount, headDim)
            keys = rawKeys.reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            queries = rope(queries, offsets: positionOffsets)
            keys = rope(keys, offsets: positionOffsets)
        }
        values = values.transposed(0, 2, 1, 3)

        var keyLengths: [Int]?
        if let cache {
            let state = cache.attentionState(appending: keys, values: values)
            keys = state!.0
            values = state!.1
            keyLengths = (cache as? LagunaRaggedKVCache)?.lastAttentionKeyLengths
        }

        let mask = precomputedMask ?? attentionMask(
            queryLength: sequenceLength,
            queryOffsets: positionOffsets,
            keyLengths: keyLengths ?? Array(repeating: keys.dim(2), count: batch),
            keyLength: keys.dim(2),
            dtype: x.dtype
        )
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3)

        if let gProj {
            let projectedGate: MLXArray
            if let bankedGate {
                projectedGate = bankedGate
            } else if useCustomKernels,
                      x.shape == [1, 1, 2_048],
                      x.dtype == .bfloat16,
                      let affine = _nativeAffineGProj,
                      affine.originalShape == [headCount, 2_048] {
                projectedGate = MLX.quantizedMM(
                    x,
                    affine.packedCodes,
                    scales: affine.scales,
                    biases: affine.biases,
                    transpose: true,
                    groupSize: 32,
                    bits: 8,
                    mode: .affine
                )
            } else {
                projectedGate = gProj(x)
            }
            let flattenedOutput = output.reshaped(batch, sequenceLength, -1)
            if useCustomKernels,
               LagunaGraphAccelerationPolicy.fusedGatedAffineOProjEnabled,
               gatePerHead,
               let affine = _nativeAffineOProj,
               let projected = LagunaGatedAffineOProj.call(
                   attentionOutput: flattenedOutput,
                   gateLogits: projectedGate,
                   codes: affine.packedCodes,
                   scales: affine.scales,
                   biases: affine.biases,
                   heads: headCount
               ) {
                return projected
            }
            let gate = MLXNN.softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output = output * MLX.expandedDimensions(gate, axis: gate.ndim)
            } else {
                output = flattenedOutput * gate
                return projectOutput(output, useCustomKernels: useCustomKernels)
            }
        }
        return projectOutput(
            output.reshaped(batch, sequenceLength, -1),
            useCustomKernels: useCustomKernels
        )
    }

    /// Final-layer multi-token specialization for callers that consume only
    /// the last hidden row. All K/V rows are produced and committed; Q,
    /// attention output, gating, and O projection run only for the final row.
    package func callLastPrefillRow(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        useProjectionBanks: Bool? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        precondition(batch == 1 && sequenceLength > 1)

        let offset = cache?.offset ?? 0
        let lastInput = x[0..., (sequenceLength - 1)..., 0...]
        let queryDimensions = headCount * headDim
        let keyValueDimensions = keyValueHeadCount * headDim

        let rawQueries: MLXArray
        let rawKeys: MLXArray
        var values: MLXArray
        let bankedGate: MLXArray?
        let useProjectionBanks = useProjectionBanks
            ?? LagunaGraphAccelerationPolicy.terminalPrefillProjectionBanksEnabled
        if useProjectionBanks,
           let queryGateWeight = _terminalPrefillQGateWeight,
           let keyValueWeight = _terminalPrefillKVWeight,
           x.dtype == .bfloat16,
           lastInput.dtype == .bfloat16,
           queryGateWeight.dtype == .bfloat16,
           keyValueWeight.dtype == .bfloat16,
           queryGateWeight.shape == [queryDimensions + headCount, 2_048],
           keyValueWeight.shape == [2 * keyValueDimensions, 2_048] {
            let queryGate = matmul(lastInput, queryGateWeight.T)
            rawQueries = queryGate[.ellipsis, 0..<queryDimensions]
            bankedGate = queryGate[
                .ellipsis,
                queryDimensions..<(queryDimensions + headCount)
            ]

            let keyValue = matmul(x, keyValueWeight.T)
            rawKeys = keyValue[.ellipsis, 0..<keyValueDimensions]
            values = keyValue[
                .ellipsis,
                keyValueDimensions..<(2 * keyValueDimensions)
            ].reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        } else {
            rawQueries = qProj(lastInput)
            rawKeys = kProj(x)
            values = vProj(x).reshaped(
                batch,
                sequenceLength,
                keyValueHeadCount,
                headDim
            )
            bankedGate = nil
        }

        var queries = qNorm(
            rawQueries.reshaped(batch, 1, headCount, headDim)
        ).transposed(0, 2, 1, 3)
        var keys = kNorm(
            rawKeys.reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        ).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset + sequenceLength - 1)
        keys = rope(keys, offset: offset)

        var keyLengths: [Int]?
        if let cache {
            let state = cache.attentionState(appending: keys, values: values)
            keys = state!.0
            values = state!.1
            keyLengths = (cache as? LagunaRaggedKVCache)?.lastAttentionKeyLengths
        }
        let queryOffset = offset + sequenceLength - 1
        let mask = attentionMask(
            queryLength: 1,
            queryOffsets: [queryOffset],
            keyLengths: keyLengths ?? [keys.dim(2)],
            keyLength: keys.dim(2),
            dtype: x.dtype
        )
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3)

        if let gProj {
            let projectedGate = bankedGate ?? gProj(lastInput)
            let gate = MLXNN.softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output = output * MLX.expandedDimensions(gate, axis: gate.ndim)
            } else {
                output = output.reshaped(batch, 1, -1) * gate
                return oProj(output)
            }
        }
        return oProj(output.reshaped(batch, 1, -1))
    }

    package func prefillMask(
        queryLength: Int,
        cache: Gemma4AttentionCache?,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let queryOffset = cache?.offset ?? 0
        let previousKeyLength = cache?.currentState()?.0.dim(2) ?? 0
        let keyLength = previousKeyLength + queryLength
        return attentionMask(
            queryLength: queryLength,
            queryOffsets: [queryOffset],
            keyLengths: [keyLength],
            keyLength: keyLength,
            dtype: dtype
        )
    }

    package func prefillRoPEAngleAtlas(length: Int) -> MLXArray? {
        rope.angleAtlas(length: length)
    }

    func attentionMask(
        queryLength: Int,
        queryOffsets: [Int],
        keyLengths: [Int],
        keyLength: Int,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(queryOffsets.count == keyLengths.count)
        guard queryLength > 1 || keyLengths.contains(where: { $0 != keyLength }) else {
            return .none
        }
        if queryOffsets.count == 1, keyLengths[0] == keyLength {
            let queryOffset = queryOffsets[0]
            let keyStart = max(0, queryOffset + queryLength - keyLength)
            let queryPositions = MLXArray(
                Int32(queryOffset)..<Int32(queryOffset + queryLength)
            ).reshaped(queryLength, 1)
            let keyPositions = MLXArray(
                Int32(keyStart)..<Int32(keyStart + keyLength)
            ).reshaped(1, keyLength)
            var allowed = keyPositions .<= queryPositions
            if let slidingWindow {
                allowed = allowed
                    .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
            }
            let typed = allowed.asType(dtype).reshaped(1, 1, queryLength, keyLength)
            let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
            let negative = zeros + MLXArray(-1e9).asType(dtype)
            return .array(MLX.where(
                typed .> MLXArray(0).asType(dtype),
                zeros,
                negative
            ))
        }

        let rowMasks = zip(queryOffsets, keyLengths).map { queryOffset, validKeyLength in
            let keyStart = max(0, queryOffset + queryLength - validKeyLength)
            let queryPositions = MLXArray(
                Int32(queryOffset)..<Int32(queryOffset + queryLength)
            ).reshaped(queryLength, 1)
            let keyIndices = MLXArray(Int32(0)..<Int32(keyLength)).reshaped(1, keyLength)
            let keyPositions = keyIndices + Int32(keyStart)
            var allowed = (keyIndices .< Int32(validKeyLength))
                .&& (keyPositions .<= queryPositions)
            if let slidingWindow {
                allowed = allowed
                    .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
            }
            return allowed
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
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(allowed, zeros, negative))
    }
}
