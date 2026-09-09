import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

extension LagunaAttention {
    package func prepareNativeAffineQKV(
        enabled: Bool? = nil,
        includeGate: Bool? = nil
    ) -> [MLXArray] {
        guard _nativeAffineQKV == nil,
              enabled
                ?? LagunaGraphAccelerationPolicy.usesNativeAffineQKV(layerIndex: layerIndex),
              headDim == 128,
              keyValueHeadCount == 8,
              headCount == 48 || headCount == 64,
              type(of: qProj) == Linear.self,
              type(of: kProj) == Linear.self,
              type(of: vProj) == Linear.self,
              qProj.bias == nil,
              kProj.bias == nil,
              vProj.bias == nil,
              qProj.weight.shape == [headCount * headDim, 2_048],
              kProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              vProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              let query = lagunaNativeAffineWeight(qProj.weight),
              let key = lagunaNativeAffineWeight(kProj.weight),
              let value = lagunaNativeAffineWeight(vProj.weight) else {
            return []
        }
        let gate: LagunaNativeAffineWeight?
        let gateEnabled = includeGate
            ?? (
                LagunaGraphAccelerationPolicy.nativeAffineGProjFoldEnabled
                    && LagunaGraphAccelerationPolicy.usesNativeAffineGProj(
                        layerIndex: layerIndex
                    )
            )
        if gateEnabled,
           gatePerHead,
           let gProj,
           type(of: gProj) == Linear.self,
           gProj.bias == nil,
           gProj.weight.shape == [headCount, 2_048] {
            gate = lagunaNativeAffineWeight(gProj.weight)
        } else {
            gate = nil
        }
        var packedBlocks = [query.packedCodes, key.packedCodes, value.packedCodes]
        var scaleBlocks = [query.scales, key.scales, value.scales]
        var biasBlocks = [query.biases, key.biases, value.biases]
        if let gate {
            packedBlocks.append(gate.packedCodes)
            scaleBlocks.append(gate.scales)
            biasBlocks.append(gate.biases)
            _nativeAffineQKVGateRows = headCount
        }
        let fused = LagunaNativeAffineWeight(
            packedCodes: concatenated(packedBlocks, axis: 0),
            scales: concatenated(scaleBlocks, axis: 0),
            biases: concatenated(biasBlocks, axis: 0),
            originalShape: [
                qProj.weight.dim(0) + kProj.weight.dim(0) + vProj.weight.dim(0)
                    + _nativeAffineQKVGateRows,
                qProj.weight.dim(1),
            ]
        )
        _nativeAffineQKV = fused
        return fused.arrays
    }

    package func prepareNativeAffineOProj(enabled: Bool? = nil) -> [MLXArray] {
        let enabled = enabled
            ?? LagunaGraphAccelerationPolicy.usesNativeAffineOProj(layerIndex: layerIndex)
        guard enabled,
              _nativeAffineOProj == nil,
              type(of: oProj) == Linear.self,
              oProj.bias == nil,
              oProj.weight.shape == [2_048, headCount * headDim],
              let affine = lagunaNativeAffineWeight(oProj.weight) else {
            return []
        }
        _nativeAffineOProj = affine
        return affine.arrays
    }

    package func prepareNativeAffineGProj(enabled: Bool? = nil) -> [MLXArray] {
        let enabled = enabled
            ?? LagunaGraphAccelerationPolicy.usesNativeAffineGProj(layerIndex: layerIndex)
        guard enabled,
              _nativeAffineQKVGateRows == 0,
              _nativeAffineGProj == nil,
              gatePerHead,
              let gProj,
              type(of: gProj) == Linear.self,
              gProj.bias == nil,
              gProj.weight.shape == [headCount, 2_048],
              let affine = lagunaNativeAffineWeight(gProj.weight) else {
            return []
        }
        _nativeAffineGProj = affine
        return affine.arrays
    }

    /// Compile and execute the fused decode tail while model loading is still
    /// untimed. The custom Metal pipeline is keyed only by the Laguna head
    /// count, so one warm-up per 48/64-head family avoids charging the first
    /// user request for PSO construction without caching request data.
    package func prepareFusedGatedAffineOProjWarmUp() -> MLXArray? {
        guard LagunaGraphAccelerationPolicy.fusedGatedAffineOProjEnabled,
              gatePerHead,
              headCount == 48 || headCount == 64,
              let affine = _nativeAffineOProj else {
            return nil
        }
        return LagunaGatedAffineOProj.call(
            attentionOutput: MLXArray.zeros([1, 1, headCount * headDim], dtype: .bfloat16),
            gateLogits: MLXArray.zeros([1, 1, headCount], dtype: .bfloat16),
            codes: affine.packedCodes,
            scales: affine.scales,
            biases: affine.biases,
            heads: headCount
        )
    }

    package func prepareFusedNormAffineQKVWarmUp(
        normWeight: MLXArray
    ) -> (rows: Int, output: MLXArray)? {
        guard LagunaGraphAccelerationPolicy.fusedNormAffineQKVEnabled,
              let affine = _nativeAffineQKV else {
            return nil
        }
        let rows = (headCount + 2 * keyValueHeadCount) * headDim
            + _nativeAffineQKVGateRows
        guard let output = LagunaNormAffineQKV.call(
            residual: MLXArray.zeros([1, 1, 2_048], dtype: .bfloat16),
            normWeight: normWeight,
            codes: affine.packedCodes,
            scales: affine.scales,
            biases: affine.biases,
            heads: headCount,
            gateRows: _nativeAffineQKVGateRows
        ) else {
            return nil
        }
        return (rows, output)
    }

    /// Retain two bias-free BF16 side banks for the terminal XS prefill layer.
    /// The final query row and per-head gate share one input, while K/V still
    /// consume every supplied row so the cache advances normally. Concatenating
    /// output rows does not change any contraction or reduction order.
    package func prepareTerminalPrefillProjectionWeights(enabled: Bool? = nil) -> [MLXArray] {
        let enabled = enabled
            ?? (
                LagunaGraphAccelerationPolicy.terminalPrefillRowEnabled
                    && LagunaGraphAccelerationPolicy.terminalPrefillProjectionBanksEnabled
            )
        guard enabled,
              _terminalPrefillQGateWeight == nil,
              _terminalPrefillKVWeight == nil,
              isTerminalLayer,
              slidingWindow != nil,
              gatePerHead,
              headDim == 128,
              keyValueHeadCount == 8,
              headCount == 48 || headCount == 64,
              let gProj,
              type(of: qProj) == Linear.self,
              type(of: kProj) == Linear.self,
              type(of: vProj) == Linear.self,
              type(of: gProj) == Linear.self,
              qProj.bias == nil,
              kProj.bias == nil,
              vProj.bias == nil,
              gProj.bias == nil,
              qProj.weight.dtype == .bfloat16,
              kProj.weight.dtype == .bfloat16,
              vProj.weight.dtype == .bfloat16,
              gProj.weight.dtype == .bfloat16,
              qProj.weight.shape == [headCount * headDim, 2_048],
              kProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              vProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              gProj.weight.shape == [headCount, 2_048] else {
            return []
        }

        let queryGate = concatenated([qProj.weight, gProj.weight], axis: 0)
        let keyValue = concatenated([kProj.weight, vProj.weight], axis: 0)
        _terminalPrefillQGateWeight = queryGate
        _terminalPrefillKVWeight = keyValue
        return [queryGate, keyValue]
    }

    /// LoRA-wrapped projections must remain the only source of Q/K/V/O values.
    /// Discard retained base-weight layouts that would otherwise bypass them.
    package func invalidateTextLoRAUnsafeAcceleration() {
        _nativeAffineQKV = nil
        _nativeAffineOProj = nil
        _nativeAffineGProj = nil
        _nativeAffineQKVGateRows = 0
        _terminalPrefillQGateWeight = nil
        _terminalPrefillKVWeight = nil
    }

}
