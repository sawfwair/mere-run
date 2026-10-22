import Foundation
import MLX
import MLXFast
import MLXNN

/// Sliding layers retain W-1 past rows while absolute RoPE positions keep advancing.
package final class FlintCache {
    package private(set) var offset = 0
    private var keyStorage: MLXArray?
    private var valueStorage: MLXArray?
    package var keys: MLXArray? {
        capacity == nil ? keyStorage : keyStorage?[.ellipsis, ..<offset, 0...]
    }
    package var values: MLXArray? {
        capacity == nil ? valueStorage : valueStorage?[.ellipsis, ..<offset, 0...]
    }
    let window: Int?
    let capacity: Int?
    package init(window: Int?, capacity: Int? = nil) {
        precondition(capacity == nil || (window == nil && capacity! > 0))
        self.window = window
        self.capacity = capacity
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let capacity {
            let nextOffset = offset + newKeys.dim(2)
            precondition(nextOffset <= capacity, "Flint cache exceeds the request's token budget.")
            if keyStorage == nil {
                keyStorage = MLXArray.zeros([newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)], dtype: newKeys.dtype)
                valueStorage = MLXArray.zeros([newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)], dtype: newValues.dtype)
            }
            keyStorage?[.ellipsis, offset..<nextOffset, 0...] = newKeys
            valueStorage?[.ellipsis, offset..<nextOffset, 0...] = newValues
            offset = nextOffset
            return (keys!, values!)
        }
        let allKeys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        let allValues = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        offset += newKeys.dim(2)
        let keep = min(allKeys.dim(2), window.map { $0 - 1 } ?? allKeys.dim(2))
        keyStorage = contiguous(allKeys[.ellipsis, (allKeys.dim(2) - keep)..., 0...])
        valueStorage = contiguous(allValues[.ellipsis, (allValues.dim(2) - keep)..., 0...])
        return (allKeys, allValues)
    }
}

package final class FlintAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: FlintProjection
    @ModuleInfo(key: "k_proj") var kProj: FlintProjection
    @ModuleInfo(key: "v_proj") var vProj: FlintProjection
    @ModuleInfo(key: "o_proj") var oProj: FlintProjection
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var sinks: MLXArray
    let config: FlintConfiguration
    let window: Int?
    let rope: RoPE?

    init(config: FlintConfiguration, index: Int) {
        self.config = config
        let path = "model.layers.\(index).self_attn"
        _qProj.wrappedValue = FlintProjection(input: config.hiddenSize, output: config.numAttentionHeads * config.headDim,
                                               policy: config.policy(for: path + ".q_proj"))
        _kProj.wrappedValue = FlintProjection(input: config.hiddenSize, output: config.numKeyValueHeads * config.headDim,
                                               policy: config.policy(for: path + ".k_proj"))
        _vProj.wrappedValue = FlintProjection(input: config.hiddenSize, output: config.numKeyValueHeads * config.headDim,
                                               policy: config.policy(for: path + ".v_proj"))
        _oProj.wrappedValue = FlintProjection(input: config.numAttentionHeads * config.headDim, output: config.hiddenSize,
                                               policy: config.policy(for: path + ".o_proj"))
        _qNorm.wrappedValue = flintRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = flintRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _sinks.wrappedValue = MLXArray.zeros([config.numAttentionHeads], dtype: .bfloat16)
        window = config.layerTypes[index] == .sliding ? config.slidingWindow : nil
        rope = config.noRopeLayerIndices.contains(index) ? nil : RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: FlintCache?) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), offset = cache?.offset ?? 0
        var q = qNorm(qProj(x).reshaped(batch, length, config.numAttentionHeads, config.headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim).transposed(0, 2, 1, 3)
        if let rope { q = rope(q, offset: offset); k = rope(k, offset: offset) }
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if length == 1 {
            // Every populated cache row precedes this query and is within the
            // retained window. Avoid rebuilding an all-zero mask for each layer.
            mask = .none
        } else {
            let queryPositions = MLXArray(offset..<(offset + length)).expandedDimensions(axis: -1)
            let keyStart = offset + length - k.dim(2)
            let keyPositions = MLXArray(keyStart..<(keyStart + k.dim(2))).expandedDimensions(axis: 0)
            var allowed = keyPositions .<= queryPositions
            if let window { allowed = allowed .&& (keyPositions .> (queryPositions - window)) }
            mask = .array(which(allowed, MLXArray(0, dtype: x.dtype), MLXArray(-Float.infinity, dtype: x.dtype)))
        }
        let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                     scale: 1 / sqrt(Float(config.headDim)), mask: mask, sinks: sinks.asType(q.dtype))
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, config.numAttentionHeads * config.headDim))
    }
}
