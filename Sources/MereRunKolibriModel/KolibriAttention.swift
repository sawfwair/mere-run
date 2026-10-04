import Foundation
import MLX
import MLXFast
import MLXNN

/// Sliding layers retain W-1 past rows while absolute RoPE positions keep advancing.
package final class KolibriCache {
    package private(set) var offset = 0
    package private(set) var keys: MLXArray?
    package private(set) var values: MLXArray?
    let window: Int?
    package init(window: Int?) { self.window = window }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let allKeys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        let allValues = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        offset += newKeys.dim(2)
        let keep = min(allKeys.dim(2), window.map { $0 - 1 } ?? allKeys.dim(2))
        keys = contiguous(allKeys[.ellipsis, (allKeys.dim(2) - keep)..., 0...])
        values = contiguous(allValues[.ellipsis, (allValues.dim(2) - keep)..., 0...])
        return (allKeys, allValues)
    }
}

package final class KolibriAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: KolibriProjection
    @ModuleInfo(key: "k_proj") var kProj: KolibriProjection
    @ModuleInfo(key: "v_proj") var vProj: KolibriProjection
    @ModuleInfo(key: "o_proj") var oProj: KolibriProjection
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let config: KolibriConfiguration
    let window: Int?
    let rope: RoPE?

    init(config: KolibriConfiguration, index: Int) {
        self.config = config
        let path = "model.layers.\(index).self_attn"
        _qProj.wrappedValue = KolibriProjection(input: config.hiddenSize, output: config.numAttentionHeads * config.headDim,
                                               policy: config.quantization?[path + ".q_proj"])
        _kProj.wrappedValue = KolibriProjection(input: config.hiddenSize, output: config.numKeyValueHeads * config.headDim,
                                               policy: config.quantization?[path + ".k_proj"])
        _vProj.wrappedValue = KolibriProjection(input: config.hiddenSize, output: config.numKeyValueHeads * config.headDim,
                                               policy: config.quantization?[path + ".v_proj"])
        _oProj.wrappedValue = KolibriProjection(input: config.numAttentionHeads * config.headDim, output: config.hiddenSize,
                                               policy: config.quantization?[path + ".o_proj"])
        _qNorm.wrappedValue = kolibriRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = kolibriRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        window = config.layerTypes[index] == .sliding ? config.slidingWindow : nil
        rope = window == nil ? nil : RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KolibriCache?) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), offset = cache?.offset ?? 0
        var q = qNorm(qProj(x).reshaped(batch, length, config.numAttentionHeads, config.headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim).transposed(0, 2, 1, 3)
        if let rope { q = rope(q, offset: offset); k = rope(k, offset: offset) }
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let queryPositions = MLXArray(offset..<(offset + length)).expandedDimensions(axis: -1)
        let keyStart = offset + length - k.dim(2)
        let keyPositions = MLXArray(keyStart..<(keyStart + k.dim(2))).expandedDimensions(axis: 0)
        var allowed = keyPositions .<= queryPositions
        if let window { allowed = allowed .&& (keyPositions .> (queryPositions - window)) }
        let mask = which(allowed, MLXArray(0, dtype: x.dtype), MLXArray(-Float.infinity, dtype: x.dtype))
        let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                     scale: 1 / sqrt(Float(config.headDim)), mask: .array(mask))
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, config.numAttentionHeads * config.headDim))
    }
}
