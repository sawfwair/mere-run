import Foundation
import MLX
import MLXFast
import MLXNN
import XCTest
@testable import MereRunGemmaModel

extension SOVDecoderKernelProfileTests {
    /// Diagnostic barriers isolate operators; these are not resident inference timings.
    func profileAttention(layer: Gemma4DecoderLayer, input: MLXArray, cache: Gemma4AttentionCache,
                          config: Gemma4TextConfig, layerIndex: Int, prefix: String) throws {
        let attention = layer.selfAttention
        let full = config.layerTypes[layerIndex] == "full_attention"
        let dim = full ? (config.globalHeadDim ?? config.headDim) : config.headDim
        let heads = full ? (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads) : config.numKeyValueHeads
        let count = input.dim(1), offset = cache.offset
        var times: [String: [Double]] = [:]
        func timed(_ name: String, _ operation: () -> [MLXArray]) -> [MLXArray] {
            var result: [MLXArray] = []
            for iteration in 0..<7 {
                let start = ProcessInfo.processInfo.systemUptime
                result = operation(); eval(result)
                if iteration >= 2 { times[name, default: []].append(ProcessInfo.processInfo.systemUptime - start) }
            }
            return result
        }
        eval(input)
        let normalized = timed("inputNorm") { [layer.inputLayerNorm(input)] }[0]
        let projected = timed("projections") {
            let key = attention.kProj(normalized)
            return [attention.qProj(normalized), key, full && config.attentionKEqV ? key : attention.vProj!(normalized)]
        }
        let normed = timed("headNorms") {
            [attention.qNorm(projected[0].reshaped(1, count, config.numAttentionHeads, dim)).transposed(0, 2, 1, 3),
             attention.kNorm(projected[1].reshaped(1, count, heads, dim)).transposed(0, 2, 1, 3),
             gemma4RMSNormNoScale(projected[2].reshaped(1, count, heads, dim), eps: config.rmsNormEps).transposed(0, 2, 1, 3)]
        }
        let params = config.ropeParameters[config.layerTypes[layerIndex]]!
        let rope: any OffsetLayer = full
            ? Gemma4ProportionalRoPE(dims: dim, base: params.ropeTheta, partialRotaryFactor: params.partialRotaryFactor ?? 1)
            : RoPE(dimensions: dim, traditional: false, base: params.ropeTheta)
        let rotated = timed("rope") { [rope(normed[0], offset: offset), rope(normed[1], offset: offset)] }
        let kv = timed("cacheAppend") {
            let view = Gemma4ForwardAttentionCache(cache.fork())
            view.append(keys: rotated[1], values: normed[2])
            let state = view.currentState()!
            return [state.0, state.1]
        }
        let mask = timed("mask") {
            let positions = MLXArray(Int32(offset)..<Int32(offset + count)).reshaped(count, 1)
            let start = max(0, offset + count - kv[0].dim(2))
            let keys = MLXArray(Int32(start)..<Int32(start + kv[0].dim(2))).reshaped(1, kv[0].dim(2))
            var allowed = keys .<= positions
            if !full { allowed = allowed .&& (keys .> (positions - Int32(config.slidingWindow))) }
            let typed = allowed.asType(input.dtype).reshaped(1, 1, count, kv[0].dim(2))
            let zeros = MLXArray.zeros(typed.shape, dtype: input.dtype)
            return [MLX.where(typed .> MLXArray(0).asType(input.dtype), zeros, zeros + MLXArray(-1e9).asType(input.dtype))]
        }[0]
        let attended = timed("sdpa") {
            [MLXFast.scaledDotProductAttention(queries: rotated[0], keys: kv[0], values: kv[1], scale: 1, mask: .array(mask))]
        }[0]
        let output = timed("outputProjection") {
            [attention.oProj(attended.transposed(0, 2, 1, 3).reshaped(1, count, config.numAttentionHeads * dim))]
        }[0]
        let expected = attention(normalized, cache: Gemma4ForwardAttentionCache(cache.fork()))
        XCTAssertEqual(abs(expected - output).max().item(Float.self), 0)
        let url = prefix + "-attention-\(layerIndex)"
        try MLX.save(arrays: ["input": normalized, "queries": rotated[0], "keys": kv[0], "values": kv[1], "mask": mask,
                             "q_raw": projected[0], "k_raw": projected[1], "v_raw": projected[2], "q_norm": attention.qNorm.weight,
                             "k_norm": attention.kNorm.weight, "q_pre_rope": normed[0], "k_pre_rope": normed[1]],
                     url: URL(fileURLWithPath: url + ".safetensors"))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(times).write(to: URL(fileURLWithPath: url + ".json"))
    }
}
