// Adapted from Gradium xn-ptts 49b9033e44ab0fbb935973eca3213757644d1bbb.
// MIT license; see THIRD_PARTY_NOTICES.md.
import Foundation
import MLX
import MLXFast
import MLXNN

struct PhononWeights {
    let arrays: [String: MLXArray]
    func tensor(_ name: String, _ shape: [Int]) throws -> MLXArray {
        guard let value = arrays[name], value.shape == shape,
              value.dtype == .float32 || value.dtype == .bfloat16 || value.dtype == .float16 else {
            throw PhononError.invalidAsset("\(name) must be a floating tensor with shape \(shape)")
        }
        return value.asType(.float32)
    }
}

struct PhononLinear {
    let weight: MLXArray
    let bias: MLXArray?
    init(_ weights: PhononWeights, _ prefix: String, _ input: Int, _ output: Int, bias: Bool = false) throws {
        weight = try weights.tensor(prefix + ".weight", [output, input])
        self.bias = try bias ? weights.tensor(prefix + ".bias", [output]) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = matmul(x, weight.T)
        return bias.map { y + $0 } ?? y
    }
}

struct PhononNorm {
    let weight, bias: MLXArray
    let epsilon: Float
    init(_ weights: PhononWeights, _ prefix: String, _ width: Int, epsilon: Float = 1e-5) throws {
        weight = try weights.tensor(prefix + ".weight", [width])
        bias = try weights.tensor(prefix + ".bias", [width])
        self.epsilon = epsilon
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: epsilon)
    }
}

struct PhononAttentionState {
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0
}

struct PhononTransformer {
    struct Layer {
        let norm1, norm2: PhononNorm
        let inProjection, outProjection, linear1, linear2: PhononLinear
        let scale1, scale2: MLXArray?
        init(_ weights: PhononWeights, _ prefix: String, width: Int, feedforward: Int, scaled: Bool) throws {
            norm1 = try PhononNorm(weights, prefix + ".norm1", width)
            norm2 = try PhononNorm(weights, prefix + ".norm2", width)
            inProjection = try PhononLinear(weights, prefix + ".self_attn.in_proj", width, 3 * width)
            outProjection = try PhononLinear(weights, prefix + ".self_attn.out_proj", width, width)
            linear1 = try PhononLinear(weights, prefix + ".linear1", width, feedforward)
            linear2 = try PhononLinear(weights, prefix + ".linear2", feedforward, width)
            scale1 = try scaled ? weights.tensor(prefix + ".layer_scale_1.scale", [width]) : nil
            scale2 = try scaled ? weights.tensor(prefix + ".layer_scale_2.scale", [width]) : nil
        }
    }
    let layers: [Layer]
    let heads: Int
    let period: Float
    let context: Int?

    init(_ weights: PhononWeights, _ prefix: String, width: Int, heads: Int, count: Int,
         feedforward: Int, period: Float, context: Int? = nil) throws {
        self.heads = heads
        self.period = period
        self.context = context
        layers = try (0..<count).map {
            try Layer(weights, prefix + ".layers.\($0)", width: width,
                      feedforward: feedforward, scaled: context != nil)
        }
    }

    func callAsFunction(_ input: MLXArray, state: inout [PhononAttentionState]) -> MLXArray {
        var x = input
        let length = x.dim(1), width = x.dim(2), headWidth = width / heads
        for (index, layer) in layers.enumerated() {
            let projection = split(layer.inProjection(layer.norm1(x)), parts: 3, axis: -1)
            let offset = state[index].offset
            let qkv = projection.map { $0.reshaped(1, length, heads, headWidth).transposed(0, 2, 1, 3) }
            let queries = MLXFast.RoPE(qkv[0], dimensions: headWidth, traditional: true, base: period, scale: 1, offset: offset)
            let newKeys = MLXFast.RoPE(qkv[1], dimensions: headWidth, traditional: true, base: period, scale: 1, offset: offset)
            let keys = state[index].keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
            let values = state[index].values.map { concatenated([$0, qkv[2]], axis: 2) } ?? qkv[2]
            let previous = keys.dim(2) - length
            let mask: MLXArray?
            if length > 1 {
                let queryPositions = MLXArray(0..<length).reshaped(length, 1) + previous
                let keyPositions = MLXArray(0..<keys.dim(2)).reshaped(1, keys.dim(2))
                var allowed = keyPositions .<= queryPositions
                if let context { allowed = allowed .&& (keyPositions .>= (queryPositions - context)) }
                mask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
            } else { mask = nil }
            let attention = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: 1 / sqrt(Float(headWidth)), mask: mask
            ).transposed(0, 2, 1, 3).reshaped(1, length, width)
            let attended = layer.outProjection(attention)
            x = x + (layer.scale1.map { attended * $0 } ?? attended)
            let feedforward = layer.linear2(gelu(layer.linear1(layer.norm2(x))))
            x = x + (layer.scale2.map { feedforward * $0 } ?? feedforward)
            let start = context.map { max(0, keys.dim(2) - $0) } ?? 0
            state[index].keys = keys[0..., 0..., start..., 0...]
            state[index].values = values[0..., 0..., start..., 0...]
            state[index].offset += length
            eval(keys, values)
        }
        return x
    }
    func makeState() -> [PhononAttentionState] { layers.map { _ in PhononAttentionState() } }
}
