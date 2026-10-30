import Foundation
import MLX
import MLXFast
import MLXNN

struct IdolLayers {
    let weights: IdolWeights
    let heads: Int

    func norm(_ x: MLXArray, _ name: String? = nil, eps: Float = 1e-6) -> MLXArray {
        let result = MLXFast.layerNorm(x.asType(.float32),
            weight: name.map { weights[$0 + ".weight"] }, bias: name.map { weights[$0 + ".bias"] }, eps: eps)
        return result.asType(x.dtype)
    }

    /// Torch F.normalize uses an L2-norm floor of 1e-12, not RMSNorm epsilon.
    func headNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        let full = x.asType(.float32)
        let length = maximum(sqrt(sum(full * full, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return (full / length * weights[name] * sqrt(Float(x.dim(-1)))).asType(x.dtype)
    }

    func ffn(_ x: MLXArray, _ name: String) -> MLXArray {
        weights.linear(geluApproximate(weights.linear(x, name + ".mlp.0", half: true)), name + ".mlp.2", half: true)
    }

    func scalarEmbedding(_ values: MLXArray, _ name: String, period: Float = 10000, twoPi: Bool = false) -> MLXArray {
        let freqs = exp(-log(period) * MLXArray(0..<128).asType(.float32) / 128)
        let scale: Float = twoPi ? 2 * Float.pi : 1
        let angles = values.reshaped(-1, 1) * freqs * scale
        return weights.linear(silu(weights.linear(concatenated([cos(angles), sin(angles)], axis: -1), name + ".mlp.0")),
                              name + ".mlp.2")
    }

    static func positionEmbedding(_ points: MLXArray, width: Int, decoder: Bool) -> MLXArray {
        let count = width / 3 / 2
        let frequencies: [Float]
        if decoder {
            frequencies = (0..<count).map { pow(2, count == 1 ? 0 : Float($0) / Float(count - 1) * 10) }
        } else {
            let residual = max(0, count - 16)
            frequencies = Array(((0..<16).map { pow(2, Float($0)) }
                + (0..<residual).map { pow(2, Float($0) / Float(residual) * 16) }).prefix(count))
        }
        let angles = points.expandedDimensions(axis: -1) * MLXArray(frequencies) * (decoder ? Float.pi : 2 * Float.pi)
        let embedded = concatenated([sin(angles), cos(angles)], axis: -1)
            .reshaped(points.dim(0), points.dim(1), 6 * count)
        return concatenated([embedded, zeros([points.dim(0), points.dim(1), width - 6 * count])], axis: -1)
    }

    func rotaryAngles(_ x: MLXArray, _ name: String) -> MLXArray {
        let h = norm(x.asType(.float32), name + ".norm", eps: 1e-5)
        let offsets = weights.linear(silu(weights.linear(h, name + ".gate_map"))
            * weights.linear(h, name + ".content_map"), name + ".final_map")
            .reshaped(x.dim(0), x.dim(1), heads, 3)
        let angles = (0..<3).map { axis in
            offsets[0..., 0..., 0..., axis].expandedDimensions(axis: -1) * weights[name + ".freqs_\(axis)"] * Float.pi
        }
        return concatenated(angles, axis: -1)
    }

    static func rotate(_ x: MLXArray, angles: MLXArray) -> MLXArray {
        let pairs = x.asType(.float32).reshaped(x.dim(0), x.dim(1), x.dim(2), -1, 2)
        let first = pairs[0..., 0..., 0..., 0..., 0], second = pairs[0..., 0..., 0..., 0..., 1]
        return stacked([first * cos(angles) - second * sin(angles), first * sin(angles) + second * cos(angles)], axis: -1)
            .reshaped(x.shape).asType(x.dtype)
    }

    func attention(_ x: MLXArray, _ name: String, context: MLXArray? = nil, angles: MLXArray? = nil) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), width = x.dim(2), dim = width / heads
        let query: MLXArray, key: MLXArray, value: MLXArray
        let outputName: String
        if let angles {
            let qkv = weights.linear(x, name + ".qkv", half: true).reshaped(batch, length, 3, heads, dim)
            query = headNorm(Self.rotate(qkv[0..., 0..., 0, 0..., 0...], angles: angles), name + ".q_norm.gamma")
            key = headNorm(Self.rotate(qkv[0..., 0..., 1, 0..., 0...], angles: angles), name + ".k_norm.gamma")
            value = qkv[0..., 0..., 2, 0..., 0...]
            outputName = ".out"
        } else if let context {
            query = headNorm(weights.linear(x, name + ".to_q", half: true).reshaped(batch, length, heads, dim),
                             name + ".q_rms_norm.gamma")
            let kv = weights.linear(context, name + ".to_kv", half: true).reshaped(batch, context.dim(1), 2, heads, dim)
            key = headNorm(kv[0..., 0..., 0, 0..., 0...], name + ".k_rms_norm.gamma")
            value = kv[0..., 0..., 1, 0..., 0...]
            outputName = ".to_out"
        } else {
            let qkv = weights.linear(x, name + ".to_qkv", half: true).reshaped(batch, length, 3, heads, dim)
            query = headNorm(qkv[0..., 0..., 0, 0..., 0...], name + ".q_rms_norm.gamma")
            key = headNorm(qkv[0..., 0..., 1, 0..., 0...], name + ".k_rms_norm.gamma")
            value = qkv[0..., 0..., 2, 0..., 0...]
            outputName = ".to_out"
        }
        let attended = MLXFast.scaledDotProductAttention(queries: query.transposed(0, 2, 1, 3),
            keys: key.transposed(0, 2, 1, 3), values: value.transposed(0, 2, 1, 3), scale: 1 / sqrt(Float(dim)), mask: .none)
        return weights.linear(attended.transposed(0, 2, 1, 3).reshaped(batch, length, width), name + outputName, half: true)
    }

    func modulatedBlock(_ x: MLXArray, _ name: String, modulation: MLXArray,
                        context: MLXArray? = nil, angles: MLXArray? = nil) -> MLXArray {
        let mods = split(modulation, parts: 6, axis: -1).map { $0.expandedDimensions(axis: 1) }
        let h = norm(x) * (1 + mods[1]) + mods[0]
        let afterAttention = x + attention(h, name + (context == nil ? ".attn" : ".cross_attn"),
                                           context: context, angles: angles) * mods[2]
        return afterAttention + ffn(norm(afterAttention) * (1 + mods[4]) + mods[3], name + ".mlp") * mods[5]
    }
}
