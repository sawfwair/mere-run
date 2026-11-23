import Foundation
import MLX

enum Inkling2Math {
    static func rms(_ x: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
        let f = x.asType(.float32)
        return f * rsqrt(square(f).mean(axis: -1, keepDims: true) + epsilon)
    }

    static func clip(_ x: MLXArray, scale: MLXArray? = nil, threshold: Float) -> MLXArray {
        let f = x.asType(.float32) * (scale?.asType(.float32) ?? MLXArray(Float(1)))
        let norm = sqrt(square(f).sum(axis: -1, keepDims: true))
        return (f * minimum(threshold / (norm + 1e-6), MLXArray(Float(1)))).asType(x.dtype)
    }

    static func softcap(_ x: MLXArray, _ cap: Float) -> MLXArray { tanh(x / cap) * cap }

    static func activation(gate: MLXArray, up: MLXArray, gateCap: Float, upCap: Float) -> MLXArray {
        let g = gate.asType(.float32)
        return (softcap(g, gateCap) * sigmoid(g) * softcap(up.asType(.float32), upCap)).asType(gate.dtype)
    }

    static func groupedAttention(
        query: MLXArray, keys: MLXArray, values: MLXArray, bias: MLXArray, allowed: MLXArray, sinks: MLXArray
    ) -> MLXArray {
        let scores = groupedAttentionDots(query: query, keys: keys) / sqrt(Float(query.dim(-1))) + bias
        let masked = MLX.where(allowed, scores, MLXArray(-Float.infinity))
        let sink = sinks.transposed(0, 2, 1).expandedDimensions(axis: -1).asType(.float32)
        let probabilities = softmax(concatenated([masked, sink], axis: -1), axis: -1)[.ellipsis, ..<keys.dim(2)]
        return groupedAttentionValues(probabilities.asType(values.dtype), values: values)
    }

    static func groupedAttentionDots(query: MLXArray, keys: MLXArray) -> MLXArray {
        let batch = query.dim(0), tokens = query.dim(1), heads = query.dim(2), width = query.dim(3)
        let kvHeads = keys.dim(1), keyCount = keys.dim(2), rowsPerGroup = heads / kvHeads * tokens
        // Stack the queries that share one KV head as matrix rows, so both matmuls can read the original KV cache.
        let rows = query.transposed(0, 2, 1, 3).reshaped(batch, kvHeads, rowsPerGroup, width).asType(.float32)
        let keyMatrix = keys.asType(.float32).swappedAxes(-1, -2)
        return matmul(rows, keyMatrix).reshaped(batch, heads, tokens, keyCount)
    }

    static func groupedAttentionValues(_ probabilities: MLXArray, values: MLXArray) -> MLXArray {
        let batch = probabilities.dim(0), heads = probabilities.dim(1), tokens = probabilities.dim(2)
        let kvHeads = values.dim(1), keyCount = values.dim(2), width = values.dim(3)
        let rows = probabilities.reshaped(batch, kvHeads, heads / kvHeads * tokens, keyCount)
        return matmul(rows, values).reshaped(batch, heads, tokens, width)
            .transposed(0, 2, 1, 3).reshaped(batch, tokens, heads * width)
    }

    /// Values as well as keys are normalized before attending over depth.
    static func residual(_ blocks: [MLXArray], partial: MLXArray, query: MLXArray, fusedDecode: Bool = false) -> [MLXArray] {
        #if os(macOS)
        if fusedDecode, blocks.count <= 4 {
            return Inkling2DecodeResidual.apply(blocks, partial: partial, query: query)
        }
        #endif
        let values = rms(stacked(blocks + [partial], axis: -2))
        let q = query.asType(.float32).reshaped(-1, partial.dim(-1))
        let scores = matmul(q, values.swappedAxes(-1, -2)) / sqrt(Float(partial.dim(-1)))
        let result = matmul(softmax(scores, axis: -1), values)
        return (0..<q.dim(0)).map { result[.ellipsis, $0, 0...] }
    }

    static func e4m3(_ byte: UInt8) -> Float {
        let e = Int((byte >> 3) & 15), m = Int(byte & 7)
        let sign: Float = byte & 128 == 0 ? 1 : -1
        if e == 15 && m == 7 { return .nan }
        return sign * (e == 0 ? Float(m) * pow(2, -9) : (1 + Float(m) / 8) * pow(2, Float(e - 7)))
    }

    static func decodeFP8(_ bytes: MLXArray) -> MLXArray {
        MLXArray((0..<256).map { e4m3(UInt8($0)) })[bytes.asType(.int32)]
    }

    static func decodeNVFP4(_ bytes: MLXArray, scales: MLXArray, global: MLXArray) -> MLXArray {
        let lut = MLXArray([Float(0), 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
        let low = bitwiseAnd(bytes, MLXArray(UInt8(15))).asType(.int32)
        let high = rightShift(bytes, MLXArray(UInt8(4))).asType(.int32)
        let shape = Array(bytes.shape.dropLast()) + [bytes.dim(-1) * 2]
        let values = stacked([lut[low], lut[high]], axis: -1).reshaped(shape)
        let blocked = values.reshaped(Array(shape.dropLast()) + [shape.last! / 16, 16])
        return (blocked * decodeFP8(scales).expandedDimensions(axis: -1) * global).reshaped(shape)
    }

    static func decodeMXFP8(_ bytes: MLXArray, scales: MLXArray) -> MLXArray {
        let scale = pow(MLXArray(Float(2)), scales.asType(.float32) - 127)
        let shape = bytes.shape
        return (decodeFP8(bytes).reshaped(Array(shape.dropLast()) + [shape.last! / 32, 32])
                * scale.expandedDimensions(axis: -1)).reshaped(shape)
    }
}
