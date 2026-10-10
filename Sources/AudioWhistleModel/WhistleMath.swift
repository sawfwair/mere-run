import Foundation
import MLX
import MLXFast

enum WhistleMath {
    static func unit(_ x: MLXArray) -> MLXArray {
        x * rsqrt(mean(x * x, axis: -1, keepDims: true) + 1e-6)
    }

    static func norm(_ x: MLXArray, _ scale: MLXArray) -> MLXArray {
        unit(x) * (1 + scale)
    }

    static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    static func kron(_ x: MLXArray, _ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let z = x.reshaped(-1, 16, 32)
        return matmul(matmul(a.T, z), b).reshaped(x.shape)
    }

    static func hadamard(_ x: MLXArray, weights: WhistleWeights, prefix: String, layer: Int) -> MLXArray {
        func weight(_ name: String) -> MLXArray { weights(prefix + name, layer: layer) }
        let conditioning = 1 + matmul(softmax(matmul(x, weight("cond_v")), axis: -1), weight("cond_u"))
        var z = kron(weight("d1") * x, weight("w1a"), weight("w1b"))
        z = take(z, weights.permutations[0], axis: -1)
        z = kron(silu(weight("d2") * conditioning * z + weight("b2")), weight("w2a"), weight("w2b"))
        z = take(z, weights.permutations[1], axis: -1)
        return weight("d4") * kron(weight("d3") * z, weight("w3a"), weight("w3b"))
    }

    /// Split-half rotary positions. Q/K have 48 channels; V has 64.
    static func rope(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let angles = MLXArray((offset..<(offset + x.dim(2))).map(Float.init)).reshaped(1, 1, -1, 1)
            * MLXArray((0..<24).map { pow(Float(100000), -Float($0 * 2) / 48) }).reshaped(1, 1, 1, 24)
        let first = x[.ellipsis, 0..<24]
        let second = x[.ellipsis, 24..<48]
        return concatenated([first * cos(angles) - second * sin(angles),
                             second * cos(angles) + first * sin(angles)], axis: -1)
    }

    static func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(48)), mask: .none)
            .transposed(0, 2, 1, 3).reshaped(-1, 512)
    }

    /// mHC mixes four residual lanes, with one favoured lane per physical layer.
    static func advance(
        _ stream: MLXArray, weights: WhistleWeights, prefix: String, layer: Int,
        block: (MLXArray) -> MLXArray
    ) -> MLXArray {
        func weight(_ name: String) -> MLXArray { weights(prefix + "mhc_" + name, layer: layer) }
        let nx = unit(stream.reshaped(-1, 2048))
        let lane = MLXArray((0..<4).map { Float($0 == layer % 4 ? 1 : 0) })
        let pre = sigmoid(weight("a_pre") * weights.project(nx, prefix + "mhc_phi_pre", layer: layer) + weight("b_pre") + 8 * lane - 4)
        let input = sum(pre.expandedDimensions(axis: -1) * stream, axis: 1)
        let delta = block(input) - input
        let post = 2 * sigmoid(weight("a_post") * weights.project(nx, prefix + "mhc_phi_post", layer: layer) + weight("b_post") - 4 * (1 - lane))
        var residual = weight("a_res") * weights.project(nx, prefix + "mhc_phi_res", layer: layer).reshaped(-1, 4, 4) + weight("b_res")
        for _ in 0..<20 {
            residual = residual - logSumExp(residual, axis: -1, keepDims: true)
            residual = residual - logSumExp(residual, axis: -2, keepDims: true)
        }
        return matmul(exp(residual), stream) + post.expandedDimensions(axis: -1) * delta.expandedDimensions(axis: 1)
    }
}
