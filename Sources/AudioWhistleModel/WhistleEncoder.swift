import Foundation
import MLX

package struct WhistleAudioContext {
    package let embedding: MLXArray
    let keys: [MLXArray]
    let values: [MLXArray]
}

package final class WhistleModel {
    let weights: WhistleWeights

    package init(weights: WhistleWeights) { self.weights = weights }

    /// Native FP32 reference path over the original checkpoint. Input is [frames, 80].
    package func encode(_ mel: MLXArray) throws -> WhistleAudioContext {
        guard mel.ndim == 2, mel.dim(1) == 80, mel.dim(0) > 0, mel.dim(0) <= 3000 else {
            throw WhistleError.invalid("expected 1...3000 frames of 80 mel bins")
        }
        var x = mel.reshaped(1, mel.dim(0), 80, 1)
        x = WhistleMath.silu(conv2d(x, weights("stem/w").transposed(3, 0, 1, 2), stride: 2, padding: 1))
        for stage in 1...2 {
            x = conv2d(x, weights("stem/dw_\(stage)").transposed(3, 0, 1, 2), stride: 2, padding: 1, groups: 128)
            x = WhistleMath.silu(weights.project(x, "stem/pw_\(stage)/kernel"))
        }
        // Training flattens channels before frequency, not the native NHWC order.
        x = weights.project(x.transposed(0, 1, 3, 2).reshaped(-1, 1280), "stem/out/kernel")
        var stream = broadcast(x.expandedDimensions(axis: 1), to: [x.dim(0), 4, 512])
        for layer in 0..<8 {
            try Task.checkCancellation()
            stream = WhistleMath.advance(stream, weights: weights, prefix: "encoder/", layer: layer) {
                self.encoderBlock($0, layer: layer)
            }
            eval(stream)
        }
        x = WhistleMath.norm(mean(stream, axis: 1), weights("encoder/final_norm/scale"))
        let frequency = MLXArray((0..<256).map { exp(-log(Float(10000)) * Float($0) / 255) })
        let phase = MLXArray((0..<x.dim(0)).map(Float.init)).reshaped(-1, 1) * frequency
        x = x + sigmoid(weights("pe_gate")) * concatenated([sin(phase), cos(phase)], axis: -1)
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for layer in 0..<8 {
            let prefix = "stack/layers/block/cross_attn/"
            let key = weights.project(x, prefix + "k_proj/kernel", layer: layer)
                .reshaped(-1, 8, 48).transposed(1, 0, 2).expandedDimensions(axis: 0)
            keys.append(WhistleMath.norm(key, weights(prefix + "k_norm/scale", layer: layer)))
            values.append(weights.project(x, prefix + "v_proj/kernel", layer: layer)
                .reshaped(-1, 8, 64).transposed(1, 0, 2).expandedDimensions(axis: 0))
        }
        eval(x, keys, values)
        return WhistleAudioContext(embedding: x, keys: keys, values: values)
    }

    private func encoderBlock(_ input: MLXArray, layer: Int) -> MLXArray {
        let prefix = "encoder/layers/block/"
        func weight(_ name: String) -> MLXArray { weights(prefix + name, layer: layer) }
        func norm(_ x: MLXArray, _ name: String) -> MLXArray { WhistleMath.norm(x, weight(name + "/scale")) }
        var x = input + 0.5 * WhistleMath.hadamard(
            norm(input, "pre_hada_norm_0"), weights: weights, prefix: prefix + "hadamard_mlp_0/", layer: layer
        )
        let normalized = norm(x, "ZCRMSNorm_0")
        let attention = selfAttention(normalized, prefix: prefix + "self_attn/", layer: layer)
        x = x + sigmoid(weight("attn_gate")) * norm(attention, "post_attn_norm")
        let projected = weights.project(norm(x, "conv_norm"), prefix + "pw1/kernel", layer: layer)
        let gated = projected[.ellipsis, 0..<512] * sigmoid(projected[.ellipsis, 512..<1024])
        let convolved = conv1d(gated.expandedDimensions(axis: 0), weight("dw").transposed(2, 0, 1), padding: 4, groups: 512)[0]
        x = x + weights.project(WhistleMath.silu(norm(convolved, "conv_out_norm")), prefix + "pw2/kernel", layer: layer)
        return x + 0.5 * WhistleMath.hadamard(
            norm(x, "pre_hada_norm"), weights: weights, prefix: prefix + "hadamard_mlp/", layer: layer
        )
    }

    private func selfAttention(_ x: MLXArray, prefix: String, layer: Int) -> MLXArray {
        func projection(_ name: String, heads: Int, dim: Int) -> MLXArray {
            weights.project(x, prefix + name + "_proj/kernel", layer: layer)
                .reshaped(-1, heads, dim).transposed(1, 0, 2).expandedDimensions(axis: 0)
        }
        let q = WhistleMath.rope(WhistleMath.norm(projection("q", heads: 8, dim: 48), weights(prefix + "q_norm/scale", layer: layer)))
        let k = WhistleMath.rope(WhistleMath.norm(projection("k", heads: 2, dim: 48), weights(prefix + "k_norm/scale", layer: layer)))
        let v = projection("v", heads: 2, dim: 64)
        let output = WhistleMath.attention(q, k, v) * sigmoid(weights.project(x, prefix + "gate_proj/kernel", layer: layer))
        return weights.project(output, prefix + "out_proj/kernel", layer: layer)
    }
}
